import copy
from dataclasses import dataclass
from typing import Any, Callable

import torch
import torch.nn as nn
from einops import rearrange
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torchdiffeq import odeint
from transformers import Qwen2Config, Qwen2ForCausalLM

from dots_tts.models.dots_tts.config import ModelConfig
from dots_tts.modules.backbone.dit import DiT
from dots_tts.modules.backbone.encoder import VAESemanticEncoder
from dots_tts.modules.voice_design import (
    VoiceDesignBranch,
    VoiceDesignOutput,
    gather_voice_hidden,
)
from dots_tts.utils.tokenizer import (
    AUDIO_COMP_SPAN_TOKEN,
    PROSODY_THINK_TOKEN,
    AUDIO_GEN_SPAN_TOKEN,
    TEXT_COND_END_TOKEN,
    VOICE_GEN_START_TOKEN,
    VOICE_PATCH_TOKEN,
    VOICE_THINK_TOKEN,
    has_token,
    require_token_id,
)
from dots_tts.utils.util import get_mask_from_lengths, mask_data


def prosody_segment_bounds(
    patches: torch.Tensor, num_slots: int, patches_per_slot: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """``[B]`` patch counts -> per-slot ``[lo, hi)`` patch ranges, ``[B, M]``.

    The single definition of how an utterance is cut into plan segments.
    Three places need it and they must agree exactly: the regression target
    (which patches a slot is scored on), the loss mask (which slots exist,
    derived from input_ids one optimizer step earlier), and the acoustic
    conditioning (which DiT positions a slot modulates). Two of them
    disagreeing is silent -- the loss is simply normalized over a different
    set of slots than it was computed on -- so they all come from here.

    Segments are an ABSOLUTE stride, not a fraction of the utterance::

        seg(n) = min(n // S, M - 1)
        lo(m)  = min(m * S, P)
        hi(m)  = P if m == M - 1 else min((m + 1) * S, P)

    A relative split (slot m = the m-th M-th of the utterance) needs the total
    patch count P, and at inference P does not exist: generation stops on EOS,
    while the plan slots are read during prefill before a single patch has been
    produced. An absolute stride needs only "which patch is this", which is
    known at every step of both training and generation. The last slot absorbs
    whatever is left, so utterances longer than ``M * S`` degrade by holding
    their final plan rather than by running off the end.
    """

    if num_slots <= 0:
        raise ValueError(f"num_slots must be positive, got {num_slots}.")
    if patches_per_slot <= 0:
        raise ValueError(f"patches_per_slot must be positive, got {patches_per_slot}.")
    slots = torch.arange(num_slots, device=patches.device).reshape(1, -1).long()
    counts = patches.reshape(-1, 1).long()
    stride = int(patches_per_slot)
    low = torch.minimum(slots * stride, counts)
    high = torch.minimum((slots + 1) * stride, counts)
    # The last slot runs to the end of the utterance, matching the clamp in
    # seg(n): every patch at or past (M-1)*S maps to it.
    high = torch.where(slots.eq(num_slots - 1), counts.expand_as(high), high)
    return low, high


def _gather_slot_hiddens(
    hidden_states: torch.Tensor,
    slot_mask: torch.Tensor,
    num_slots: int,
    token_name: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pack a row's ``num_slots`` slot hidden states, ``[B, K, H]`` + ``[B]``.

    A row carries either all of them or none. A partial count means the
    tokenized data and the model disagree on the layout, which is silent
    everywhere else -- the sequence is simply shifted -- so it is caught
    here rather than diagnosed from bad audio a day later.
    """

    if num_slots <= 0:
        raise RuntimeError(f"This model has no {token_name} slots.")
    if hidden_states.dim() != 3:
        raise ValueError(
            "Expected [batch, seq, hidden] hidden states, got "
            f"{tuple(hidden_states.shape)}."
        )
    batch_size, _, hidden_size = hidden_states.shape
    slot_mask = slot_mask.to(device=hidden_states.device).bool()
    counts = slot_mask.sum(dim=1)
    unexpected = counts[(counts != 0) & (counts != num_slots)]
    if unexpected.numel() > 0:
        raise ValueError(
            f"Every sample must carry 0 or {num_slots} {token_name}; got "
            f"counts {sorted(set(unexpected.tolist()))}. The tokenized data "
            "and the model disagree on the slot layout."
        )
    packed = hidden_states.new_zeros((batch_size, num_slots, hidden_size))
    row_mask = counts.eq(num_slots)
    for index in range(batch_size):
        if bool(row_mask[index]):
            packed[index] = hidden_states[index][slot_mask[index]]
    return packed, row_mask


@dataclass(frozen=True)
class DotsTtsForwardOutput:
    llm_logits: torch.Tensor
    pred: torch.Tensor
    target: torch.Tensor
    eos_out: torch.Tensor
    voice_design: VoiceDesignOutput | None = None
    voice_metrics: dict[str, float] | None = None
    # Per-refinement-step squared error of the think chain, ``[B, K]``.
    # None when the model carries no think slots.
    voice_think_loss: torch.Tensor | None = None
    # Per-segment plan error off the post-text slots, ``[B, M]``.
    voice_prosody_loss: torch.Tensor | None = None


class DotsTtsCore(nn.Module):
    # region Module construction
    def __init__(
        self,
        config: ModelConfig,
        llm_config: Qwen2Config,
        tokenizer=None,
        *,
        latent_stats_path,
    ):
        super().__init__()
        self.config = config
        self.fm_hidden_size = config.DiT.hidden_size
        self.hidden_patch_size = 1
        self.cfg_droprate = config.get("cfg_droprate", 0.2)
        self.latent_patch_size = config.patch_size
        self.latent_dim = config.latent_dim
        self.xvec_dim = config.campplus_embedding_size
        self.xvec_drop_rate = config.get("xvec_drop_rate", 0.2)

        # Setup tokenizer
        self.tokenizer = tokenizer
        if self.tokenizer is None:
            raise RuntimeError("Tokenizer must be provided before building the model.")
        if llm_config is None:
            raise RuntimeError("LLM config must be provided before building the model.")
        self.pad_token_id = getattr(self.tokenizer, "pad_token_id", None)
        self.audio_gen_span_id = require_token_id(self.tokenizer, AUDIO_GEN_SPAN_TOKEN)
        self.audio_comp_span_id = require_token_id(
            self.tokenizer, AUDIO_COMP_SPAN_TOKEN
        )
        self.text_cond_end_id = require_token_id(self.tokenizer, TEXT_COND_END_TOKEN)

        # Setup LLM with language modeling head so we can obtain logits directly
        llm_config = copy.deepcopy(llm_config)
        llm_config.vocab_size = len(self.tokenizer)
        self.llm = Qwen2ForCausalLM._from_config(
            llm_config,
            dtype=torch.float32,
        )
        self.llm_hidden_size = self.llm.config.hidden_size

        self.patch_encoder = VAESemanticEncoder(
            in_dim=self.latent_dim,
            out_dim=self.llm_hidden_size,
            config=config,
        )

        # Setup Flow matching related modules
        self.hidden_proj = nn.Linear(self.llm_hidden_size, self.fm_hidden_size)
        self.latent_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.coordinate_proj = nn.Linear(self.latent_dim, self.fm_hidden_size)
        self.xvec_proj = nn.Sequential(
            nn.Linear(self.xvec_dim, self.fm_hidden_size),
            nn.LayerNorm(self.fm_hidden_size),
        )
        self.meanflow_config = config.meanflow if config.meanflow is not None else None
        self.mode = (
            "meanflow"
            if self.meanflow_config is not None and self.meanflow_config.enabled
            else "flow_matching"
        )
        dit_mode = (
            "meanflow"
            if self.mode == "meanflow"
            and self.meanflow_config.use_duration_embedding
            else "flow_matching"
        )
        self.velocity_field_predictor = DiT(
            in_dim=self.fm_hidden_size,
            out_dim=self.latent_dim,
            transformer_config=config.DiT,
            mode=dit_mode,
        )

        # Setup eos predictor
        self.eos_proj = nn.Sequential(
            nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
            nn.SiLU(),
            nn.Linear(self.llm_hidden_size, 2),
        )

        # Voice-design branch (instruction -> voice code -> g_cond).
        voice_design_config = config.get("voice_design", None)
        self.voice_design_enabled = bool(
            voice_design_config is not None and voice_design_config.enabled
        )
        self.instruction_residual_only = bool(
            self.voice_design_enabled and voice_design_config.instruction_residual_only
        )
        self.voice_gen_start_id: int | None = None
        self.voice_patch_id: int | None = None
        self.voice_design: VoiceDesignBranch | None = None
        self.voice_proj: nn.Linear | None = None
        self.voice_embed_proj: nn.Linear | None = None
        self.instruction_g_proj: nn.Sequential | None = None
        self.semantic_patch_proj: nn.Sequential | None = None
        self.voice_think_id: int | None = None
        self.num_think_slots: int = 0
        self.think_head: nn.Module | None = None
        self.think_g_proj: nn.Linear | None = None
        self.prosody_think_id: int | None = None
        self.num_prosody_slots: int = 0
        self.prosody_head: nn.Module | None = None
        self.plan_g_proj: nn.Linear | None = None
        self.prosody_patches_per_slot: int = 4
        self.g_cond_source: str = "xvector"
        if self.voice_design_enabled:
            for token in (VOICE_GEN_START_TOKEN, VOICE_PATCH_TOKEN):
                if not has_token(self.tokenizer, token):
                    raise RuntimeError(
                        "voice_design is enabled but the tokenizer has no "
                        f"{token}. Extend the artifact tokenizer with "
                        "add_voice_design_tokens() and resize the LLM embedding "
                        "before building the model."
                    )
            self.voice_gen_start_id = require_token_id(
                self.tokenizer, VOICE_GEN_START_TOKEN
            )
            self.voice_patch_id = require_token_id(self.tokenizer, VOICE_PATCH_TOKEN)
            voice_dim = (
                0
                if voice_design_config.voice_source == "none"
                else int(voice_design_config.voice_dim)
                if voice_design_config.voice_source == "qformer"
                else self.latent_dim
            )
            self.voice_design = None if self.instruction_residual_only else VoiceDesignBranch(
                voice_design_config,
                lm_hidden_size=self.llm_hidden_size,
                anchor_dim=self.xvec_dim,
                voice_dim=voice_dim,
                latent_dim=self.latent_dim,
            )
            self.voice_patch_dropout = float(
                getattr(voice_design_config, "voice_patch_dropout", 0.0) or 0.0
            )
            # The QFormer half of the latent has no pretrained consumer, so it
            # gets its own projection into g_cond. Zero-initialized: at step 0
            # the model behaves exactly as if only the CAM++ anchor existed, and
            # this channel opens only as far as the acoustic loss pays for it.
            if voice_dim and not self.instruction_residual_only:
                self.voice_proj = nn.Linear(voice_dim, self.fm_hidden_size)
                nn.init.zeros_(self.voice_proj.weight)
                nn.init.zeros_(self.voice_proj.bias)
            # The voice latent's route into the LM context. This is the analogue
            # of patch_encoder for audio patches: it turns the (normalized) code
            # into the input embedding at <|voice_patch|>, which is what lets
            # every text and audio position downstream attend to the voice that
            # was actually chosen.
            self.voice_embed_proj = None if self.instruction_residual_only else nn.Linear(
                self.voice_design.code_dim, self.llm_hidden_size
            )

            # --- think chain ---------------------------------------------
            # K ordinary token positions between the instruction and
            # <|voice_gen_start|>. Slot t predicts an increment to the voice
            # code and the running sum is scored against the same teacher
            # code the branch uses, so the K slots form a coarse-to-fine
            # refinement of one timbre estimate rather than K copies of it.
            #
            # Nothing is injected back into the inputs: the think positions
            # attend causally to each other, so slot t already sees slot
            # t-1's hidden state. That is what keeps the prefill a two-chunk
            # pass and inference the same shape as before.
            #
            # The slots sit BEFORE the text, so they cannot know what will
            # be said. That is why the target is timbre (time-invariant) and
            # not a per-segment acoustic plan: the latter is not predictable
            # from this position and would collapse onto its own mean.
            self.num_think_slots = int(
                getattr(voice_design_config, "num_think_slots", 0) or 0
            )
            if self.num_think_slots > 0:
                if not has_token(self.tokenizer, VOICE_THINK_TOKEN):
                    raise RuntimeError(
                        "num_think_slots > 0 but the tokenizer has no "
                        f"{VOICE_THINK_TOKEN}. Extend the artifact tokenizer "
                        "with add_voice_design_tokens() before building."
                    )
                self.voice_think_id = require_token_id(
                    self.tokenizer, VOICE_THINK_TOKEN
                )
                think_code_dim = self.voice_design.code_dim
                self.think_head = nn.Sequential(
                    nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.llm_hidden_size, think_code_dim),
                )
                # Only the downstream projection is zero-initialized. If both
                # linear maps start at zero, acoustic loss cannot train either
                # input-dependent map when the auxiliary think loss is off.
                nn.init.normal_(self.think_head[-1].weight, std=0.01)
                nn.init.zeros_(self.think_head[-1].bias)
                # The chain's route into the acoustic condition. Zero-init
                # for the same reason as voice_proj, and additive so the
                # existing anchor path is untouched.
                self.think_g_proj = nn.Linear(think_code_dim, self.fm_hidden_size)
                nn.init.zeros_(self.think_g_proj.weight)
                nn.init.zeros_(self.think_g_proj.bias)

            # --- post-text plan slots ------------------------------------
            # M slots between the text and the first audio token. Slot m
            # regresses onto the mean normalized latent of segment m of the
            # target audio, and its prediction modulates the DiT at exactly
            # the positions that segment occupies.
            #
            # This replaces the single predicted voice code rather than
            # supplementing it. That code was one vector per utterance whose
            # teacher was a time-pooled speaker summary, so nothing an
            # instruction says about how a delivery *moves* could survive
            # it; and its presence let the acoustic head lean on a global
            # constant instead of developing the per-position path. These M
            # vectors are the acoustic representation, they are positioned in
            # time, and the timbre is their time-average -- one mechanism
            # where there were two competing ones.
            #
            # They sit after the text on purpose: a pre-text slot cannot know
            # what is about to be said, so the only thing it can carry is
            # something time-invariant.
            self.num_prosody_slots = int(
                getattr(voice_design_config, "num_prosody_slots", 0) or 0
            )
            self.prosody_patches_per_slot = int(
                getattr(voice_design_config, "prosody_patches_per_slot", 4) or 4
            )
            # "xvector": g_cond is the CAM++ anchor path, as it has always
            # been. "plan": every x-vector-derived term is scaled out and the
            # plan is the only global conditioning, so its time-average is
            # the timbre. Cloning is unaffected either way -- it supplies its
            # own g_cond from the reference at inference.
            self.g_cond_source = str(
                getattr(voice_design_config, "g_cond_source", "xvector")
            )
            if self.num_prosody_slots > 0:
                if not has_token(self.tokenizer, PROSODY_THINK_TOKEN):
                    raise RuntimeError(
                        "num_prosody_slots > 0 but the tokenizer has no "
                        f"{PROSODY_THINK_TOKEN}. Extend the artifact "
                        "tokenizer with add_voice_design_tokens() first."
                    )
                self.prosody_think_id = require_token_id(
                    self.tokenizer, PROSODY_THINK_TOKEN
                )
                self.prosody_head = nn.Sequential(
                    nn.Linear(self.llm_hidden_size, self.llm_hidden_size),
                    nn.SiLU(),
                    nn.Linear(self.llm_hidden_size, self.latent_dim),
                )
                # Keep the head nonzero so the zero-initialized downstream
                # projection can learn even in an auxiliary-loss ablation.
                # plan_g_proj, not this head, makes the initial DiT residual 0.
                nn.init.normal_(self.prosody_head[-1].weight, std=0.01)
                nn.init.zeros_(self.prosody_head[-1].bias)
                # The plan's route into the acoustic head's own modulation,
                # applied PER DiT POSITION. This is the thing g_cond cannot
                # do: `c = c + g_cond` adds one vector to every patch, so
                # nothing carried in it can say 'louder here, slower there'.
                # Zero-initialized, so a run resumed from a checkpoint
                # without it reproduces that checkpoint exactly at step 0.
                self.plan_g_proj = nn.Linear(
                    self.latent_dim, self.fm_hidden_size
                )
                nn.init.zeros_(self.plan_g_proj.weight)
                nn.init.zeros_(self.plan_g_proj.bias)

            if getattr(voice_design_config, "instruction_conditioning", False):
                # Keep the existing code and its consumers intact. A single
                # zero output layer makes the added residual initially inert,
                # while its nonzero input features let it learn immediately.
                width = int(voice_design_config.instruction_condition_hidden_size)
                self.instruction_g_proj = nn.Sequential(
                    nn.LayerNorm(self.llm_hidden_size),
                    nn.Linear(self.llm_hidden_size, width),
                    nn.SiLU(),
                    nn.Linear(width, self.fm_hidden_size, bias=False),
                )
                nn.init.zeros_(self.instruction_g_proj[-1].weight)

            if getattr(voice_design_config, "semantic_patch_conditioning", False):
                self.semantic_patch_proj = nn.Sequential(
                    nn.LayerNorm(self.llm_hidden_size),
                    nn.Linear(self.llm_hidden_size, voice_design_config.semantic_patch_hidden_size),
                    nn.SiLU(),
                    nn.Linear(voice_design_config.semantic_patch_hidden_size,
                              self.llm_hidden_size, bias=False),
                )
                nn.init.zeros_(self.semantic_patch_proj[-1].weight)

        if self.instruction_residual_only:
            # Never train the unused target-speaker projection in this arm.
            self.xvec_proj.requires_grad_(False)

        # Helpers
        self.fm_helper = FlowMatchingHelper(sigma=config.get("fm_sigma", 0.0))
        self.causal_helper = CausalHelper()
        self.io_helper = IOHelper(latent_stats_path=latent_stats_path)
        self.audio_span_token_ids: list[int] = [
            self.audio_gen_span_id,
            self.audio_comp_span_id,
        ]
    # endregion Module construction

    # region Training forward path
    def forward(self, data: dict[str, Any]) -> DotsTtsForwardOutput:
        input_ids: torch.Tensor = data["input_ids"]
        input_ids_lengths: torch.Tensor = data["input_ids_lengths"]
        input_span_mask: torch.Tensor = data["input_span_mask"]
        output_span_mask: torch.Tensor = data["output_span_mask"]
        batch_size = input_ids.size(0)
        device = input_ids.device

        latents: torch.Tensor | None = data.get("latents")
        latents_sampled: torch.Tensor | None = data.get("latents_sampled")
        latent_lengths: torch.Tensor | None = data.get("latent_lengths")
        has_latents = latents is not None or latents_sampled is not None

        patch_embeddings: torch.Tensor | None
        valid_patch_counts: torch.Tensor | None
        if has_latents:
            if latents_sampled is None:
                latents_sampled = self.io_helper.sample_from_latent(latents)
            patch_embeddings = self.patch_encoder(
                latents_sampled, x_lens=latent_lengths
            )
            valid_patch_counts = latent_lengths // self.latent_patch_size
            latents_sampled = self.io_helper.normalize(latents_sampled).to(
                device=self.latent_proj.weight.device,
                dtype=self.latent_proj.weight.dtype,
            )
        else:
            latents_sampled = None
            patch_embeddings = None
            valid_patch_counts = torch.zeros(
                batch_size, dtype=torch.long, device=device
            )

        input_span_counts = input_span_mask.sum(dim=1)
        if input_span_counts.sum() > 0 and patch_embeddings is None:
            raise RuntimeError(
                "Found audio span tokens but no latents provided to compute patch embeddings."
            )

        # Token embeddings with audio span replacement
        inputs_embeds = self.llm.get_input_embeddings()(input_ids)
        if patch_embeddings is not None:
            inputs_embeds = inputs_embeds.clone()
            patch_embeddings = patch_embeddings.to(inputs_embeds.dtype)
            for b in range(batch_size):
                span_num = input_span_counts[b].item()
                if span_num == 0:
                    continue
                expected = valid_patch_counts[b].item()
                if expected != span_num:
                    raise RuntimeError(
                        f"Mismatch between span tokens ({span_num}) and latent patches ({expected}) for sample {b}."
                    )
                indices = input_span_mask[b].nonzero(as_tuple=False).squeeze(-1)
                inputs_embeds[b, indices, :] = patch_embeddings[b, :span_num, :]

        # ---- Voice patch (patch 0) --------------------------------------
        # The <|voice_patch|> input embedding has to be settled BEFORE the main
        # forward, exactly like an audio patch's. Teacher forcing supplies it in
        # most rows; the scheduled-sampling rows need the model's own prediction,
        # which needs a hidden state, which needs a forward — so those rows get
        # one short no-grad pass over the instruction prefix. That prefix is a
        # few dozen tokens next to a sequence dominated by audio spans, and the
        # pass is skipped outright when the draw selects nobody.
        voice_gen_mask: torch.Tensor | None = None
        voice_patch_mask: torch.Tensor | None = None
        voice_rows: torch.Tensor | None = None
        teacher_code: torch.Tensor | None = None
        chosen_code: torch.Tensor | None = None
        predicted_code_ratio = 0.0
        semantic_hidden: torch.Tensor | None = None
        anchor_for_cond: torch.Tensor | None = data.get("xvector")
        if self.instruction_residual_only:
            # Neither target CAM++ nor pooled VAE features are conditions.
            anchor_for_cond = None
            voice_gen_mask = input_ids.eq(int(self.voice_gen_start_id))
            voice_patch_mask = input_ids.eq(int(self.voice_patch_id))
            gen_counts, patch_counts = voice_gen_mask.sum(1), voice_patch_mask.sum(1)
            if bool(((gen_counts != patch_counts) | (gen_counts > 1)).any()):
                raise ValueError("Residual-only rows require one marker and one patch, or neither for plain TTS")
            voice_rows = voice_gen_mask.any(1) & voice_patch_mask.any(1)
            if bool((voice_rows & (
                voice_patch_mask.long().argmax(1) != voice_gen_mask.long().argmax(1) + 1
            )).any()):
                raise ValueError("Residual-only patch must immediately follow the instruction marker")
            semantic_hidden = self._voice_prefix_hidden(
                inputs_embeds, voice_gen_mask, voice_rows
            )
            inputs_embeds = self.inject_voice_patch(
                inputs_embeds, voice_patch_mask, None, voice_rows,
                condition_hidden=semantic_hidden,
            )
        if anchor_for_cond is not None:
            anchor_for_cond = anchor_for_cond.to(self.xvec_proj[0].weight)
        if self.voice_design is not None:
            voice_gen_mask = data.get("voice_gen_mask")
            if voice_gen_mask is None:
                voice_gen_mask = input_ids.eq(int(self.voice_gen_start_id))
            voice_patch_mask = data.get("voice_patch_mask")
            if voice_patch_mask is None:
                voice_patch_mask = input_ids.eq(int(self.voice_patch_id))
            voice_rows = voice_gen_mask.any(dim=1) & voice_patch_mask.any(dim=1)
            if anchor_for_cond is None:
                # No audio in this batch, so there is no teacher. Keep the shapes
                # and the graph, drop the supervision.
                voice_rows = torch.zeros_like(voice_rows)

            teacher_anchor = (
                anchor_for_cond
                if anchor_for_cond is not None
                else inputs_embeds.new_zeros((batch_size, self.xvec_dim))
            )
            teacher_voice = self.voice_design.voice_target(
                latents_sampled, latent_lengths
            )
            teacher_code = self.voice_design.encode_target(
                teacher_anchor,
                teacher_voice,
                # Those zero placeholders are not observations. Folding a batch
                # of zeros into the running statistics would drive the variance
                # to the eps floor and make every later target explode.
                update_stats=anchor_for_cond is not None,
            )

            voice_step = data.get("voice_design_step")
            chosen_code = teacher_code
            # A cloning row's latent comes from the reference encoder, not from
            # the flow head, and its voice losses are zeroed out below — the flow
            # head must not be scored against a target it was handed.
            prompt_code: torch.Tensor | None = data.get("voice_prompt_code")
            if prompt_code is not None:
                prompt_rows = data["voice_prompt_mask"].to(voice_rows.device).bool()
                chosen_code = torch.where(
                    prompt_rows.reshape(-1, 1, 1),
                    prompt_code.to(chosen_code),
                    chosen_code,
                )
                voice_rows = voice_rows & ~prompt_rows

            if self.semantic_patch_proj is not None:
                # Must retain gradients even on teacher-code rows. Only the
                # instruction prefix is visible, never the voice patch/text/audio.
                # Plain rows still execute a zero-weight path for DDP.
                semantic_hidden = self._voice_prefix_hidden(
                    inputs_embeds, voice_gen_mask,
                    voice_gen_mask.any(dim=1) & voice_patch_mask.any(dim=1),
                )

            use_predicted = (
                torch.rand(batch_size, device=device)
                < float(self.voice_design.predicted_code_prob(voice_step))
            ) & voice_rows
            if bool(use_predicted.any()):
                if semantic_hidden is None:
                    predicted_code = self._sample_voice_code_from_prefix(
                        inputs_embeds, voice_gen_mask, voice_rows
                    )
                else:
                    # Reuse the prefix, but do not backpropagate through the
                    # stochastic ODE sampler. The semantic route stays live.
                    with torch.no_grad():
                        predicted_code = self.voice_design.sample_code(
                            semantic_hidden.detach(),
                            num_steps=int(self.voice_design.config.train_sample_steps),
                            guidance_scale=1.0,
                        )
                chosen_code = torch.where(
                    use_predicted.reshape(-1, 1, 1),
                    predicted_code.to(chosen_code),
                    chosen_code,
                )
                predicted_code_ratio = float(
                    use_predicted.float().mean().detach().cpu()
                )

            inputs_embeds = self.inject_voice_patch(
                inputs_embeds,
                voice_patch_mask,
                chosen_code,
                voice_patch_mask.any(dim=1),
                condition_hidden=semantic_hidden,
            )

        # LLM forward pass to obtain logits & hidden states
        _llm_attn_mask, llm_seq_mask, _ = self.causal_helper.create_causal_mask_and_pos(
            seq_lens=input_ids_lengths, max_len=input_ids.size(1)
        )
        llm_outputs = self.llm(
            inputs_embeds=inputs_embeds,
            attention_mask=llm_seq_mask.long(),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        llm_logits = llm_outputs.logits  # [B, L, V]
        llm_hidden = llm_outputs.hidden_states[-1]  # [B, L, H]

        # eos prediction, before cfg masking
        eos = self.eos_proj(llm_hidden.detach())

        # Voice-design branch. It runs on every batch, including batches with no
        # instruction rows, because the training run uses
        # find_unused_parameters=False: a parameter that leaves the autograd
        # graph on some ranks and not others changes the number of gradient
        # all-reduces and hangs the job. Rows without a voice patch contribute
        # zero-weighted losses instead of being skipped.
        voice_output: VoiceDesignOutput | None = None
        voice_metrics: dict[str, float] | None = None
        voice_for_cond: torch.Tensor | None = None
        think_loss: torch.Tensor | None = None
        think_for_cond: torch.Tensor | None = None
        prosody_loss: torch.Tensor | None = None
        prosody_plan: torch.Tensor | None = None
        instruction_cond: torch.Tensor | None = None
        if self.instruction_residual_only:
            instruction_cond = self.instruction_to_g_cond(semantic_hidden, voice_rows)
            voice_metrics = {}
        if self.voice_design is not None:
            condition_hidden, _ = gather_voice_hidden(llm_hidden, voice_gen_mask)
            # Include reference-conditioned rows too, matching inference. Mask
            # AFTER projection: learned biases must not affect plain-TTS rows.
            instruction_cond = self.instruction_to_g_cond(
                condition_hidden,
                voice_gen_mask.any(dim=1) & voice_patch_mask.any(dim=1),
            )
            voice_output = self.voice_design(
                condition_hidden,
                teacher_code,
                voice_rows,
                step=data.get("voice_design_step"),
            )
            # g_cond is built from the SAME draw that went into the
            # <|voice_patch|> embedding. Deriving one from the teacher and the
            # other from the prediction would show the LM one voice while the
            # acoustic head hears another, and the model would learn to ignore
            # whichever of the two is less reliable.
            decoded_anchor, voice_for_cond = self.voice_design.decode_code(chosen_code)
            has_voice = voice_patch_mask.any(dim=1)
            if anchor_for_cond is not None:
                anchor_for_cond = torch.where(
                    has_voice.reshape(-1, 1),
                    decoded_anchor.to(anchor_for_cond),
                    anchor_for_cond,
                )
            if voice_for_cond is not None:
                voice_for_cond = voice_for_cond * has_voice.reshape(-1, 1).to(
                    voice_for_cond
                )
            if self.think_head is not None:
                think_mask = data.get("voice_think_mask")
                if think_mask is None:
                    think_mask = input_ids.eq(int(self.voice_think_id))
                think_hiddens, think_rows = self.gather_think_hiddens(
                    llm_hidden, think_mask
                )
                think_states = self.think_chain(think_hiddens)
                # The teacher is detached: the chain has to move toward the
                # target, not drag the target toward what the chain already
                # emits. That direction of travel is the difference between
                # refinement and collapse.
                think_target = teacher_code.detach().to(think_states)
                think_loss = (think_states - think_target).pow(2).mean(dim=-1)
                # Row weighting lives in the loss mask, which the trainer
                # collapses a step ahead of the forward; only the g_cond
                # contribution has to be zeroed here, because it bypasses
                # the loss machinery entirely.
                think_keep = (think_rows & voice_rows).reshape(-1, 1)
                think_for_cond = think_states[:, -1, :] * think_keep.to(
                    think_states.dtype
                )
            voice_metrics = {
                "voice_row_ratio": float(voice_rows.float().mean().detach().cpu()),
                "voice_predicted_code_ratio": predicted_code_ratio,
                "voice_predicted_code_prob": float(voice_output.predicted_code_prob),
                "voice_align_weight": float(voice_output.align_weight),
                # Number of rows the off-diagonal cosine below actually had to
                # work with. Read it first: with 0 or 1 voice rows there are no
                # pairs, and the metric reports exactly 0.0 — which is also the
                # value that means "maximally diverse". Without this you cannot
                # tell a healthy batch from an empty one.
                "voice_offdiag_rows": float(voice_output.row_mask.sum().detach().cpu()),
                # Mean off-diagonal cosine of the predicted codes. Approaching 1
                # means every instruction in the batch produced the same code,
                # which is the collapse that makes timbre stop tracking the
                # prompt long before any audio metric notices.
                "voice_offdiag_cosine": float(
                    (
                        voice_output.offdiag_cosine.reshape(-1)
                        * voice_output.row_mask.reshape(-1)
                    ).sum()
                    / voice_output.row_mask.sum().clamp_min(1.0)
                ),
            }
            if think_loss is not None:
                # First and last step of the refinement chain, on voice rows
                # only. The gap between them is the whole claim of this
                # design: if they track each other the extra slots are
                # decoration, and K should come back down.
                think_weight = voice_rows.to(think_loss.dtype)
                think_denominator = think_weight.sum().clamp_min(1.0)
                voice_metrics["voice_think_first"] = float(
                    (think_loss[:, 0].detach() * think_weight).sum()
                    / think_denominator
                )
                voice_metrics["voice_think_final"] = float(
                    (think_loss[:, -1].detach() * think_weight).sum()
                    / think_denominator
                )

        # Post-text plan slots. Outside the voice-design block because the
        # target comes from the audio, not from the branch's teacher code --
        # and unconditional once the head exists, so its gradient bucket is
        # filled on every rank in every step. With
        # find_unused_parameters=False a parameter that leaves the graph on
        # one rank and not another hangs the job.
        if self.prosody_head is not None:
            prosody_mask = data.get("voice_prosody_mask")
            if prosody_mask is None:
                prosody_mask = input_ids.eq(int(self.prosody_think_id))
            prosody_hiddens, _prosody_rows = self.gather_prosody_hiddens(
                llm_hidden, prosody_mask
            )
            prosody_pred = self.prosody_head(prosody_hiddens)
            if latents_sampled is None or latent_lengths is None:
                # No audio in this batch: keep the shapes and the graph,
                # drop the supervision. The loss mask is zero on these rows.
                prosody_target = torch.zeros_like(prosody_pred)
                prosody_valid = torch.zeros(
                    prosody_pred.shape[:2],
                    dtype=torch.bool,
                    device=prosody_pred.device,
                )
            else:
                prosody_target, prosody_valid = self.prosody_plan_targets(
                    latents_sampled, latent_lengths, valid_patch_counts
                )
                prosody_target = prosody_target.to(prosody_pred)
            prosody_loss = (prosody_pred - prosody_target).pow(2).mean(dim=-1)
            # Rows without a full set of slots contribute nothing: their
            # gathered hidden states are zeros, and a trained head does not
            # map zeros to zeros.
            prosody_plan = prosody_pred * _prosody_rows.reshape(-1, 1, 1).to(
                prosody_pred.dtype
            )
            if voice_metrics is not None:
                plan_weight = prosody_valid.to(prosody_loss.dtype)
                plan_denominator = plan_weight.sum().clamp_min(1.0)
                voice_metrics["voice_prosody_error"] = float(
                    (prosody_loss.detach() * plan_weight).sum() / plan_denominator
                )
                # Error of a zero predictor. Compare against this baseline;
                # low absolute error alone may just reflect a small target.
                # Beating it is not sufficient evidence of instruction use
                # or disentangled prosody; those require generation tests.
                voice_metrics["voice_prosody_target_var"] = float(
                    (
                        prosody_target.detach().pow(2).mean(dim=-1) * plan_weight
                    ).sum()
                    / plan_denominator
                )
                # The one number worth watching: error divided by the variance
                # of what it is predicting. 1.0 means the head is still emitting
                # the zero it was initialized to and has learned nothing, so a
                # falling voice_prosody_loss on its own proves nothing -- the
                # target's own scale is small, and the weighted term can shrink
                # simply because the batch held flatter utterances.
                voice_metrics["voice_prosody_ratio"] = float(
                    voice_metrics["voice_prosody_error"]
                    / max(voice_metrics["voice_prosody_target_var"], 1e-8)
                )

        # Flow matching forward
        total_patches = int(output_span_mask.sum().item())
        if total_patches > 0 and latents_sampled is None:
            raise RuntimeError("Flow matching requested but latents are missing.")
        if total_patches > 0:
            if anchor_for_cond is None and not self.instruction_residual_only:
                raise KeyError(
                    "Flow matching requires an 'xvector' entry; "
                    "prepare_training_inputs builds it from the batch audio."
                )
            xvec_cond = (
                torch.zeros_like(instruction_cond) if self.instruction_residual_only
                else self.xvec_proj(anchor_for_cond)
            )
            if self.voice_proj is not None and voice_for_cond is not None:
                # Already zeroed on rows without a voice patch, so a plain TTS
                # row in a mixed batch stays on the stock g_cond path exactly.
                xvec_cond = xvec_cond + self.voice_proj(
                    voice_for_cond.to(self.voice_proj.weight.dtype)
                ).to(xvec_cond)
            if self.think_g_proj is not None and think_for_cond is not None:
                # Zero-initialized, and already zeroed on rows without a full
                # set of think slots, so a plain-TTS row in a mixed batch
                # stays on the stock g_cond path exactly. Added before the
                # CFG drop below, so classifier-free guidance drops the
                # instruction's timbre as one object rather than half of it.
                xvec_cond = xvec_cond + self.think_g_proj(
                    think_for_cond.to(self.think_g_proj.weight.dtype)
                ).to(xvec_cond)
            if self.g_cond_source == "plan":
                # Scaled out rather than skipped. A parameter that leaves the
                # autograd graph on some ranks and not others changes the
                # number of gradient all-reduces and hangs the job under
                # find_unused_parameters=False, so xvec_proj and friends stay
                # in the graph contributing exactly zero.
                #
                # What remains is a model whose only global acoustic
                # conditioning is the time-average of the plan added below --
                # no separately predicted speaker vector for the instruction
                # to have to squeeze through, and nothing for the
                # per-position path to lose attention to.
                xvec_cond = xvec_cond * 0.0
            if instruction_cond is not None:
                xvec_cond = xvec_cond + instruction_cond.to(xvec_cond)
                voice_metrics["voice_instruction_cond_rms"] = float(
                    instruction_cond.detach().float().square().mean().sqrt().cpu()
                )
            vocal_mask = data.get("vocal_mask")
            if vocal_mask is None:
                vocal_mask = torch.ones((batch_size,), device=device, dtype=torch.bool)
            xvec_drop_mask = (
                torch.empty((batch_size,), device=device, dtype=torch.float32).uniform_(
                    0, 1
                )
                < self.xvec_drop_rate
            )
            xvec_drop_mask = xvec_drop_mask & vocal_mask
            xvec_cond = mask_data(xvec_cond, xvec_drop_mask)

            hiddens_for_fm = torch.where(
                output_span_mask.unsqueeze(-1), llm_hidden, inputs_embeds
            )

            # Prepare DiT inputs
            (
                fm_seq,
                target,
                fm_attn_mask,
                fm_seq_mask,
                fm_pos_ids,
                times,
                fm_prefix_lengths,
                fm_gen_lengths,
                fm_gen_patch_size,
            ) = self.io_helper.prepare_inputs_for_dit(
                hiddens=hiddens_for_fm,
                hidden_lens=input_ids_lengths,
                latents=latents_sampled,
                latent_lens=latent_lengths,
                hidden_proj=self.hidden_proj,
                latent_proj=self.latent_proj,
                noisy_proj=self.coordinate_proj,
                span_mask=output_span_mask,
                hidden_patch_size=self.hidden_patch_size,
                latent_patch_size=self.latent_patch_size,
                fm_helper=self.fm_helper,
                cfg_droprate=self.cfg_droprate,
            )

            g_cond_for_dit = xvec_cond
            if self.plan_g_proj is not None and prosody_plan is not None:
                plan_vectors = self.plan_g_proj(
                    prosody_plan.to(self.plan_g_proj.weight.dtype)
                ).to(xvec_cond)
                # Dropped by the SAME classifier-free mask as the global
                # condition: both are instruction-derived, and dropping one
                # half of the voice while keeping the other would train a
                # combination that inference never produces.
                plan_vectors = mask_data(plan_vectors, xvec_drop_mask)
                slot_index, slot_valid = self.plan_slot_positions(
                    valid_patch_counts,
                    fm_prefix_lengths,
                    fm_gen_lengths,
                    fm_gen_patch_size,
                    int(fm_seq.size(1)),
                )
                gathered = torch.gather(
                    plan_vectors,
                    1,
                    slot_index.unsqueeze(-1).expand(-1, -1, plan_vectors.size(-1)),
                )
                gathered = gathered * slot_valid.unsqueeze(-1).to(gathered.dtype)
                # Promoting g_cond to [B, T, H] is what makes the DiT's
                # adaLN per-position; the global half broadcasts unchanged.
                g_cond_for_dit = xvec_cond.unsqueeze(1) + gathered

            # Predict velocity field
            vt = self.velocity_field_predictor(
                x=fm_seq,
                timesteps=times,
                pos_ids=fm_pos_ids,
                mask=fm_seq_mask,
                attn_mask=fm_attn_mask,
                return_hidden_stats=False,
                g_cond=g_cond_for_dit,
            )

            # Get predictions and targets
            pred = self.io_helper.get_dit_outputs(
                pred_v=vt,
                fm_prefix_lengths=fm_prefix_lengths,
                fm_gen_lengths=fm_gen_lengths,
                fm_gen_patch_size=fm_gen_patch_size,
                latent_patch_size=self.latent_patch_size,
            )
        else:
            # Dummy forward for velocity_field_predictor to keep gradients connected in DDP
            dummy_length = self.latent_patch_size
            dummy_seq_h = llm_hidden.new_zeros((1, dummy_length, self.llm_hidden_size))
            dummy_seq_h = self.hidden_proj(dummy_seq_h) * 0.0  # dummy op for ddp
            dummy_seq_l = llm_hidden.new_zeros((1, dummy_length, self.latent_dim))
            dummy_seq_l = self.latent_proj(dummy_seq_l) * 0.0  # dummy op for ddp
            dummy_seq_c = llm_hidden.new_zeros((1, dummy_length, self.latent_dim))
            dummy_seq_c = self.coordinate_proj(dummy_seq_c) * 0.0  # dummy op for ddp
            dummy_seq = dummy_seq_h + dummy_seq_l + dummy_seq_c
            dummy_times = torch.zeros((1,), device=device, dtype=torch.float32)
            dummy_attn_mask = torch.ones(
                (1, dummy_length, dummy_length), device=device, dtype=torch.bool
            )
            dummy_out = self.velocity_field_predictor(
                x=dummy_seq,
                timesteps=dummy_times,
                attn_mask=dummy_attn_mask,
            )
            pred = dummy_out[:, -self.latent_patch_size :, :]
            if instruction_cond is not None:
                # Text-only batches still participate in DDP all-reduces.
                pred = pred + instruction_cond.sum().to(pred) * 0.0
            if semantic_hidden is not None:
                # Link the patch projection through the main LLM even when
                # there are no acoustic targets in this micro-batch.
                pred = pred + llm_hidden.sum().to(pred) * 0.0
            target = pred.detach()

        return DotsTtsForwardOutput(
            llm_logits=llm_logits,
            pred=pred,
            target=target,
            eos_out=eos,
            voice_design=voice_output,
            voice_metrics=voice_metrics,
            voice_think_loss=think_loss,
            voice_prosody_loss=prosody_loss,
        )
    # endregion Training forward path

    # region Voice-design shared helpers
    def instruction_to_g_cond(
        self,
        condition_hidden: torch.Tensor,
        row_mask: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Shared train/infer projection of the pre-voice instruction state.

        The causal voice_gen_start position cannot see the injected voice patch,
        target text or target audio in the standard voice-design template.
        This is a direct condition, not a disentangled semantic representation
        or an instruction-specific CFG implementation.
        """
        if self.instruction_g_proj is None:
            return None
        if condition_hidden.ndim != 3 or condition_hidden.size(1) != 1:
            raise ValueError("instruction condition must have shape [batch, 1, hidden]")
        residual = self.instruction_g_proj(
            condition_hidden[:, 0].to(self.instruction_g_proj[1].weight)
        )
        if row_mask is not None:
            if row_mask.shape != residual.shape[:1]:
                raise ValueError("instruction row_mask must have shape [batch]")
            residual = residual * row_mask.to(residual).unsqueeze(-1)
        return residual

    def gather_think_hiddens(
        self,
        hidden_states: torch.Tensor,
        think_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The K pre-text ``<|voice_think|>`` hidden states."""

        return _gather_slot_hiddens(
            hidden_states,
            think_mask,
            int(self.num_think_slots),
            VOICE_THINK_TOKEN,
        )

    def gather_prosody_hiddens(
        self,
        hidden_states: torch.Tensor,
        prosody_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """The M post-text ``<|prosody_think|>`` hidden states."""

        return _gather_slot_hiddens(
            hidden_states,
            prosody_mask,
            int(self.num_prosody_slots),
            PROSODY_THINK_TOKEN,
        )

    def plan_slot_positions(
        self,
        valid_patch_counts: torch.Tensor,
        fm_prefix_lengths: torch.Tensor,
        fm_gen_lengths: torch.Tensor,
        fm_gen_patch_size: int,
        total_length: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """DiT position -> plan slot, ``[B, T]`` long + ``[B, T]`` bool.

        The DiT sequence is ``[prefix: P equal chunks][gen: P chunks of
        fm_gen_patch_size][padding]``; both halves are indexed by the same
        patch n. Position -> patch -> slot uses the same
        ``seg(n) = min(n // S, M - 1)`` that ``prosody_segment_bounds``
        inverts, so the slot that modulates a patch is the slot that was
        scored on it -- and neither direction needs the total patch count,
        which is what makes this reproducible at generation time.
        """

        num_slots = int(self.num_prosody_slots)
        if num_slots <= 0:
            raise RuntimeError("This model has no prosody slots.")
        stride = int(self.prosody_patches_per_slot)
        device = valid_patch_counts.device
        patches = valid_patch_counts.reshape(-1).long()
        prefix = fm_prefix_lengths.reshape(-1).to(device).long()
        generated = fm_gen_lengths.reshape(-1).to(device).long()
        # clamp_min(1) only guards the division; rows with no patches are
        # excluded wholesale by `valid` below.
        safe_patches = patches.clamp_min(1)
        prefix_chunk = (prefix // safe_patches).clamp_min(1)
        position = torch.arange(total_length, device=device).unsqueeze(0)
        in_prefix = position < prefix.unsqueeze(1)
        in_generated = (position >= prefix.unsqueeze(1)) & (
            position < (prefix + generated).unsqueeze(1)
        )
        patch_index = torch.where(
            in_prefix,
            position // prefix_chunk.unsqueeze(1),
            (position - prefix.unsqueeze(1)).clamp_min(0)
            // int(fm_gen_patch_size),
        )
        patch_index = torch.minimum(
            patch_index.clamp_min(0), (safe_patches - 1).unsqueeze(1)
        )
        slot = (patch_index // stride).clamp(max=num_slots - 1)
        valid = (in_prefix | in_generated) & patches.gt(0).unsqueeze(1)
        return slot, valid

    def prosody_plan_targets(
        self,
        latents_sampled: torch.Tensor,
        latent_lengths: torch.Tensor,
        valid_patch_counts: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-segment mean of the normalized latent, ``[B, M, D]``+``[B, M]``.

        Segment m covers patches ``[lo(m), hi(m))`` from
        ``prosody_segment_bounds``. Splitting on patches rather than on raw
        frames is what lets the loss mask be derived from ``input_ids``
        alone -- the trainer collapses masks into denominators a full
        optimizer step before any forward runs, so a mask that needed the
        audio would be a step out of date.

        The target is the segment mean itself, not its deviation from the
        utterance mean. This is an acoustic latent target, not a guaranteed
        timbre/prosody disentanglement. The x-vector route is retained when
        g_cond_source="xvector" and suppressed when g_cond_source="plan".
        """

        num_slots = int(self.num_prosody_slots)
        if num_slots <= 0:
            raise RuntimeError("This model has no prosody slots.")
        batch_size, num_frames, _ = latents_sampled.shape
        device = latents_sampled.device
        patch_size = int(self.latent_patch_size)
        low, high = prosody_segment_bounds(
            valid_patch_counts.to(device=device),
            num_slots,
            int(self.prosody_patches_per_slot),
        )
        valid = high > low
        frames = torch.arange(num_frames, device=device).reshape(1, 1, -1)
        weights = (frames >= (low * patch_size).unsqueeze(-1)) & (
            frames < (high * patch_size).unsqueeze(-1)
        )
        weights = weights.to(latents_sampled.dtype)
        counts = weights.sum(dim=-1, keepdim=True).clamp_min(1.0)
        segments = torch.bmm(weights, latents_sampled) / counts
        segments = segments * valid.unsqueeze(-1).to(segments.dtype)
        del batch_size, latent_lengths
        return segments.detach(), valid

    def think_chain(self, think_hiddens: torch.Tensor) -> torch.Tensor:
        """``[B, K, H]`` -> cumulative code estimates ``[B, K, D]``.

        Row t is the estimate of the voice code after t+1 refinement steps.
        The head predicts increments and the cumulative sum is what is
        scored, so an early slot that gets the coarse timbre roughly right
        leaves the later ones a small correction to make instead of a fresh
        guess -- which is the only reason K slots beat one.
        """

        if self.think_head is None:
            raise RuntimeError("This model has no think slots.")
        return self.think_head(think_hiddens).cumsum(dim=1)

    def inject_voice_patch(
        self,
        inputs_embeds: torch.Tensor,
        voice_patch_mask: torch.Tensor,
        code: torch.Tensor | None,
        row_mask: torch.Tensor,
        *,
        condition_hidden: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Write the voice latent into the ``<|voice_patch|>`` input embedding.

        The direct analogue of replacing an audio span's embedding with its
        encoded latent patch. Used by both the training forward and the
        inference prefill; they have to agree exactly, or the hidden states the
        branch reads at inference come from embeddings it never saw in training.
        """

        if not self.instruction_residual_only and (self.voice_design is None or self.voice_embed_proj is None):
            return inputs_embeds
        voice_patch_mask = voice_patch_mask.to(device=inputs_embeds.device).bool()
        counts = voice_patch_mask.sum(dim=1)
        unexpected = counts[counts > 1]
        if unexpected.numel() > 0:
            raise ValueError(
                "Every sample must carry at most one <|voice_patch|>; got counts "
                f"{sorted(set(unexpected.tolist()))}."
            )
        # The projection is applied unconditionally, before any early return.
        # It has no loss of its own — this write is its ONLY route into the
        # autograd graph — so skipping it on a micro-batch that happens to hold
        # no voice rows would leave its gradient bucket unfilled on that rank,
        # and with find_unused_parameters=False that hangs the job. Rare, which
        # is what makes it nasty: one all-plain-TTS batch on one rank is enough.
        row_weight = (
            row_mask.to(device=inputs_embeds.device).bool()
            & counts.eq(1)
        ).to(inputs_embeds.dtype).reshape(-1, 1)
        # Drop the patch on a fraction of rows during training. The patch sits
        # between the instruction and the text and is a ready-made summary of
        # the instruction, so attention drifts to it and away from the
        # instruction tokens themselves -- and that summary regresses onto a
        # speaker-identity target, which is prosody-free by construction. The
        # model then follows style instructions WORSE than if the patch were not
        # there. Zeroing it at random keeps the direct instruction path alive;
        # g_cond still carries the voice, so timbre reproducibility is unharmed.
        # `keep` multiplies rather than slices so the executed graph never
        # depends on batch contents (find_unused_parameters=False).
        patch_dropout = float(getattr(self, "voice_patch_dropout", 0.0) or 0.0)
        if self.training and patch_dropout > 0.0:
            keep = (
                torch.rand(
                    inputs_embeds.size(0), 1, device=inputs_embeds.device
                )
                >= patch_dropout
            ).to(inputs_embeds.dtype)
            row_weight = row_weight * keep
        embedded = (
            inputs_embeds.new_zeros((inputs_embeds.size(0), self.llm_hidden_size))
            if self.instruction_residual_only else self.voice_embed_proj(
                code.to(self.voice_embed_proj.weight.dtype)
            )[:, 0].to(inputs_embeds.dtype)
        )
        if self.semantic_patch_proj is not None:
            if condition_hidden is None:
                raise ValueError("Semantic voice patch requires instruction prefix hidden states")
            embedded = embedded + self.instruction_to_voice_patch(condition_hidden).to(embedded)
        # The existing patch dropout/mask applies to the whole replacement.
        embedded = embedded * row_weight
        if float(row_weight.sum()) == 0.0:
            # Keep the graph edge without writing anything: adding a zero-valued
            # slice of `embedded` would still touch positions that must not move.
            return inputs_embeds + embedded.sum() * 0.0
        inputs_embeds = inputs_embeds.clone()
        for index in range(inputs_embeds.size(0)):
            if float(row_weight[index]) == 0.0:
                continue
            position = int(voice_patch_mask[index].nonzero()[0].item())
            inputs_embeds[index, position, :] = embedded[index]
        return inputs_embeds

    @torch.no_grad()
    def _sample_voice_code_from_prefix(
        self,
        inputs_embeds: torch.Tensor,
        voice_gen_mask: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Predicted voice latent from a short pass over the instruction prefix.

        Scheduled sampling is circular otherwise: the ``<|voice_patch|>`` input
        embedding must be settled before the main forward, but the prediction it
        would carry comes from a hidden state that forward produces. Running the
        prefix — everything up to and including ``<|voice_gen_start|>`` — breaks
        the cycle for a few dozen tokens instead of the whole sequence. It is
        no-grad on purpose: the flow loss takes its gradient from the main
        forward's hidden state, and this pass exists only to draw a sample.
        """

        if self.voice_design is None:
            raise RuntimeError("voice_design is not enabled on this model.")
        condition_hidden = self._voice_prefix_hidden(inputs_embeds, voice_gen_mask, row_mask)
        return self.voice_design.sample_code(
            condition_hidden,
            num_steps=int(self.voice_design.config.train_sample_steps),
            guidance_scale=1.0,
        )

    def instruction_to_voice_patch(self, condition_hidden: torch.Tensor) -> torch.Tensor:
        """Deterministic semantic residual [B,H], shared by train and prefill."""
        if self.semantic_patch_proj is None:
            raise RuntimeError("semantic_patch_conditioning is disabled")
        if condition_hidden.ndim != 3 or condition_hidden.size(1) != 1:
            raise ValueError("condition_hidden must have shape [batch, 1, hidden]")
        return self.semantic_patch_proj(
            condition_hidden[:, 0].to(self.semantic_patch_proj[1].weight)
        )

    def _voice_prefix_hidden(
        self, inputs_embeds: torch.Tensor, voice_gen_mask: torch.Tensor,
        row_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Differentiable instruction-only prefill; caller controls no_grad."""
        positions = voice_gen_mask.float().argmax(dim=1).long()
        prefix_lengths = torch.where(
            row_mask.to(positions.device),
            positions + 1,
            torch.ones_like(positions),
        )
        max_prefix = int(prefix_lengths.max().item())
        _, seq_mask, _ = self.causal_helper.create_causal_mask_and_pos(
            seq_lens=prefix_lengths, max_len=max_prefix
        )
        outputs = self.llm(
            inputs_embeds=inputs_embeds[:, :max_prefix],
            attention_mask=seq_mask.long(),
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
        )
        condition_hidden, _ = gather_voice_hidden(
            outputs.hidden_states[-1], voice_gen_mask[:, :max_prefix]
        )
        return condition_hidden

    # endregion Voice-design shared helpers

    # region Voice-design inference
    @torch.no_grad()
    def sample_voice_code(
        self,
        condition_hidden: torch.Tensor,
        *,
        num_steps: int | None = None,
        guidance_scale: float | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        """Draw one voice latent from the ``<|voice_gen_start|>`` hidden state."""

        if self.voice_design is None:
            raise RuntimeError("voice_design is not enabled on this model.")
        return self.voice_design.sample_code(
            condition_hidden,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            generator=generator,
        )

    def voice_code_to_g_cond(self, code: torch.Tensor) -> torch.Tensor:
        """Normalized voice latent -> ``g_cond``.

        The inference twin of the training routing: split the latent into the
        CAM++ anchor and the QFormer half, run the anchor through the
        *pretrained* ``xvec_proj`` and the QFormer half through its own
        zero-initialized projection.

        Note that the reference-audio path scales its x-vector by
        ``speaker_scale`` before projection; a predicted anchor must NOT be
        scaled, because it was trained against unscaled CAM++ targets and
        scaling it would move it off distribution.
        """

        if self.voice_design is None:
            raise RuntimeError("voice_design is not enabled on this model.")
        anchor, voice = self.voice_design.decode_code(code)
        g_cond = self.xvec_proj(anchor.to(self.xvec_proj[0].weight.dtype))
        if self.voice_proj is not None and voice is not None:
            g_cond = g_cond + self.voice_proj(
                voice.to(self.voice_proj.weight.dtype)
            ).to(g_cond)
        return g_cond

    # endregion Voice-design inference

    # region Autoregressive and flow-matching inference steps
    def _llm_base_model(self) -> nn.Module:
        base_model = getattr(self.llm, "base_model", None)
        if base_model is not None and base_model is not self.llm:
            return base_model

        base_model_prefix = getattr(self.llm, "base_model_prefix", None)
        if base_model_prefix:
            prefixed_model = getattr(self.llm, base_model_prefix, None)
            if prefixed_model is not None and prefixed_model is not self.llm:
                return prefixed_model

        raise RuntimeError(
            "Qwen2ForCausalLM did not expose a base transformer model. "
            "Add an adapter for this architecture before optimized inference."
        )

    @staticmethod
    def _last_hidden_state(outputs: Any) -> torch.Tensor | None:
        hidden = getattr(outputs, "last_hidden_state", None)
        if hidden is not None:
            return hidden
        if isinstance(outputs, tuple) and outputs:
            first = outputs[0]
            if isinstance(first, torch.Tensor):
                return first
        return None

    def _compute_lm_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        output_embeddings = self.llm.get_output_embeddings()
        if output_embeddings is None:
            raise RuntimeError("LLM does not expose output embeddings.")
        return output_embeddings(hidden)

    @torch.no_grad()
    def fm_solver_step(
        self,
        t: torch.Tensor,
        z: torch.Tensor,
        *,
        input_sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        hidden_size: int,
        patch_size: int,
        g_cond: torch.Tensor | None,
        guidance_scale: torch.Tensor | float,
    ) -> torch.Tensor:
        batch_size = input_sequence.size(0)
        if input_sequence.shape != cfg_sequence.shape:
            raise ValueError(
                "FM input_sequence and cfg_sequence must share the same shape."
            )
        if input_sequence.size(1) < patch_size:
            raise ValueError(
                "FM input sequence must reserve at least one latent patch slot."
            )
        latent_start = input_sequence.size(1) - patch_size
        z = self.coordinate_proj(z)
        z_c = input_sequence.clone()
        z_c[:, latent_start:] = z
        z_branches = [z_c]
        g_cond_t = (
            None if g_cond is None else g_cond.to(device=z_c.device, dtype=z_c.dtype)
        )
        g_cond_branches = None if g_cond_t is None else [g_cond_t]

        z_cfg = cfg_sequence.clone()
        z_cfg[:, latent_start:] = z
        z_branches.append(z_cfg)
        if g_cond_branches is not None:
            g_cond_branches.append(torch.zeros_like(g_cond_t))

        z_z = torch.cat(z_branches, dim=0)
        t_t = t.reshape(1).repeat(len(z_branches))
        if g_cond_branches is not None:
            g_cond_t = torch.cat(g_cond_branches, dim=0)
        vt = self.velocity_field_predictor(
            x=z_z,
            timesteps=t_t,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            g_cond=g_cond_t,
            hidden_size=patch_size * 2 + hidden_size,
            patch_size=patch_size + 1,
        )
        vt = vt[:, latent_start:]
        vt_c = vt[:batch_size]
        vt_u = vt[batch_size:]
        if not torch.is_tensor(guidance_scale):
            guidance_scale = vt_c.new_tensor(float(guidance_scale))
        else:
            guidance_scale = guidance_scale.to(device=vt_c.device, dtype=vt_c.dtype)
        return vt_c + guidance_scale * (vt_c - vt_u)

    @torch.no_grad()
    def step_llm(
        self,
        inputs_embeds: torch.Tensor | None = None,
        input_ids: torch.Tensor | None = None,
        past_key_values: Any | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, Any | None]:
        provided = int(inputs_embeds is not None) + int(input_ids is not None)
        if provided != 1:
            raise ValueError(
                "Exactly one of inputs_embeds or input_ids must be provided to step_llm()."
            )

        if inputs_embeds is not None:
            pass
        else:
            inputs_embeds = self.llm.get_input_embeddings()(input_ids)

        outputs = self.llm(
            inputs_embeds=inputs_embeds,
            past_key_values=past_key_values,
            use_cache=True,
            output_hidden_states=True,
            return_dict=True,
        )

        hidden = outputs.hidden_states[-1]
        logits = outputs.logits
        past_key_values = outputs.past_key_values

        return inputs_embeds, hidden, logits, past_key_values

    @torch.no_grad()
    def _meanflow_step_fm(
        self,
        *,
        input_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        patch_size: int,
        g_cond: torch.Tensor | None = None,
        nfe: int = 2,
        solver_step: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if nfe <= 0:
            raise ValueError(f"MeanFlow nfe must be positive, got {nfe}.")
        batch_size = input_sequence.size(0)
        device = input_sequence.device
        dtype = input_sequence.dtype
        solver_step = self.meanflow_solver_step if solver_step is None else solver_step
        z = (
            torch.randn(
                (batch_size, patch_size, self.latent_dim),
                device=device,
                dtype=dtype,
            )
        )
        times = torch.linspace(0.0, 1.0, nfe + 1, device=device, dtype=dtype)

        for step in range(nfe):
            t = times[step].expand(batch_size)
            dt = (times[step + 1] - times[step]).expand(batch_size)
            z = solver_step(
                z,
                t=t,
                dt=dt,
                input_sequence=input_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                patch_size=patch_size,
                g_cond=g_cond,
            ).clone()
        return z

    @torch.no_grad()
    def meanflow_solver_step(
        self,
        z: torch.Tensor,
        *,
        t: torch.Tensor,
        dt: torch.Tensor,
        input_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        patch_size: int,
        g_cond: torch.Tensor | None,
    ) -> torch.Tensor:
        if input_sequence.size(1) < patch_size:
            raise ValueError(
                "MeanFlow input sequence must reserve at least one latent patch slot."
            )
        latent_start = input_sequence.size(1) - patch_size
        z_proj = self.coordinate_proj(z)
        z_c = input_sequence.clone()
        z_c[:, latent_start:] = z_proj
        vt = self.velocity_field_predictor(
            x=z_c,
            timesteps=t,
            duration=dt,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            g_cond=g_cond,
        )
        velocity = vt[:, latent_start:]
        return z + velocity * dt.view(-1, 1, 1)

    @torch.no_grad()
    def _flow_matching_step_fm(
        self,
        *,
        input_sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        hidden_size: int,
        patch_size: int,
        g_cond: torch.Tensor | None = None,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 3.0,
        solver_step: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        batch_size = input_sequence.size(0)
        num_evals = 0
        solver_step = self.fm_solver_step if solver_step is None else solver_step
        guidance_scale_tensor = input_sequence.new_tensor(float(guidance_scale))

        # Prepare ODE solver
        def solver(t, z):
            nonlocal num_evals
            num_evals += 1
            return solver_step(
                t,
                z,
                input_sequence=input_sequence,
                cfg_sequence=cfg_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                hidden_size=hidden_size,
                patch_size=patch_size,
                g_cond=g_cond,
                guidance_scale=guidance_scale_tensor,
            )

        # Prepare noise as initial coordinate
        noise = torch.randn(
            (batch_size, patch_size, self.latent_dim),
            dtype=input_sequence.dtype,
            device=input_sequence.device,
        )
        # Solve
        times = torch.tensor(
            [0.0, 1.0], dtype=input_sequence.dtype, device=input_sequence.device
        )
        if ode_method in ["euler", "midpoint", "rk4"]:  # fixed step size methods
            options = {"step_size": 1.0 / num_steps}
        else:
            logger.warning(
                "Using adaptive step size ODE solver for FM, NFE is not guaranteed: "
                "ode_method={}",
                ode_method,
            )
            options = {}
        trajectory = odeint(
            func=solver,
            y0=noise,
            t=times,
            atol=1e-5,
            rtol=1e-5,
            method=ode_method,
            options=options,
        )
        # print(f"Expected NFE: {num_steps}, Actual NFE: {num_evals}")
        return trajectory[-1]

    @torch.no_grad()
    def step_fm(
        self,
        input_sequence: torch.Tensor,
        cfg_sequence: torch.Tensor,
        attn_mask: torch.Tensor,
        pos_ids: torch.Tensor | None,
        hidden_size: int,
        patch_size: int,
        g_cond: torch.Tensor | None = None,
        ode_method: str = "euler",
        num_steps: int = 10,
        guidance_scale: float = 3.0,
        solver_step: Callable[..., torch.Tensor] | None = None,
    ) -> torch.Tensor:
        if self.mode == "meanflow":
            return self._meanflow_step_fm(
                input_sequence=input_sequence,
                attn_mask=attn_mask,
                pos_ids=pos_ids,
                patch_size=patch_size,
                g_cond=g_cond,
                nfe=num_steps,
                solver_step=solver_step,
            )

        return self._flow_matching_step_fm(
            input_sequence=input_sequence,
            cfg_sequence=cfg_sequence,
            attn_mask=attn_mask,
            pos_ids=pos_ids,
            hidden_size=hidden_size,
            patch_size=patch_size,
            g_cond=g_cond,
            ode_method=ode_method,
            num_steps=num_steps,
            guidance_scale=guidance_scale,
            solver_step=solver_step,
        )
    # endregion Autoregressive and flow-matching inference steps


class FlowMatchingHelper:
    """
    Base helper for computing x_t and u_t, given target x_1 and noise x_0
    ref:  Flow matching for generative modeling, Lipman
    """

    def __init__(self, sigma=1e-5):
        self.sigma = sigma

    def compute_mu_t(self, x1, t):
        return t * x1

    def compute_sigma_t(self, t):
        return 1 - (1 - self.sigma) * t

    def sample_x_t(self, x0, x1, t):
        mu_t = self.compute_mu_t(x1, t)
        sigma_t = self.compute_sigma_t(t)
        return mu_t + sigma_t * x0

    def compute_u_t(self, x0, x1):
        return x1 - (1 - self.sigma) * x0

    def compute_xt_ut(self, x1, t=None, x0=None):
        if x0 is None:
            x0 = torch.randn_like(x1, device=x1.device)
        if t is None:
            t = torch.rand(x1.size(0), dtype=x1.dtype, device=x1.device)
        times = t
        t = t.reshape(-1, *([1] * (x1.dim() - 1)))
        xt = self.sample_x_t(x0, x1, t)
        ut = self.compute_u_t(x0, x1)
        return xt, ut, times


class CausalHelper:
    def create_causal_mask_and_pos(self, seq_lens, max_len):
        seq_mask = get_mask_from_lengths(seq_lens, max_len=max_len).unsqueeze(1)
        causal_mask = (
            torch.ones((max_len, max_len), device=seq_lens.device).triu(1).bool()
        )
        causal_mask = ~causal_mask.unsqueeze(0)
        attn_mask = seq_mask & causal_mask
        return attn_mask, seq_mask.squeeze(1), None

    def create_causal_chunk_mask_and_pos(
        self,
        batch_size,
        C_lens,
        Z_lens,
        span_mask,
        patch_size=8,
    ):
        device = C_lens.device
        total_lens = C_lens + Z_lens
        attn_mask = torch.zeros(
            (batch_size, total_lens.max(), total_lens.max()),
            device=device,
            dtype=torch.bool,
        )
        pos_ids = []
        # | C2C |     |
        # | Z2C | Z2Z |
        for i in range(batch_size):
            C_len = C_lens[i]
            Z_len = Z_lens[i]

            # C2C parts are standard causal attention
            attn_mask[i, :C_len, :C_len] = (
                torch.ones((C_len, C_len), device=device, dtype=torch.bool)
                .triu(1)
                .logical_not()
            )
            # Position ids in C parts are 0, 1, 2, ..., n
            c_pos = torch.arange(C_len, device=device, dtype=torch.float32)

            # Z2Z parts are block diag attention
            assert Z_len % patch_size == 0, "Z_len must be multiple of patch_size"
            attn_mask[i, C_len : C_len + Z_len, C_len : C_len + Z_len] = (
                torch.block_diag(
                    *[
                        torch.ones(
                            (patch_size, patch_size), device=device, dtype=torch.bool
                        )
                    ]
                    * (Z_len // patch_size)
                )
            )

            # Z2C parts is full attention before current patch latents
            # build according to span_mask
            j_indices = torch.arange(Z_len, device=device)
            patch_indices = j_indices // patch_size
            patch_in_c_indices = torch.where(span_mask[i])[0][patch_indices]
            attn_mask[
                i,
                C_len + j_indices.unsqueeze(1),
                torch.arange(C_len, device=device).unsqueeze(0),
            ] = torch.arange(C_len, device=device).unsqueeze(
                0
            ) < patch_in_c_indices.unsqueeze(1)
            # Position ids in Z parts start from current patch latents index in C parts
            z_pos = (patch_in_c_indices + j_indices % patch_size).to(torch.float32)
            pos_ids.append(torch.cat([c_pos, z_pos]))
        seq_mask = get_mask_from_lengths(total_lens, max_len=total_lens.max().item())
        pos_ids = pad_sequence(pos_ids, batch_first=True, padding_value=0.0).to(
            C_lens.device
        )
        return attn_mask, seq_mask, pos_ids


class IOHelper:
    def __init__(self, latent_stats_path=None):
        if latent_stats_path is not None:
            latent_stats = torch.load(latent_stats_path, weights_only=False)
            self.global_mean = torch.as_tensor(latent_stats["mean"])
            self.global_var = torch.as_tensor(latent_stats["var"])
        else:
            self.global_mean = None
            self.global_var = None

    def normalize(self, x):
        if self.global_mean is not None and self.global_var is not None:
            x = (x - self.global_mean.to(x.device)) / torch.sqrt(
                self.global_var.to(x.device)
            )
        return x

    def denormalize(self, x):
        if self.global_mean is not None and self.global_var is not None:
            x = x * torch.sqrt(self.global_var.to(x.device)) + self.global_mean.to(
                x.device
            )
        return x

    @staticmethod
    def sample_from_latent(latent):
        mean, log_std = latent.chunk(2, 1)
        z = mean + torch.randn_like(mean) * torch.exp(log_std)
        return z.transpose(1, 2)

    @staticmethod
    def prepare_inputs_for_dit(
        hiddens,
        hidden_lens,
        latents,
        latent_lens,
        hidden_proj,
        latent_proj,
        noisy_proj,
        span_mask,
        hidden_patch_size,
        latent_patch_size,
        fm_helper,
        cfg_droprate=-1,
    ):
        assert hidden_patch_size == 1, "Hidden patch size > 1 is not supported."

        B, _, _, device = *hiddens.shape, hiddens.device

        # Gather span hidden states for flow matching using span_mask
        span_hidden_list = []
        for b in range(B):
            indices = span_mask[b].nonzero(as_tuple=False).squeeze(-1)
            span_hidden_list.append(hiddens[b, indices, :])
        hiddens = pad_sequence(span_hidden_list, batch_first=True, padding_value=0.0)
        hidden_lens = torch.tensor(
            [t.size(0) for t in span_hidden_list], device=device, dtype=torch.long
        )

        # Update span_mask to be all True for the new lengths
        max_len = hiddens.size(1)
        span_mask = torch.arange(max_len, device=device).expand(
            B, max_len
        ) < hidden_lens.unsqueeze(1)

        # Prepare history latents
        history_latents = latent_proj(latents)
        fm_dim = history_latents.shape[-1]
        assert (latent_patch_size * history_latents.size(1) % latents.size(1)) == 0
        latent_history_patch_size = (
            latent_patch_size * history_latents.size(1) // latents.size(1)
        )

        # Prepare llm hidden with cfg masking
        cfg_mask = (
            torch.empty((B,), dtype=torch.float, device=latents.device).uniform_(0, 1)
            < cfg_droprate
        )
        hiddens = hidden_proj(mask_data(hiddens, cfg_mask))

        # Prepare noise latents
        xt, ut, times = fm_helper.compute_xt_ut(latents)
        projected_noise = noisy_proj(xt)

        # Initialize empty fm_seq
        hist_chunk_size = hidden_patch_size + latent_history_patch_size
        valid_patch_counts = latent_lens // latent_patch_size
        fm_prefix_lengths = hidden_lens + valid_patch_counts * (
            hist_chunk_size - hidden_patch_size
        )
        fm_gen_lengths = latent_lens + valid_patch_counts * hidden_patch_size
        fm_gen_patch_size = hidden_patch_size + latent_patch_size
        fm_seq_lengths = fm_prefix_lengths + fm_gen_lengths
        fm_seq = torch.zeros(
            (B, fm_seq_lengths.max().item(), fm_dim),
            dtype=history_latents.dtype,
            device=device,
        )
        fm_target = []
        patch_context_lengths = []
        history_latent_span_mask = torch.zeros(
            (B, fm_seq_lengths.max().item()), dtype=torch.bool, device=device
        )  # to mark start positions of each history latents

        # Fill fm_seq
        for b in range(B):
            # Step 1: Interleave hiddens at span positions with patched_latents
            interleaved = []
            span_mask_b = span_mask[b, : hidden_lens[b]]
            interleaved.append(
                hiddens[b, : hidden_lens[b]][span_mask_b].reshape(
                    valid_patch_counts[b], hidden_patch_size, fm_dim
                )
            )
            interleaved.append(
                history_latents[
                    b, : valid_patch_counts[b] * latent_history_patch_size, :
                ].reshape(valid_patch_counts[b], latent_history_patch_size, fm_dim)
            )
            interleaved = torch.cat(interleaved, dim=1)
            interleaved = rearrange(
                interleaved, "n h d -> (n h) d"
            )  # [num_spans*hist_chunk_size, D]

            # Step 2: Build mapping from input positions to fm positions
            position_increment = torch.where(
                span_mask_b, hist_chunk_size, 1
            )  # span->hist_chunk_size, non-span->1
            fm_seq_positions = (
                torch.cumsum(position_increment, dim=0) - position_increment
            )

            # Step 3: Scatter non-span hiddens
            non_span_mask = ~span_mask_b
            non_span_indices = fm_seq_positions[non_span_mask]  # [num_non_spans]
            fm_seq[b, non_span_indices, :] = hiddens[b, : hidden_lens[b]][
                non_span_mask, :
            ]

            # Step 4: Scatter interleaved span tokens
            span_indices = fm_seq_positions[span_mask_b]  # [num_spans]
            span_indices_expanded = torch.stack(
                [span_indices + i for i in range(hist_chunk_size)], dim=1
            )  # [num_spans, hist_chunk_size]
            span_indices_flat = span_indices_expanded.reshape(
                -1
            )  # [num_spans*hist_chunk_size]
            fm_seq[b, span_indices_flat, :] = interleaved
            history_latent_span_mask[b, span_indices] = True
            patch_context_lengths.append(span_indices.clone())

            # Step 5: Fill with noise latents at the end
            noise_part = []
            span_mask_b = span_mask[b, : hidden_lens[b]]
            noise_part.append(
                hiddens[b, : hidden_lens[b]][span_mask_b].reshape(
                    valid_patch_counts[b], hidden_patch_size, fm_dim
                )
            )
            noise_part.append(
                projected_noise[b, : latent_lens[b], :].reshape(
                    valid_patch_counts[b], latent_patch_size, fm_dim
                )
            )
            noise_part = torch.cat(noise_part, dim=1)
            noise_part = rearrange(noise_part, "n h d -> (n h) d")
            noise_start = fm_seq_positions[-1] + position_increment[-1]
            noise_end = noise_start + fm_gen_lengths[b]
            fm_seq[b, noise_start:noise_end, :] = noise_part

            # Step 6: prepare fm_target
            ut_b = ut[b, : latent_lens[b], :]
            fm_target.append(rearrange(ut_b, "(n p) d -> n p d", p=latent_patch_size))

        # Construct fm_attn_mask and fm_pos_ids
        fm_attn_mask, fm_seq_mask, fm_pos_ids = (
            CausalHelper().create_causal_chunk_mask_and_pos(
                batch_size=B,
                C_lens=fm_prefix_lengths,
                Z_lens=fm_gen_lengths,
                span_mask=history_latent_span_mask,
                patch_size=fm_gen_patch_size,
            )
        )
        fm_prefix_lengths = fm_prefix_lengths.unsqueeze(1)
        fm_gen_lengths = fm_gen_lengths.unsqueeze(1)
        fm_target = torch.cat(fm_target, dim=0)
        results = [
            fm_seq,
            fm_target,
            fm_attn_mask,
            fm_seq_mask,
            fm_pos_ids,
            times,
            fm_prefix_lengths,
            fm_gen_lengths,
            fm_gen_patch_size,
        ]
        return tuple(results)

    @staticmethod
    def prepare_meanflow_inputs_for_dit(
        *,
        hiddens: torch.Tensor,
        latents: torch.Tensor,
        latent_lens: torch.Tensor,
        hidden_proj,
        latent_proj,
        noisy_proj,
        span_mask: torch.Tensor,
        hidden_patch_size: int,
        latent_patch_size: int,
        cfg_mask: torch.Tensor,
        noise_latents: torch.Tensor,
    ) -> dict[str, Any]:
        if hidden_patch_size != 1:
            raise ValueError("MeanFlow training only supports hidden_patch_size=1.")

        batch_size = hiddens.size(0)
        device = hiddens.device

        span_hidden_list = []
        for batch_idx in range(batch_size):
            indices = span_mask[batch_idx].nonzero(as_tuple=False).squeeze(-1)
            span_hidden_list.append(hiddens[batch_idx, indices, :])
        hiddens = pad_sequence(span_hidden_list, batch_first=True, padding_value=0.0)
        hidden_lens = torch.tensor(
            [item.size(0) for item in span_hidden_list],
            device=device,
            dtype=torch.long,
        )

        history_latents = latent_proj(latents)
        fm_dim = history_latents.shape[-1]
        latent_history_patch_size = (
            latent_patch_size * history_latents.size(1) // latents.size(1)
        )

        hiddens = hidden_proj(mask_data(hiddens, cfg_mask))
        projected_noise = noisy_proj(noise_latents)

        hist_chunk_size = hidden_patch_size + latent_history_patch_size
        valid_patch_counts = latent_lens // latent_patch_size
        fm_prefix_lengths = hidden_lens + valid_patch_counts * (
            hist_chunk_size - hidden_patch_size
        )
        fm_gen_lengths = latent_lens + valid_patch_counts * hidden_patch_size
        fm_gen_patch_size = hidden_patch_size + latent_patch_size
        fm_seq_lengths = fm_prefix_lengths + fm_gen_lengths
        fm_seq = torch.zeros(
            (batch_size, int(fm_seq_lengths.max().item()), fm_dim),
            dtype=history_latents.dtype,
            device=device,
        )
        history_latent_span_mask = torch.zeros(
            (batch_size, int(fm_seq_lengths.max().item())),
            dtype=torch.bool,
            device=device,
        )
        noise_region_starts = []

        for batch_idx in range(batch_size):
            patch_count = int(valid_patch_counts[batch_idx].item())
            hidden_len = int(hidden_lens[batch_idx].item())
            if patch_count <= 0 or hidden_len <= 0:
                noise_region_starts.append(0)
                continue

            hidden_block = hiddens[batch_idx, :hidden_len].reshape(
                patch_count,
                hidden_patch_size,
                fm_dim,
            )
            history_block = history_latents[
                batch_idx,
                : patch_count * latent_history_patch_size,
                :,
            ].reshape(patch_count, latent_history_patch_size, fm_dim)
            interleaved = rearrange(
                torch.cat([hidden_block, history_block], dim=1),
                "n h d -> (n h) d",
            )

            span_indices = torch.arange(patch_count, device=device) * hist_chunk_size
            span_indices_expanded = torch.stack(
                [span_indices + idx for idx in range(hist_chunk_size)],
                dim=1,
            )
            fm_seq[batch_idx, span_indices_expanded.reshape(-1), :] = interleaved
            history_latent_span_mask[batch_idx, span_indices] = True

            noise_start = patch_count * hist_chunk_size
            noise_region_starts.append(noise_start)
            noise_part = torch.cat(
                [
                    hidden_block,
                    projected_noise[
                        batch_idx,
                        : patch_count * latent_patch_size,
                        :,
                    ].reshape(patch_count, latent_patch_size, fm_dim),
                ],
                dim=1,
            )
            noise_part = rearrange(noise_part, "n h d -> (n h) d")
            noise_end = noise_start + int(fm_gen_lengths[batch_idx].item())
            fm_seq[batch_idx, noise_start:noise_end, :] = noise_part

        fm_attn_mask, fm_seq_mask, fm_pos_ids = (
            CausalHelper().create_causal_chunk_mask_and_pos(
                batch_size=batch_size,
                C_lens=fm_prefix_lengths,
                Z_lens=fm_gen_lengths,
                span_mask=history_latent_span_mask,
                patch_size=fm_gen_patch_size,
            )
        )
        return {
            "fm_seq": fm_seq,
            "fm_attn_mask": fm_attn_mask,
            "fm_seq_mask": fm_seq_mask,
            "fm_pos_ids": fm_pos_ids,
            "fm_prefix_lengths": fm_prefix_lengths.unsqueeze(1),
            "fm_gen_lengths": fm_gen_lengths.unsqueeze(1),
            "fm_gen_patch_size": fm_gen_patch_size,
            "noise_region_starts": torch.tensor(
                noise_region_starts,
                device=device,
                dtype=torch.long,
            ),
            "noise_chunk_size": fm_gen_patch_size,
            "noise_inner_offset": hidden_patch_size,
            "valid_patch_counts": valid_patch_counts,
            "latent_lens": latent_lens,
            "latent_patch_size": latent_patch_size,
        }

    @staticmethod
    def replace_noise_latents_in_fm_seq(
        prefix_data: dict[str, Any],
        new_noise_latents: torch.Tensor,
        noisy_proj,
    ) -> torch.Tensor:
        projected = noisy_proj(new_noise_latents)
        fm_seq = prefix_data["fm_seq"].clone()
        starts = prefix_data["noise_region_starts"]
        chunk_size = int(prefix_data["noise_chunk_size"])
        inner_offset = int(prefix_data["noise_inner_offset"])
        latent_patch_size = int(prefix_data["latent_patch_size"])
        valid_patch_counts = prefix_data["valid_patch_counts"]

        for batch_idx in range(fm_seq.size(0)):
            patch_count = int(valid_patch_counts[batch_idx].item())
            if patch_count <= 0:
                continue
            base = int(starts[batch_idx].item())
            for patch_idx in range(patch_count):
                src_start = patch_idx * latent_patch_size
                dst_start = base + patch_idx * chunk_size + inner_offset
                fm_seq[
                    batch_idx,
                    dst_start : dst_start + latent_patch_size,
                    :,
                ] = projected[
                    batch_idx,
                    src_start : src_start + latent_patch_size,
                    :,
                ]
        return fm_seq

    @staticmethod
    def get_dit_outputs(
        pred_v,
        fm_prefix_lengths,
        fm_gen_lengths,
        fm_gen_patch_size,
        latent_patch_size,
    ):
        B, P = fm_prefix_lengths.shape
        fm_pred = []
        for b in range(B):
            p_offset = 0
            for p in range(P):
                latents_b = pred_v[
                    b,
                    p_offset + fm_prefix_lengths[b][p] : p_offset
                    + fm_prefix_lengths[b][p]
                    + fm_gen_lengths[b][p],
                ]
                latents_b = rearrange(
                    latents_b, "(n p) d -> n p d", p=fm_gen_patch_size
                )
                # extract only the latent parts
                latents_b = latents_b[:, -latent_patch_size:, :]
                fm_pred.append(latents_b)
                p_offset += fm_prefix_lengths[b][p] + fm_gen_lengths[b][p]
        return torch.cat(fm_pred, dim=0)
