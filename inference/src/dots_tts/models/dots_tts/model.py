from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from loguru import logger
from safetensors.torch import load_file, save_file
from transformers import AutoTokenizer, Qwen2Config

from dots_tts.models.dots_tts.config import ModelConfig
from dots_tts.models.dots_tts.core import (
    DotsTtsCore,
    DotsTtsForwardOutput,
    prosody_segment_bounds,
)
from dots_tts.modules.backbone.dit_inference import (
    DiTInferenceContext,
    DiTSolver,
    DiTSolverState,
)
from dots_tts.modules.backbone.encoder_inference import (
    SemanticEncoderInference,
)
from dots_tts.modules.backbone.llm_inference import LLMInference, LLMInferenceState
from dots_tts.modules.backbone.scm_inference import SCMDiTSolver
from dots_tts.modules.speaker.encoder import SpeakerXVectorFeatures
from dots_tts.modules.vocoder.bigvgan import AudioVAE
from dots_tts.modules.vocoder.vocoder_inference import VocoderInference
from dots_tts.training.losses import LossMasks, LossTerm, LossTerms
from dots_tts.utils.logging import categorized_log as logc
from dots_tts.utils.profiling import measure_inference
from dots_tts.utils.tokenizer import (
    AUDIO_GEN_START_TOKEN,
    add_voice_design_tokens,
    has_voice_design_tokens,
    has_token,
    require_token_id,
)
from dots_tts.utils.util import get_dtype


@dataclass
class _GenerateState:
    llm_hiddens: torch.Tensor | None = None
    llm_state: LLMInferenceState = field(default_factory=LLMInferenceState)
    patch_encoder_state: Any | None = None
    fm_seq_len: int = 0
    fm_capacity: int = 0
    fm_sequence: torch.Tensor | None = None
    fm_cfg_sequence: torch.Tensor | None = None
    fm_null_g_cond: torch.Tensor | None = None
    fm_dit_state: DiTSolverState = field(default_factory=DiTSolverState)
    end_flag: bool = False
    # Hidden state read off <|voice_gen_start|> during prefill. It is the only
    # thing the voice flow head needs, which is why that position has to sit
    # inside the prefilled prefix.
    voice_condition_hidden: torch.Tensor | None = None
    # The voice latent actually used this run, kept so g_cond and the
    # <|voice_patch|> embedding are provably the same draw.
    voice_code: torch.Tensor | None = None
    # Cumulative think-chain estimate, read off the first prefill chunk.
    # The think slots sit before <|voice_gen_start|>, so their hidden states
    # already exist by the time the voice code is drawn -- no second pass.
    think_code: torch.Tensor | None = None
    # The acoustic plan, already projected into the DiT's conditioning
    # space: [B, M, fm_hidden]. Read off the prosody slots during prefill,
    # which is the only time they exist -- they sit before the first audio
    # token, so no extra forward pass is needed to obtain them.
    plan_vectors: torch.Tensor | None = None


@dataclass(frozen=True)
class _PromptConditioning:
    prompt_patches: torch.Tensor | None = None
    prompt_latents: torch.Tensor | None = None
    g_cond: torch.Tensor | None = None


@dataclass
class _PromptFeatureCacheEntry:
    speaker_embedding: torch.Tensor | None = None
    prompt_latent_distribution: torch.Tensor | None = None


class DotsTtsModel(nn.Module):
    """Full train/infer model assembly around the dots.tts core network."""

    _GENERATE_LENGTH_BUCKETS = (64, 128, 256, 512)
    _optimize_enabled = True
    _PROMPT_FEATURE_CACHE_MAX_ENTRIES = 256
    VOCODER_STREAM_INITIAL_UNMERGED_PATCHES = 2
    DEFAULT_MAX_SEQUENCE_LENGTH = 2048
    CONFIG_FILENAME = "config.json"
    HF_MODEL_TYPE = "dots_tts"
    HF_ARCHITECTURES = ["DotsTTSForConditionalGeneration"]
    LATENT_STATS_FILENAME = "latent_stats.pt"
    LLM_CONFIG_FILENAME = "llm_config.json"
    MODEL_FILENAME = "model.safetensors"
    VOCODER_FILENAME = "vocoder.safetensors"
    SPEAKER_ENCODER_FILENAME = "speaker_encoder.safetensors"
    _ARTIFACT_ALIASES = (("llm.lm_head.weight", "llm.model.embed_tokens.weight"),)
    REQUIRED_ARTIFACT_FILES = (
        CONFIG_FILENAME,
        LATENT_STATS_FILENAME,
        LLM_CONFIG_FILENAME,
        MODEL_FILENAME,
        VOCODER_FILENAME,
        SPEAKER_ENCODER_FILENAME,
    )

    # region Module assembly and checkpoint IO
    def __init__(
        self,
        config: ModelConfig,
        tokenizer,
        latent_stats_path: str | Path,
        llm_config: Qwen2Config,
    ):
        super().__init__()
        self.config = config
        self.tokenizer = tokenizer
        self.latent_stats_path = (
            None if latent_stats_path is None else Path(latent_stats_path)
        )
        self.audio_gen_start_id = require_token_id(
            self.tokenizer, AUDIO_GEN_START_TOKEN
        )

        self.core = DotsTtsCore(
            config,
            llm_config=llm_config,
            tokenizer=tokenizer,
            latent_stats_path=self.latent_stats_path,
        )
        self.vocoder = AudioVAE(config.vocoder).eval()
        self.vocoder.remove_weight_norm()
        self.hop_size = self.vocoder.hop_size
        self.xvector_extractor = SpeakerXVectorFeatures(
            sample_rate=self.vocoder.sample_rate,
            campplus_embedding_size=config.campplus_embedding_size,
            max_audio_seconds=config.xvec_max_audio_seconds,
        ).eval()

        for param in self.vocoder.parameters():
            param.requires_grad = False
        for param in self.xvector_extractor.parameters():
            param.requires_grad = False
        self._optimize_enabled = True
        self._llm_max_sequence_length = self.DEFAULT_MAX_SEQUENCE_LENGTH
        self._static_generate_workspaces: dict[tuple[Any, ...], dict[str, Any]] = {}
        self._prompt_feature_cache: OrderedDict[
            str, _PromptFeatureCacheEntry
        ] = OrderedDict()
        self._fm_dit_solvers: dict[tuple[str, bool], DiTSolver] = {}
        self._llm_inference: LLMInference | None = None
        self._llm_inference_core_id: int | None = None
        self._patch_encoder_inference: SemanticEncoderInference | None = None
        self._patch_encoder_inference_encoder_id: int | None = None
        self._vocoder_inference: VocoderInference | None = None
        self._vocoder_inference_vocoder_id: int | None = None
        self.latest_voice_metrics: dict[str, float] = {}

    def set_optimize(
        self,
        optimize: bool,
        *,
        max_sequence_length: int | None = None,
    ) -> None:
        if max_sequence_length is not None:
            requested_max_sequence_length = int(max_sequence_length)
            if requested_max_sequence_length <= 0:
                raise ValueError("max_sequence_length must be positive.")
            if requested_max_sequence_length != self._llm_max_sequence_length:
                llm_inference = getattr(self, "_llm_inference", None)
                if llm_inference is not None:
                    llm_inference.clear()
            self._llm_max_sequence_length = requested_max_sequence_length
        self._optimize_enabled = bool(optimize)
        if not self._optimize_enabled:
            self._fm_dit_solvers.clear()
            llm_inference = getattr(self, "_llm_inference", None)
            if llm_inference is not None:
                llm_inference.clear()
            patch_encoder_inference = getattr(self, "_patch_encoder_inference", None)
            if patch_encoder_inference is not None:
                patch_encoder_inference.clear()
            vocoder_inference = getattr(self, "_vocoder_inference", None)
            if vocoder_inference is not None:
                vocoder_inference.clear()

    def set_cfg_droprate(
        self,
        cfg_droprate: float | None = None,
        xvec_drop_rate: float | None = None,
    ) -> None:
        if cfg_droprate is not None:
            self.config.cfg_droprate = cfg_droprate
            self.core.config.cfg_droprate = cfg_droprate
            self.core.cfg_droprate = cfg_droprate

        if xvec_drop_rate is not None:
            self.config.xvec_drop_rate = xvec_drop_rate
            self.core.config.xvec_drop_rate = xvec_drop_rate
            self.core.xvec_drop_rate = xvec_drop_rate

    @classmethod
    def _resolve_generate_length_bucket(
        cls,
        max_generate_length: int,
    ) -> int:
        requested = int(max_generate_length)
        if requested <= 0:
            raise ValueError("max_generate_length must be positive.")
        for bucket in cls._GENERATE_LENGTH_BUCKETS:
            if requested <= bucket:
                return bucket
        raise ValueError(
            "max_generate_length exceeds the largest supported compile bucket: "
            f"max_generate_length={requested} "
            f"max_supported={cls._GENERATE_LENGTH_BUCKETS[-1]}."
        )

    def _get_llm_inference(self) -> LLMInference:
        core_id = id(self.core)
        adapter = getattr(self, "_llm_inference", None)
        if adapter is None or getattr(self, "_llm_inference_core_id", None) != core_id:
            adapter = LLMInference(self.core)
            self._llm_inference = adapter
            self._llm_inference_core_id = core_id
        return adapter

    def _get_vocoder_inference(self) -> VocoderInference:
        vocoder_id = id(self.vocoder)
        adapter = getattr(self, "_vocoder_inference", None)
        if (
            adapter is None
            or getattr(self, "_vocoder_inference_vocoder_id", None) != vocoder_id
        ):
            adapter = VocoderInference(self.vocoder)
            self._vocoder_inference = adapter
            self._vocoder_inference_vocoder_id = vocoder_id
        return adapter

    def _allocate_generate_state(
        self,
        *,
        max_audio_patch_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> _GenerateState:
        state_dtype = dtype if device.type == "cuda" else torch.float32
        requested_audio_patch_count = int(max_audio_patch_count)
        if requested_audio_patch_count <= 0:
            raise ValueError("max_audio_patch_count must be positive.")
        state_audio_patch_count = (
            self._resolve_generate_length_bucket(requested_audio_patch_count)
            if self._optimize_enabled
            else requested_audio_patch_count
        )
        fm_capacity = state_audio_patch_count * (
            self.core.hidden_patch_size + self.core.latent_patch_size
        )
        workspace_key = (
            state_audio_patch_count,
            str(device),
            state_dtype,
        )
        workspace = self._static_generate_workspaces.get(workspace_key)
        if workspace is None:
            workspace = {
                "fm_sequence": torch.zeros(
                    (1, fm_capacity, self.core.fm_hidden_size),
                    dtype=state_dtype,
                    device=device,
                ),
                "fm_cfg_sequence": torch.zeros(
                    (1, fm_capacity, self.core.fm_hidden_size),
                    dtype=state_dtype,
                    device=device,
                ),
                "fm_null_g_cond": torch.zeros(
                    (1, self.core.fm_hidden_size),
                    dtype=state_dtype,
                    device=device,
                ),
            }
            self._static_generate_workspaces[workspace_key] = workspace
        else:
            workspace["fm_sequence"].zero_()
            workspace["fm_cfg_sequence"].zero_()

        llm_state = self._get_llm_inference().init_state(
            optimize=self._optimize_enabled,
            max_sequence_length=self._llm_max_sequence_length,
            device=device,
            dtype=self.core.llm.get_input_embeddings().weight.dtype,
        )

        return _GenerateState(
            llm_state=llm_state,
            patch_encoder_state=None,
            fm_seq_len=0,
            fm_capacity=fm_capacity,
            fm_sequence=workspace["fm_sequence"],
            fm_cfg_sequence=workspace["fm_cfg_sequence"],
            fm_null_g_cond=workspace["fm_null_g_cond"],
        )

    @staticmethod
    def _tensor_storage_signature(tensor: torch.Tensor) -> tuple:
        return (
            tensor.untyped_storage().data_ptr(),
            tensor.storage_offset(),
            tuple(tensor.size()),
            tuple(tensor.stride()),
            tensor.dtype,
        )

    @classmethod
    def _build_artifact_state_dict(cls, module) -> dict[str, torch.Tensor]:
        state_dict = module.state_dict()
        skip_keys = set()

        for redundant_key, canonical_key in cls._ARTIFACT_ALIASES:
            redundant_tensor = state_dict.get(redundant_key)
            canonical_tensor = state_dict.get(canonical_key)
            if (
                redundant_tensor is not None
                and canonical_tensor is not None
                and cls._tensor_storage_signature(redundant_tensor)
                == cls._tensor_storage_signature(canonical_tensor)
            ):
                skip_keys.add(redundant_key)

        cleaned_state_dict = {}
        seen_storage = set()
        for key, value in state_dict.items():
            if key in skip_keys:
                continue

            storage_signature = cls._tensor_storage_signature(value)
            if storage_signature in seen_storage:
                continue

            seen_storage.add(storage_signature)
            cleaned_state_dict[key] = value.detach().cpu().contiguous()

        return cleaned_state_dict

    @classmethod
    def _restore_artifact_state_dict(cls, state_dict: dict, module) -> dict:
        restored_state_dict = dict(state_dict)
        for redundant_key, canonical_key in cls._ARTIFACT_ALIASES:
            if (
                canonical_key in restored_state_dict
                and redundant_key not in restored_state_dict
                and redundant_key in module.state_dict()
            ):
                restored_state_dict[redundant_key] = restored_state_dict[canonical_key]
        return restored_state_dict

    @classmethod
    def _save_artifact_module(cls, module, path: Path) -> None:
        save_file(cls._build_artifact_state_dict(module), path)

    @classmethod
    def _load_artifact_module(cls, module, path: Path):
        state_dict = load_file(path, device="cpu")
        restored_state_dict = cls._restore_artifact_state_dict(state_dict, module)
        mismatch = module.load_state_dict(restored_state_dict, strict=False)
        if mismatch.missing_keys or mismatch.unexpected_keys:
            raise RuntimeError(f"Failed to load {path}: {mismatch}")
        return module

    _EMBEDDING_KEYS = ("llm.model.embed_tokens.weight", "llm.lm_head.weight")

    @classmethod
    def _grow_embedding_rows(cls, state_dict: dict, module) -> dict:
        """Pad saved embedding matrices up to the current (extended) vocabulary.

        ``load_state_dict`` turns a row-count mismatch into a hard error even
        under ``strict=False``, so the rows have to be grown before the load
        rather than resized after it. New rows are initialized from the mean of
        the pretrained embeddings plus a little noise: a freshly random row
        lands far outside the region the LLM's first layer expects, and the
        symmetry has to be broken or the three new tokens stay interchangeable.
        """

        grown_state_dict = dict(state_dict)
        target_state_dict = module.state_dict()
        for key in cls._EMBEDDING_KEYS:
            saved = grown_state_dict.get(key)
            wanted = target_state_dict.get(key)
            if saved is None or wanted is None or saved.shape == wanted.shape:
                continue
            if saved.dim() != 2 or saved.size(1) != wanted.size(1):
                raise RuntimeError(
                    f"Cannot reconcile {key}: saved shape {tuple(saved.shape)} vs "
                    f"model shape {tuple(wanted.shape)}."
                )
            if saved.size(0) > wanted.size(0):
                raise RuntimeError(
                    f"Saved {key} has more rows than the model expects: "
                    f"{saved.size(0)} > {wanted.size(0)}. The tokenizer shrank."
                )
            grown = saved.new_empty(wanted.shape)
            grown[: saved.size(0)] = saved
            mean_row = saved.float().mean(dim=0)
            noise_scale = float(saved.float().std().item()) * 0.02
            new_rows = mean_row.unsqueeze(0).repeat(wanted.size(0) - saved.size(0), 1)
            new_rows = new_rows + torch.randn_like(new_rows) * noise_scale
            grown[saved.size(0) :] = new_rows.to(grown.dtype)
            grown_state_dict[key] = grown
            logger.info(
                logc("model", "Grew {} from {} to {} rows for new special tokens."),
                key,
                saved.size(0),
                wanted.size(0),
            )
        return grown_state_dict

    @classmethod
    def _load_core_allowing_new_modules(
        cls,
        module,
        path: Path,
        *,
        allowed_missing_prefixes: tuple[str, ...],
    ):
        state_dict = load_file(path, device="cpu")
        state_dict = cls._restore_artifact_state_dict(state_dict, module)
        state_dict = cls._grow_embedding_rows(state_dict, module)
        mismatch = module.load_state_dict(state_dict, strict=False)
        unexpected = list(mismatch.unexpected_keys)
        missing = [
            key
            for key in mismatch.missing_keys
            if not key.startswith(allowed_missing_prefixes)
        ]
        if missing or unexpected:
            raise RuntimeError(
                f"Failed to load {path}: missing={missing} unexpected={unexpected}"
            )
        return module

    @classmethod
    def _validate_pretrained_directory(
        cls, pretrained_model_name_or_path: str | Path
    ) -> Path:
        pretrained_path = Path(pretrained_model_name_or_path).expanduser().resolve()
        missing_files = [
            name
            for name in cls.REQUIRED_ARTIFACT_FILES
            if not (pretrained_path / name).is_file()
        ]
        if missing_files:
            # A training checkpoint directory holds the save_pretrained artifact
            # in a `model/` subdirectory, next to optimizer shards that inference
            # has no use for. Pointing at the outer directory is the natural
            # mistake, so name the directory that would have worked instead of
            # only listing what is absent.
            hint = ""
            nested = pretrained_path / "model"
            if all(
                (nested / name).is_file() for name in cls.REQUIRED_ARTIFACT_FILES
            ):
                hint = (
                    f"\nThis looks like a training checkpoint: use {nested} "
                    "instead, or the matching directory under `exports/`, which "
                    "checkpoint cleanup does not delete."
                )
            raise FileNotFoundError(
                f"Pretrained path {pretrained_path} is missing required files: "
                f"{missing_files}{hint}"
            )
        return pretrained_path

    @classmethod
    def _load_pretrained_config(cls, pretrained_path: Path) -> ModelConfig:
        return ModelConfig.model_validate(
            json.loads(
                (pretrained_path / cls.CONFIG_FILENAME).read_text(encoding="utf-8")
            )
        )

    @staticmethod
    def _save_llm_config(llm_config: Qwen2Config, path: Path) -> None:
        path.write_text(
            json.dumps(llm_config.to_dict(), ensure_ascii=True, indent=2),
            encoding="utf-8",
        )

    @staticmethod
    def _load_llm_config(path: Path) -> Qwen2Config:
        return Qwen2Config.from_dict(json.loads(path.read_text(encoding="utf-8")))

    def _tie_llm_weights(self) -> None:
        if hasattr(self.core.llm, "tie_weights"):
            self.core.llm.tie_weights()

    def save_pretrained(self, save_directory: str | Path) -> Path:
        save_directory = Path(save_directory)
        save_directory.mkdir(parents=True, exist_ok=True)

        config_payload = self.config.to_declared_dict()
        config_payload["model_type"] = self.HF_MODEL_TYPE
        config_payload["architectures"] = list(self.HF_ARCHITECTURES)
        (save_directory / self.CONFIG_FILENAME).write_text(
            json.dumps(config_payload, ensure_ascii=True, indent=2),
            encoding="utf-8",
        )
        self._save_llm_config(
            self.core.llm.config,
            save_directory / self.LLM_CONFIG_FILENAME,
        )
        self.tokenizer.save_pretrained(save_directory)
        shutil.copy2(
            self.latent_stats_path,
            save_directory / self.LATENT_STATS_FILENAME,
        )
        self._save_artifact_module(self.core, save_directory / self.MODEL_FILENAME)
        self._save_artifact_module(self.vocoder, save_directory / self.VOCODER_FILENAME)
        self._save_artifact_module(
            self.xvector_extractor,
            save_directory / self.SPEAKER_ENCODER_FILENAME,
        )
        return save_directory

    def _load_pretrained_artifacts(self, pretrained_path: Path) -> None:
        self.latent_stats_path = pretrained_path / self.LATENT_STATS_FILENAME
        self.core.io_helper = type(self.core.io_helper)(
            latent_stats_path=self.latent_stats_path
        )
        self._load_artifact_module(self.core, pretrained_path / self.MODEL_FILENAME)
        self._tie_llm_weights()
        self._load_artifact_module(
            self.vocoder, pretrained_path / self.VOCODER_FILENAME
        )
        self._load_artifact_module(
            self.xvector_extractor,
            pretrained_path / self.SPEAKER_ENCODER_FILENAME,
        )
        self.core.eval()
        self.vocoder.eval()
        self.xvector_extractor.eval()

    def load_pretrained_weights(
        self, pretrained_model_name_or_path: str | Path
    ) -> None:
        pretrained_path = self._validate_pretrained_directory(
            pretrained_model_name_or_path
        )
        saved_config = self._load_pretrained_config(pretrained_path)
        if saved_config.to_declared_dict() != self.config.to_declared_dict():
            raise ValueError(
                f"Pretrained config at {pretrained_path} does not match the current model."
            )
        saved_llm_config = self._load_llm_config(
            pretrained_path / self.LLM_CONFIG_FILENAME
        )
        if saved_llm_config.to_dict() != self.core.llm.config.to_dict():
            raise ValueError(
                f"Pretrained LLM config at {pretrained_path} does not match the current model."
            )
        self._load_pretrained_artifacts(pretrained_path)

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: str | Path):
        logger.debug(
            logc("model", "DotsTtsModel load started: pretrained_path={}"),
            pretrained_model_name_or_path,
        )
        pretrained_model_name_or_path = cls._validate_pretrained_directory(
            pretrained_model_name_or_path
        )
        config = cls._load_pretrained_config(pretrained_model_name_or_path)
        llm_config = cls._load_llm_config(
            pretrained_model_name_or_path / cls.LLM_CONFIG_FILENAME
        )
        logger.debug(
            logc(
                "model",
                "DotsTtsModel config loaded: pretrained_path={} sample_rate={} patch_size={}",
            ),
            pretrained_model_name_or_path,
            config.vocoder.sample_rate,
            config.patch_size,
        )
        tokenizer = AutoTokenizer.from_pretrained(
            str(pretrained_model_name_or_path),
            local_files_only=True,
        )
        model = cls(
            config,
            tokenizer=tokenizer,
            latent_stats_path=pretrained_model_name_or_path / cls.LATENT_STATS_FILENAME,
            llm_config=llm_config,
        )
        model._load_pretrained_artifacts(pretrained_model_name_or_path)
        logger.info(
            logc("model", "DotsTtsModel load completed: pretrained_path={}"),
            pretrained_model_name_or_path,
        )
        return model.eval()

    @staticmethod
    def _load_voice_extractor_checkpoint(model, voice_design) -> None:
        """Install Stage-1 Voice-QFormer weights, if a checkpoint was given.

        Loaded here rather than in the branch constructor so that only the
        Stage-2 bootstrap touches the filesystem: once the model is saved, the
        extractor lives in the artifact's own state dict and later loads —
        including inference on a machine that never had the Stage-1 file — do
        not need the path to still resolve.
        """

        checkpoint = getattr(voice_design, "extractor_checkpoint", None)
        extractor = getattr(model.core.voice_design, "extractor", None)
        if not checkpoint:
            if extractor is not None and bool(voice_design.freeze_extractor):
                raise ValueError(
                    "voice_source='qformer' with freeze_extractor=true needs "
                    "extractor_checkpoint: a frozen randomly-initialized teacher "
                    "would define the target as noise and the flow head would "
                    "dutifully learn it. Run scripts/train_voice_extractor.py, "
                    "or set voice_source='pooled'."
                )
            return
        if extractor is None:
            raise ValueError(
                "extractor_checkpoint was given but voice_source is "
                f"{voice_design.voice_source!r}, which builds no extractor."
            )
        path = Path(checkpoint)
        if not path.is_file():
            raise FileNotFoundError(f"extractor_checkpoint does not exist: {path}")
        payload = torch.load(str(path), map_location="cpu", weights_only=False)
        if getattr(voice_design, "pooled_residual", False):
            settings = payload.get("settings", {})
            if payload.get("kind") != "bounded_pool_probe_v1":
                raise ValueError("Expected a bounded residual-pool checkpoint, not a QFormer checkpoint")
            for key, expected in (("latent_dim", model.core.latent_dim),
                                  ("hidden", voice_design.pooled_residual_hidden),
                                  ("alpha", voice_design.pooled_residual_alpha)):
                if settings.get(key) != expected:
                    raise ValueError(f"Residual-pool {key} mismatch: {settings.get(key)} != {expected}")
            extractor.load_state_dict(payload["state_dict"], strict=True)
            extractor.requires_grad_(False)
            logger.info("Loaded frozen residual pooling adapter: {}", path)
            return
        state = payload.get("extractor", payload)
        missing, unexpected = extractor.load_state_dict(state, strict=False)
        if missing or unexpected:
            raise RuntimeError(
                f"Stage-1 checkpoint does not match the configured Voice-QFormer: "
                f"missing={sorted(missing)} unexpected={sorted(unexpected)}. The "
                "qformer_* and voice_dim settings must match Stage-1."
            )
        logger.info(
            logc("model", "Loaded Voice-QFormer teacher: path={} voice_dim={}"),
            path,
            voice_design.voice_dim,
        )

    @classmethod
    def from_pretrained_for_voice_design(
        cls,
        pretrained_model_name_or_path: str | Path,
        voice_design: Any = None,
    ):
        """Bootstrap a voice-design model from a stock dots.tts artifact.

        A released checkpoint has neither the voice-design tokens nor the branch
        weights, so a first voice-design run cannot go through
        ``from_pretrained``: the tokenizer has to grow by the voice token pair
        (``<|voice_gen_start|>`` and ``<|voice_patch|>``), the LLM embedding has
        to grow with it, and the branch has to start fresh. Once
        this model is saved with ``save_pretrained`` the artifact is
        self-consistent again and every later load — including resume — uses the
        ordinary path.

        Calling this on an artifact that *already* carries the tokens is
        harmless: nothing is added, nothing is grown, and the branch weights load
        normally.
        """

        from dots_tts.models.dots_tts.config import VoiceDesignConfig

        pretrained_path = cls._validate_pretrained_directory(
            pretrained_model_name_or_path
        )
        config = cls._load_pretrained_config(pretrained_path)
        llm_config = cls._load_llm_config(pretrained_path / cls.LLM_CONFIG_FILENAME)
        tokenizer = AutoTokenizer.from_pretrained(
            str(pretrained_path), local_files_only=True
        )
        added = add_voice_design_tokens(tokenizer)
        if not has_voice_design_tokens(tokenizer):
            raise RuntimeError(
                "Failed to register the voice-design tokens on the artifact tokenizer."
            )

        if voice_design is None:
            voice_design = config.get("voice_design", None) or VoiceDesignConfig()
        elif isinstance(voice_design, dict):
            voice_design = VoiceDesignConfig.model_validate(voice_design)
        voice_design.enabled = True
        config.voice_design = voice_design

        model = cls(
            config,
            tokenizer=tokenizer,
            latent_stats_path=pretrained_path / cls.LATENT_STATS_FILENAME,
            llm_config=llm_config,
        )
        model.latent_stats_path = pretrained_path / cls.LATENT_STATS_FILENAME
        model.core.io_helper = type(model.core.io_helper)(
            latent_stats_path=model.latent_stats_path
        )
        cls._load_core_allowing_new_modules(
            model.core,
            pretrained_path / cls.MODEL_FILENAME,
            # Every module voice design adds to `core`, not just the branch.
            # voice_proj and voice_embed_proj live directly on DotsTtsCore, so
            # omitting them here makes the very first bootstrap raise.
            allowed_missing_prefixes=(
                "voice_design.",
                "voice_proj.",
                "voice_embed_proj.",
                "instruction_g_proj.",
                "semantic_patch_proj.",
                # The think chain is new in the same sense the branch once
                # was: a checkpoint trained without think slots carries none
                # of it. The downstream projection starts at zero; the head
                # does not, so the acoustic gradient can open this path.
                # Extra LM tokens mean this is not exact baseline equivalence.
                "think_head.",
                "think_g_proj.",
                "prosody_head.",
                "plan_g_proj.",
            ),
        )
        model._tie_llm_weights()
        cls._load_artifact_module(model.vocoder, pretrained_path / cls.VOCODER_FILENAME)
        cls._load_artifact_module(
            model.xvector_extractor,
            pretrained_path / cls.SPEAKER_ENCODER_FILENAME,
        )
        cls._load_voice_extractor_checkpoint(model, voice_design)
        logger.info(
            logc(
                "model",
                "Voice-design model prepared: pretrained_path={} added_tokens={} "
                "vocab_size={} voice_source={} voice_dim={}",
            ),
            pretrained_path,
            added,
            len(tokenizer),
            voice_design.voice_source,
            voice_design.voice_dim,
        )
        return model

    # endregion Module assembly and checkpoint IO

    # region Training batch preparation
    @torch.no_grad()
    def prepare_training_inputs(self, data: dict[str, Any]) -> dict[str, Any]:
        self.vocoder.eval()
        self.xvector_extractor.eval()
        processed = dict(data)
        if data.get("features_precomputed"):
            if data.get("latents_sampled") is None:
                raise ValueError(
                    "Precomputed batch is missing the sampled AudioVAE latent."
                )
            if data.get("latent_lengths") is None:
                raise ValueError("Precomputed batch is missing latent_lengths.")
            if data.get("xvector") is None:
                raise ValueError("Precomputed batch is missing the CAM++ xvector.")
            processed["latents"] = None
            return processed

        sample: torch.Tensor | None = data.get("sample")
        sample_lengths: torch.Tensor | None = data.get("sample_lengths")

        if sample is not None:
            latents = self.vocoder.extract_latents(sample)
            processed["latents"] = latents
            if sample_lengths is not None:
                processed["latent_lengths"] = sample_lengths // self.hop_size
            else:
                processed["latent_lengths"] = torch.full(
                    (latents.size(0),),
                    latents.size(-1),
                    dtype=torch.long,
                    device=latents.device,
                )
            processed["latents_sampled"] = self.core.io_helper.sample_from_latent(
                latents
            )
            fbank = data.get("fbank")
            fbank_lengths = data.get("fbank_lengths")
            processed["xvector"] = self.xvector_extractor(
                sample,
                audio_lengths=sample_lengths,
                fbank=fbank,
                fbank_lengths=fbank_lengths,
            )
        else:
            processed["latents"] = None
            processed["latent_lengths"] = None

        return processed

    def _build_audio_span_mask(self, token_ids: torch.Tensor) -> torch.Tensor:
        span_mask = torch.zeros_like(token_ids, dtype=torch.bool)
        for token_id in self.core.audio_span_token_ids:
            span_mask = span_mask | (token_ids == token_id)
        return span_mask

    def _build_voice_masks(
        self, token_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        gen_start_id = getattr(self.core, "voice_gen_start_id", None)
        patch_id = getattr(self.core, "voice_patch_id", None)
        if gen_start_id is None or patch_id is None:
            return None
        return token_ids == int(gen_start_id), token_ids == int(patch_id)

    def _build_prosody_mask(
        self, token_ids: torch.Tensor
    ) -> torch.Tensor | None:
        """Positions holding ``<|prosody_think|>``, or None without them."""

        prosody_id = getattr(self.core, "prosody_think_id", None)
        if prosody_id is None:
            return None
        return token_ids == int(prosody_id)

    def _build_voice_think_mask(
        self, token_ids: torch.Tensor
    ) -> torch.Tensor | None:
        """Positions holding ``<|voice_think|>``, or None on a model without them."""

        think_id = getattr(self.core, "voice_think_id", None)
        if think_id is None:
            return None
        return token_ids == int(think_id)

    def _build_voice_loss_masks(
        self,
        voice_masks: tuple[torch.Tensor, torch.Tensor] | None,
        reference: torch.Tensor,
        prompt_mask: torch.Tensor | None = None,
        think_mask: torch.Tensor | None = None,
        prosody_mask: torch.Tensor | None = None,
        input_span_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Masks for the voice terms, derived from ``input_ids`` alone.

        The trainer collapses these into denominators one full optimizer step
        *before* running any forward pass, so they cannot depend on model
        output. Deriving them from the token layout keeps them exactly in step
        with what the branch will produce.

        A cloning row is excluded here: its voice latent was handed to the model
        by the reference encoder, so scoring the flow head against it would be
        scoring it on an answer it was given.
        """

        if voice_masks is None or self.core.voice_design is None:
            return {}
        voice_gen_mask, voice_patch_mask = voice_masks
        rows = voice_gen_mask.any(dim=1) & voice_patch_mask.any(dim=1)
        if prompt_mask is not None:
            rows = rows & ~prompt_mask.to(rows.device).bool()
        # The voice latent is one token, so every voice term is [B, 1]: a row
        # either carries a voice patch or it does not.
        rows = rows.to(reference.dtype).unsqueeze(1).contiguous()
        masks = {
            "voice_flow_loss": rows,
            "voice_qformer_loss": rows.clone(),
            "voice_align_loss": rows.clone(),
            "voice_relation_loss": rows.clone(),
        }
        num_think = int(getattr(self.core, "num_think_slots", 0) or 0)
        if think_mask is not None and num_think > 0:
            # A row counts toward the chain only if it carries the full K
            # slots AND a voice patch: the chain is scored against the
            # branch's teacher code, which a row without a patch never has.
            # Unlike the other voice terms this mask is [B, K] -- one weight
            # per refinement step, so an early step is normalized the same
            # way as a late one.
            think_rows = (
                think_mask.sum(dim=1)
                .eq(num_think)
                .to(device=rows.device, dtype=rows.dtype)
                .unsqueeze(1)
            )
            masks["voice_think_loss"] = (
                (rows * think_rows).expand(-1, num_think).contiguous()
            )
        num_prosody = int(getattr(self.core, "num_prosody_slots", 0) or 0)
        if (
            prosody_mask is not None
            and input_span_mask is not None
            and num_prosody > 0
        ):
            # Segment m of a row holding P audio patches covers patches
            # [m*P//M, (m+1)*P//M), so a row with fewer patches than slots
            # leaves some segments empty. Those slots are masked out rather
            # than scored against a target that does not exist.
            #
            # Deliberately NOT multiplied by `rows`: the plan's target comes
            # from the audio, not from the voice branch's teacher code, so a
            # cloning row -- excluded from every other voice term because its
            # latent was handed to it -- still has a real plan to learn.
            patches = input_span_mask.sum(dim=1).to(device=rows.device).long()
            # Same function the target and the DiT conditioning use. Three
            # copies of this arithmetic is exactly how a mask ends up
            # normalizing over a different set of slots than it scored.
            low, high = prosody_segment_bounds(
                patches,
                num_prosody,
                int(getattr(self.core, "prosody_patches_per_slot", 4) or 4),
            )
            non_empty = high > low
            has_slots = (
                prosody_mask.sum(dim=1)
                .eq(num_prosody)
                .to(device=rows.device)
                .unsqueeze(1)
            )
            masks["voice_prosody_loss"] = (
                (non_empty & has_slots).to(reference.dtype).contiguous()
            )
        return masks

    def _prepare_loss_metadata(self, data: dict[str, Any]) -> dict[str, Any]:
        input_ids: torch.Tensor = data["input_ids"]
        labels: torch.Tensor = data["labels"]
        loss_mask: torch.Tensor = data["loss_mask"]
        input_span_mask = self._build_audio_span_mask(input_ids)
        output_span_mask = self._build_audio_span_mask(labels)
        output_span_mask_float = output_span_mask.to(loss_mask.dtype)
        llm_loss_mask = loss_mask * (1.0 - output_span_mask_float)
        fm_loss_mask = loss_mask * output_span_mask_float
        patch_counts = output_span_mask.sum(dim=1)
        max_patch_count = max(1, int(patch_counts.max().item()))
        fm_patch_mask = loss_mask.new_zeros((loss_mask.size(0), max_patch_count))
        for batch_idx in range(loss_mask.size(0)):
            count = int(patch_counts[batch_idx].item())
            if count <= 0:
                continue
            fm_patch_mask[batch_idx, :count] = fm_loss_mask[batch_idx].masked_select(
                output_span_mask[batch_idx]
            )

        voice_masks = self._build_voice_masks(input_ids)
        voice_think_mask = self._build_voice_think_mask(input_ids)
        voice_prosody_mask = self._build_prosody_mask(input_ids)
        loss_masks = {
            "ce_loss": llm_loss_mask,
            "fm_loss": fm_patch_mask,
            "eos_loss": self._build_eos_loss_mask(fm_loss_mask),
        }
        loss_masks.update(
            self._build_voice_loss_masks(
                voice_masks,
                loss_mask,
                data.get("voice_prompt_mask"),
                think_mask=voice_think_mask,
                prosody_mask=voice_prosody_mask,
                input_span_mask=input_span_mask,
            )
        )
        metadata = {
            "input_span_mask": input_span_mask,
            "output_span_mask": output_span_mask,
            "loss_masks": loss_masks,
        }
        if voice_masks is not None:
            metadata["voice_gen_mask"], metadata["voice_patch_mask"] = voice_masks
        if voice_think_mask is not None:
            metadata["voice_think_mask"] = voice_think_mask
        if voice_prosody_mask is not None:
            metadata["voice_prosody_mask"] = voice_prosody_mask
        return metadata

    @staticmethod
    def _build_eos_loss_mask(eos_loss_mask: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len = eos_loss_mask.shape
        mask = eos_loss_mask.to(dtype=torch.bool)
        target = torch.zeros((batch_size, seq_len), dtype=torch.bool, device=mask.device)
        mask_counts = mask.sum(dim=1, keepdim=True)
        cumulative = mask.long().cumsum(dim=1)
        target[mask & (cumulative == mask_counts)] = True

        mask_counts_flat = mask_counts.squeeze(1)
        neg_counts = (mask_counts_flat - 1).clamp_min(0).to(eos_loss_mask.dtype)
        pos_weight = torch.where(
            neg_counts > 0,
            torch.full_like(neg_counts, 0.5),
            torch.ones_like(neg_counts),
        ).unsqueeze(1)
        neg_weight = torch.where(
            neg_counts > 0,
            0.5 / neg_counts,
            torch.zeros_like(neg_counts),
        ).unsqueeze(1)

        positive_mask = target & mask
        negative_mask = mask & ~positive_mask
        return torch.where(
            positive_mask,
            pos_weight,
            negative_mask.to(eos_loss_mask.dtype) * neg_weight,
        )
    # endregion Training batch preparation

    # region Training loss assembly and forward
    @staticmethod
    def _compute_ce_loss_term(
        llm_logits: torch.Tensor,
        llm_labels: torch.Tensor,
        llm_loss_mask: torch.Tensor,
    ) -> LossTerm:
        vocab_size = llm_logits.size(-1)
        ce_loss = F.cross_entropy(
            llm_logits.view(-1, vocab_size),
            llm_labels.view(-1),
            reduction="none",
        ).view_as(llm_labels)
        return LossTerm(loss=ce_loss, mask=llm_loss_mask.to(ce_loss.dtype))

    @staticmethod
    def _compute_fm_loss_term(
        pred: torch.Tensor,
        target: torch.Tensor,
        fm_patch_mask: torch.Tensor,
    ) -> LossTerm:
        batch_size, max_patch_count = fm_patch_mask.shape
        fm_loss = (pred - target).pow(2).mean(dim=2).mean(dim=1)
        loss = fm_loss.new_zeros((batch_size, max_patch_count))
        patch_counts = fm_patch_mask.gt(0).sum(dim=1).tolist()
        expected_count = int(sum(patch_counts))
        if expected_count > 0 and int(fm_loss.numel()) != expected_count:
            raise RuntimeError(
                "Flow-matching loss count mismatch: "
                f"expected {expected_count}, got {int(fm_loss.numel())}."
            )

        offset = 0
        for batch_idx, patch_count in enumerate(patch_counts):
            if patch_count <= 0:
                continue
            next_offset = offset + int(patch_count)
            loss[batch_idx, :patch_count] = fm_loss[offset:next_offset]
            offset = next_offset
        return LossTerm(loss=loss, mask=fm_patch_mask.to(loss.dtype))

    @staticmethod
    def _compute_eos_loss_term(
        eos_out: torch.Tensor,
        eos_loss_mask: torch.Tensor,
    ) -> LossTerm:
        batch_size, seq_len, _ = eos_out.shape
        weights = eos_loss_mask.to(device=eos_out.device)
        mask = weights.gt(0)
        target = torch.zeros(
            (batch_size, seq_len),
            dtype=torch.long,
            device=eos_out.device,
        )
        mask_counts = mask.sum(dim=1, keepdim=True)
        cumulative = mask.long().cumsum(dim=1)
        target[mask & (cumulative == mask_counts)] = 1

        logits = rearrange(eos_out, "b n c -> b c n")
        ce_per_token = F.cross_entropy(logits, target, reduction="none")
        return LossTerm(loss=ce_per_token, mask=weights.to(ce_per_token.dtype))

    @staticmethod
    def _compute_eos_loss_stats(
        eos_out: torch.Tensor,
        eos_loss_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        weights = DotsTtsModel._build_eos_loss_mask(eos_loss_mask)
        term = DotsTtsModel._compute_eos_loss_term(eos_out, weights)
        mask = term.mask.to(device=term.loss.device, dtype=term.loss.dtype)
        eos_loss_sum = (term.loss * mask).sum(dim=1)
        eos_sample_count = eos_loss_mask.to(device=term.loss.device).gt(0).any(
            dim=1
        ).to(term.loss.dtype)
        return eos_loss_sum, eos_sample_count

    @staticmethod
    def _compute_voice_loss_terms(
        outputs: DotsTtsForwardOutput,
        loss_masks: LossMasks,
    ) -> LossTerms:
        voice = outputs.voice_design
        if voice is None or "voice_flow_loss" not in loss_masks:
            return {}

        def _term(loss: torch.Tensor, name: str, scale: float = 1.0) -> LossTerm:
            mask = loss_masks[name].to(device=loss.device, dtype=loss.dtype)
            if mask.shape != loss.shape:
                raise RuntimeError(
                    f"{name} mask shape {tuple(mask.shape)} does not match the "
                    f"branch output shape {tuple(loss.shape)}."
                )
            return LossTerm(loss=loss * float(scale), mask=mask)

        terms: LossTerms = {
            "voice_flow_loss": _term(voice.flow_anchor_loss, "voice_flow_loss"),
            "voice_qformer_loss": _term(voice.flow_voice_loss, "voice_qformer_loss"),
            # The annealed factor is folded into the loss rather than the mask:
            # the mask is also the normalizer, so scaling it would divide the
            # factor straight back out.
            "voice_align_loss": _term(
                voice.align_loss, "voice_align_loss", voice.align_weight
            ),
            "voice_relation_loss": _term(voice.relation_loss, "voice_relation_loss"),
        }
        # Guarded on the mask, not just on the tensor: a numerator without a
        # matching normalizer is a KeyError in the reduction, and the mask is
        # built one optimizer step ahead of this forward.
        if (
            outputs.voice_think_loss is not None
            and "voice_think_loss" in loss_masks
        ):
            terms["voice_think_loss"] = _term(
                outputs.voice_think_loss, "voice_think_loss"
            )
        if (
            outputs.voice_prosody_loss is not None
            and "voice_prosody_loss" in loss_masks
        ):
            terms["voice_prosody_loss"] = _term(
                outputs.voice_prosody_loss, "voice_prosody_loss"
            )
        return terms

    def _compute_loss_terms(
        self,
        outputs: DotsTtsForwardOutput,
        *,
        labels: torch.Tensor,
        loss_masks: LossMasks,
    ) -> LossTerms:
        terms: LossTerms = {
            "ce_loss": self._compute_ce_loss_term(
                outputs.llm_logits,
                labels,
                loss_masks["ce_loss"],
            ),
            "fm_loss": self._compute_fm_loss_term(
                outputs.pred,
                outputs.target,
                loss_masks["fm_loss"],
            ),
            "eos_loss": self._compute_eos_loss_term(
                outputs.eos_out,
                loss_masks["eos_loss"],
            ),
        }
        terms.update(self._compute_voice_loss_terms(outputs, loss_masks))
        return terms

    def prepare_training_batch(self, data: dict[str, Any]) -> dict[str, Any]:
        prepared = dict(data)
        prepared.update(self._prepare_loss_metadata(prepared))
        return prepared

    def forward(self, data: dict[str, Any]) -> LossTerms:
        loss_masks: LossMasks = data["loss_masks"]
        processed = self.prepare_training_inputs(data)
        processed["input_span_mask"] = data["input_span_mask"]
        processed["output_span_mask"] = data["output_span_mask"]
        for key in (
            "voice_gen_mask",
            "voice_patch_mask",
            "voice_think_mask",
            "voice_prosody_mask",
            "voice_prompt_mask",
            "voice_prompt_code",
        ):
            if key in data:
                processed[key] = data[key]
        outputs = self.core(processed)
        # Diagnostics that must not be weighted into the total loss. Read them
        # via accelerator.unwrap_model(model).latest_voice_metrics after the
        # forward; the collapse detector in particular fails long before any
        # audio metric moves.
        self.latest_voice_metrics = outputs.voice_metrics or {}
        return self._compute_loss_terms(
            outputs,
            labels=processed["labels"],
            loss_masks=loss_masks,
        )
    # endregion Training loss assembly and forward

    # region Inference helpers
    def _get_patch_encoder_inference(self) -> SemanticEncoderInference:
        encoder = self.core.patch_encoder
        encoder_id = id(encoder)
        adapter = getattr(self, "_patch_encoder_inference", None)
        if (
            adapter is None
            or getattr(self, "_patch_encoder_inference_encoder_id", None) != encoder_id
        ):
            adapter = SemanticEncoderInference(encoder)
            self._patch_encoder_inference = adapter
            self._patch_encoder_inference_encoder_id = encoder_id
        return adapter

    def _get_llm_inference(self) -> LLMInference:
        core_id = id(self.core)
        adapter = getattr(self, "_llm_inference", None)
        if adapter is None or getattr(self, "_llm_inference_core_id", None) != core_id:
            adapter = LLMInference(self.core)
            self._llm_inference = adapter
            self._llm_inference_core_id = core_id
        return adapter

    def _get_dit_solver(self, *, solver_mode: str) -> DiTSolver:
        key = (
            solver_mode,
            bool(self._optimize_enabled),
        )
        solver = self._fm_dit_solvers.get(key)
        if solver is None:
            context = DiTInferenceContext.from_core(self.core)
            if solver_mode == "scm":
                sampling = self.config.sampling
                if sampling is None:
                    raise RuntimeError("sCM solver requires artifact sampling config.")
                solver = SCMDiTSolver(
                    context,
                    optimize=self._optimize_enabled,
                    bucket_resolver=self._resolve_generate_length_bucket,
                    tau_mid=sampling.tau_mid,
                )
            elif solver_mode in {"flow_matching", "meanflow"}:
                solver = DiTSolver(
                    context,
                    optimize=self._optimize_enabled,
                    bucket_resolver=self._resolve_generate_length_bucket,
                    meanflow=solver_mode == "meanflow",
                )
            else:
                raise ValueError(f"Unsupported DiT solver mode: {solver_mode!r}.")
            self._fm_dit_solvers[key] = solver
        return solver

    def _get_vocoder_inference(self) -> VocoderInference:
        vocoder_id = id(self.vocoder)
        adapter = getattr(self, "_vocoder_inference", None)
        if (
            adapter is None
            or getattr(self, "_vocoder_inference_vocoder_id", None) != vocoder_id
        ):
            adapter = VocoderInference(self.vocoder)
            self._vocoder_inference = adapter
            self._vocoder_inference_vocoder_id = vocoder_id
        return adapter

    # endregion Inference helpers

    # region Prompt conditioning and decode state helpers
    def _prepare_prompt_audio_for_conditioning(
        self,
        prompt_audio: torch.Tensor,
    ) -> tuple[torch.Tensor, str]:
        if prompt_audio.ndim == 1:
            prompt_audio = prompt_audio.unsqueeze(0)
        prompt_audio = prompt_audio.detach().cpu().contiguous()

        samples_per_patch = self.config.patch_size * self.hop_size
        target_len = (
            math.ceil(prompt_audio.size(1) / samples_per_patch) * samples_per_patch
        )
        pad_len = target_len - prompt_audio.size(1)
        if pad_len > 0:
            prompt_audio = F.pad(prompt_audio, (0, pad_len))

        digest = hashlib.sha1()
        digest.update(str(tuple(prompt_audio.shape)).encode("ascii"))
        digest.update(str(prompt_audio.dtype).encode("ascii"))
        digest.update(prompt_audio.numpy().tobytes())
        return prompt_audio, digest.hexdigest()

    def _get_prompt_feature_cache_entry(
        self,
        cache_key: str,
    ) -> _PromptFeatureCacheEntry | None:
        entry = self._prompt_feature_cache.get(cache_key)
        if entry is not None:
            self._prompt_feature_cache.move_to_end(cache_key)
        return entry

    def _store_prompt_feature_cache_entry(
        self,
        cache_key: str,
        entry: _PromptFeatureCacheEntry,
    ) -> None:
        if (
            entry.speaker_embedding is None
            and entry.prompt_latent_distribution is None
        ):
            return
        self._prompt_feature_cache[cache_key] = entry
        self._prompt_feature_cache.move_to_end(cache_key)
        while (
            len(self._prompt_feature_cache) > self._PROMPT_FEATURE_CACHE_MAX_ENTRIES
        ):
            self._prompt_feature_cache.popitem(last=False)

    def _can_cache_speaker_embedding(self, prompt_sample_count: int) -> bool:
        max_audio_seconds = self.xvector_extractor.max_audio_seconds
        if max_audio_seconds <= 0:
            return True
        max_input_length = round(
            self.xvector_extractor.sample_rate * max_audio_seconds
        )
        return int(prompt_sample_count) <= max_input_length

    @torch.no_grad()
    def _prepare_prompt_conditioning(
        self,
        prompt_audio: torch.Tensor | None,
        *,
        use_prompt_prefill: bool,
        speaker_scale: float = 1.5,
    ) -> _PromptConditioning:
        if prompt_audio is None:
            logger.debug(
                logc(
                    "conditioning",
                    "Prompt conditioning skipped: no prompt audio provided.",
                )
            )
            return _PromptConditioning()

        self.vocoder.eval()
        self.xvector_extractor.eval()
        device = next(self.core.parameters()).device
        prompt_audio, cache_key = self._prepare_prompt_audio_for_conditioning(
            prompt_audio
        )
        prompt_sample_count = int(prompt_audio.shape[-1])
        cache_entry = self._get_prompt_feature_cache_entry(cache_key)
        if cache_entry is None:
            cache_entry = _PromptFeatureCacheEntry()
        prompt_audio = prompt_audio.to(device=device)

        can_cache_speaker = self._can_cache_speaker_embedding(prompt_sample_count)
        speaker_embedding = (
            cache_entry.speaker_embedding if can_cache_speaker else None
        )
        if speaker_embedding is None:
            with measure_inference("speaker_encoder", phase="prompt_conditioning"):
                speaker_embedding = self.xvector_extractor(prompt_audio[None, :])
            if can_cache_speaker:
                cache_entry.speaker_embedding = speaker_embedding.detach()
        else:
            logger.debug(
                logc(
                    "conditioning",
                    "Prompt speaker cache hit: key={} prompt_samples={}",
                ),
                cache_key[:12],
                prompt_sample_count,
            )
        g_cond = self.core.xvec_proj(speaker_embedding * float(speaker_scale))
        if not use_prompt_prefill:
            self._store_prompt_feature_cache_entry(cache_key, cache_entry)
            logger.debug(
                logc(
                    "conditioning",
                    "Reference-audio-only conditioning prepared: prompt_samples={} speaker_scale={} device={}",
                ),
                prompt_sample_count,
                speaker_scale,
                device,
            )
            return _PromptConditioning(g_cond=g_cond)

        prompt_latents = cache_entry.prompt_latent_distribution
        if prompt_latents is None:
            with measure_inference("latent_encoder", phase="prompt_conditioning"):
                prompt_latents = self._get_vocoder_inference().extract_latents(
                    prompt_audio[None, :]
                )
            cache_entry.prompt_latent_distribution = prompt_latents.detach()
        else:
            logger.debug(
                logc(
                    "conditioning",
                    "Prompt latent cache hit: key={} prompt_samples={}",
                ),
                cache_key[:12],
                prompt_sample_count,
            )
        self._store_prompt_feature_cache_entry(cache_key, cache_entry)
        prompt_latents_sampled = self.core.io_helper.sample_from_latent(prompt_latents)
        prompt_latents_sampled = prompt_latents_sampled[:, : -self.config.patch_size]
        prompt_patches = rearrange(
            self.core.io_helper.normalize(prompt_latents_sampled),
            "b (s p) d -> b s p d",
            p=self.config.patch_size,
        )
        logger.debug(
            logc(
                "conditioning",
                "Prompt conditioning prepared: prompt_samples={} prompt_patch_count={} "
                "speaker_scale={} device={}",
            ),
            prompt_sample_count,
            prompt_patches.size(1),
            speaker_scale,
            device,
        )
        return _PromptConditioning(
            prompt_patches=prompt_patches,
            prompt_latents=prompt_latents_sampled,
            g_cond=g_cond,
        )

    def _prepare_patch_encoder_input(
        self,
        latents: torch.Tensor,
        *,
        already_normalized: bool = False,
    ) -> torch.Tensor:
        if self.core.patch_encoder.expects_normalized_input:
            return (
                latents
                if already_normalized
                else self.core.io_helper.normalize(latents)
            )
        return (
            self.core.io_helper.denormalize(latents) if already_normalized else latents
        )

    def _prefill_prompt_latents(
        self,
        prompt_latents: torch.Tensor | None,
        *,
        state: _GenerateState,
    ) -> torch.Tensor | None:
        if prompt_latents is None:
            return None
        if prompt_latents.size(1) == 0:
            return prompt_latents.new_zeros(
                (prompt_latents.size(0), 0, self.core.llm_hidden_size)
            )
        patch_encoder_input = self._prepare_patch_encoder_input(prompt_latents)
        state_dtype = (
            state.fm_sequence.dtype
            if state.fm_sequence is not None
            else patch_encoder_input.dtype
        )
        with measure_inference("patch_encoder", phase="prompt_prefill"):
            prompt_patch_embeddings, state.patch_encoder_state = (
                self._get_patch_encoder_inference().prefill_with_state(
                    patch_encoder_input,
                    state.patch_encoder_state,
                    optimize=self._optimize_enabled,
                    bucket_resolver=self._resolve_generate_length_bucket,
                    dtype=state_dtype,
                )
            )
        return prompt_patch_embeddings

    def _append_to_fm_buffer(
        self,
        buffer: torch.Tensor | None,
        state: _GenerateState,
        chunk: torch.Tensor,
    ) -> tuple[int, int]:
        if buffer is None:
            raise RuntimeError("FM static buffer is not initialized.")
        start = state.fm_seq_len
        end = start + chunk.size(1)
        if end > state.fm_capacity:
            raise RuntimeError(
                "FM StaticBuffer capacity exceeded: "
                f"next_length={end} capacity={state.fm_capacity}."
            )
        buffer[:, start:end].copy_(chunk.to(buffer.dtype))
        return start, end

    def _append_hidden_chunk(
        self, state: _GenerateState, hidden_chunk: torch.Tensor
    ) -> None:
        last_hidden = hidden_chunk[:, -self.core.hidden_patch_size :, :]
        projected = self.core.hidden_proj(last_hidden)
        null_projected = self.core.hidden_proj(torch.zeros_like(last_hidden))
        _start, end = self._append_to_fm_buffer(
            state.fm_sequence,
            state,
            projected,
        )
        cfg_buffer = state.fm_cfg_sequence
        if cfg_buffer is None:
            raise RuntimeError("FM cfg static buffer is not initialized.")
        cfg_buffer[:, state.fm_seq_len : end].copy_(null_projected.to(cfg_buffer.dtype))
        state.fm_seq_len = end

    def _append_history_chunk(
        self, state: _GenerateState, latent_chunk: torch.Tensor
    ) -> None:
        history_latent = self.core.latent_proj(latent_chunk)
        _start, end = self._append_to_fm_buffer(
            state.fm_sequence,
            state,
            history_latent,
        )
        cfg_buffer = state.fm_cfg_sequence
        if cfg_buffer is None:
            raise RuntimeError("FM cfg static buffer is not initialized.")
        cfg_buffer[:, state.fm_seq_len : end].copy_(history_latent.to(cfg_buffer.dtype))
        state.fm_seq_len = end

    def _consume_text_schedule(
        self,
        generation_schedule: torch.Tensor,
        *,
        position: int,
        next_audio_position: int,
        state: _GenerateState,
        profile_step: int | None = None,
    ) -> int:
        with measure_inference("LLM", phase="text_schedule", step=profile_step):
            text_chunk = generation_schedule[:, position:next_audio_position]
            _, state.llm_hiddens, _logits = self._get_llm_inference().step(
                state.llm_state,
                input_ids=text_chunk,
                request_logits=False,
                compile_static=True,
                optimize=self._optimize_enabled,
                max_sequence_length=self._llm_max_sequence_length,
            )
        self._append_hidden_chunk(state, state.llm_hiddens)
        return next_audio_position

    def _locate_prefill_boundary(
        self,
        *,
        span_positions: torch.Tensor,
        prompt_patch_count: int,
    ) -> tuple[int, torch.Tensor]:
        if span_positions.numel() > prompt_patch_count:
            return int(span_positions[prompt_patch_count].item()), span_positions[
                :prompt_patch_count
            ]
        raise RuntimeError(
            "Prefill boundary discovery failed despite prior schedule validation."
        )

    @staticmethod
    def _find_audio_span_positions(
        generation_schedule: torch.Tensor,
        *,
        audio_placeholder_ids: set[int],
    ) -> torch.Tensor:
        schedule = generation_schedule[0]
        placeholder_ids = torch.tensor(
            sorted(audio_placeholder_ids),
            device=schedule.device,
            dtype=schedule.dtype,
        )
        return torch.nonzero(
            torch.isin(schedule, placeholder_ids),
            as_tuple=False,
        ).squeeze(-1)

    @staticmethod
    def _next_token_is_audio_span(
        generation_schedule: torch.Tensor,
        *,
        position: int,
        audio_placeholder_ids: set[int],
    ) -> bool:
        next_position = position + 1
        if next_position >= generation_schedule.size(1):
            return False
        return (
            int(generation_schedule[0, next_position].item()) in audio_placeholder_ids
        )

    def _build_prefill_inputs_embeds(
        self,
        generation_schedule: torch.Tensor,
        *,
        prompt_patch_embeddings: torch.Tensor | None,
        prompt_span_positions: torch.Tensor,
    ) -> torch.Tensor:
        inputs_embeds = self.core.llm.get_input_embeddings()(
            generation_schedule
        ).clone()
        if prompt_span_positions.numel() > 0:
            if prompt_patch_embeddings is None:
                raise RuntimeError(
                    "Prompt patch embeddings are required when prefill includes prompt audio spans."
                )
            patch_embeddings = prompt_patch_embeddings[
                :, : prompt_span_positions.numel()
            ].to(inputs_embeds.dtype)
            if patch_embeddings.size(1) != prompt_span_positions.numel():
                raise RuntimeError(
                    f"Prompt patch embeddings ({patch_embeddings.size(1)}) do not match prompt span count ({prompt_span_positions.numel()})."
                )
            inputs_embeds[:, prompt_span_positions, :] = patch_embeddings
        return inputs_embeds

    def _find_voice_positions(
        self, generation_schedule: torch.Tensor
    ) -> tuple[int, int] | None:
        """``(voice_gen_start, voice_patch)`` positions in the schedule."""

        gen_start_id = getattr(self.core, "voice_gen_start_id", None)
        patch_id = getattr(self.core, "voice_patch_id", None)
        if gen_start_id is None or patch_id is None:
            return None
        row = generation_schedule[0]
        gen_positions = row.eq(int(gen_start_id)).nonzero(as_tuple=False).squeeze(-1)
        patch_positions = row.eq(int(patch_id)).nonzero(as_tuple=False).squeeze(-1)
        if gen_positions.numel() == 0 and patch_positions.numel() == 0:
            return None
        if gen_positions.numel() != 1 or patch_positions.numel() != 1:
            raise ValueError(
                "generation_schedule must carry exactly one <|voice_gen_start|> "
                f"and one <|voice_patch|>; got {int(gen_positions.numel())} and "
                f"{int(patch_positions.numel())}."
            )
        gen_position = int(gen_positions.item())
        patch_position = int(patch_positions.item())
        if patch_position != gen_position + 1:
            raise ValueError(
                "<|voice_patch|> must immediately follow <|voice_gen_start|>; got "
                f"positions {gen_position} and {patch_position}."
            )
        return gen_position, patch_position

    def _resolve_voice_design_g_cond(
        self,
        state: _GenerateState,
        prompt_g_cond: torch.Tensor | None,
        *,
        voice_num_steps: int | None,
        voice_guidance_scale: float | None,
        generator: torch.Generator | None,
    ) -> torch.Tensor | None:
        """Combine the reference-audio and instruction timbre paths.

        ``state.voice_code`` was already drawn during prefill — it had to be, to
        become the ``<|voice_patch|>`` input embedding — so this reuses that
        exact draw rather than sampling again. Sampling twice would hand the
        acoustic head a different voice from the one the LM saw.

        With no reference audio the instruction owns ``g_cond`` outright. With
        reference audio the reference keeps the speaker — that is what the
        caller asked for by supplying it — and only the QFormer half of the
        predicted latent is layered on, so a prompt like "same speaker, but
        cheerful" changes delivery without drifting off the reference speaker.
        """

        del voice_num_steps, voice_guidance_scale, generator
        if self.core.instruction_residual_only:
            if prompt_g_cond is not None:
                raise ValueError("Residual-only inference does not accept reference speaker conditioning")
            if state.voice_condition_hidden is None:
                return None
            return self.core.instruction_to_g_cond(state.voice_condition_hidden)
        if getattr(self.core, "g_cond_source", "xvector") == "plan":
            # The acoustic plan is the only global conditioning, added
            # per segment in the decode loop. Mirrors training, where
            # every x-vector-derived term is scaled to zero. A cloning
            # request still gets its reference conditioning: that arrives
            # as prompt_g_cond and is returned untouched.
            return prompt_g_cond

        # The think chain is additive and independent of which branch below
        # supplies the base, mirroring training, where it is added to
        # xvec_cond after the anchor and the QFormer half. Keeping it outside
        # the branches is what stops the reference-audio path and the
        # instruction-only path from diverging on it.
        think_residual: torch.Tensor | None = None
        if state.think_code is not None and self.core.think_g_proj is not None:
            think_residual = self.core.think_g_proj(
                state.think_code.reshape(state.think_code.size(0), -1).to(
                    self.core.think_g_proj.weight.dtype
                )
            )

        instruction_residual = None
        if state.voice_condition_hidden is not None:
            instruction_residual = self.core.instruction_to_g_cond(
                state.voice_condition_hidden
            )

        def _with_instruction(base: torch.Tensor | None) -> torch.Tensor | None:
            if base is None:
                return base
            for residual in (think_residual, instruction_residual):
                if residual is not None:
                    base = base + residual.to(base)
            return base

        if state.voice_code is None or self.core.voice_design is None:
            return _with_instruction(prompt_g_cond)
        if prompt_g_cond is None:
            return _with_instruction(self.core.voice_code_to_g_cond(state.voice_code))
        _anchor, voice = self.core.voice_design.decode_code(state.voice_code)
        if self.core.voice_proj is None or voice is None:
            return _with_instruction(prompt_g_cond)
        residual = self.core.voice_proj(voice.to(self.core.voice_proj.weight.dtype))
        return _with_instruction(prompt_g_cond + residual.to(prompt_g_cond))


    def _prefill(
        self,
        generation_schedule: torch.Tensor,
        *,
        state: _GenerateState,
        span_positions: torch.Tensor,
        prompt_patches: torch.Tensor | None,
        prompt_patch_embeddings: torch.Tensor | None,
        audio_placeholder_ids: set[int],
        voice_positions: tuple[int, int] | None = None,
        voice_num_steps: int | None = None,
        voice_guidance_scale: float | None = None,
        voice_generator: torch.Generator | None = None,
        voice_prompt_code: torch.Tensor | None = None,
    ) -> int:
        prompt_patch_count = (
            0 if prompt_patches is None else int(prompt_patches.size(1))
        )
        prefill_end, prompt_span_positions = self._locate_prefill_boundary(
            span_positions=span_positions,
            prompt_patch_count=prompt_patch_count,
        )
        if prefill_end == 0:
            return 0
        inputs_embeds = self._build_prefill_inputs_embeds(
            generation_schedule[:, :prefill_end],
            prompt_patch_embeddings=prompt_patch_embeddings,
            prompt_span_positions=prompt_span_positions,
        )
        if voice_positions is None:
            with measure_inference("LLM", phase="prefill"):
                _, llm_hiddens, _logits = self._get_llm_inference().step(
                    state.llm_state,
                    inputs_embeds=inputs_embeds,
                    request_logits=False,
                    optimize=self._optimize_enabled,
                    max_sequence_length=self._llm_max_sequence_length,
                )
        else:
            # Prefill in two chunks around the voice patch. The KV cache carries
            # over between them, so the total work equals a single prefill —
            # there is no second pass over anything, only a split in where the
            # one pass pauses to draw the voice.
            gen_position, patch_position = voice_positions
            if patch_position >= prefill_end:
                raise RuntimeError(
                    "The voice patch must fall inside the prefilled prefix: "
                    f"voice_patch_position={patch_position} "
                    f"prefill_end={prefill_end}. Put {{voice}} before the text "
                    "and audio placeholders in the template."
                )
            with measure_inference("LLM", phase="prefill"):
                _, head_hiddens, _logits = self._get_llm_inference().step(
                    state.llm_state,
                    inputs_embeds=inputs_embeds[:, : gen_position + 1],
                    request_logits=False,
                    optimize=self._optimize_enabled,
                    max_sequence_length=self._llm_max_sequence_length,
                )
            condition_hidden = head_hiddens[:, -1:, :]
            state.voice_condition_hidden = condition_hidden
            num_think = int(getattr(self.core, "num_think_slots", 0) or 0)
            if num_think > 0 and getattr(self.core, "think_head", None) is not None:
                # The think slots are the num_think positions immediately
                # before <|voice_gen_start|>, so this chunk already holds
                # their hidden states: one matmul, no extra forward.
                if gen_position < num_think:
                    raise RuntimeError(
                        f"Expected {num_think} <|voice_think|> positions before "
                        f"<|voice_gen_start|>, which sits at {gen_position}. The "
                        "generation schedule was built with a different "
                        "num_think_slots than this checkpoint was trained with."
                    )
                think_span = generation_schedule[
                    0, gen_position - num_think : gen_position
                ]
                if not bool(
                    think_span.eq(int(self.core.voice_think_id)).all()
                ):
                    raise RuntimeError(
                        "The positions before <|voice_gen_start|> are not all "
                        "<|voice_think|>; the generation schedule and the "
                        "checkpoint disagree on the voice layout."
                    )
                state.think_code = self.core.think_chain(
                    head_hiddens[:, gen_position - num_think : gen_position, :]
                )[:, -1:, :]
            # Cloning mode fills the same slot the design mode samples into.
            if self.core.instruction_residual_only and voice_prompt_code is not None:
                raise ValueError("Residual-only inference does not accept a reference voice code")
            if self.core.instruction_residual_only:
                state.voice_code = None
            elif voice_prompt_code is not None:
                state.voice_code = voice_prompt_code.to(condition_hidden)
            else:
                state.voice_code = self.core.sample_voice_code(
                    condition_hidden,
                    num_steps=voice_num_steps,
                    guidance_scale=voice_guidance_scale,
                    generator=voice_generator,
                )
            tail = inputs_embeds[:, gen_position + 1 :]
            patch_mask = torch.zeros(
                tail.shape[:2], dtype=torch.bool, device=tail.device
            )
            patch_mask[:, patch_position - (gen_position + 1)] = True
            tail = self.core.inject_voice_patch(
                tail,
                patch_mask,
                state.voice_code,
                torch.ones(tail.size(0), dtype=torch.bool, device=tail.device),
                condition_hidden=condition_hidden,
            )
            with measure_inference("LLM", phase="prefill"):
                _, tail_hiddens, _logits = self._get_llm_inference().step(
                    state.llm_state,
                    inputs_embeds=tail,
                    request_logits=False,
                    optimize=self._optimize_enabled,
                    max_sequence_length=self._llm_max_sequence_length,
                )
            llm_hiddens = torch.cat([head_hiddens, tail_hiddens], dim=1)
        state.plan_vectors = self._read_acoustic_plan(
            generation_schedule, llm_hiddens, prefill_end
        )
        state.llm_hiddens = llm_hiddens[:, -1:, :]

        cursor = 0
        for prompt_index, span_position in enumerate(prompt_span_positions.tolist()):
            if span_position > cursor:
                self._append_hidden_chunk(
                    state, llm_hiddens[:, span_position - 1 : span_position, :]
                )
            self._append_history_chunk(state, prompt_patches[:, prompt_index])
            if self._next_token_is_audio_span(
                generation_schedule,
                position=span_position,
                audio_placeholder_ids=audio_placeholder_ids,
            ):
                self._append_hidden_chunk(
                    state, llm_hiddens[:, span_position : span_position + 1, :]
                )
            cursor = span_position + 1
        if prefill_end > cursor:
            self._append_hidden_chunk(
                state, llm_hiddens[:, prefill_end - 1 : prefill_end, :]
            )
        return prefill_end

    def _read_acoustic_plan(
        self,
        generation_schedule: torch.Tensor,
        llm_hiddens: torch.Tensor,
        prefill_end: int,
    ) -> torch.Tensor | None:
        """Project the prosody slots' hidden states into DiT conditioning.

        The slots sit between the text and the first audio token, so their
        hidden states are already in the prefill output: this is one matmul,
        not a second pass. The result is ``[B, M, fm_hidden]`` -- one
        conditioning vector per segment of the utterance about to be
        generated, which the decode loop hands to the DiT segment by
        segment.
        """

        head = getattr(self.core, "prosody_head", None)
        projection = getattr(self.core, "plan_g_proj", None)
        if head is None or projection is None:
            return None
        num_slots = int(getattr(self.core, "num_prosody_slots", 0) or 0)
        slot_id = getattr(self.core, "prosody_think_id", None)
        if num_slots <= 0 or slot_id is None:
            return None
        row = generation_schedule[0, :prefill_end]
        positions = row.eq(int(slot_id)).nonzero(as_tuple=False).squeeze(-1)
        if int(positions.numel()) != num_slots:
            # Refuse rather than condition on a partial plan: a schedule
            # holding a different number of slots than the checkpoint was
            # trained with produces audio that is merely worse, with no
            # error anywhere to explain it.
            raise RuntimeError(
                f"Expected {num_slots} <|prosody_think|> positions inside the "
                f"prefilled prefix, found {int(positions.numel())}. The "
                "generation schedule and the checkpoint disagree on "
                "num_prosody_slots."
            )
        with torch.no_grad():
            plan = head(llm_hiddens[:, positions, :])
            return projection(plan.to(projection.weight.dtype))

    def _decode_next_audio(
        self,
        state: _GenerateState,
        *,
        g_cond: torch.Tensor | None,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        profile_step: int | None = None,
        g_cond_version: int = 0,
    ) -> torch.Tensor:
        with measure_inference("FM", phase="decode", step=profile_step):
            sequence = state.fm_sequence
            if sequence is None:
                raise RuntimeError("FM static buffer is not initialized.")
            null_g_cond = state.fm_null_g_cond
            if null_g_cond is None:
                raise RuntimeError("FM null conditioning buffer is not initialized.")
            sampling = self.config.sampling
            if sampling is not None:
                sampling.resolve(
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                )
            solver_mode = sampling.solver if sampling is not None else self.core.mode
            cfg_sequence = state.fm_cfg_sequence
            if solver_mode == "flow_matching" and cfg_sequence is None:
                raise RuntimeError("FM cfg static buffer is not initialized.")
            solver_cfg_sequence = (
                cfg_sequence if solver_mode == "flow_matching" else None
            )
            audio_patch = self._get_dit_solver(solver_mode=solver_mode).decode_next(
                state.fm_dit_state,
                sequence=sequence,
                cfg_sequence=solver_cfg_sequence,
                fm_seq_len=state.fm_seq_len,
                null_g_cond=null_g_cond,
                g_cond=g_cond,
                nfe=num_steps,
                ode_method=ode_method,
                guidance_scale=guidance_scale,
                g_cond_version=int(g_cond_version),
            )
            return audio_patch

    def _consume_audio_patch(
        self,
        state: _GenerateState,
        *,
        audio_patch: torch.Tensor,
        profile_step: int | None = None,
    ) -> None:
        audio_patch_for_llm = self._prepare_patch_encoder_input(
            audio_patch,
            already_normalized=True,
        )
        self._append_history_chunk(state, audio_patch)
        state_dtype = (
            state.fm_sequence.dtype
            if state.fm_sequence is not None
            else audio_patch_for_llm.dtype
        )
        with measure_inference("patch_encoder", phase="decode", step=profile_step):
            llm_embedding, state.patch_encoder_state = (
                self._get_patch_encoder_inference().decode_patch_with_state(
                    audio_patch_for_llm,
                    state.patch_encoder_state,
                    optimize=self._optimize_enabled,
                    bucket_resolver=self._resolve_generate_length_bucket,
                    dtype=state_dtype,
                )
            )
        with measure_inference("LLM", phase="decode", step=profile_step):
            _, state.llm_hiddens, _logits = self._get_llm_inference().step(
                state.llm_state,
                inputs_embeds=llm_embedding,
                request_logits=False,
                compile_static=True,
                optimize=self._optimize_enabled,
                max_sequence_length=self._llm_max_sequence_length,
            )

    def _plan_conditioning(
        self,
        state: _GenerateState,
        *,
        g_cond: torch.Tensor | None,
        patch_index: int,
    ) -> tuple[torch.Tensor | None, int]:
        """Condition every history unit and the current noisy patch separately.

        The inference DiT sequence is [history units][current hidden][noise].
        Training modulates history unit n with plan seg(n); broadcasting the
        current plan across history changes that computation at each boundary.
        Derive the absolute patch offset from the FM buffer, including prompt
        history, rather than the number of newly decoded patches.
        """

        plan = state.plan_vectors
        if plan is None:
            return g_cond, 0
        del patch_index
        stride = int(getattr(self.core, "prosody_patches_per_slot", 4) or 4)
        hidden_tokens = int(self.core.hidden_patch_size)
        latent_tokens = int(self.core.latent_patch_size)
        unit_size = hidden_tokens + latent_tokens
        history_length = int(state.fm_seq_len) - hidden_tokens
        if history_length < 0 or history_length % unit_size:
            raise RuntimeError(
                "Acoustic plan requires unit-aligned FM history: "
                f"fm_seq_len={state.fm_seq_len}, unit_size={unit_size}."
            )
        total_length = int(state.fm_seq_len) + latent_tokens
        positions = torch.arange(total_length, device=plan.device)
        slots = (positions // unit_size // stride).clamp(max=plan.size(1) - 1)
        conditioning = plan[:, slots, :]
        if g_cond is None:
            return conditioning, total_length
        return g_cond.unsqueeze(1) + conditioning.to(g_cond), total_length

    def _decode(
        self,
        generation_schedule: torch.Tensor,
        *,
        position: int,
        state: _GenerateState,
        audio_placeholder_ids: set[int],
        span_positions: torch.Tensor,
        g_cond: torch.Tensor | None,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        eos_threshold: float,
        suppress_first_eos_check: bool = False,
    ) -> Iterator[torch.Tensor]:
        span_cursor = torch.searchsorted(
            span_positions,
            torch.tensor(
                position,
                device=span_positions.device,
                dtype=span_positions.dtype,
            ),
        ).item()
        decoded_audio_count = 0
        while position < generation_schedule.size(1):
            token_id = int(generation_schedule[0, position].item())
            if token_id in audio_placeholder_ids:
                profile_step = decoded_audio_count + 1
                should_check_eos = not (
                    suppress_first_eos_check and decoded_audio_count == 0
                )
                stop_after_current_audio = (
                    self._should_stop_after_current_audio(
                        state,
                        eos_threshold=eos_threshold,
                    )
                    if should_check_eos
                    else False
                )
                # The plan is time-varying, so the conditioning handed to
                # the DiT changes with the segment this patch falls in --
                # the same seg(n) = min(n // S, M-1) the training target
                # was cut with, and the reason that rule uses an absolute
                # stride: here the utterance's total length is not known,
                # generation stops on EOS.
                patch_g_cond, patch_g_cond_version = self._plan_conditioning(
                    state, g_cond=g_cond, patch_index=decoded_audio_count
                )
                audio_patch = self._decode_next_audio(
                    state,
                    g_cond=patch_g_cond,
                    ode_method=ode_method,
                    num_steps=num_steps,
                    guidance_scale=guidance_scale,
                    profile_step=profile_step,
                    g_cond_version=patch_g_cond_version,
                )
                self._consume_audio_patch(
                    state,
                    audio_patch=audio_patch,
                    profile_step=profile_step,
                )
                decoded_audio_count += 1
                if self._next_token_is_audio_span(
                    generation_schedule,
                    position=position,
                    audio_placeholder_ids=audio_placeholder_ids,
                ):
                    self._append_hidden_chunk(state, state.llm_hiddens)
                position += 1
                span_cursor += 1
                yield audio_patch
                if stop_after_current_audio:
                    state.end_flag = True
                    return
                continue
            next_audio_position = (
                int(span_positions[span_cursor].item())
                if span_cursor < span_positions.numel()
                else generation_schedule.size(1)
            )
            position = self._consume_text_schedule(
                generation_schedule,
                position=position,
                next_audio_position=next_audio_position,
                state=state,
                profile_step=decoded_audio_count + 1,
            )

    def _should_stop_after_current_audio(
        self, state: _GenerateState, *, eos_threshold: float
    ) -> bool:
        if state.llm_hiddens is None:
            return False
        eos = (
            self.core.eos_proj(state.llm_hiddens).softmax(dim=-1)[:, -1, 1]
            > eos_threshold
        )
        return state.end_flag or bool(eos.item())

    # endregion Prompt conditioning and decode state helpers

    # region Public generation APIs
    @torch.no_grad()
    def _generate_latents_stream(
        self,
        data: dict[str, Any],
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
        voice_num_steps: int | None = None,
        voice_guidance_scale: float | None = None,
        voice_generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        dtype = get_dtype(precision)
        device = next(self.core.parameters()).device
        use_amp = device.type == "cuda" and dtype in {torch.float16, torch.bfloat16}
        with torch.autocast(device_type=device.type, dtype=dtype, enabled=use_amp):
            generation_schedule: torch.Tensor = data["generation_schedule"]
            if generation_schedule.size(0) != 1:
                raise ValueError(
                    "DotsTtsModel.generate expects batch size 1 for generation_schedule."
                )
            if self._optimize_enabled:
                max_sequence_length = int(self._llm_max_sequence_length)
                schedule_length = int(generation_schedule.size(1))
                if schedule_length > max_sequence_length:
                    raise ValueError(
                        "generation_schedule length exceeds max_sequence_length for "
                        "optimized LLM StaticCache inference: "
                        f"schedule_length={schedule_length} "
                        f"max_sequence_length={max_sequence_length}."
                    )

            use_prompt_prefill = data.get("prompt_audio") is not None and bool(
                data.get("prompt_text")
            )
            prompt_conditioning = self._prepare_prompt_conditioning(
                data.get("prompt_audio"),
                use_prompt_prefill=use_prompt_prefill,
                speaker_scale=speaker_scale,
            )
            has_prompt_prefill = prompt_conditioning.prompt_patches is not None
            prompt_patch_count = (
                0
                if not has_prompt_prefill
                else int(prompt_conditioning.prompt_patches.size(1))
            )
            audio_placeholder_ids = set(self.core.audio_span_token_ids)
            span_positions = self._find_audio_span_positions(
                generation_schedule,
                audio_placeholder_ids=audio_placeholder_ids,
            )
            span_count = int(span_positions.numel())
            minimum_required_spans = prompt_patch_count + 1
            if span_count < minimum_required_spans:
                raise ValueError(
                    f"generation_schedule provides {span_count} audio spans, but prompt prefill requires "
                    f"{prompt_patch_count} spans and generation requires at least one additional decode span."
                )
            logger.debug(
                logc(
                    "decode",
                    "Latent generation prepared: schedule_audio_spans={} prompt_patch_count={} "
                    "minimum_required_spans={}",
                ),
                span_count,
                prompt_patch_count,
                minimum_required_spans,
            )

            state = self._allocate_generate_state(
                max_audio_patch_count=span_count,
                device=device,
                dtype=dtype,
            )
            prompt_patch_embeddings = self._prefill_prompt_latents(
                prompt_conditioning.prompt_latents,
                state=state,
            )
            position = self._prefill(
                generation_schedule,
                state=state,
                span_positions=span_positions,
                prompt_patches=prompt_conditioning.prompt_patches,
                prompt_patch_embeddings=prompt_patch_embeddings,
                audio_placeholder_ids=audio_placeholder_ids,
                voice_positions=self._find_voice_positions(generation_schedule),
                voice_num_steps=voice_num_steps,
                voice_guidance_scale=voice_guidance_scale,
                voice_generator=voice_generator,
            )
            g_cond = self._resolve_voice_design_g_cond(
                state,
                prompt_conditioning.g_cond,
                voice_num_steps=voice_num_steps,
                voice_guidance_scale=voice_guidance_scale,
                generator=voice_generator,
            )

            payload_patch_count = 0
            should_drop_regenerated_prompt_patch = has_prompt_prefill
            for audio_patch in self._decode(
                generation_schedule,
                position=position,
                state=state,
                audio_placeholder_ids=audio_placeholder_ids,
                span_positions=span_positions,
                g_cond=g_cond,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                eos_threshold=eos_threshold,
                suppress_first_eos_check=has_prompt_prefill,
            ):
                if should_drop_regenerated_prompt_patch:
                    should_drop_regenerated_prompt_patch = False
                    continue
                payload_patch_count += 1
                if payload_patch_count == 1 or payload_patch_count % 10 == 0:
                    logger.debug(
                        logc(
                            "decode",
                            "Latent generation progress: payload_audio_patches={}",
                        ),
                        payload_patch_count,
                    )
                yield self.core.io_helper.denormalize(audio_patch)

            if payload_patch_count == 0:
                if has_prompt_prefill:
                    raise RuntimeError(
                        "Generation produced no payload latents after discarding the regenerated prompt-tail patch. "
                        "This usually means EOS triggered immediately after prompt continuation "
                        "or the generation schedule did not provide an effective decode span."
                    )
                raise RuntimeError(
                    "Generation produced no decodable latents. "
                    "This usually means EOS triggered before the first decode patch "
                    "or the generation schedule did not provide an effective decode span."
                )
            logger.debug(
                logc("decode", "Latent generation completed: payload_audio_patches={}"),
                payload_patch_count,
            )

    @torch.no_grad()
    def generate_audio_stream(
        self,
        data: dict[str, Any],
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float = 1.5,
        eos_threshold: float = 0.8,
        vocoder_merge_steps: int = 1,
        voice_num_steps: int | None = None,
        voice_guidance_scale: float | None = None,
        voice_generator: torch.Generator | None = None,
    ) -> Iterator[torch.Tensor]:
        merge_steps = vocoder_merge_steps if self._optimize_enabled else 1
        if merge_steps < 1:
            raise ValueError(
                f"vocoder_merge_steps must be >= 1, got {vocoder_merge_steps}."
            )
        vocoder_inference = self._get_vocoder_inference()
        stream_state = vocoder_inference.init_stream_state(
            batch_size=1,
            chunk_size=int(self.core.latent_patch_size) * merge_steps,
        )
        pending_latent_patches: list[torch.Tensor] = []
        pending_start_index = 0

        for latent_patch_index, latent_patch in enumerate(
            self._generate_latents_stream(
                data,
                precision=precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
                eos_threshold=eos_threshold,
                voice_num_steps=voice_num_steps,
                voice_guidance_scale=voice_guidance_scale,
                voice_generator=voice_generator,
            ),
            start=1,
        ):
            # Keep the initial cadence unchanged for first-packet latency.
            if (
                merge_steps == 1
                or latent_patch_index <= self.VOCODER_STREAM_INITIAL_UNMERGED_PATCHES
            ):
                audio_chunk = vocoder_inference.stream_step(
                    latent_patch,
                    stream_state=stream_state,
                    optimize=self._optimize_enabled,
                    profile_step=latent_patch_index,
                    use_compiled=True,
                )
            else:
                if not pending_latent_patches:
                    pending_start_index = latent_patch_index
                pending_latent_patches.append(latent_patch)
                if len(pending_latent_patches) < merge_steps:
                    continue
                audio_chunk = vocoder_inference.stream_step(
                    torch.cat(pending_latent_patches, dim=1),
                    stream_state=stream_state,
                    optimize=self._optimize_enabled,
                    profile_step=f"{pending_start_index}-{latent_patch_index}",
                    use_compiled=True,
                )
                pending_latent_patches = []
                pending_start_index = 0
            if audio_chunk.size(-1) > 0:
                yield audio_chunk

        if pending_latent_patches:
            final_index = pending_start_index + len(pending_latent_patches) - 1
            audio_chunk = vocoder_inference.stream_step(
                torch.cat(pending_latent_patches, dim=1),
                stream_state=stream_state,
                optimize=self._optimize_enabled,
                profile_step=f"{pending_start_index}-{final_index}",
                use_compiled=False,
            )
            if audio_chunk.size(-1) > 0:
                yield audio_chunk

        final_chunk = vocoder_inference.flush(stream_state)
        if final_chunk.size(-1) > 0:
            yield final_chunk

    @torch.no_grad()
    def generate_audio(
        self,
        data: dict[str, Any],
        *,
        precision: str,
        ode_method: str,
        num_steps: int,
        guidance_scale: float,
        speaker_scale: float = 1.5,
        voice_num_steps: int | None = None,
        voice_guidance_scale: float | None = None,
        voice_generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        latent_patches = list(
            self._generate_latents_stream(
                data,
                precision=precision,
                ode_method=ode_method,
                num_steps=num_steps,
                guidance_scale=guidance_scale,
                speaker_scale=speaker_scale,
                voice_num_steps=voice_num_steps,
                voice_guidance_scale=voice_guidance_scale,
                voice_generator=voice_generator,
            )
        )
        logger.debug(
            logc("vocoder", "Vocoder decode started: latent_patch_count={}"),
            len(latent_patches),
        )
        audio = self._get_vocoder_inference().decode_latents(
            torch.cat(latent_patches, dim=1)
        )
        logger.debug(
            logc("vocoder", "Vocoder decode completed: waveform_samples={}"),
            audio.shape[-1],
        )
        return audio

    # endregion Public generation APIs
