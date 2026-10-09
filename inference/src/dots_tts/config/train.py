from __future__ import annotations

from pydantic import Field

from dots_tts.config.base import StrictConfigBase
from dots_tts.models.dots_tts.config import VoiceDesignConfig


class TrainConfig(StrictConfigBase):
    pretrained_model_path: str
    # Present -> load through DotsTtsModel.from_pretrained_for_voice_design,
    # which extends the tokenizer, grows the LLM embedding and attaches a fresh
    # voice-design branch. Absent -> ordinary fine-tuning, unchanged.
    voice_design: VoiceDesignConfig | None = None
    # Modules whose names start with any of these prefixes keep requires_grad;
    # everything else is frozen. Empty/None trains the whole model, which is the
    # historical behaviour.
    trainable_module_prefixes: list[str] | None = None
    output_dir: str
    seed: int = 42
    learning_rate: float
    # The voice-design branch is randomly initialized while the backbone is
    # pretrained, so they cannot share a learning rate: the backbone's ~1e-5
    # would leave a fresh flow DiT essentially frozen, and the branch's ~1e-4
    # would tear the backbone apart. Null falls back to `learning_rate`.
    voice_design_learning_rate: float | None = None
    # Both are applied through DotsTtsModel.set_cfg_droprate, where None
    # means "leave the artifact's own value alone". They decide how often
    # each conditioning path is dropped during training, and therefore how
    # much the acoustic head is forced to rely on the other one:
    #   cfg_droprate    -- drops the PER-FRAME LM hidden conditioning
    #   xvec_drop_rate  -- drops the GLOBAL g_cond vector
    # xvec_drop_rate=1.0 zeroes g_cond on every row, which is the only way
    # to train a model whose inference has no global conditioning at all.
    cfg_droprate: float | None = None
    xvec_drop_rate: float | None = None
    weight_decay: float = 0.01
    warmup_steps: int = 0
    max_train_steps: int
    gradient_accumulation_steps: int = Field(default=1, ge=1)
    grad_clip_norm: float = 1.0
    # Trade compute for activation memory: each transformer block is
    # recomputed in the backward pass rather than kept. Costs roughly a third
    # more compute per step, but lets the audio budget per micro-batch grow
    # several-fold -- and on a 2B model a small batch is latency-bound, so the
    # larger batch usually more than pays the recompute back.
    gradient_checkpointing: bool = False
    # null disables step-based saving entirely, which is what you want when
    # save_on_epoch_end is the only trigger you care about.
    save_interval: int | None = Field(default=1000, ge=1)
    # Also save whenever the data stream finishes an epoch, i.e. whenever
    # train_data.num_tokens_per_epoch is exhausted. The epoch rolls over inside
    # batch preparation, which is mid-accumulation, so the save is deferred to
    # the next completed optimizer step -- a boundary every rank agrees on.
    save_on_epoch_end: bool = False
    max_checkpoints_to_keep: int = 10
    # Hardlink each checkpoint's `model/` directory into `<output_dir>/exports/
    # step-XXXXXXXX/` so it survives max_checkpoints_to_keep cleanup. A ZeRO-2
    # checkpoint is mostly optimizer shards; the model directory alone is what
    # inference needs, and hardlinks cost nothing until the checkpoint is
    # deleted. Turn this on whenever you intend to sample from intermediate
    # steps but only want to keep a few resumable checkpoints.
    export_inference_model: bool = False
    log_interval: int = Field(default=10, ge=1)
    eval_interval: int | None = Field(default=None, ge=1)
    max_eval_batches: int | None = None
    run_eval_on_start: bool = False

    # Accelerate trackers. "tensorboard" alone reproduces the previous
    # behaviour; add "wandb" to mirror every logged scalar to Weights & Biases.
    report_to: list[str] = Field(default_factory=lambda: ["tensorboard"])
    wandb_project: str = "dots-tts"
    wandb_run_name: str | None = None
    wandb_entity: str | None = None
    wandb_group: str | None = None
    wandb_tags: list[str] | None = None
    # "online" | "offline" | "disabled". Exported as WANDB_MODE before the
    # tracker initializes, which is the only point where wandb reads it.
    wandb_mode: str | None = None


__all__ = ["TrainConfig"]
