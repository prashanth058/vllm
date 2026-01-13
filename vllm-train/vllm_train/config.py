"""Configuration classes for vLLM-Train."""

from dataclasses import dataclass, field
from typing import Literal


@dataclass
class LoRATrainConfig:
    """Configuration for LoRA fine-tuning.

    This config is designed to be compatible with both HuggingFace PEFT
    and vLLM's LoRA serving.

    Attributes:
        rank: LoRA rank (r). Higher = more capacity but more memory.
        alpha: LoRA alpha for scaling. Typically alpha = 2 * rank.
        target_modules: Which modules to apply LoRA to.
        dropout: Dropout probability for LoRA layers.
        learning_rate: Learning rate for training.
        batch_size: Per-device batch size.
        gradient_accumulation_steps: Number of steps to accumulate gradients.
        num_epochs: Number of training epochs.
        warmup_ratio: Ratio of total steps for warmup.
        max_grad_norm: Maximum gradient norm for clipping.
        output_dir: Directory to save checkpoints and final model.
        save_steps: Save checkpoint every N steps (0 = only at end).
        logging_steps: Log metrics every N steps.
        eval_steps: Evaluate every N steps (0 = no evaluation during training).
        bf16: Use bfloat16 mixed precision.
        gradient_checkpointing: Use gradient checkpointing to save memory.
    """

    # LoRA configuration
    rank: int = 8
    alpha: int = 16
    target_modules: list[str] = field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ]
    )
    dropout: float = 0.0

    # Training configuration
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Output configuration
    output_dir: str = "./lora_output"
    save_steps: int = 0
    logging_steps: int = 10
    eval_steps: int = 0

    # Precision
    bf16: bool = True
    gradient_checkpointing: bool = False

    def to_peft_config(self):
        """Convert to PEFT LoraConfig."""
        from peft import LoraConfig, TaskType

        return LoraConfig(
            r=self.rank,
            lora_alpha=self.alpha,
            target_modules=self.target_modules,
            lora_dropout=self.dropout,
            task_type=TaskType.CAUSAL_LM,
            bias="none",
        )

    def to_vllm_lora_config(self):
        """Convert to vLLM LoRAConfig for inference."""
        from vllm.config import LoRAConfig

        return LoRAConfig(
            max_lora_rank=self.rank,
            max_loras=1,
            lora_dtype="auto",
        )


@dataclass
class RLHFConfig:
    """Configuration for RLHF training.

    Attributes:
        lora_config: LoRA configuration for the policy model.
        reward_model: Path or name of the reward model.
        kl_coef: KL divergence coefficient for PPO.
        clip_range: PPO clip range.
        value_clip_range: Value function clip range.
        ppo_epochs: Number of PPO epochs per batch.
        temperature: Sampling temperature for generation.
        max_new_tokens: Maximum number of tokens to generate.
        batch_size: Number of prompts per batch.
        mini_batch_size: Mini-batch size for PPO updates.
        num_steps: Total number of training steps.
        save_steps: Save checkpoint every N steps.
    """

    # LoRA config for policy
    lora_config: LoRATrainConfig = field(default_factory=LoRATrainConfig)

    # Reward model
    reward_model: str = ""

    # PPO hyperparameters
    kl_coef: float = 0.1
    clip_range: float = 0.2
    value_clip_range: float = 0.2
    ppo_epochs: int = 4

    # Generation settings
    temperature: float = 0.7
    max_new_tokens: int = 512

    # Training settings
    batch_size: int = 8
    mini_batch_size: int = 4
    num_steps: int = 1000
    save_steps: int = 100


@dataclass
class DPOConfig:
    """Configuration for Direct Preference Optimization.

    Attributes:
        lora_config: LoRA configuration for the policy model.
        beta: DPO beta parameter (inverse temperature).
        label_smoothing: Label smoothing for DPO loss.
        loss_type: Type of DPO loss ('sigmoid', 'hinge', 'ipo').
        reference_free: Whether to use reference-free DPO.
    """

    lora_config: LoRATrainConfig = field(default_factory=LoRATrainConfig)

    # DPO hyperparameters
    beta: float = 0.1
    label_smoothing: float = 0.0
    loss_type: Literal["sigmoid", "hinge", "ipo"] = "sigmoid"
    reference_free: bool = False
