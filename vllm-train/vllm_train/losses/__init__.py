"""Loss functions for training."""

from vllm_train.losses.causal_lm import CausalLMLoss
from vllm_train.losses.dpo import DPOLoss

__all__ = [
    "CausalLMLoss",
    "DPOLoss",
]
