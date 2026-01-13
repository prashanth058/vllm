from __future__ import annotations

import functools
import threading
from contextlib import contextmanager
from typing import Any, Callable

import torch

_TRAINING_MODE = threading.local()


def is_training_enabled() -> bool:
    return getattr(_TRAINING_MODE, "enabled", False)


def set_training_mode(enabled: bool) -> None:
    _TRAINING_MODE.enabled = enabled


@contextmanager
def training_mode():
    prev = is_training_enabled()
    set_training_mode(True)
    try:
        yield
    finally:
        set_training_mode(prev)


def conditional_inference_mode(fn: Callable) -> Callable:
    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        if is_training_enabled():
            return fn(*args, **kwargs)
        with torch.inference_mode():
            return fn(*args, **kwargs)
    return wrapper


def patch_inference_mode_decorator(target_module: Any, fn_name: str) -> None:
    original_fn = getattr(target_module, fn_name)
    if hasattr(original_fn, "_vllm_train_patched"):
        return

    unwrapped = original_fn
    while hasattr(unwrapped, "__wrapped__"):
        unwrapped = unwrapped.__wrapped__

    patched = conditional_inference_mode(unwrapped)
    patched._vllm_train_patched = True
    setattr(target_module, fn_name, patched)


def patch_model_eval(model: torch.nn.Module) -> None:
    original_eval = model.eval

    def conditional_eval(self: torch.nn.Module = model) -> torch.nn.Module:
        if is_training_enabled():
            return self
        return original_eval()

    model.eval = conditional_eval


def enable_lora_gradients(model: torch.nn.Module, lora_keywords: list[str] | None = None) -> int:
    if lora_keywords is None:
        lora_keywords = ["lora_", "lora_a", "lora_b"]

    count = 0
    for name, param in model.named_parameters():
        if any(kw in name.lower() for kw in lora_keywords):
            param.requires_grad = True
            count += 1
        else:
            param.requires_grad = False

    return count


def patch_vllm_for_training() -> dict[str, bool]:
    patched = {}

    try:
        from vllm.v1.worker import gpu_model_runner
        patch_inference_mode_decorator(gpu_model_runner.GPUModelRunner, "execute_model")
        patched["gpu_model_runner.execute_model"] = True
    except (ImportError, AttributeError):
        patched["gpu_model_runner.execute_model"] = False

    try:
        from vllm.lora.ops.triton_ops import lora_shrink_op
        if hasattr(lora_shrink_op, "_lora_shrink"):
            patch_inference_mode_decorator(lora_shrink_op, "_lora_shrink")
            patched["lora_shrink_op._lora_shrink"] = True
    except (ImportError, AttributeError):
        patched["lora_shrink_op._lora_shrink"] = False

    try:
        from vllm.lora.ops.triton_ops import lora_expand_op
        if hasattr(lora_expand_op, "_lora_expand"):
            patch_inference_mode_decorator(lora_expand_op, "_lora_expand")
            patched["lora_expand_op._lora_expand"] = True
    except (ImportError, AttributeError):
        patched["lora_expand_op._lora_expand"] = False

    return patched


class VLLMTrainingContext:
    def __init__(self, model: torch.nn.Module | None = None):
        self.model = model
        self._patched = False
        self._original_training_state: dict[str, bool] = {}

    def __enter__(self) -> VLLMTrainingContext:
        set_training_mode(True)

        if not self._patched:
            patch_vllm_for_training()
            self._patched = True

        if self.model is not None:
            for name, module in self.model.named_modules():
                self._original_training_state[name] = module.training
            self.model.train()

        return self

    def __exit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
        set_training_mode(False)

        if self.model is not None:
            for name, module in self.model.named_modules():
                if name in self._original_training_state:
                    module.training = self._original_training_state[name]
