# vLLM-Train Architecture: Public API Integration Strategy

## Overview

This document outlines how `vllm-train` will integrate with vLLM using **only public interfaces** to ensure:
1. Easy vLLM upgrades (minimal breaking changes)
2. Clean separation of concerns
3. No need for vLLM patches or forks

---

## vLLM Public API Surface Analysis

### ✅ Stable Public APIs (Safe to Use)

| API | Location | Stability | Usage in vllm-train |
|-----|----------|-----------|---------------------|
| `LLM` | `vllm.LLM` | Stable | Base class access to engine |
| `EngineArgs` | `vllm.EngineArgs` | Stable | Configuration |
| `SamplingParams` | `vllm.SamplingParams` | Stable | Generation during RLHF |
| `LoRARequest` | `vllm.lora.request.LoRARequest` | Stable | Loading trained adapters |
| `LoRAConfig` | `vllm.config.LoRAConfig` | Stable | LoRA configuration |
| `VllmConfig` | `vllm.config.VllmConfig` | Stable | Full config access |
| `ModelConfig` | `vllm.config.ModelConfig` | Stable | Model configuration |
| `get_model_loader` | `vllm.model_executor.model_loader` | Stable | Model loading |
| `register_model_loader` | `vllm.model_executor.model_loader` | Stable | Custom loader registration |
| `BaseModelLoader` | `vllm.model_executor.model_loader` | Stable | Extension point |
| `RequestOutput` | `vllm.outputs.RequestOutput` | Stable | Inference results |

### ⚠️ Semi-Public APIs (Use with Caution)

| API | Location | Risk | Mitigation |
|-----|----------|------|------------|
| `llm.llm_engine` | `LLM.llm_engine` | Medium | Accessed via public LLM class |
| `llm_engine.vllm_config` | Engine attribute | Medium | Standard config access pattern |
| `ModelRegistry` | `vllm.ModelRegistry` | Medium | For model class lookup |

### ❌ Internal APIs (Avoid Direct Use)

| API | Location | Why Avoid | Alternative |
|-----|----------|-----------|-------------|
| `BasevLLMParameter` | `vllm.model_executor.parameter` | Internal impl | Wrap with our own parameter class |
| `initialize_model` | `model_loader.utils` | Internal | Use `BaseModelLoader.load_model()` |
| `GPUModelRunner.model` | Worker internals | Not exposed | Access via custom loader |
| LoRA layer internals | `vllm.lora.layers.*` | Implementation detail | Create our own trainable layers |

---

## Integration Strategy

### Strategy 1: Custom Model Loader (Recommended)

Use vLLM's `register_model_loader` to create a training-aware loader:

```python
from vllm.model_executor.model_loader import register_model_loader, BaseModelLoader

@register_model_loader("trainable")
class TrainableModelLoader(BaseModelLoader):
    """Custom loader that keeps gradients enabled for LoRA params."""

    def load_model(self, vllm_config, model_config):
        # Load base model using parent (this calls .eval())
        model = super().load_model(vllm_config, model_config)

        # Re-enable training mode for LoRA layers
        model.train()  # Override the .eval() call

        # Inject trainable LoRA layers
        inject_trainable_lora(model, self.lora_config)

        return model
```

**Pros:**
- Uses official extension point
- No monkey-patching
- Will survive vLLM upgrades

**Cons:**
- Need to override `.eval()` behavior

### Strategy 2: Wrapper Pattern

Wrap vLLM's LLM class without modifying it:

```python
from vllm import LLM

class TrainableLLM:
    """Wrapper that adds training capabilities to vLLM's LLM."""

    def __init__(self, model: str, **kwargs):
        # Use vLLM for inference
        self._llm = LLM(model=model, enable_lora=True, **kwargs)

        # Load a separate training copy of the model
        self._train_model = self._load_trainable_model(model)

    def generate(self, prompts, **kwargs):
        """Delegate inference to vLLM."""
        return self._llm.generate(prompts, **kwargs)

    def train_step(self, batch):
        """Training on separate model instance."""
        return self._train_model.forward(batch)

    def sync_to_inference(self):
        """Copy trained LoRA weights to inference model."""
        # Save LoRA, reload in vLLM
        pass
```

**Pros:**
- Zero changes to vLLM
- Complete isolation
- Can use any training framework for training part

**Cons:**
- Duplicate memory for two model copies
- Sync overhead

### Strategy 3: HuggingFace + vLLM Hybrid (Pragmatic)

Use HuggingFace/PEFT for training, vLLM for inference:

```python
from transformers import AutoModelForCausalLM
from peft import get_peft_model, LoraConfig
from vllm import LLM
from vllm.lora.request import LoRARequest

class HybridTrainer:
    def __init__(self, model_name: str):
        # HuggingFace for training
        self.hf_model = AutoModelForCausalLM.from_pretrained(model_name)
        self.hf_model = get_peft_model(self.hf_model, LoraConfig(...))

        # vLLM for inference (lazy init)
        self._vllm = None
        self.model_name = model_name

    def train(self, dataset):
        """Train using HuggingFace/PEFT."""
        # Standard HF training loop
        pass

    def save_lora(self, path: str):
        """Save in format compatible with vLLM."""
        self.hf_model.save_pretrained(path)

    def generate(self, prompts, lora_path: str = None):
        """Use vLLM for fast inference."""
        if self._vllm is None:
            self._vllm = LLM(self.model_name, enable_lora=True)

        lora_request = None
        if lora_path:
            lora_request = LoRARequest("trained", 1, lora_path)

        return self._vllm.generate(prompts, lora_request=lora_request)
```

**Pros:**
- Uses battle-tested training (PEFT)
- Uses battle-tested inference (vLLM)
- LoRA format is compatible
- No hacks needed

**Cons:**
- Two separate frameworks
- Memory overhead during transition
- Not truly "unified"

---

## Recommended Approach: Hybrid Strategy

Based on the analysis, **Strategy 3 (Hybrid)** is recommended for Phase 1, with evolution to Strategy 1 for deeper integration:

### Phase 1: HuggingFace Training + vLLM Inference

```
┌─────────────────────────────────────────────────────────────────┐
│                       vllm-train (Phase 1)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Training Side (HuggingFace/PEFT)     Inference Side (vLLM)     │
│  ┌─────────────────────────┐          ┌─────────────────────┐  │
│  │ AutoModelForCausalLM    │          │ vllm.LLM             │  │
│  │ + PEFT LoRA             │  ──────▶ │ + LoRARequest        │  │
│  │                         │  (save)  │                      │  │
│  └─────────────────────────┘          └─────────────────────┘  │
│                                                                 │
│  Benefits:                                                      │
│  • No vLLM internal dependencies                                │
│  • PEFT LoRA format works with vLLM                             │
│  • Easy to upgrade either side                                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Phase 2: Custom Loader Integration

```
┌─────────────────────────────────────────────────────────────────┐
│                       vllm-train (Phase 2)                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  @register_model_loader("trainable")                            │
│  class TrainableModelLoader(BaseModelLoader):                   │
│      """Uses vLLM's official extension point"""                 │
│                                                                 │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │                    Single Model Instance                 │   │
│  │  ┌──────────────┐    ┌──────────────┐                   │   │
│  │  │ Base Model   │    │ LoRA Layers  │                   │   │
│  │  │ (frozen)     │    │ (trainable)  │                   │   │
│  │  │ .eval()      │    │ .train()     │                   │   │
│  │  └──────────────┘    └──────────────┘                   │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                 │
│  Benefits:                                                      │
│  • Single model instance                                        │
│  • Seamless train/inference switching                           │
│  • Uses vLLM's TP infrastructure                                │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Public APIs We Will Use

### From vllm (top-level)
```python
from vllm import (
    LLM,                    # Main inference class
    EngineArgs,             # Configuration
    SamplingParams,         # For generation
    RequestOutput,          # Output type
)
```

### From vllm.config
```python
from vllm.config import (
    LoRAConfig,             # LoRA settings
    VllmConfig,             # Full config
    ModelConfig,            # Model settings
)
```

### From vllm.lora
```python
from vllm.lora.request import LoRARequest  # For loading trained adapters
```

### From vllm.model_executor.model_loader
```python
from vllm.model_executor.model_loader import (
    register_model_loader,   # Extension point for custom loaders
    BaseModelLoader,         # Base class for our loader
    get_model_loader,        # For getting standard loaders
)
```

---

## What We Will NOT Use (Internal APIs)

```python
# ❌ Don't use these directly
from vllm.model_executor.parameter import BasevLLMParameter  # Internal
from vllm.model_executor.model_loader.utils import initialize_model  # Internal
from vllm.lora.layers import *  # Implementation details
from vllm.v1.worker.gpu_model_runner import GPUModelRunner  # Internal
```

---

## File Structure

```
vllm-train/
├── vllm_train/
│   ├── __init__.py
│   ├── config.py              # LoRATrainConfig, extends vllm.config.LoRAConfig
│   ├── trainer.py             # Main LoRATrainer class
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # Training data loading
│   ├── lora/
│   │   ├── __init__.py
│   │   ├── layers.py          # Our own trainable LoRA layers
│   │   └── utils.py           # LoRA weight manipulation
│   ├── losses/
│   │   ├── __init__.py
│   │   ├── causal_lm.py       # CausalLM loss
│   │   └── dpo.py             # DPO loss
│   └── integration/
│       ├── __init__.py
│       ├── vllm_inference.py  # Wrapper for vLLM inference
│       └── hf_training.py     # Wrapper for HF/PEFT training
├── examples/
│   ├── lora_finetune.py
│   ├── rlhf_ppo.py
│   └── distillation.py
├── tests/
├── pyproject.toml
└── README.md
```

---

## Version Compatibility

We will support vLLM versions via:

```python
# vllm_train/compat.py
import vllm

VLLM_VERSION = tuple(map(int, vllm.__version__.split('.')[:2]))

if VLLM_VERSION >= (0, 8):
    from vllm.config import LoRAConfig
else:
    # Fallback for older versions
    from vllm.config.lora import LoRAConfig
```

---

## Summary

| Aspect | Approach |
|--------|----------|
| Training | HuggingFace/PEFT (Phase 1) → Custom LoRA layers (Phase 2) |
| Inference | vLLM's public `LLM` class |
| LoRA Format | PEFT-compatible (works with vLLM) |
| Extension Point | `register_model_loader` decorator |
| Internal APIs | Avoided entirely |
| vLLM Upgrades | Should work with minor version bumps |
