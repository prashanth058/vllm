# vLLM-Train Implementation Plan

## Project Goals

Build a LoRA fine-tuning library on top of vLLM that:
1. Uses only vLLM public APIs for easy upgrades
2. Enables "train once, serve immediately" workflow
3. Supports RLHF with vLLM's fast inference for rollouts
4. Works as a standalone repo (no vLLM fork required)

---

## Phase 1: Hybrid Training (Week 1-2)

### Goal
Get basic LoRA training working using HuggingFace/PEFT, with vLLM for inference.

### Components

#### 1.1 Project Setup
```
vllm-train/
├── vllm_train/
│   ├── __init__.py
│   ├── config.py
│   ├── trainer.py
│   └── utils.py
├── pyproject.toml
└── README.md
```

#### 1.2 Core Classes

**LoRATrainConfig** (`config.py`)
```python
@dataclass
class LoRATrainConfig:
    # LoRA settings
    rank: int = 8
    alpha: int = 16
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj"
    ])
    dropout: float = 0.0

    # Training settings
    learning_rate: float = 2e-4
    batch_size: int = 4
    gradient_accumulation_steps: int = 4
    num_epochs: int = 3
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0

    # Output
    output_dir: str = "./lora_output"
```

**LoRATrainer** (`trainer.py`)
```python
class LoRATrainer:
    """Hybrid trainer: HuggingFace for training, vLLM for inference."""

    def __init__(
        self,
        model_name: str,
        train_config: LoRATrainConfig,
    ):
        self.model_name = model_name
        self.config = train_config

        # Training model (HuggingFace + PEFT)
        self._hf_model = None
        self._tokenizer = None

        # Inference model (vLLM) - lazy loaded
        self._vllm = None

    def setup_training(self):
        """Initialize HuggingFace model with PEFT LoRA."""
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from peft import get_peft_model, LoraConfig

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            device_map="auto",
        )

        peft_config = LoraConfig(
            r=self.config.rank,
            lora_alpha=self.config.alpha,
            target_modules=self.config.target_modules,
            lora_dropout=self.config.dropout,
            task_type="CAUSAL_LM",
        )
        self._hf_model = get_peft_model(self._hf_model, peft_config)

    def train(self, dataset):
        """Train using HuggingFace Trainer."""
        from transformers import Trainer, TrainingArguments

        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            bf16=True,
            logging_steps=10,
            save_strategy="epoch",
        )

        trainer = Trainer(
            model=self._hf_model,
            args=training_args,
            train_dataset=dataset,
            tokenizer=self._tokenizer,
        )
        trainer.train()

    def save_lora(self, path: str):
        """Save LoRA weights in PEFT format (vLLM compatible)."""
        self._hf_model.save_pretrained(path)

    def generate(self, prompts: list[str], lora_path: str = None, **kwargs):
        """Generate using vLLM (fast inference)."""
        from vllm import LLM, SamplingParams
        from vllm.lora.request import LoRARequest

        if self._vllm is None:
            self._vllm = LLM(
                model=self.model_name,
                enable_lora=True,
                max_lora_rank=self.config.rank,
            )

        sampling_params = SamplingParams(**kwargs)
        lora_request = None
        if lora_path:
            lora_request = LoRARequest("trained_lora", 1, lora_path)

        return self._vllm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )
```

#### 1.3 Deliverables
- [ ] Basic project structure
- [ ] `LoRATrainConfig` dataclass
- [ ] `LoRATrainer` class with HF training
- [ ] vLLM inference integration
- [ ] Example: `examples/basic_lora_finetune.py`
- [ ] Tests for config and basic training flow

---

## Phase 2: Data Loading & Loss Functions (Week 2-3)

### Goal
Add proper data loading, multiple loss functions, and evaluation.

### Components

#### 2.1 Data Loading (`data/`)

**TrainingDataset** (`data/dataset.py`)
```python
class CausalLMDataset(torch.utils.data.Dataset):
    """Dataset for causal language modeling."""

    def __init__(
        self,
        data: list[dict],  # {"text": "..."} or {"prompt": "...", "completion": "..."}
        tokenizer,
        max_length: int = 2048,
    ):
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __getitem__(self, idx):
        item = self.data[idx]
        if "text" in item:
            text = item["text"]
        else:
            text = f"{item['prompt']}{item['completion']}"

        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encodings["input_ids"].squeeze(),
            "attention_mask": encodings["attention_mask"].squeeze(),
            "labels": encodings["input_ids"].squeeze(),
        }
```

**PreferenceDataset** (`data/preference.py`)
```python
class PreferenceDataset(torch.utils.data.Dataset):
    """Dataset for preference learning (DPO/RLHF)."""

    def __init__(
        self,
        data: list[dict],  # {"prompt": "...", "chosen": "...", "rejected": "..."}
        tokenizer,
        max_length: int = 2048,
    ):
        ...
```

#### 2.2 Loss Functions (`losses/`)

**CausalLMLoss** (`losses/causal_lm.py`)
```python
class CausalLMLoss(nn.Module):
    """Standard causal language modeling loss."""

    def forward(self, logits, labels, attention_mask=None):
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=-100,
        )
        return loss
```

**DPOLoss** (`losses/dpo.py`)
```python
class DPOLoss(nn.Module):
    """Direct Preference Optimization loss."""

    def __init__(self, beta: float = 0.1):
        super().__init__()
        self.beta = beta

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> torch.Tensor:
        chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
        rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)
        return -F.logsigmoid(chosen_rewards - rejected_rewards).mean()
```

#### 2.3 Deliverables
- [ ] `CausalLMDataset` with various input formats
- [ ] `PreferenceDataset` for DPO/RLHF
- [ ] Data collators for padding
- [ ] `CausalLMLoss`
- [ ] `DPOLoss`
- [ ] Evaluation metrics (perplexity, accuracy)
- [ ] Example: `examples/dpo_training.py`

---

## Phase 3: RLHF Integration (Week 3-4)

### Goal
Enable RLHF with vLLM for fast rollout generation.

### Components

#### 3.1 RLHF Trainer (`rlhf/`)

**RLHFTrainer** (`rlhf/trainer.py`)
```python
class RLHFTrainer:
    """RLHF trainer using vLLM for rollout generation."""

    def __init__(
        self,
        model_name: str,
        reward_model_name: str,
        config: RLHFConfig,
    ):
        self.model_name = model_name
        self.config = config

        # Policy model (HF + PEFT for training)
        self.policy_trainer = LoRATrainer(model_name, config.lora_config)
        self.policy_trainer.setup_training()

        # vLLM for fast generation
        self._vllm = LLM(model_name, enable_lora=True)

        # Reward model (vLLM for inference)
        self._reward_model = LLM(reward_model_name)

    def generate_rollouts(self, prompts: list[str]) -> list[str]:
        """Generate responses using vLLM (fast!)."""
        sampling_params = SamplingParams(
            temperature=self.config.temperature,
            max_tokens=self.config.max_new_tokens,
        )
        outputs = self._vllm.generate(prompts, sampling_params)
        return [o.outputs[0].text for o in outputs]

    def compute_rewards(self, prompts: list[str], responses: list[str]) -> torch.Tensor:
        """Compute rewards using reward model."""
        # Use vLLM's scoring API if available, else custom
        ...

    def ppo_step(self, prompts: list[str]):
        """Single PPO training step."""
        # 1. Generate rollouts
        responses = self.generate_rollouts(prompts)

        # 2. Compute rewards
        rewards = self.compute_rewards(prompts, responses)

        # 3. Compute advantages
        advantages = self._compute_advantages(rewards)

        # 4. PPO update
        self._ppo_update(prompts, responses, advantages)

    def train(self, prompt_dataset, num_steps: int):
        """Full RLHF training loop."""
        for step in range(num_steps):
            batch = self._sample_batch(prompt_dataset)
            self.ppo_step(batch)

            if step % self.config.save_steps == 0:
                self.save_checkpoint(f"step_{step}")
```

#### 3.2 Deliverables
- [ ] `RLHFConfig` dataclass
- [ ] `RLHFTrainer` with PPO
- [ ] vLLM integration for fast rollouts
- [ ] Reward model integration
- [ ] Example: `examples/rlhf_ppo.py`

---

## Phase 4: Custom LoRA Layers (Week 4-5)

### Goal
Build our own trainable LoRA layers that can work with vLLM models directly.

### Components

#### 4.1 Trainable LoRA (`lora/`)

**TrainableLoRALinear** (`lora/layers.py`)
```python
class TrainableLoRALinear(nn.Module):
    """Trainable LoRA layer that wraps a frozen linear layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_a = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_b = nn.Parameter(torch.zeros(out_features, rank))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_a, a=math.sqrt(5))
        nn.init.zeros_(self.lora_b)

    def forward(self, x: torch.Tensor, base_output: torch.Tensor) -> torch.Tensor:
        """Add LoRA contribution to base output."""
        lora_out = self.dropout(x) @ self.lora_a.T @ self.lora_b.T
        return base_output + lora_out * self.scaling
```

**LoRAInjector** (`lora/injector.py`)
```python
class LoRAInjector:
    """Inject trainable LoRA layers into a frozen model."""

    def __init__(self, config: LoRATrainConfig):
        self.config = config
        self.lora_layers = {}

    def inject(self, model: nn.Module) -> nn.Module:
        """Inject LoRA layers into target modules."""
        for name, module in model.named_modules():
            if self._should_inject(name, module):
                lora_layer = self._create_lora_layer(module)
                self.lora_layers[name] = lora_layer
                self._wrap_module(model, name, module, lora_layer)
        return model

    def get_trainable_params(self) -> list[nn.Parameter]:
        """Get only the trainable LoRA parameters."""
        params = []
        for lora in self.lora_layers.values():
            params.extend([lora.lora_a, lora.lora_b])
        return params
```

#### 4.2 Deliverables
- [ ] `TrainableLoRALinear` layer
- [ ] `LoRAInjector` for model modification
- [ ] PEFT-compatible save/load
- [ ] Integration with custom model loader
- [ ] Tests for gradient flow

---

## Phase 5: vLLM Deep Integration (Week 5-6)

### Goal
Use vLLM's `register_model_loader` for deeper integration.

### Components

#### 5.1 Custom Model Loader (`integration/`)

**TrainableModelLoader** (`integration/loader.py`)
```python
from vllm.model_executor.model_loader import register_model_loader, BaseModelLoader

@register_model_loader("trainable")
class TrainableModelLoader(BaseModelLoader):
    """Model loader that enables training mode for LoRA."""

    def __init__(self, load_config, lora_train_config: LoRATrainConfig = None):
        super().__init__(load_config)
        self.lora_train_config = lora_train_config

    def load_model(self, vllm_config, model_config):
        # Load base model (this calls .eval())
        model = super().load_model(vllm_config, model_config)

        if self.lora_train_config:
            # Inject trainable LoRA
            injector = LoRAInjector(self.lora_train_config)
            model = injector.inject(model)

            # Put LoRA layers in training mode
            for lora in injector.lora_layers.values():
                lora.train()

        return model
```

#### 5.2 Unified Engine

**TrainableEngine** (`engine.py`)
```python
class TrainableEngine:
    """Unified engine for training and inference."""

    def __init__(self, model_name: str, train_config: LoRATrainConfig):
        from vllm import LLM
        from vllm.config import LoadConfig

        # Register our custom loader
        load_config = LoadConfig(load_format="trainable")

        self.llm = LLM(
            model=model_name,
            load_format="trainable",
            enable_lora=True,
        )

        # Access the model for training
        self.model = self._get_model_from_engine()
        self.optimizer = self._create_optimizer()

    def train_step(self, batch) -> dict:
        """Single training step."""
        self.model.train()
        loss = self._compute_loss(batch)
        loss.backward()
        self.optimizer.step()
        self.optimizer.zero_grad()
        return {"loss": loss.item()}

    def generate(self, prompts, **kwargs):
        """Generate using vLLM."""
        self.model.eval()
        return self.llm.generate(prompts, **kwargs)
```

#### 5.3 Deliverables
- [ ] `TrainableModelLoader` registered with vLLM
- [ ] `TrainableEngine` for unified train/infer
- [ ] Tensor parallel training support
- [ ] Example: `examples/unified_training.py`

---

## Testing Strategy

### Unit Tests
```
tests/
├── test_config.py           # Config validation
├── test_data.py             # Data loading
├── test_losses.py           # Loss functions
├── test_lora_layers.py      # LoRA gradient flow
└── test_trainer.py          # Training loop
```

### Integration Tests
```
tests/integration/
├── test_hf_vllm_hybrid.py   # HF training + vLLM inference
├── test_lora_save_load.py   # PEFT format compatibility
└── test_rlhf_pipeline.py    # End-to-end RLHF
```

### Benchmarks
```
benchmarks/
├── training_throughput.py   # Tokens/sec during training
├── inference_latency.py     # vLLM inference speed
└── memory_usage.py          # GPU memory profiling
```

---

## Dependencies

```toml
[project]
dependencies = [
    "vllm>=0.6.0",
    "transformers>=4.40.0",
    "peft>=0.10.0",
    "torch>=2.0.0",
    "datasets>=2.0.0",
    "accelerate>=0.25.0",
    "trl>=0.8.0",  # For RLHF utilities
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0.0",
    "pytest-cov>=4.0.0",
    "black>=23.0.0",
    "ruff>=0.1.0",
]
```

---

## Success Criteria

### Phase 1
- [ ] Can train LoRA adapter using HF/PEFT
- [ ] Can load trained adapter in vLLM for inference
- [ ] Training loss decreases on small dataset

### Phase 2
- [ ] Multiple data formats supported
- [ ] DPO training works
- [ ] Evaluation metrics match baseline

### Phase 3
- [ ] RLHF with vLLM rollouts faster than HF-only
- [ ] Reward model integration works
- [ ] PPO training converges

### Phase 4
- [ ] Custom LoRA layers have correct gradients
- [ ] Save/load compatible with PEFT format
- [ ] Memory usage comparable to PEFT

### Phase 5
- [ ] Single model instance for train+infer
- [ ] Tensor parallel training works
- [ ] No performance regression vs pure inference

---

## Timeline

| Phase | Duration | Deliverable |
|-------|----------|-------------|
| Phase 1 | Week 1-2 | Hybrid trainer working |
| Phase 2 | Week 2-3 | Data + losses complete |
| Phase 3 | Week 3-4 | RLHF pipeline working |
| Phase 4 | Week 4-5 | Custom LoRA layers |
| Phase 5 | Week 5-6 | Deep vLLM integration |

Total: ~6 weeks for full implementation
