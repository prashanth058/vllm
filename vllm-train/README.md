# vLLM-Train

LoRA fine-tuning library built on top of vLLM. Train once, serve immediately.

## Features

- **Hybrid Training**: Uses HuggingFace/PEFT for training, vLLM for inference
- **Zero-Cost Transition**: Trained LoRA adapters work directly with vLLM
- **PEFT Compatible**: Saves in standard PEFT format
- **Fast Inference**: Leverages vLLM's optimized inference engine

## Installation

```bash
pip install vllm-train
```

Or install from source:

```bash
git clone https://github.com/vllm-project/vllm-train
cd vllm-train
pip install -e .
```

## Quick Start

```python
from vllm_train import LoRATrainer, LoRATrainConfig

# Configure training
config = LoRATrainConfig(
    rank=16,
    alpha=32,
    learning_rate=2e-4,
    num_epochs=3,
)

# Initialize trainer
trainer = LoRATrainer("meta-llama/Llama-3.1-8B", config)
trainer.setup_training()

# Train on your dataset
trainer.train(train_dataset)

# Save the adapter
trainer.save_lora("./my_lora")

# Use vLLM for fast inference
outputs = trainer.generate(
    ["What is machine learning?"],
    lora_path="./my_lora",
    max_tokens=256,
)
```

## Examples

### Basic LoRA Fine-tuning

```bash
python examples/basic_lora_finetune.py \
    --model meta-llama/Llama-3.2-1B \
    --dataset tatsu-lab/alpaca \
    --output ./my_lora
```

### Configuration Options

```python
from vllm_train import LoRATrainConfig

config = LoRATrainConfig(
    # LoRA settings
    rank=16,                    # LoRA rank
    alpha=32,                   # LoRA alpha (scaling)
    target_modules=[            # Modules to apply LoRA
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ],
    dropout=0.05,               # LoRA dropout

    # Training settings
    learning_rate=2e-4,
    batch_size=4,
    gradient_accumulation_steps=4,
    num_epochs=3,
    warmup_ratio=0.1,
    max_grad_norm=1.0,

    # Precision
    bf16=True,
    gradient_checkpointing=True,

    # Output
    output_dir="./lora_output",
)
```

## Architecture

vLLM-Train uses a hybrid approach:

```
┌─────────────────────────────────────────────────────────────────┐
│                         vLLM-Train                              │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  Training (HuggingFace/PEFT)          Inference (vLLM)          │
│  ┌─────────────────────────┐          ┌─────────────────────┐  │
│  │ AutoModelForCausalLM    │          │ vllm.LLM             │  │
│  │ + PEFT LoRA             │  ──────▶ │ + LoRARequest        │  │
│  │                         │  (save)  │                      │  │
│  └─────────────────────────┘          └─────────────────────┘  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

This approach:
- Uses battle-tested training code (HuggingFace Trainer)
- Uses battle-tested inference (vLLM)
- No vLLM modifications required
- Easy to upgrade either component

## Roadmap

- [x] Phase 1: Hybrid training (HF + vLLM)
- [ ] Phase 2: Data loading & loss functions
- [ ] Phase 3: RLHF integration
- [ ] Phase 4: Custom LoRA layers
- [ ] Phase 5: Deep vLLM integration

## License

Apache 2.0
