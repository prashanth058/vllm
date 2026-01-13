#!/usr/bin/env python3
"""Basic LoRA fine-tuning example.

This example demonstrates how to:
1. Fine-tune a model with LoRA using HuggingFace/PEFT
2. Save the trained adapter
3. Use vLLM for fast inference with the trained adapter

Usage:
    python examples/basic_lora_finetune.py \
        --model meta-llama/Llama-3.2-1B \
        --dataset tatsu-lab/alpaca \
        --output ./my_lora

Requirements:
    pip install vllm-train datasets
"""

import argparse
import logging

from datasets import load_dataset

from vllm_train import LoRATrainer, LoRATrainConfig

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def prepare_dataset(dataset_name: str, tokenizer, max_length: int = 512):
    """Prepare the training dataset."""
    logger.info(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(dataset_name, split="train[:1000]")  # Use subset for demo

    def format_example(example):
        """Format Alpaca-style examples."""
        if example.get("input"):
            text = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Input:\n{example['input']}\n\n"
                f"### Response:\n{example['output']}"
            )
        else:
            text = (
                f"### Instruction:\n{example['instruction']}\n\n"
                f"### Response:\n{example['output']}"
            )
        return {"text": text}

    dataset = dataset.map(format_example, remove_columns=dataset.column_names)

    def tokenize(example):
        return tokenizer(
            example["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    dataset = dataset.map(tokenize, remove_columns=["text"])
    return dataset


def main():
    parser = argparse.ArgumentParser(description="LoRA fine-tuning example")
    parser.add_argument(
        "--model",
        type=str,
        default="meta-llama/Llama-3.2-1B",
        help="Model name or path",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="tatsu-lab/alpaca",
        help="Dataset name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./lora_output",
        help="Output directory for LoRA adapter",
    )
    parser.add_argument("--rank", type=int, default=16, help="LoRA rank")
    parser.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    parser.add_argument("--epochs", type=int, default=1, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument(
        "--skip-inference",
        action="store_true",
        help="Skip vLLM inference test",
    )
    args = parser.parse_args()

    # Create config
    config = LoRATrainConfig(
        rank=args.rank,
        alpha=args.alpha,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        output_dir=args.output,
        gradient_accumulation_steps=4,
        logging_steps=10,
    )

    # Initialize trainer
    logger.info(f"Initializing trainer with model: {args.model}")
    trainer = LoRATrainer(args.model, config)

    # Setup training (loads model with LoRA)
    trainer.setup_training()

    # Prepare dataset
    dataset = prepare_dataset(args.dataset, trainer.tokenizer)
    logger.info(f"Dataset size: {len(dataset)}")

    # Train
    trainer.train(dataset)

    # Save LoRA adapter
    lora_path = f"{args.output}/final_adapter"
    trainer.save_lora(lora_path)
    logger.info(f"LoRA adapter saved to: {lora_path}")

    # Test inference with vLLM
    if not args.skip_inference:
        logger.info("Testing inference with vLLM...")

        test_prompts = [
            "### Instruction:\nExplain what machine learning is in simple terms.\n\n### Response:\n",
            "### Instruction:\nWrite a haiku about programming.\n\n### Response:\n",
        ]

        # Generate without LoRA (base model)
        logger.info("Generating with base model...")
        base_outputs = trainer.generate(
            test_prompts,
            max_tokens=128,
            temperature=0.7,
        )

        # Generate with trained LoRA
        logger.info("Generating with trained LoRA...")
        lora_outputs = trainer.generate(
            test_prompts,
            lora_path=lora_path,
            max_tokens=128,
            temperature=0.7,
        )

        # Print comparison
        for i, prompt in enumerate(test_prompts):
            print(f"\n{'='*60}")
            print(f"Prompt: {prompt[:50]}...")
            print(f"\nBase model output:\n{base_outputs[i]}")
            print(f"\nLoRA output:\n{lora_outputs[i]}")

    logger.info("Done!")


if __name__ == "__main__":
    main()
