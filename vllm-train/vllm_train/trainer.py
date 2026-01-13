"""Main trainer class for LoRA fine-tuning."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from vllm_train.config import LoRATrainConfig

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import PreTrainedModel, PreTrainedTokenizer
    from vllm import LLM

logger = logging.getLogger(__name__)


class LoRATrainer:
    """Hybrid trainer: HuggingFace/PEFT for training, vLLM for inference.

    This trainer uses HuggingFace Transformers and PEFT for the training loop,
    and vLLM for fast inference. The trained LoRA adapters are saved in PEFT
    format, which is compatible with vLLM's LoRA serving.

    Example:
        ```python
        from vllm_train import LoRATrainer, LoRATrainConfig

        config = LoRATrainConfig(
            rank=16,
            alpha=32,
            learning_rate=2e-4,
            num_epochs=3,
        )

        trainer = LoRATrainer("meta-llama/Llama-3.1-8B", config)
        trainer.setup_training()
        trainer.train(dataset)
        trainer.save_lora("./my_lora")

        # Use vLLM for fast inference
        outputs = trainer.generate(
            ["What is machine learning?"],
            lora_path="./my_lora",
            max_tokens=256,
        )
        ```
    """

    def __init__(
        self,
        model_name: str,
        config: LoRATrainConfig,
        tokenizer_name: str | None = None,
    ):
        """Initialize the trainer.

        Args:
            model_name: HuggingFace model name or path.
            config: Training configuration.
            tokenizer_name: Optional tokenizer name (defaults to model_name).
        """
        self.model_name = model_name
        self.config = config
        self.tokenizer_name = tokenizer_name or model_name

        # Models (lazy loaded)
        self._hf_model: PreTrainedModel | None = None
        self._tokenizer: PreTrainedTokenizer | None = None
        self._vllm: LLM | None = None

        # Training state
        self._is_setup = False

    @property
    def tokenizer(self) -> PreTrainedTokenizer:
        """Get the tokenizer, loading if necessary."""
        if self._tokenizer is None:
            from transformers import AutoTokenizer

            self._tokenizer = AutoTokenizer.from_pretrained(
                self.tokenizer_name,
                trust_remote_code=True,
            )
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
        return self._tokenizer

    @property
    def model(self) -> PreTrainedModel:
        """Get the training model."""
        if self._hf_model is None:
            raise RuntimeError("Model not initialized. Call setup_training() first.")
        return self._hf_model

    def setup_training(self, device_map: str = "auto") -> None:
        """Initialize the HuggingFace model with PEFT LoRA.

        Args:
            device_map: Device map for model loading ('auto', 'cuda:0', etc.)
        """
        from peft import get_peft_model
        from transformers import AutoModelForCausalLM

        logger.info(f"Loading model: {self.model_name}")

        # Load base model
        dtype = torch.bfloat16 if self.config.bf16 else torch.float16
        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            torch_dtype=dtype,
            device_map=device_map,
            trust_remote_code=True,
        )

        # Enable gradient checkpointing if requested
        if self.config.gradient_checkpointing:
            self._hf_model.gradient_checkpointing_enable()

        # Apply LoRA
        peft_config = self.config.to_peft_config()
        self._hf_model = get_peft_model(self._hf_model, peft_config)

        # Log trainable parameters
        trainable_params, total_params = self._count_parameters()
        logger.info(
            f"Trainable params: {trainable_params:,} / {total_params:,} "
            f"({100 * trainable_params / total_params:.2f}%)"
        )

        self._is_setup = True

    def _count_parameters(self) -> tuple[int, int]:
        """Count trainable and total parameters."""
        trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.model.parameters())
        return trainable, total

    def train(
        self,
        train_dataset: Dataset,
        eval_dataset: Dataset | None = None,
        data_collator: Any | None = None,
    ) -> None:
        """Train the model using HuggingFace Trainer.

        Args:
            train_dataset: Training dataset.
            eval_dataset: Optional evaluation dataset.
            data_collator: Optional data collator for batching.
        """
        if not self._is_setup:
            raise RuntimeError("Call setup_training() before train()")

        from transformers import (
            DataCollatorForLanguageModeling,
            Trainer,
            TrainingArguments,
        )

        # Default data collator
        if data_collator is None:
            data_collator = DataCollatorForLanguageModeling(
                tokenizer=self.tokenizer,
                mlm=False,
            )

        # Training arguments
        training_args = TrainingArguments(
            output_dir=self.config.output_dir,
            per_device_train_batch_size=self.config.batch_size,
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            num_train_epochs=self.config.num_epochs,
            learning_rate=self.config.learning_rate,
            warmup_ratio=self.config.warmup_ratio,
            max_grad_norm=self.config.max_grad_norm,
            bf16=self.config.bf16,
            logging_steps=self.config.logging_steps,
            save_strategy="steps" if self.config.save_steps > 0 else "epoch",
            save_steps=self.config.save_steps if self.config.save_steps > 0 else 500,
            evaluation_strategy="steps" if eval_dataset and self.config.eval_steps > 0 else "no",
            eval_steps=self.config.eval_steps if self.config.eval_steps > 0 else None,
            remove_unused_columns=False,
            report_to="none",  # Disable wandb etc by default
        )

        # Create trainer
        trainer = Trainer(
            model=self._hf_model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
        )

        # Train
        logger.info("Starting training...")
        trainer.train()
        logger.info("Training complete!")

    def save_lora(self, path: str) -> None:
        """Save LoRA weights in PEFT format (vLLM compatible).

        Args:
            path: Directory to save the adapter.
        """
        if self._hf_model is None:
            raise RuntimeError("No model to save. Train first.")

        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self._hf_model.save_pretrained(path)
        self.tokenizer.save_pretrained(path)

        logger.info(f"LoRA adapter saved to: {path}")

    def _get_vllm(self) -> LLM:
        """Get or create vLLM engine for inference."""
        if self._vllm is None:
            from vllm import LLM

            logger.info("Initializing vLLM for inference...")
            self._vllm = LLM(
                model=self.model_name,
                enable_lora=True,
                max_lora_rank=self.config.rank,
                trust_remote_code=True,
            )
        return self._vllm

    def generate(
        self,
        prompts: list[str],
        lora_path: str | None = None,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs: Any,
    ) -> list[str]:
        """Generate text using vLLM (fast inference).

        Args:
            prompts: List of prompts to generate from.
            lora_path: Path to trained LoRA adapter (optional).
            max_tokens: Maximum tokens to generate.
            temperature: Sampling temperature.
            top_p: Top-p sampling parameter.
            **kwargs: Additional arguments for SamplingParams.

        Returns:
            List of generated texts.
        """
        from vllm import SamplingParams
        from vllm.lora.request import LoRARequest

        vllm = self._get_vllm()

        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            **kwargs,
        )

        lora_request = None
        if lora_path:
            lora_request = LoRARequest(
                lora_name="trained_lora",
                lora_int_id=1,
                lora_path=lora_path,
            )

        outputs = vllm.generate(
            prompts,
            sampling_params=sampling_params,
            lora_request=lora_request,
        )

        return [output.outputs[0].text for output in outputs]

    def __repr__(self) -> str:
        return (
            f"LoRATrainer("
            f"model={self.model_name!r}, "
            f"rank={self.config.rank}, "
            f"alpha={self.config.alpha})"
        )
