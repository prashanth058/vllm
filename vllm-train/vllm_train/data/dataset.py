"""Dataset classes for training."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch
from torch.utils.data import Dataset

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


class CausalLMDataset(Dataset):
    """Dataset for causal language modeling.

    Supports multiple input formats:
    - {"text": "..."} - Full text for training
    - {"prompt": "...", "completion": "..."} - Prompt-completion pairs
    - {"messages": [...]} - Chat format (will be converted using chat template)

    Example:
        ```python
        data = [
            {"text": "The quick brown fox jumps over the lazy dog."},
            {"prompt": "What is 2+2?", "completion": " 4"},
        ]
        dataset = CausalLMDataset(data, tokenizer, max_length=512)
        ```
    """

    def __init__(
        self,
        data: list[dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
        mask_prompt: bool = False,
    ):
        """Initialize the dataset.

        Args:
            data: List of data items.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
            mask_prompt: If True, mask prompt tokens in labels (for completion-only loss).
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mask_prompt = mask_prompt

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        # Handle different input formats
        if "text" in item:
            text = item["text"]
            prompt_len = 0
        elif "prompt" in item and "completion" in item:
            prompt = item["prompt"]
            completion = item["completion"]
            text = prompt + completion
            prompt_len = len(self.tokenizer.encode(prompt, add_special_tokens=False))
        elif "messages" in item:
            text = self.tokenizer.apply_chat_template(
                item["messages"],
                tokenize=False,
                add_generation_prompt=False,
            )
            prompt_len = 0
        else:
            raise ValueError(f"Unknown data format: {item.keys()}")

        # Tokenize
        encodings = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        input_ids = encodings["input_ids"].squeeze(0)
        attention_mask = encodings["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        # Mask prompt tokens if requested
        if self.mask_prompt and prompt_len > 0:
            labels[:prompt_len] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }


class PreferenceDataset(Dataset):
    """Dataset for preference learning (DPO/RLHF).

    Expected format:
    - {"prompt": "...", "chosen": "...", "rejected": "..."}

    Example:
        ```python
        data = [
            {
                "prompt": "What is the capital of France?",
                "chosen": " Paris is the capital of France.",
                "rejected": " London is the capital of France.",
            },
        ]
        dataset = PreferenceDataset(data, tokenizer, max_length=512)
        ```
    """

    def __init__(
        self,
        data: list[dict[str, str]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = 2048,
    ):
        """Initialize the dataset.

        Args:
            data: List of preference pairs.
            tokenizer: HuggingFace tokenizer.
            max_length: Maximum sequence length.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        item = self.data[idx]

        prompt = item["prompt"]
        chosen = item["chosen"]
        rejected = item["rejected"]

        # Tokenize chosen
        chosen_text = prompt + chosen
        chosen_encodings = self.tokenizer(
            chosen_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        # Tokenize rejected
        rejected_text = prompt + rejected
        rejected_encodings = self.tokenizer(
            rejected_text,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors="pt",
        )

        # Get prompt length for masking
        prompt_encodings = self.tokenizer(
            prompt,
            add_special_tokens=False,
            return_tensors="pt",
        )
        prompt_len = prompt_encodings["input_ids"].size(1)

        # Create labels (mask prompt)
        chosen_labels = chosen_encodings["input_ids"].squeeze(0).clone()
        chosen_labels[:prompt_len] = -100

        rejected_labels = rejected_encodings["input_ids"].squeeze(0).clone()
        rejected_labels[:prompt_len] = -100

        return {
            "chosen_input_ids": chosen_encodings["input_ids"].squeeze(0),
            "chosen_attention_mask": chosen_encodings["attention_mask"].squeeze(0),
            "chosen_labels": chosen_labels,
            "rejected_input_ids": rejected_encodings["input_ids"].squeeze(0),
            "rejected_attention_mask": rejected_encodings["attention_mask"].squeeze(0),
            "rejected_labels": rejected_labels,
        }


def create_causal_lm_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    split: str = "train",
    text_column: str = "text",
    **kwargs: Any,
) -> CausalLMDataset:
    """Create a CausalLMDataset from a HuggingFace dataset.

    Args:
        data_path: HuggingFace dataset path or local path.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        split: Dataset split to use.
        text_column: Name of the text column.
        **kwargs: Additional arguments for load_dataset.

    Returns:
        CausalLMDataset instance.
    """
    from datasets import load_dataset

    dataset = load_dataset(data_path, split=split, **kwargs)

    # Convert to list of dicts
    data = [{"text": item[text_column]} for item in dataset]

    return CausalLMDataset(data, tokenizer, max_length)


def create_preference_dataset(
    data_path: str,
    tokenizer: PreTrainedTokenizer,
    max_length: int = 2048,
    split: str = "train",
    prompt_column: str = "prompt",
    chosen_column: str = "chosen",
    rejected_column: str = "rejected",
    **kwargs: Any,
) -> PreferenceDataset:
    """Create a PreferenceDataset from a HuggingFace dataset.

    Args:
        data_path: HuggingFace dataset path or local path.
        tokenizer: HuggingFace tokenizer.
        max_length: Maximum sequence length.
        split: Dataset split to use.
        prompt_column: Name of the prompt column.
        chosen_column: Name of the chosen response column.
        rejected_column: Name of the rejected response column.
        **kwargs: Additional arguments for load_dataset.

    Returns:
        PreferenceDataset instance.
    """
    from datasets import load_dataset

    dataset = load_dataset(data_path, split=split, **kwargs)

    # Convert to list of dicts
    data = [
        {
            "prompt": item[prompt_column],
            "chosen": item[chosen_column],
            "rejected": item[rejected_column],
        }
        for item in dataset
    ]

    return PreferenceDataset(data, tokenizer, max_length)
