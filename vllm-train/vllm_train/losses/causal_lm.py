"""Causal Language Modeling loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalLMLoss(nn.Module):
    """Standard causal language modeling loss.

    Computes cross-entropy loss for next-token prediction, shifting
    the logits and labels appropriately.

    Example:
        ```python
        loss_fn = CausalLMLoss()
        loss = loss_fn(logits, labels)
        ```
    """

    def __init__(self, ignore_index: int = -100, reduction: str = "mean"):
        """Initialize the loss function.

        Args:
            ignore_index: Index to ignore in loss computation.
            reduction: Reduction method ('mean', 'sum', 'none').
        """
        super().__init__()
        self.ignore_index = ignore_index
        self.reduction = reduction

    def forward(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Compute the causal LM loss.

        Args:
            logits: Model output logits [batch, seq_len, vocab_size].
            labels: Target token IDs [batch, seq_len].
            attention_mask: Optional attention mask [batch, seq_len].

        Returns:
            Loss tensor.
        """
        # Shift for causal LM: predict token t+1 from position t
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        # Flatten for cross entropy
        vocab_size = shift_logits.size(-1)
        shift_logits = shift_logits.view(-1, vocab_size)
        shift_labels = shift_labels.view(-1)

        # Compute loss
        loss = F.cross_entropy(
            shift_logits,
            shift_labels,
            ignore_index=self.ignore_index,
            reduction=self.reduction,
        )

        return loss


def compute_log_probs(
    logits: torch.Tensor,
    labels: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
    average: bool = True,
) -> torch.Tensor:
    """Compute log probabilities of labels given logits.

    This is useful for DPO and other preference-based methods.

    Args:
        logits: Model output logits [batch, seq_len, vocab_size].
        labels: Target token IDs [batch, seq_len].
        attention_mask: Optional attention mask [batch, seq_len].
        average: If True, average over sequence; else sum.

    Returns:
        Log probabilities [batch].
    """
    # Shift for causal LM
    shift_logits = logits[..., :-1, :].contiguous()
    shift_labels = labels[..., 1:].contiguous()

    # Compute per-token log probs
    log_probs = F.log_softmax(shift_logits, dim=-1)
    per_token_log_probs = log_probs.gather(
        dim=-1, index=shift_labels.unsqueeze(-1)
    ).squeeze(-1)

    # Mask out padding/ignored tokens
    if attention_mask is not None:
        shift_mask = attention_mask[..., 1:].contiguous()
        per_token_log_probs = per_token_log_probs * shift_mask

    # Mask out -100 labels
    valid_mask = shift_labels != -100
    per_token_log_probs = per_token_log_probs * valid_mask

    # Aggregate
    if average:
        # Average over valid tokens
        num_valid = valid_mask.sum(dim=-1).clamp(min=1)
        return per_token_log_probs.sum(dim=-1) / num_valid
    else:
        return per_token_log_probs.sum(dim=-1)
