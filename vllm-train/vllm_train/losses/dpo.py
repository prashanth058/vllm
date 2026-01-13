"""Direct Preference Optimization (DPO) loss."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class DPOLoss(nn.Module):
    """Direct Preference Optimization loss.

    Implements the DPO loss from "Direct Preference Optimization:
    Your Language Model is Secretly a Reward Model" (Rafailov et al., 2023).

    The loss is:
        -log(sigmoid(beta * (log(pi(y_w|x)/pi_ref(y_w|x)) - log(pi(y_l|x)/pi_ref(y_l|x)))))

    where y_w is the preferred (chosen) response and y_l is the rejected response.

    Example:
        ```python
        loss_fn = DPOLoss(beta=0.1)
        loss = loss_fn(
            policy_chosen_logps=chosen_logps,
            policy_rejected_logps=rejected_logps,
            ref_chosen_logps=ref_chosen_logps,
            ref_rejected_logps=ref_rejected_logps,
        )
        ```
    """

    def __init__(
        self,
        beta: float = 0.1,
        label_smoothing: float = 0.0,
        loss_type: str = "sigmoid",
        reference_free: bool = False,
    ):
        """Initialize the DPO loss.

        Args:
            beta: Temperature parameter (inverse KL penalty strength).
            label_smoothing: Label smoothing factor.
            loss_type: Type of loss ('sigmoid', 'hinge', 'ipo').
            reference_free: If True, don't use reference model log probs.
        """
        super().__init__()
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        self.reference_free = reference_free

    def forward(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor | None = None,
        ref_rejected_logps: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Compute the DPO loss.

        Args:
            policy_chosen_logps: Log probs of chosen responses under policy [batch].
            policy_rejected_logps: Log probs of rejected responses under policy [batch].
            ref_chosen_logps: Log probs of chosen responses under reference [batch].
            ref_rejected_logps: Log probs of rejected responses under reference [batch].

        Returns:
            Tuple of (loss, metrics_dict).
        """
        if self.reference_free:
            # Reference-free DPO: just compare policy log probs
            chosen_rewards = self.beta * policy_chosen_logps
            rejected_rewards = self.beta * policy_rejected_logps
        else:
            if ref_chosen_logps is None or ref_rejected_logps is None:
                raise ValueError(
                    "Reference log probs required unless reference_free=True"
                )
            # Standard DPO: compute implicit rewards
            chosen_rewards = self.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.beta * (policy_rejected_logps - ref_rejected_logps)

        # Compute loss based on type
        if self.loss_type == "sigmoid":
            loss = self._sigmoid_loss(chosen_rewards, rejected_rewards)
        elif self.loss_type == "hinge":
            loss = self._hinge_loss(chosen_rewards, rejected_rewards)
        elif self.loss_type == "ipo":
            loss = self._ipo_loss(chosen_rewards, rejected_rewards)
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}")

        # Metrics
        reward_margin = (chosen_rewards - rejected_rewards).detach()
        metrics = {
            "loss": loss.detach(),
            "reward_margin": reward_margin.mean(),
            "chosen_rewards": chosen_rewards.mean().detach(),
            "rejected_rewards": rejected_rewards.mean().detach(),
            "accuracy": (reward_margin > 0).float().mean(),
        }

        return loss, metrics

    def _sigmoid_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Standard sigmoid (logistic) DPO loss."""
        logits = chosen_rewards - rejected_rewards

        if self.label_smoothing > 0:
            # Soft labels
            losses = (
                -F.logsigmoid(logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-logits) * self.label_smoothing
            )
        else:
            losses = -F.logsigmoid(logits)

        return losses.mean()

    def _hinge_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Hinge loss variant."""
        margin = chosen_rewards - rejected_rewards
        return F.relu(1 - margin).mean()

    def _ipo_loss(
        self,
        chosen_rewards: torch.Tensor,
        rejected_rewards: torch.Tensor,
    ) -> torch.Tensor:
        """Identity Preference Optimization (IPO) loss."""
        # IPO uses squared hinge-like loss
        margin = chosen_rewards - rejected_rewards
        return ((margin - 1 / (2 * self.beta)) ** 2).mean()
