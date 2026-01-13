from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

import torch
import torch.nn.functional as F
from torch.optim import AdamW

from vllm_train.patch import VLLMTrainingContext, enable_lora_gradients

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer


@dataclass
class DPOConfig:
    beta: float = 0.1
    learning_rate: float = 1e-4
    micro_batch_size: int = 4
    max_length: int = 512
    gradient_accumulation_steps: int = 1
    max_grad_norm: float = 1.0
    label_smoothing: float = 0.0
    loss_type: str = "sigmoid"
    reference_free: bool = False


@dataclass
class DPOPair:
    prompt: str
    chosen: str
    rejected: str
    metadata: dict[str, Any] = field(default_factory=dict)


class EfficientDPOTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: DPOConfig | None = None,
        lora_params: list[torch.nn.Parameter] | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config or DPOConfig()

        if lora_params is not None:
            self.lora_params = list(lora_params)
        else:
            self.lora_params = [p for p in model.parameters() if p.requires_grad]

        self.optimizer = AdamW(self.lora_params, lr=self.config.learning_rate)
        self.training_context = VLLMTrainingContext(model)
        self._step_count = 0

    def compute_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        with self.training_context:
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        masked_log_probs = token_log_probs * shift_mask
        sequence_log_probs = masked_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)

        return sequence_log_probs

    def compute_ref_logps(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        with torch.no_grad():
            outputs = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
            )
            logits = outputs.logits

        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        shift_mask = attention_mask[..., 1:].contiguous()

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = torch.gather(
            log_probs, dim=-1, index=shift_labels.unsqueeze(-1)
        ).squeeze(-1)

        masked_log_probs = token_log_probs * shift_mask
        sequence_log_probs = masked_log_probs.sum(dim=-1) / shift_mask.sum(dim=-1).clamp(min=1)

        return sequence_log_probs

    def tokenize_pair(self, pair: DPOPair) -> dict[str, torch.Tensor]:
        chosen_text = pair.prompt + pair.chosen
        rejected_text = pair.prompt + pair.rejected

        chosen_enc = self.tokenizer(
            chosen_text,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        rejected_enc = self.tokenizer(
            rejected_text,
            max_length=self.config.max_length,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )

        prompt_enc = self.tokenizer(
            pair.prompt,
            max_length=self.config.max_length,
            truncation=True,
            return_tensors="pt",
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        chosen_labels = chosen_enc["input_ids"].clone()
        chosen_labels[:, :prompt_len] = -100

        rejected_labels = rejected_enc["input_ids"].clone()
        rejected_labels[:, :prompt_len] = -100

        device = next(self.model.parameters()).device
        return {
            "chosen_input_ids": chosen_enc["input_ids"].to(device),
            "chosen_attention_mask": chosen_enc["attention_mask"].to(device),
            "chosen_labels": chosen_labels.to(device),
            "rejected_input_ids": rejected_enc["input_ids"].to(device),
            "rejected_attention_mask": rejected_enc["attention_mask"].to(device),
            "rejected_labels": rejected_labels.to(device),
        }

    def compute_dpo_loss(
        self,
        policy_chosen_logps: torch.Tensor,
        policy_rejected_logps: torch.Tensor,
        ref_chosen_logps: torch.Tensor,
        ref_rejected_logps: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        if self.config.reference_free:
            chosen_rewards = self.config.beta * policy_chosen_logps
            rejected_rewards = self.config.beta * policy_rejected_logps
        else:
            chosen_rewards = self.config.beta * (policy_chosen_logps - ref_chosen_logps)
            rejected_rewards = self.config.beta * (policy_rejected_logps - ref_rejected_logps)

        logits = chosen_rewards - rejected_rewards

        if self.config.loss_type == "sigmoid":
            if self.config.label_smoothing > 0:
                loss = (
                    -F.logsigmoid(logits) * (1 - self.config.label_smoothing)
                    - F.logsigmoid(-logits) * self.config.label_smoothing
                ).mean()
            else:
                loss = -F.logsigmoid(logits).mean()
        elif self.config.loss_type == "hinge":
            loss = F.relu(1 - logits).mean()
        elif self.config.loss_type == "ipo":
            loss = ((logits - 1 / (2 * self.config.beta)) ** 2).mean()
        else:
            raise ValueError(f"Unknown loss type: {self.config.loss_type}")

        metrics = {
            "loss": loss.item(),
            "reward_margin": logits.mean().item(),
            "chosen_rewards": chosen_rewards.mean().item(),
            "rejected_rewards": rejected_rewards.mean().item(),
            "accuracy": (logits > 0).float().mean().item(),
        }

        return loss, metrics

    def train_step(self, pairs: list[DPOPair]) -> dict[str, float]:
        if not pairs:
            return {}

        all_chosen_logps = []
        all_rejected_logps = []
        all_ref_chosen_logps = []
        all_ref_rejected_logps = []

        for pair in pairs:
            batch = self.tokenize_pair(pair)

            ref_chosen_logps = self.compute_ref_logps(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            ref_rejected_logps = self.compute_ref_logps(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

            policy_chosen_logps = self.compute_logps(
                batch["chosen_input_ids"],
                batch["chosen_attention_mask"],
                batch["chosen_labels"],
            )
            policy_rejected_logps = self.compute_logps(
                batch["rejected_input_ids"],
                batch["rejected_attention_mask"],
                batch["rejected_labels"],
            )

            all_chosen_logps.append(policy_chosen_logps)
            all_rejected_logps.append(policy_rejected_logps)
            all_ref_chosen_logps.append(ref_chosen_logps)
            all_ref_rejected_logps.append(ref_rejected_logps)

        policy_chosen = torch.cat(all_chosen_logps)
        policy_rejected = torch.cat(all_rejected_logps)
        ref_chosen = torch.cat(all_ref_chosen_logps)
        ref_rejected = torch.cat(all_ref_rejected_logps)

        loss, metrics = self.compute_dpo_loss(
            policy_chosen, policy_rejected, ref_chosen, ref_rejected
        )

        loss = loss / self.config.gradient_accumulation_steps
        loss.backward()

        self._step_count += 1
        if self._step_count % self.config.gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(self.lora_params, self.config.max_grad_norm)
            self.optimizer.step()
            self.optimizer.zero_grad()

        return metrics

    def train_on_correction(
        self,
        prompt: str,
        rejected: str,
        chosen: str,
    ) -> dict[str, float]:
        pair = DPOPair(prompt=prompt, chosen=chosen, rejected=rejected)
        return self.train_step([pair])

    def save_checkpoint(self, path: str) -> None:
        state = {
            "optimizer": self.optimizer.state_dict(),
            "step_count": self._step_count,
            "config": self.config,
        }
        torch.save(state, path)

    def load_checkpoint(self, path: str) -> None:
        state = torch.load(path, weights_only=False)
        self.optimizer.load_state_dict(state["optimizer"])
        self._step_count = state["step_count"]
