from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import torch

from vllm_train.dpo.async_trainer import AsyncDPOConfig, AsyncDPOTrainer
from vllm_train.dpo.efficient_dpo import DPOConfig
from vllm_train.patch import enable_lora_gradients, patch_vllm_for_training

if TYPE_CHECKING:
    from vllm import LLM

logger = logging.getLogger(__name__)


@dataclass
class ContinualConfig:
    model: str = ""
    lora_rank: int = 16
    lora_alpha: int = 32
    target_modules: list[str] = field(default_factory=lambda: [
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj",
    ])
    learning_rate: float = 1e-4
    beta: float = 0.1
    micro_batch_size: int = 4
    async_training: bool = True
    device: str = "cuda"
    dtype: Literal["float16", "bfloat16", "float32"] = "bfloat16"
    max_model_len: int = 4096
    trust_remote_code: bool = True


class ContinualLLM:
    def __init__(self, model_or_config: str | ContinualConfig):
        if isinstance(model_or_config, str):
            self.config = ContinualConfig(model=model_or_config)
        else:
            self.config = model_or_config

        self._vllm: LLM | None = None
        self._hf_model: torch.nn.Module | None = None
        self._tokenizer: Any = None
        self._trainer: AsyncDPOTrainer | None = None
        self._lora_path: Path | None = None
        self._initialized = False

        patch_vllm_for_training()

    def _init_models(self) -> None:
        if self._initialized:
            return

        from peft import LoraConfig, get_peft_model
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info(f"Loading model: {self.config.model}")

        dtype_map = {
            "float16": torch.float16,
            "bfloat16": torch.bfloat16,
            "float32": torch.float32,
        }
        dtype = dtype_map[self.config.dtype]

        self._hf_model = AutoModelForCausalLM.from_pretrained(
            self.config.model,
            torch_dtype=dtype,
            device_map=self.config.device,
            trust_remote_code=self.config.trust_remote_code,
        )

        lora_config = LoraConfig(
            r=self.config.lora_rank,
            lora_alpha=self.config.lora_alpha,
            target_modules=self.config.target_modules,
            lora_dropout=0.0,
            bias="none",
            task_type="CAUSAL_LM",
        )
        self._hf_model = get_peft_model(self._hf_model, lora_config)

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.config.model,
            trust_remote_code=self.config.trust_remote_code,
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        enable_lora_gradients(self._hf_model)

        dpo_config = DPOConfig(
            beta=self.config.beta,
            learning_rate=self.config.learning_rate,
            micro_batch_size=self.config.micro_batch_size,
        )
        async_config = AsyncDPOConfig(dpo_config=dpo_config)
        self._trainer = AsyncDPOTrainer(
            self._hf_model, self._tokenizer, async_config
        )

        if self.config.async_training:
            self._trainer.start()

        self._initialized = True
        logger.info("ContinualLLM initialized")

    def _get_vllm(self) -> LLM:
        if self._vllm is None:
            from vllm import LLM

            self._vllm = LLM(
                model=self.config.model,
                enable_lora=True if self._lora_path else False,
                max_lora_rank=self.config.lora_rank,
                max_model_len=self.config.max_model_len,
                trust_remote_code=self.config.trust_remote_code,
            )
        return self._vllm

    def generate(
        self,
        prompts: str | list[str],
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
        use_vllm: bool = False,
        **kwargs: Any,
    ) -> str | list[str]:
        single_input = isinstance(prompts, str)
        if single_input:
            prompts = [prompts]

        if use_vllm:
            outputs = self._generate_vllm(prompts, max_tokens, temperature, top_p, **kwargs)
        else:
            self._init_models()
            outputs = self._generate_hf(prompts, max_tokens, temperature, top_p, **kwargs)

        return outputs[0] if single_input else outputs

    def _generate_hf(
        self,
        prompts: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs: Any,
    ) -> list[str]:
        outputs = []
        for prompt in prompts:
            inputs = self._tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self._hf_model.device) for k, v in inputs.items()}

            with torch.no_grad():
                generated = self._hf_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=temperature if temperature > 0 else None,
                    top_p=top_p,
                    do_sample=temperature > 0,
                    pad_token_id=self._tokenizer.pad_token_id,
                    **kwargs,
                )

            output_tokens = generated[0][inputs["input_ids"].shape[1]:]
            output = self._tokenizer.decode(output_tokens, skip_special_tokens=True)
            outputs.append(output)

        return outputs

    def _generate_vllm(
        self,
        prompts: list[str],
        max_tokens: int,
        temperature: float,
        top_p: float,
        **kwargs: Any,
    ) -> list[str]:
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
        if self._lora_path:
            lora_request = LoRARequest(
                lora_name="continual_lora",
                lora_int_id=1,
                lora_path=str(self._lora_path),
            )

        outputs = vllm.generate(prompts, sampling_params, lora_request=lora_request)
        return [o.outputs[0].text for o in outputs]

    def learn_from_correction(
        self,
        prompt: str,
        bad_response: str,
        good_response: str,
        immediate: bool = True,
    ) -> dict[str, float] | None:
        self._init_models()
        return self._trainer.add_correction(
            prompt, bad_response, good_response, immediate=immediate
        )

    def learn_from_regenerate(
        self,
        prompt: str,
        rejected_response: str,
        accepted_response: str,
    ) -> None:
        self._init_models()
        self._trainer.add_regenerate(prompt, rejected_response, accepted_response)

    def learn_from_thumbs_down(self, prompt: str, response: str) -> None:
        self._init_models()
        self._trainer.add_thumbs_down(prompt, response)

    def save_lora(self, path: str) -> None:
        self._init_models()
        save_path = Path(path)
        save_path.mkdir(parents=True, exist_ok=True)

        self._hf_model.save_pretrained(save_path)
        self._tokenizer.save_pretrained(save_path)
        self._lora_path = save_path

        logger.info(f"LoRA saved to: {save_path}")

    def load_lora(self, path: str) -> None:
        self._lora_path = Path(path)
        logger.info(f"LoRA path set to: {self._lora_path}")

    def get_training_stats(self) -> dict[str, Any]:
        if self._trainer is None:
            return {}
        return self._trainer.get_stats()

    def get_training_metrics(self, last_n: int = 10) -> list[dict[str, Any]]:
        if self._trainer is None:
            return []
        return self._trainer.get_metrics(last_n)

    def __enter__(self) -> ContinualLLM:
        self._init_models()
        return self

    def __exit__(self, *args: Any) -> None:
        if self._trainer:
            self._trainer.stop()

    def __repr__(self) -> str:
        return f"ContinualLLM(model={self.config.model!r}, lora_rank={self.config.lora_rank})"
