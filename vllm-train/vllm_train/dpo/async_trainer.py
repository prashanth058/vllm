from __future__ import annotations

import asyncio
import logging
import queue
import threading
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from vllm_train.dpo.efficient_dpo import DPOConfig, DPOPair, EfficientDPOTrainer
from vllm_train.dpo.feedback_collector import FeedbackCollector, FeedbackConfig

if TYPE_CHECKING:
    import torch
    from transformers import PreTrainedTokenizer

logger = logging.getLogger(__name__)


@dataclass
class AsyncDPOConfig:
    dpo_config: DPOConfig = field(default_factory=DPOConfig)
    feedback_config: FeedbackConfig = field(default_factory=FeedbackConfig)
    training_interval: float = 0.1
    max_queue_size: int = 100
    enable_logging: bool = True
    checkpoint_interval: int = 100


class AsyncDPOTrainer:
    def __init__(
        self,
        model: torch.nn.Module,
        tokenizer: PreTrainedTokenizer,
        config: AsyncDPOConfig | None = None,
    ):
        self.config = config or AsyncDPOConfig()
        self._trainer = EfficientDPOTrainer(
            model, tokenizer, self.config.dpo_config
        )
        self._queue: queue.Queue[list[DPOPair]] = queue.Queue(
            maxsize=self.config.max_queue_size
        )
        self._feedback_collector = FeedbackCollector(
            self.config.feedback_config,
            on_batch_ready=self._on_batch_ready,
        )
        self._running = False
        self._thread: threading.Thread | None = None
        self._metrics_history: list[dict[str, Any]] = []
        self._lock = threading.Lock()
        self._total_updates = 0
        self._checkpoint_path: str | None = None

    def start(self) -> None:
        if self._running:
            return

        self._running = True
        self._thread = threading.Thread(target=self._training_loop, daemon=True)
        self._thread.start()
        logger.info("Async DPO trainer started")

    def stop(self, wait: bool = True) -> None:
        self._running = False
        if self._thread and wait:
            self._thread.join(timeout=5.0)
        logger.info("Async DPO trainer stopped")

    def _training_loop(self) -> None:
        while self._running:
            try:
                batch = self._queue.get(timeout=self.config.training_interval)
                self._process_batch(batch)
            except queue.Empty:
                continue
            except Exception as e:
                logger.exception(f"Training error: {e}")

    def _process_batch(self, batch: list[DPOPair]) -> None:
        start = time.time()
        metrics = self._trainer.train_step(batch)
        elapsed = time.time() - start

        with self._lock:
            self._total_updates += 1
            metrics["elapsed_ms"] = elapsed * 1000
            metrics["batch_size"] = len(batch)
            metrics["update_id"] = self._total_updates
            self._metrics_history.append(metrics)

            if len(self._metrics_history) > 1000:
                self._metrics_history = self._metrics_history[-500:]

        if self.config.enable_logging and self._total_updates % 10 == 0:
            logger.info(
                f"DPO update {self._total_updates}: "
                f"loss={metrics.get('loss', 0):.4f} "
                f"acc={metrics.get('accuracy', 0):.2%} "
                f"time={elapsed*1000:.1f}ms"
            )

        if (
            self._checkpoint_path
            and self._total_updates % self.config.checkpoint_interval == 0
        ):
            self._trainer.save_checkpoint(
                f"{self._checkpoint_path}/step_{self._total_updates}.pt"
            )

    def _on_batch_ready(self, pairs: list[DPOPair]) -> None:
        try:
            self._queue.put_nowait(pairs)
        except queue.Full:
            logger.warning("Training queue full, dropping batch")

    def add_correction(
        self,
        prompt: str,
        wrong_response: str,
        correct_response: str,
        immediate: bool = False,
    ) -> dict[str, float] | None:
        if immediate:
            pair = DPOPair(
                prompt=prompt, chosen=correct_response, rejected=wrong_response
            )
            return self._trainer.train_step([pair])

        self._feedback_collector.add_correction(
            prompt, wrong_response, correct_response
        )
        return None

    def add_regenerate(
        self,
        prompt: str,
        rejected_response: str,
        accepted_response: str,
    ) -> None:
        self._feedback_collector.add_regenerate(
            prompt, rejected_response, accepted_response
        )

    def add_thumbs_down(self, prompt: str, response: str) -> None:
        self._feedback_collector.add_thumbs_down(prompt, response)

    def train_batch(self, pairs: list[DPOPair]) -> dict[str, float]:
        return self._trainer.train_step(pairs)

    def get_metrics(self, last_n: int = 10) -> list[dict[str, Any]]:
        with self._lock:
            return self._metrics_history[-last_n:]

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "total_updates": self._total_updates,
                "queue_size": self._queue.qsize(),
                "running": self._running,
                "feedback_stats": self._feedback_collector.get_stats(),
            }

    def set_checkpoint_path(self, path: str) -> None:
        self._checkpoint_path = path

    async def add_correction_async(
        self,
        prompt: str,
        wrong_response: str,
        correct_response: str,
    ) -> dict[str, float]:
        pair = DPOPair(
            prompt=prompt, chosen=correct_response, rejected=wrong_response
        )
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, self._trainer.train_step, [pair]
        )

    def __enter__(self) -> AsyncDPOTrainer:
        self.start()
        return self

    def __exit__(self, *args: Any) -> None:
        self.stop()
