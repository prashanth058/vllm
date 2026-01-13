from __future__ import annotations

import heapq
import threading
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable

from vllm_train.dpo.efficient_dpo import DPOPair


class FeedbackType(Enum):
    CORRECTION = "correction"
    REGENERATE = "regenerate"
    THUMBS_DOWN = "thumbs_down"
    THUMBS_UP = "thumbs_up"
    IMPLICIT = "implicit"


@dataclass
class FeedbackConfig:
    priorities: dict[FeedbackType, int] = field(default_factory=lambda: {
        FeedbackType.CORRECTION: 100,
        FeedbackType.REGENERATE: 50,
        FeedbackType.THUMBS_DOWN: 30,
        FeedbackType.THUMBS_UP: 10,
        FeedbackType.IMPLICIT: 5,
    })
    weights: dict[FeedbackType, float] = field(default_factory=lambda: {
        FeedbackType.CORRECTION: 10.0,
        FeedbackType.REGENERATE: 2.0,
        FeedbackType.THUMBS_DOWN: 1.0,
        FeedbackType.THUMBS_UP: 0.5,
        FeedbackType.IMPLICIT: 0.1,
    })
    max_buffer_size: int = 1000
    batch_trigger_size: int = 4


@dataclass(order=True)
class PrioritizedFeedback:
    priority: int
    timestamp: float = field(compare=False)
    feedback_type: FeedbackType = field(compare=False)
    prompt: str = field(compare=False)
    response: str = field(compare=False)
    correction: str | None = field(compare=False, default=None)
    metadata: dict[str, Any] = field(compare=False, default_factory=dict)


class FeedbackCollector:
    def __init__(
        self,
        config: FeedbackConfig | None = None,
        on_batch_ready: Callable[[list[DPOPair]], None] | None = None,
    ):
        self.config = config or FeedbackConfig()
        self.on_batch_ready = on_batch_ready
        self._buffer: list[PrioritizedFeedback] = []
        self._lock = threading.Lock()
        self._stats = {
            "total_collected": 0,
            "total_batches": 0,
            "by_type": {t: 0 for t in FeedbackType},
        }

    def add_correction(
        self,
        prompt: str,
        wrong_response: str,
        correct_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._add_feedback(
            FeedbackType.CORRECTION,
            prompt,
            wrong_response,
            correct_response,
            metadata or {},
        )

    def add_regenerate(
        self,
        prompt: str,
        rejected_response: str,
        accepted_response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._add_feedback(
            FeedbackType.REGENERATE,
            prompt,
            rejected_response,
            accepted_response,
            metadata or {},
        )

    def add_thumbs_down(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._add_feedback(
            FeedbackType.THUMBS_DOWN,
            prompt,
            response,
            None,
            metadata or {},
        )

    def add_thumbs_up(
        self,
        prompt: str,
        response: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        self._add_feedback(
            FeedbackType.THUMBS_UP,
            prompt,
            response,
            None,
            metadata or {},
        )

    def _add_feedback(
        self,
        feedback_type: FeedbackType,
        prompt: str,
        response: str,
        correction: str | None,
        metadata: dict[str, Any],
    ) -> None:
        priority = -self.config.priorities[feedback_type]
        feedback = PrioritizedFeedback(
            priority=priority,
            timestamp=time.time(),
            feedback_type=feedback_type,
            prompt=prompt,
            response=response,
            correction=correction,
            metadata=metadata,
        )

        with self._lock:
            heapq.heappush(self._buffer, feedback)
            self._stats["total_collected"] += 1
            self._stats["by_type"][feedback_type] += 1

            if len(self._buffer) > self.config.max_buffer_size:
                heapq.heappop(self._buffer)

            if len(self._buffer) >= self.config.batch_trigger_size:
                self._trigger_batch()

    def _trigger_batch(self) -> None:
        if not self.on_batch_ready:
            return

        pairs = []
        to_remove = []

        for i, feedback in enumerate(self._buffer):
            if feedback.correction is not None:
                pair = DPOPair(
                    prompt=feedback.prompt,
                    chosen=feedback.correction,
                    rejected=feedback.response,
                    metadata={
                        "feedback_type": feedback.feedback_type.value,
                        "weight": self.config.weights[feedback.feedback_type],
                        **feedback.metadata,
                    },
                )
                pairs.append(pair)
                to_remove.append(i)

            if len(pairs) >= self.config.batch_trigger_size:
                break

        for i in reversed(to_remove):
            self._buffer[i] = self._buffer[-1]
            self._buffer.pop()

        heapq.heapify(self._buffer)

        if pairs:
            self._stats["total_batches"] += 1
            self.on_batch_ready(pairs)

    def get_batch(self, size: int | None = None) -> list[DPOPair]:
        size = size or self.config.batch_trigger_size
        pairs = []

        with self._lock:
            while self._buffer and len(pairs) < size:
                feedback = heapq.heappop(self._buffer)
                if feedback.correction is not None:
                    pair = DPOPair(
                        prompt=feedback.prompt,
                        chosen=feedback.correction,
                        rejected=feedback.response,
                        metadata={
                            "feedback_type": feedback.feedback_type.value,
                            "weight": self.config.weights[feedback.feedback_type],
                            **feedback.metadata,
                        },
                    )
                    pairs.append(pair)

        return pairs

    def get_stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                **self._stats,
                "buffer_size": len(self._buffer),
            }

    def clear(self) -> None:
        with self._lock:
            self._buffer.clear()
