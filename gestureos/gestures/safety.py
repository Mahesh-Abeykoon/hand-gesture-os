from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Deque, Dict

from .types import GestureEvent


class CooldownGate:
    def __init__(self, cooldowns: Dict[str, float]):
        self.cooldowns = cooldowns
        self.last: Dict[str, float] = defaultdict(lambda: 0.0)

    def allow(self, action: str) -> bool:
        now = time.perf_counter()
        cd = float(self.cooldowns.get(action, 0.3))
        if now - self.last[action] >= cd:
            self.last[action] = now
            return True
        return False


class GestureDebouncer:
    """Requires a gesture to be observed repeatedly before firing."""

    def __init__(self, window: int = 5, required: int = 3, threshold: float = 0.72):
        self.window = window
        self.required = required
        self.threshold = threshold
        self.samples: Deque[str] = deque(maxlen=window)

    def update_threshold(self, threshold: float) -> None:
        self.threshold = threshold

    def stable(self, event: GestureEvent) -> bool:
        if event.confidence < self.threshold:
            self.samples.append("none")
            return False
        self.samples.append(event.gesture.value)
        return sum(1 for s in self.samples if s == event.gesture.value) >= self.required
