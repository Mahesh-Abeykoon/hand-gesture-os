from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional, Tuple

import cv2
import numpy as np


@dataclass
class FallbackPoint:
    point: Tuple[float, float]
    confidence: float
    age: float


class OpticalFlowPointTracker:
    """
    Short-term point tracker used when MediaPipe drops the hand for a few frames.

    This does not replace hand landmarks. It only bridges tiny detection gaps so
    the cursor/pen does not freeze when fingers partially occlude each other or
    the detector misses one frame.
    """

    def __init__(self, max_age: float = 0.22):
        self.max_age = max_age
        self.prev_gray: Optional[np.ndarray] = None
        self.prev_pt: Optional[np.ndarray] = None
        self.last_seen = 0.0

    def reset(self) -> None:
        self.prev_gray = None
        self.prev_pt = None
        self.last_seen = 0.0

    def update(self, frame_bgr: np.ndarray, point_norm: Tuple[float, float]) -> None:
        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        x = float(np.clip(point_norm[0], 0.0, 1.0) * w)
        y = float(np.clip(point_norm[1], 0.0, 1.0) * h)
        self.prev_gray = gray
        self.prev_pt = np.array([[[x, y]]], dtype=np.float32)
        self.last_seen = time.perf_counter()

    def predict(self, frame_bgr: np.ndarray) -> Optional[FallbackPoint]:
        if self.prev_gray is None or self.prev_pt is None:
            return None
        now = time.perf_counter()
        age = now - self.last_seen
        if age > self.max_age:
            self.reset()
            return None

        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        nxt, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            gray,
            self.prev_pt,
            None,
            winSize=(25, 25),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 18, 0.025),
        )
        if nxt is None or status is None or int(status[0][0]) != 1:
            self.reset()
            return None

        h, w = frame_bgr.shape[:2]
        x, y = float(nxt[0][0][0]), float(nxt[0][0][1])
        if x < -20 or y < -20 or x > w + 20 or y > h + 20:
            self.reset()
            return None

        self.prev_gray = gray
        self.prev_pt = nxt.astype(np.float32)
        # Confidence decays quickly; good enough for pointer bridging, not for clicks.
        flow_error = float(err[0][0]) if err is not None else 0.0
        conf = max(0.15, min(0.72, 0.72 - age * 2.2 - flow_error / 80.0))
        return FallbackPoint((float(np.clip(x / w, 0.0, 1.0)), float(np.clip(y / h, 0.0, 1.0))), conf, age)
