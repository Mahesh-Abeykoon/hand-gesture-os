from __future__ import annotations

import cv2
import time
from typing import Optional, Tuple


class Camera:
    def __init__(self, index: int = 0, width: int = 1280, height: int = 720, target_fps: int = 60):
        self.index = index
        self.width = width
        self.height = height
        self.target_fps = target_fps
        self.cap: Optional[cv2.VideoCapture] = None
        self.last_ts = time.perf_counter()
        self.fps = 0.0

    def open(self) -> None:
        self.cap = cv2.VideoCapture(self.index, cv2.CAP_DSHOW if hasattr(cv2, "CAP_DSHOW") else 0)
        if not self.cap.isOpened():
            raise RuntimeError(f"Unable to open webcam index {self.index}")
        # Force MJPG compression for high speed and high frame rates (essential on Windows/DirectShow)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
        self.cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    def read(self) -> Tuple[bool, Optional[object], float]:
        if self.cap is None:
            self.open()
        ok, frame = self.cap.read()
        now = time.perf_counter()
        dt = max(1e-6, now - self.last_ts)
        self.last_ts = now
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt) if self.fps else 1.0 / dt
        if ok and frame is not None:
            frame = cv2.flip(frame, 1)
        return ok, frame, dt

    def release(self) -> None:
        if self.cap is not None:
            self.cap.release()
            self.cap = None
