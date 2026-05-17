from __future__ import annotations

import numpy as np
from typing import Tuple


class Kalman2D:
    """Constant-velocity Kalman filter for cursor-grade smoothing."""

    def __init__(self, process_noise: float = 1e-3, measurement_noise: float = 6e-2):
        self.x = np.zeros((4, 1), dtype=np.float32)  # x, y, vx, vy
        self.P = np.eye(4, dtype=np.float32)
        self.Q = np.eye(4, dtype=np.float32) * process_noise
        self.R = np.eye(2, dtype=np.float32) * measurement_noise
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=np.float32)
        self.initialized = False

    def reset(self) -> None:
        self.initialized = False
        self.x[:] = 0
        self.P = np.eye(4, dtype=np.float32)

    def update(self, point: Tuple[float, float], dt: float = 1 / 60) -> Tuple[float, float]:
        if not self.initialized:
            self.x[0, 0], self.x[1, 0] = point
            self.initialized = True
            return point
        F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], dtype=np.float32)
        self.x = F @ self.x
        self.P = F @ self.P @ F.T + self.Q
        z = np.array([[point[0]], [point[1]]], dtype=np.float32)
        y = z - self.H @ self.x
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ y
        self.P = (np.eye(4, dtype=np.float32) - K @ self.H) @ self.P
        return float(self.x[0, 0]), float(self.x[1, 0])


class AdaptiveSmoother:
    """EMA + Kalman hybrid. Lower alpha is smoother; higher alpha is more responsive."""

    def __init__(self, alpha: float = 0.32):
        self.alpha = alpha
        self.last = None
        self.kalman = Kalman2D()

    def set_alpha(self, alpha: float) -> None:
        self.alpha = float(max(0.05, min(0.95, alpha)))

    def reset(self) -> None:
        self.last = None
        self.kalman.reset()

    def update(self, point: Tuple[float, float], dt: float) -> Tuple[float, float]:
        kx, ky = self.kalman.update(point, dt)
        if self.last is None:
            self.last = (kx, ky)
        lx, ly = self.last
        smoothed = (lx + self.alpha * (kx - lx), ly + self.alpha * (ky - ly))
        self.last = smoothed
        return smoothed
