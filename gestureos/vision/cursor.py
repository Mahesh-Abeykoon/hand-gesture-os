from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional, Tuple

from gestureos.utils.math_utils import clamp


class LowPassFilter:
    def __init__(self, alpha: float):
        self.alpha = alpha
        self.initialized = False
        self.s = 0.0

    def filter(self, value: float, alpha: Optional[float] = None) -> float:
        if alpha is not None:
            self.alpha = alpha
        if not self.initialized:
            self.s = value
            self.initialized = True
            return value
        self.s = self.alpha * value + (1.0 - self.alpha) * self.s
        return self.s

    def reset(self) -> None:
        self.initialized = False
        self.s = 0.0


class OneEuroFilter:
    """Adaptive real-time smoothing filter."""

    def __init__(self, min_cutoff: float = 1.2, beta: float = 0.025, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self.x_filter = LowPassFilter(1.0)
        self.dx_filter = LowPassFilter(1.0)
        self.last_x: Optional[float] = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def filter(self, x: float, dt: float) -> float:
        dx = 0.0 if self.last_x is None else (x - self.last_x) / max(dt, 1e-6)
        self.last_x = x
        edx = self.dx_filter.filter(dx, self._alpha(self.d_cutoff, dt))
        cutoff = self.min_cutoff + self.beta * abs(edx)
        return self.x_filter.filter(x, self._alpha(cutoff, dt))

    def reset(self) -> None:
        self.x_filter.reset()
        self.dx_filter.reset()
        self.last_x = None


@dataclass
class CursorSettings:
    margin_x: float = 0.16
    margin_y: float = 0.12
    deadzone: float = 0.0025
    gain: float = 1.12
    min_cutoff: float = 1.25
    beta: float = 0.035
    relative_gain: float = 2.35
    relative_deadzone: float = 0.0012
    max_step: float = 0.026
    raw_jump_limit: float = 0.055
    raw_hard_reset_jump: float = 0.23
    raw_step_soft_limit: float = 0.035


class CursorMapper:
    """
    Robust webcam-to-cursor mapper.

    The previous version still allowed occasional MediaPipe landmark jumps to pass
    through. This version adds:
    - raw landmark jump rejection/clamping
    - warm-up frames after reacquisition
    - vector-level max speed clamp
    - lower acceleration defaults
    - relative touchpad mode for precise control
    """

    def __init__(self, settings: CursorSettings | None = None):
        self.settings = settings or CursorSettings()
        self.fx = OneEuroFilter(self.settings.min_cutoff, self.settings.beta)
        self.fy = OneEuroFilter(self.settings.min_cutoff, self.settings.beta)
        self.rfx = OneEuroFilter(1.85, 0.025)
        self.rfy = OneEuroFilter(1.85, 0.025)
        self.last: Optional[Tuple[float, float]] = None
        self.last_relative: Optional[Tuple[float, float]] = None
        self.last_raw_relative: Optional[Tuple[float, float]] = None
        self.warmup_frames = 0

    def update_settings(self, smoothing: float) -> None:
        smoothing = clamp(smoothing, 0.05, 0.95)
        self.settings.min_cutoff = 2.4 - 1.75 * smoothing
        self.settings.beta = 0.055 - 0.042 * smoothing
        self.settings.relative_deadzone = 0.0014 * smoothing + 0.00055
        # Conservative gain reduces wrong jumps. User can lower smoothing for speed.
        self.settings.relative_gain = 2.95 - 1.05 * smoothing
        self.settings.max_step = 0.036 - 0.018 * smoothing
        self.settings.raw_jump_limit = 0.075 - 0.030 * smoothing
        self.fx.min_cutoff = self.fy.min_cutoff = self.settings.min_cutoff
        self.fx.beta = self.fy.beta = self.settings.beta
        self.rfx.min_cutoff = self.rfy.min_cutoff = 2.35 - 1.25 * smoothing
        self.rfx.beta = self.rfy.beta = 0.050 - 0.035 * smoothing

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()
        self.rfx.reset()
        self.rfy.reset()
        self.last = None
        self.last_relative = None
        self.last_raw_relative = None
        self.warmup_frames = 2

    def map(self, p: Tuple[float, float], dt: float) -> Tuple[float, float]:
        x, y = p
        s = self.settings
        x = (x - s.margin_x) / max(1e-6, 1.0 - 2.0 * s.margin_x)
        y = (y - s.margin_y) / max(1e-6, 1.0 - 2.0 * s.margin_y)
        x = clamp((x - 0.5) * s.gain + 0.5, 0.0, 1.0)
        y = clamp((y - 0.5) * s.gain + 0.5, 0.0, 1.0)
        x = self.fx.filter(x, dt)
        y = self.fy.filter(y, dt)
        if self.last is not None:
            lx, ly = self.last
            if abs(x - lx) < s.deadzone:
                x = lx
            if abs(y - ly) < s.deadzone:
                y = ly
        self.last = (x, y)
        return x, y

    def _sanitize_raw(self, p: Tuple[float, float]) -> Tuple[float, float]:
        x, y = p
        x = clamp(x, 0.0, 1.0)
        y = clamp(y, 0.0, 1.0)
        if self.last_raw_relative is None:
            self.last_raw_relative = (x, y)
            self.warmup_frames = max(self.warmup_frames, 2)
            return x, y
        lx, ly = self.last_raw_relative
        dx, dy = x - lx, y - ly
        mag = math.hypot(dx, dy)
        s = self.settings
        if mag > s.raw_hard_reset_jump:
            # This is almost certainly a hand identity/landmark jump. Do not move.
            self.last_raw_relative = (lx, ly)
            self.warmup_frames = 2
            return lx, ly
        if mag > s.raw_jump_limit:
            # Clamp sudden landmark jumps instead of allowing cursor teleport.
            scale = s.raw_jump_limit / max(mag, 1e-6)
            x = lx + dx * scale
            y = ly + dy * scale
        self.last_raw_relative = (x, y)
        return x, y

    def map_relative(self, p: Tuple[float, float], dt: float) -> Tuple[float, float]:
        """Return normalized screen delta, not an absolute screen position."""
        x, y = self._sanitize_raw(p)
        x = self.rfx.filter(x, dt)
        y = self.rfy.filter(y, dt)
        if self.last_relative is None:
            self.last_relative = (x, y)
            self.warmup_frames = max(self.warmup_frames, 2)
            return (0.0, 0.0)

        lx, ly = self.last_relative
        self.last_relative = (x, y)
        dx, dy = x - lx, y - ly

        if self.warmup_frames > 0:
            self.warmup_frames -= 1
            return (0.0, 0.0)

        mag = math.hypot(dx, dy)
        if mag < self.settings.relative_deadzone:
            return (0.0, 0.0)

        # Gentle acceleration: precise when slow, faster when intentional.
        accel = 1.0 + min(1.25, mag * 55.0)
        dx *= self.settings.relative_gain * accel
        dy *= self.settings.relative_gain * accel

        mag = math.hypot(dx, dy)
        if mag > self.settings.max_step:
            scale = self.settings.max_step / max(mag, 1e-6)
            dx *= scale
            dy *= scale
        return dx, dy
