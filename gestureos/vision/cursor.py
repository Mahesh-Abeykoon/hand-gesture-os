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
    relative_gain: float = 2.85
    relative_deadzone: float = 0.00085
    max_step: float = 0.055


class CursorMapper:
    """
    Cursor mapping for webcam hands.

    Two modes are provided:
    - map(): absolute mapping, useful for presentation demos.
    - map_relative(): touchpad-style relative movement, much better for precise mouse control.

    High-quality webcam cursor control should generally use relative mode because
    absolute fingertip coordinates from a webcam are naturally noisy and camera
    perspective makes edge mapping inaccurate. Relative mode behaves like a
    laptop touchpad: small hand deltas become mouse deltas, with adaptive
    smoothing and acceleration.
    """

    def __init__(self, settings: CursorSettings | None = None):
        self.settings = settings or CursorSettings()
        self.fx = OneEuroFilter(self.settings.min_cutoff, self.settings.beta)
        self.fy = OneEuroFilter(self.settings.min_cutoff, self.settings.beta)
        self.rfx = OneEuroFilter(2.2, 0.055)
        self.rfy = OneEuroFilter(2.2, 0.055)
        self.last: Optional[Tuple[float, float]] = None
        self.last_relative: Optional[Tuple[float, float]] = None

    def update_settings(self, smoothing: float) -> None:
        smoothing = clamp(smoothing, 0.05, 0.95)
        self.settings.min_cutoff = 2.8 - 2.1 * smoothing
        self.settings.beta = 0.08 - 0.065 * smoothing
        self.settings.relative_deadzone = 0.0017 * smoothing + 0.00035
        self.settings.relative_gain = 3.45 - 1.20 * smoothing
        self.fx.min_cutoff = self.fy.min_cutoff = self.settings.min_cutoff
        self.fx.beta = self.fy.beta = self.settings.beta
        # Relative mode should be smoother but still responsive.
        self.rfx.min_cutoff = self.rfy.min_cutoff = 3.2 - 1.4 * smoothing
        self.rfx.beta = self.rfy.beta = 0.08 - 0.045 * smoothing

    def reset(self) -> None:
        self.fx.reset()
        self.fy.reset()
        self.rfx.reset()
        self.rfy.reset()
        self.last = None
        self.last_relative = None

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

    def map_relative(self, p: Tuple[float, float], dt: float) -> Tuple[float, float]:
        """Return normalized screen delta, not an absolute screen position."""
        x, y = p
        x = self.rfx.filter(x, dt)
        y = self.rfy.filter(y, dt)
        if self.last_relative is None:
            self.last_relative = (x, y)
            return (0.0, 0.0)

        lx, ly = self.last_relative
        self.last_relative = (x, y)
        dx, dy = x - lx, y - ly

        mag = math.hypot(dx, dy)
        if mag < self.settings.relative_deadzone:
            return (0.0, 0.0)

        # Acceleration curve: small movement stays precise; large movement crosses screen quickly.
        accel = 1.0 + min(2.3, mag * 95.0)
        dx *= self.settings.relative_gain * accel
        dy *= self.settings.relative_gain * accel
        dx = clamp(dx, -self.settings.max_step, self.settings.max_step)
        dy = clamp(dy, -self.settings.max_step, self.settings.max_step)
        return dx, dy
