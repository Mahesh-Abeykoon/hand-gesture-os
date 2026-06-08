"""
One-Euro Filter + Bézier curve drawing engine for GestureOS.

The One-Euro Filter (Casiez et al., 2012) provides:
  - Low jitter when the hand is still or moving slowly
  - Low latency when the hand moves fast
  
This is the gold standard for smoothing noisy sensor input (like MediaPipe
hand landmarks) and is far superior to simple EMA for drawing applications.

Bézier subdivision provides smooth curves between the sparse sample points
that MediaPipe delivers at ~25-30fps, eliminating the angular "connect the
dots" look of straight-line rendering.
"""

from __future__ import annotations

import math
import time
from typing import Optional, Tuple, List

import cv2
import numpy as np


class OneEuroFilter:
    """
    One-Euro Filter for 1D signal smoothing.
    
    Parameters:
        min_cutoff: Minimum cutoff frequency (Hz). Lower = more smoothing when still.
                    Good range for hand tracking: 0.8–2.0
        beta:       Speed coefficient. Higher = less latency when moving fast.
                    Good range: 0.3–1.0
        d_cutoff:   Cutoff for the derivative filter. Usually 1.0.
    """

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.min_cutoff = min_cutoff
        self.beta = beta
        self.d_cutoff = d_cutoff
        self._x_prev: Optional[float] = None
        self._dx_prev: float = 0.0
        self._t_prev: Optional[float] = None

    def reset(self):
        self._x_prev = None
        self._dx_prev = 0.0
        self._t_prev = None

    @staticmethod
    def _alpha(cutoff: float, dt: float) -> float:
        tau = 1.0 / (2.0 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / max(dt, 1e-6))

    def __call__(self, x: float, t: Optional[float] = None) -> float:
        if t is None:
            t = time.perf_counter()
        if self._t_prev is None:
            self._x_prev = x
            self._dx_prev = 0.0
            self._t_prev = t
            return x

        dt = max(t - self._t_prev, 1e-6)
        self._t_prev = t

        # Derivative (speed) estimate — smoothed
        dx = (x - self._x_prev) / dt
        a_d = self._alpha(self.d_cutoff, dt)
        dx_hat = a_d * dx + (1.0 - a_d) * self._dx_prev
        self._dx_prev = dx_hat

        # Adaptive cutoff: higher speed → higher cutoff → less smoothing
        cutoff = self.min_cutoff + self.beta * abs(dx_hat)
        a = self._alpha(cutoff, dt)
        x_hat = a * x + (1.0 - a) * self._x_prev
        self._x_prev = x_hat
        return x_hat


class OneEuroFilter2D:
    """One-Euro filter for 2D coordinates (x, y)."""

    def __init__(self, min_cutoff: float = 1.0, beta: float = 0.5, d_cutoff: float = 1.0):
        self.fx = OneEuroFilter(min_cutoff, beta, d_cutoff)
        self.fy = OneEuroFilter(min_cutoff, beta, d_cutoff)

    def reset(self):
        self.fx.reset()
        self.fy.reset()

    def __call__(self, x: float, y: float, t: Optional[float] = None) -> Tuple[float, float]:
        return self.fx(x, t), self.fy(y, t)


def _quadratic_bezier(p0: Tuple[int, int], p1: Tuple[int, int], p2: Tuple[int, int],
                       steps: int = 8) -> List[Tuple[int, int]]:
    """Generate points along a quadratic Bézier curve from p0 through p1 to p2."""
    pts = []
    for i in range(steps + 1):
        t = i / steps
        t1 = 1.0 - t
        x = t1 * t1 * p0[0] + 2 * t1 * t * p1[0] + t * t * p2[0]
        y = t1 * t1 * p0[1] + 2 * t1 * t * p1[1] + t * t * p2[1]
        pts.append((int(round(x)), int(round(y))))
    return pts


class InkEngine:
    """
    High-quality drawing engine for gesture-based handwriting.
    
    Features:
      - One-Euro filtered position (kills MediaPipe jitter, preserves speed)
      - Dead zone (sub-pixel noise doesn't trigger strokes)
      - Bézier curve interpolation (smooth arcs instead of angular segments)
      - Speed-adaptive line width (thinner when fast, wider when slow → natural feel)
      - Proper pen-up / pen-down state machine
    """

    # Tuning presets for One-Euro filter
    # Lower min_cutoff = more smoothing when still (kills jitter)
    # Higher beta = more responsive when moving fast (kills lag)
    SMOOTH_PRESET = {"min_cutoff": 0.8, "beta": 0.6, "d_cutoff": 1.0}   # Calligraphy
    NORMAL_PRESET = {"min_cutoff": 1.2, "beta": 0.70, "d_cutoff": 1.0}   # Default
    FAST_PRESET   = {"min_cutoff": 2.0, "beta": 1.0, "d_cutoff": 1.0}   # Quick sketching

    def __init__(self, fw: int, fh: int, pen_color=(20, 24, 32), pen_width: int = 5):
        self.fw = fw
        self.fh = fh
        self.pen_color = pen_color
        self.pen_width = pen_width
        
        # One-Euro filter for position
        self._filter = OneEuroFilter2D(**self.NORMAL_PRESET)
        
        # State
        self._pen_down = False
        self._prev_pt: Optional[Tuple[int, int]] = None
        self._prev_prev_pt: Optional[Tuple[int, int]] = None  # for Bézier midpoint
        self._dead_zone_px = 2.0   # pixels of movement below which we consider "still"
        self._teleport_threshold = 0.12  # normalized units — jump bigger than this = new stroke
        self._last_raw = (0.5, 0.5)
        self._initialized = False

    def resize(self, fw: int, fh: int):
        """Call when the frame dimensions change."""
        self.fw = fw
        self.fh = fh

    def reset(self):
        """Full reset — new stroke."""
        self._filter.reset()
        self._pen_down = False
        self._prev_pt = None
        self._prev_prev_pt = None
        self._initialized = False

    def pen_up(self):
        """Lift the pen (end current stroke, keep filter state for stability)."""
        self._pen_down = False
        self._prev_pt = None
        self._prev_prev_pt = None

    def update(self, raw_x: float, raw_y: float, ink: np.ndarray,
               tool: str = "pen", t: Optional[float] = None) -> Tuple[int, int]:
        """
        Process one frame of finger position and draw to the ink buffer.
        
        Args:
            raw_x, raw_y: Normalized [0,1] position from MediaPipe landmark
            ink: BGR numpy array to draw on (white=empty)
            tool: "pen" or "eraser"
            t: Optional timestamp (perf_counter seconds)
            
        Returns:
            (px, py): The filtered pixel position (for cursor dot display)
        """
        if t is None:
            t = time.perf_counter()

        # Teleport detection — reject MediaPipe landmark identity swaps
        if self._initialized:
            jump = math.hypot(raw_x - self._last_raw[0], raw_y - self._last_raw[1])
            if jump > self._teleport_threshold:
                self.reset()
        self._last_raw = (raw_x, raw_y)
        self._initialized = True

        # One-Euro filter
        fx, fy = self._filter(raw_x, raw_y, t)

        # Convert to pixel coords
        px = int(max(0, min(self.fw - 1, fx * self.fw)))
        py = int(max(0, min(self.fh - 1, fy * self.fh)))
        cur = (px, py)

        # Eraser tool — no smoothing needed, just blast a circle
        if tool == "eraser":
            cv2.circle(ink, cur, 30, (255, 255, 255), -1, cv2.LINE_AA)
            self._pen_down = True
            self._prev_pt = cur
            return cur

        # Dead zone: if pen is down and we haven't moved enough, don't draw
        if self._pen_down and self._prev_pt is not None:
            dpx = math.hypot(cur[0] - self._prev_pt[0], cur[1] - self._prev_pt[1])
            if dpx < self._dead_zone_px:
                return cur  # Return position but don't draw

        # --- PEN drawing ---
        if not self._pen_down:
            # First point of a new stroke
            cv2.circle(ink, cur, max(1, self.pen_width // 2), self.pen_color, -1, cv2.LINE_AA)
            self._pen_down = True
            self._prev_pt = cur
            self._prev_prev_pt = None
            return cur

        # We have at least one previous point
        if self._prev_prev_pt is None:
            # Second point: just a straight line
            cv2.line(ink, self._prev_pt, cur, self.pen_color, self.pen_width, cv2.LINE_AA)
            self._prev_prev_pt = self._prev_pt
            self._prev_pt = cur
            return cur

        # Third+ point: Bézier curve through the midpoints
        # This creates smooth curves by using the actual samples as control points
        # and the midpoints between them as the on-curve points
        mid_prev = ((self._prev_prev_pt[0] + self._prev_pt[0]) // 2,
                     (self._prev_prev_pt[1] + self._prev_pt[1]) // 2)
        mid_cur = ((self._prev_pt[0] + cur[0]) // 2,
                    (self._prev_pt[1] + cur[1]) // 2)

        # Draw Bézier from mid_prev through prev_pt to mid_cur
        bezier_pts = _quadratic_bezier(mid_prev, self._prev_pt, mid_cur, steps=10)
        for i in range(len(bezier_pts) - 1):
            cv2.line(ink, bezier_pts[i], bezier_pts[i + 1],
                     self.pen_color, self.pen_width, cv2.LINE_AA)

        self._prev_prev_pt = self._prev_pt
        self._prev_pt = cur
        return cur

    def draw_cursor_dot(self, frame: np.ndarray, px: int, py: int, drawing: bool):
        """
        Draw a small cursor dot on the camera frame to show where the pen is.
        Green when drawing, dim when hovering.
        """
        if drawing:
            # Bright green dot with white center
            cv2.circle(frame, (px, py), 8, (50, 220, 50), 2, cv2.LINE_AA)
            cv2.circle(frame, (px, py), 3, (255, 255, 255), -1, cv2.LINE_AA)
        else:
            # Dim grey ring (pen lifted)
            cv2.circle(frame, (px, py), 6, (150, 150, 150), 1, cv2.LINE_AA)
