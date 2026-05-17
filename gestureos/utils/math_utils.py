from __future__ import annotations

import math
from collections import deque
from typing import Iterable, Tuple

Point = Tuple[float, float]


def dist(a: Point, b: Point) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def clamp(x: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, x))


def lerp(a: float, b: float, t: float) -> float:
    return a + (b - a) * t


def moving_average(values: Iterable[Point]) -> Point:
    vals = list(values)
    if not vals:
        return (0.0, 0.0)
    return (sum(v[0] for v in vals) / len(vals), sum(v[1] for v in vals) / len(vals))


class VelocityTracker:
    def __init__(self, maxlen: int = 8):
        self.points = deque(maxlen=maxlen)

    def add(self, t: float, p: Point) -> None:
        self.points.append((t, p))

    def velocity(self) -> Point:
        if len(self.points) < 2:
            return (0.0, 0.0)
        t0, p0 = self.points[0]
        t1, p1 = self.points[-1]
        dt = max(1e-6, t1 - t0)
        return ((p1[0] - p0[0]) / dt, (p1[1] - p0[1]) / dt)

    def displacement(self) -> Point:
        if len(self.points) < 2:
            return (0.0, 0.0)
        return (self.points[-1][1][0] - self.points[0][1][0], self.points[-1][1][1] - self.points[0][1][1])
