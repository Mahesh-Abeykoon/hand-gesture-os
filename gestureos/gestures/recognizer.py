from __future__ import annotations

import time
from typing import Dict, List, Tuple

from gestureos.utils.math_utils import clamp, dist, VelocityTracker
from gestureos.vision.hand_tracker import HandResult
from .types import Gesture, GestureEvent

# MediaPipe landmark indexes
WRIST = 0
THUMB_TIP, THUMB_IP, THUMB_MCP = 4, 3, 2
INDEX_TIP, INDEX_PIP, INDEX_MCP = 8, 6, 5
MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP = 12, 10, 9
RING_TIP, RING_PIP, RING_MCP = 16, 14, 13
PINKY_TIP, PINKY_PIP, PINKY_MCP = 20, 18, 17


class GestureRecognizer:
    """
    Robust landmark-based recognizer.

    Improvements over the first version:
    - Uses palm-size-normalized distances instead of hard-coded frame distances.
    - Uses pinch hysteresis, so clicks do not flicker when fingers are near threshold.
    - Emits pinch phases in metadata: start, hold, release.
    - Separates short fist play/pause from long closed-palm deactivate.
    - Swipe detection uses palm velocity and displacement history.
    """

    def __init__(self, sensitivity: float = 0.72):
        self.sensitivity = sensitivity
        self.index_tracker = VelocityTracker(maxlen=10)
        self.palm_tracker = VelocityTracker(maxlen=8)
        self.prev_scroll_y = None

        self.pinch_active = False
        self.pinch_started_at = 0.0
        self.last_pinch_release = 0.0
        self.double_pending = False

        self.fist_started_at = 0.0
        self.prev_is_fist = False

    def set_sensitivity(self, value: float) -> None:
        self.sensitivity = clamp(value, 0.2, 0.95)

    def _p(self, lm: List[Tuple[float, float, float]], idx: int) -> Tuple[float, float]:
        return (lm[idx][0], lm[idx][1])

    def _palm_size(self, lm: List[Tuple[float, float, float]]) -> float:
        # Stable scale: wrist to middle MCP + index MCP to pinky MCP.
        return max(0.035, (dist(self._p(lm, WRIST), self._p(lm, MIDDLE_MCP)) + dist(self._p(lm, INDEX_MCP), self._p(lm, PINKY_MCP))) / 2.0)

    def _finger_states(self, lm: List[Tuple[float, float, float]], handedness: str) -> Dict[str, bool]:
        wrist = self._p(lm, WRIST)
        palm = self._p(lm, MIDDLE_MCP)
        palm_size = self._palm_size(lm)

        def extended(tip: int, pip: int, mcp: int) -> bool:
            tip_p, pip_p, mcp_p = self._p(lm, tip), self._p(lm, pip), self._p(lm, mcp)
            # Extended if tip is farther from wrist than PIP and generally points away from palm.
            return dist(tip_p, wrist) > dist(pip_p, wrist) + 0.08 * palm_size and dist(tip_p, palm) > dist(mcp_p, palm) + 0.16 * palm_size

        # Thumb is side-facing, so y-only checks are unreliable.
        thumb_tip = self._p(lm, THUMB_TIP)
        thumb_ip = self._p(lm, THUMB_IP)
        thumb_mcp = self._p(lm, THUMB_MCP)
        thumb_side = thumb_tip[0] < thumb_ip[0] if handedness == "Right" else thumb_tip[0] > thumb_ip[0]
        thumb_extended = thumb_side and dist(thumb_tip, thumb_mcp) > 0.55 * palm_size

        return {
            "thumb": thumb_extended,
            "index": extended(INDEX_TIP, INDEX_PIP, INDEX_MCP),
            "middle": extended(MIDDLE_TIP, MIDDLE_PIP, MIDDLE_MCP),
            "ring": extended(RING_TIP, RING_PIP, RING_MCP),
            "pinky": extended(PINKY_TIP, PINKY_PIP, PINKY_MCP),
        }

    def _pinch_norm(self, lm: List[Tuple[float, float, float]], tip_idx: int = INDEX_TIP) -> float:
        return dist(self._p(lm, THUMB_TIP), self._p(lm, tip_idx)) / self._palm_size(lm)

    def recognize(self, hand: HandResult) -> GestureEvent:
        lm = hand.landmarks
        now = time.perf_counter()
        states = self._finger_states(lm, hand.handedness)
        fingers_up = sum(states.values())
        palm_center = self._p(lm, MIDDLE_MCP)
        index_tip = self._p(lm, INDEX_TIP)
        self.index_tracker.add(now, index_tip)
        self.palm_tracker.add(now, palm_center)

        pinch_n = self._pinch_norm(lm)
        middle_pinch_n = self._pinch_norm(lm, MIDDLE_TIP)
        ring_pinch_n = self._pinch_norm(lm, RING_TIP)

        close_th = 0.34 - (self.sensitivity - 0.5) * 0.10
        open_th = close_th + 0.10  # hysteresis band
        pinch_now = pinch_n < (open_th if self.pinch_active else close_th)

        # Swipe first, only when not pinching to avoid false slide changes.
        dx, dy = self.palm_tracker.displacement()
        vx, _vy = self.palm_tracker.velocity()
        if not self.pinch_active and abs(dx) > 0.17 and abs(vx) > 1.05 and abs(dy) < 0.13:
            return GestureEvent(Gesture.SWIPE_RIGHT if dx > 0 else Gesture.SWIPE_LEFT, 0.90, palm_center, dx)

        is_three_pinch = pinch_n < close_th and middle_pinch_n < close_th * 1.18 and ring_pinch_n < close_th * 1.35
        if is_three_pinch:
            self.pinch_active = False
            return GestureEvent(Gesture.THREE_FINGER_PINCH, 0.94, index_tip, pinch_n)

        # Pinch state machine: start/hold/release. Release fires click more accurately than continuous contact.
        if pinch_now and not self.pinch_active:
            self.pinch_active = True
            self.pinch_started_at = now
            return GestureEvent(Gesture.PINCH, 0.88, index_tip, pinch_n, {"phase": "start", "duration": 0.0})
        if pinch_now and self.pinch_active:
            hold = now - self.pinch_started_at
            if hold > 0.46:
                return GestureEvent(Gesture.PINCH_HOLD, 0.93, index_tip, hold, {"phase": "hold", "duration": hold})
            return GestureEvent(Gesture.PINCH, 0.86, index_tip, pinch_n, {"phase": "hold", "duration": hold})
        if not pinch_now and self.pinch_active:
            duration = now - self.pinch_started_at
            self.pinch_active = False
            if duration < 0.46:
                if now - self.last_pinch_release < 0.34:
                    self.last_pinch_release = 0.0
                    return GestureEvent(Gesture.DOUBLE_PINCH, 0.94, index_tip, duration, {"phase": "release", "duration": duration})
                self.last_pinch_release = now
                return GestureEvent(Gesture.PINCH, 0.91, index_tip, duration, {"phase": "release", "duration": duration})

        if fingers_up == 5:
            self.prev_is_fist = False
            return GestureEvent(Gesture.OPEN_PALM, 0.96, palm_center)

        if fingers_up == 0:
            if not self.prev_is_fist:
                self.fist_started_at = now
            self.prev_is_fist = True
            hold = now - self.fist_started_at
            if hold > 1.10:
                return GestureEvent(Gesture.CLOSED_PALM, 0.95, palm_center, hold)
            return GestureEvent(Gesture.FIST, 0.91, palm_center, hold)
        self.prev_is_fist = False

        # Thumb-up: thumb extended, other fingers folded, thumb tip above wrist.
        if states["thumb"] and not any(states[k] for k in ["index", "middle", "ring", "pinky"]):
            if lm[THUMB_TIP][1] < lm[WRIST][1] - 0.10:
                return GestureEvent(Gesture.THUMB_UP, 0.90, self._p(lm, THUMB_TIP))

        # Relaxed index detection for cursor mode. Strict finger-state classifiers
        # often flicker when the user is trying to use the hand like a touchpad.
        index_ready = (
            lm[INDEX_TIP][1] < lm[INDEX_PIP][1] + 0.025
            and dist(self._p(lm, INDEX_TIP), self._p(lm, WRIST)) > dist(self._p(lm, INDEX_PIP), self._p(lm, WRIST)) - 0.015
        )
        middle_ready = states["middle"] or (lm[MIDDLE_TIP][1] < lm[MIDDLE_PIP][1] + 0.018)

        if index_ready and middle_ready and not states["ring"] and not states["pinky"]:
            y = (lm[INDEX_TIP][1] + lm[MIDDLE_TIP][1]) / 2
            if self.prev_scroll_y is None:
                self.prev_scroll_y = y
            delta = self.prev_scroll_y - y
            self.prev_scroll_y = y
            return GestureEvent(Gesture.TWO_FINGER_SCROLL, 0.89, index_tip, delta)

        if index_ready and not states["ring"] and not states["pinky"]:
            self.prev_scroll_y = None
            return GestureEvent(Gesture.INDEX_POINTER, 0.93, index_tip)

        self.prev_scroll_y = None
        return GestureEvent(Gesture.NONE, 0.0, palm_center)
