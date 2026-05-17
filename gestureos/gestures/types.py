from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional, Tuple


class Gesture(str, Enum):
    NONE = "none"
    OPEN_PALM = "open_palm"
    CLOSED_PALM = "closed_palm"
    FIST = "fist"
    THUMB_UP = "thumb_up"
    PINCH = "pinch"
    DOUBLE_PINCH = "double_pinch"
    THREE_FINGER_PINCH = "three_finger_pinch"
    PINCH_HOLD = "pinch_hold"
    INDEX_POINTER = "index_pointer"
    TWO_FINGER_SCROLL = "two_finger_scroll"
    SWIPE_LEFT = "swipe_left"
    SWIPE_RIGHT = "swipe_right"


@dataclass
class GestureEvent:
    gesture: Gesture
    confidence: float
    position: Tuple[float, float] = (0.0, 0.0)
    value: float = 0.0
    metadata: Optional[Dict[str, Any]] = None
