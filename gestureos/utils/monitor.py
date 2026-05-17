from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple


@dataclass
class ScreenInfo:
    width: int
    height: int


def get_screen_size() -> Tuple[int, int]:
    try:
        import pyautogui
        size = pyautogui.size()
        return int(size.width), int(size.height)
    except Exception:
        return 1920, 1080
