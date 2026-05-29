from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Tuple

from gestureos.gestures.types import Gesture, GestureEvent
from gestureos.settings.config import AppConfig
from gestureos.utils.math_utils import clamp
from gestureos.utils.monitor import get_screen_size
from gestureos.utils.sound import beep
from .volume import VolumeController

try:
    import pyautogui
    pyautogui.FAILSAFE = False
    pyautogui.PAUSE = 0
except Exception:  # pragma: no cover
    pyautogui = None


@dataclass
class ActionResult:
    executed: bool
    message: str


class ActionController:
    def __init__(self, config: AppConfig):
        self.config = config
        self.volume = VolumeController()
        self.screen_w, self.screen_h = get_screen_size()
        self.dragging = False
        self.last_volume_anchor: Optional[float] = None
        self._rel_x_accum = 0.0
        self._rel_y_accum = 0.0

    def _hotkey(self, *keys: str) -> None:
        if pyautogui:
            pyautogui.hotkey(*keys)

    def _press(self, key: str) -> None:
        if pyautogui:
            pyautogui.press(key)

    def pointer(self, pos: Tuple[float, float]) -> ActionResult:
        if not pyautogui:
            return ActionResult(False, "PyAutoGUI unavailable")
        x = clamp(pos[0], 0.02, 0.98)
        y = clamp(pos[1], 0.02, 0.98)
        pyautogui.moveTo(int(x * self.screen_w), int(y * self.screen_h), duration=0, _pause=False)
        return ActionResult(True, "Pointer move")

    def get_cursor_position(self) -> Tuple[int, int]:
        if pyautogui:
            return pyautogui.position()
        return (0, 0)

    def pointer_absolute_pixel(self, x: int, y: int) -> ActionResult:
        if pyautogui:
            pyautogui.moveTo(x, y, duration=0, _pause=False)
            return ActionResult(True, "Pointer absolute pixel move")
        return ActionResult(False, "PyAutoGUI unavailable")


    def pointer_relative(self, delta: Tuple[float, float]) -> ActionResult:
        """Touchpad-style relative pointer movement."""
        if not pyautogui:
            return ActionResult(False, "PyAutoGUI unavailable")
        # Keep fractional pixel remainders so slow precise movement is not lost
        # by int() rounding. This makes the cursor feel much less choppy.
        self._rel_x_accum += delta[0] * self.screen_w
        self._rel_y_accum += delta[1] * self.screen_h
        dx = int(self._rel_x_accum)
        dy = int(self._rel_y_accum)
        self._rel_x_accum -= dx
        self._rel_y_accum -= dy
        # Safety clamp: never allow a one-frame cursor teleport.
        max_px = max(18, int(min(self.screen_w, self.screen_h) * 0.035))
        dx = max(-max_px, min(max_px, dx))
        dy = max(-max_px, min(max_px, dy))
        if dx or dy:
            pyautogui.moveRel(dx, dy, duration=0, _pause=False)
        return ActionResult(True, "Pointer relative move")

    def left_click(self) -> ActionResult:
        if pyautogui:
            pyautogui.click(button="left")
            beep(self.config.sound_feedback)
            return ActionResult(True, "Left click")
        return ActionResult(False, "PyAutoGUI unavailable")

    def double_click(self) -> ActionResult:
        if pyautogui:
            pyautogui.doubleClick()
            beep(self.config.sound_feedback)
            return ActionResult(True, "Double click")
        return ActionResult(False, "PyAutoGUI unavailable")

    def right_click(self) -> ActionResult:
        if pyautogui:
            pyautogui.click(button="right")
            beep(self.config.sound_feedback)
            return ActionResult(True, "Right click")
        return ActionResult(False, "PyAutoGUI unavailable")

    def drag(self, pos: Tuple[float, float], holding: bool) -> ActionResult:
        if not pyautogui:
            return ActionResult(False, "PyAutoGUI unavailable")
        if holding and not self.dragging:
            pyautogui.mouseDown(_pause=False)
            self.dragging = True
        self.pointer(pos)
        if not holding and self.dragging:
            pyautogui.mouseUp(_pause=False)
            self.dragging = False
        return ActionResult(True, "Drag")

    def drag_relative(self, delta: Tuple[float, float], holding: bool) -> ActionResult:
        if not pyautogui:
            return ActionResult(False, "PyAutoGUI unavailable")
        if holding and not self.dragging:
            pyautogui.mouseDown(_pause=False)
            self.dragging = True
        self.pointer_relative(delta)
        if not holding and self.dragging:
            pyautogui.mouseUp(_pause=False)
            self.dragging = False
        return ActionResult(True, "Drag relative")


    def mouse_down(self) -> ActionResult:
        if pyautogui:
            if not self.dragging:
                pyautogui.mouseDown(_pause=False)
                self.dragging = True
            return ActionResult(True, "Mouse down")
        return ActionResult(False, "PyAutoGUI unavailable")

    def mouse_up(self) -> ActionResult:
        if pyautogui:
            if self.dragging:
                pyautogui.mouseUp(_pause=False)
                self.dragging = False
            return ActionResult(True, "Mouse up")
        return ActionResult(False, "PyAutoGUI unavailable")

    def play_pause(self) -> ActionResult:
        self._press("playpause")
        beep(self.config.sound_feedback)
        return ActionResult(True, "Play/Pause")

    def mute(self) -> ActionResult:
        if not self.volume.mute_toggle():
            self._press("volumemute")
        beep(self.config.sound_feedback)
        return ActionResult(True, "Mute toggle")

    def next(self) -> ActionResult:
        # Browser tab / PowerPoint / Google Slides friendly.
        self._press("right")
        beep(self.config.sound_feedback)
        return ActionResult(True, "Next slide/tab")

    def previous(self) -> ActionResult:
        self._press("left")
        beep(self.config.sound_feedback)
        return ActionResult(True, "Previous slide/tab")

    def scroll(self, delta: float) -> ActionResult:
        if pyautogui:
            pyautogui.scroll(int(clamp(delta * 2800, -8, 8)))
            return ActionResult(True, "Scroll")
        return ActionResult(False, "PyAutoGUI unavailable")

    def volume_by_position(self, y: float) -> ActionResult:
        scalar = clamp(1.0 - y, 0.0, 1.0)
        if not self.volume.set_volume_scalar(scalar):
            # Fallback to keyboard volume buttons for non-Windows.
            self._press("volumeup" if scalar > 0.5 else "volumedown")
        return ActionResult(True, f"Volume {int(scalar * 100)}%")

    def execute_custom(self, action: str) -> ActionResult:
        if action.startswith("hotkey:"):
            keys = action.split(":", 1)[1].split("+")
            self._hotkey(*keys)
            return ActionResult(True, f"Hotkey {'+'.join(keys)}")
        if action.startswith("press:"):
            self._press(action.split(":", 1)[1])
            return ActionResult(True, action)
        return ActionResult(False, f"Unknown custom action: {action}")
