from __future__ import annotations

from typing import Callable, List, Optional, Tuple

import cv2

from gestureos.actions.controller import ActionController
from gestureos.gestures.recognizer import GestureRecognizer
from gestureos.gestures.safety import CooldownGate, GestureDebouncer
from gestureos.gestures.trainer import GestureTrainer
from gestureos.gestures.types import Gesture, GestureEvent
from gestureos.settings.config import AppConfig
from gestureos.utils.logging import get_logger
from gestureos.vision.camera import Camera
from gestureos.vision.cursor import CursorMapper
from gestureos.vision.hand_tracker import HandTracker


class GestureEngine:
    """Coordinates camera, tracking, recognition, safety, and actions."""

    def __init__(self, config: AppConfig, on_log: Optional[Callable[[str], None]] = None):
        self.config = config
        self.log = get_logger("GestureOS.Engine")
        self.on_log = on_log or (lambda msg: None)
        self.camera = Camera(config.camera_index, config.camera_width, config.camera_height, config.target_fps)
        self.tracker = HandTracker(max_hands=2 if config.dual_hand_mode else 1)
        self.recognizer = GestureRecognizer(config.sensitivity)
        self.debouncer = GestureDebouncer(threshold=config.confidence_threshold)
        self.cooldown = CooldownGate(config.cooldowns)
        self.actions = ActionController(config)
        self.cursor = CursorMapper()
        self.cursor.update_settings(config.cursor_smoothing)
        self.trainer = GestureTrainer(config.profiles_path)
        self.history: List[str] = []
        self.last_event = GestureEvent(Gesture.NONE, 0.0)
        self.last_hands = []
        self.last_cursor_pos = (0.5, 0.5)

    def close(self) -> None:
        self.camera.release()

    def _emit(self, message: str) -> None:
        self.history.append(message)
        self.history[:] = self.history[-200:]
        self.on_log(message)
        self.log.info(message)

    def update_config(self) -> None:
        self.recognizer.set_sensitivity(self.config.sensitivity)
        self.debouncer.update_threshold(self.config.confidence_threshold)
        self.cursor.update_settings(self.config.cursor_smoothing)

    def _active_gate(self, event: GestureEvent) -> bool:
        if not self.config.activation_required:
            return True
        if event.gesture == Gesture.OPEN_PALM and self.debouncer.stable(event) and self.cooldown.allow("activate"):
            self.config.gesture_mode_active = True
            self.cursor.reset()
            self._emit("Gesture mode activated")
            return False
        if event.gesture == Gesture.CLOSED_PALM and self.debouncer.stable(event) and self.cooldown.allow("deactivate"):
            self.config.gesture_mode_active = False
            self.actions.drag(self.last_cursor_pos, False)
            self._emit("Gesture mode deactivated")
            return False
        return self.config.gesture_mode_active

    def _pointer_pos(self, event: GestureEvent, dt: float) -> Tuple[float, float]:
        self.last_cursor_pos = self.cursor.map(event.position, dt)
        return self.last_cursor_pos

    def _pointer_delta(self, event: GestureEvent, dt: float) -> Tuple[float, float]:
        return self.cursor.map_relative(event.position, dt)

    def _execute(self, event: GestureEvent, dt: float) -> None:
        toggles = self.config.gesture_toggles
        g = event.gesture
        if not self._active_gate(event):
            return

        metadata = event.metadata or {}
        phase = metadata.get("phase")
        duration = float(metadata.get("duration", 0.0) or 0.0)
        stable = self.debouncer.stable(event)
        pos = self.last_cursor_pos
        delta = (0.0, 0.0)
        if g in (Gesture.INDEX_POINTER, Gesture.PINCH_HOLD):
            # Touchpad-style relative cursor movement is much more precise than
            # absolute webcam mapping. Absolute pos is still updated for fallback.
            pos = self._pointer_pos(event, dt)
            delta = self._pointer_delta(event, dt)
        elif g not in (Gesture.PINCH, Gesture.DOUBLE_PINCH, Gesture.THREE_FINGER_PINCH):
            # Reset the relative filter when the cursor gesture is not active;
            # this avoids a jump when the user returns to pointer mode.
            self.cursor.reset()

        # Volume control: intentional pinch/hold in the left control strip.
        # This prevents normal clicks from becoming volume changes.
        if g in (Gesture.PINCH, Gesture.PINCH_HOLD) and toggles.get("volume", True) and event.position[0] < 0.18:
            if self.cooldown.allow("volume"):
                self._emit(self.actions.volume_by_position(event.position[1]).message)
            return

        # Mouse movement is continuous and should not wait for debouncing.
        if g == Gesture.INDEX_POINTER and toggles.get("pointer", True):
            self.actions.pointer_relative(delta)

        # Clicks happen on pinch release, not while fingers touch. This is much more reliable.
        elif g == Gesture.PINCH and phase == "release" and duration < 0.46:
            if toggles.get("left_click", True) and self.cooldown.allow("left_click"):
                self._emit(self.actions.left_click().message)

        elif g == Gesture.DOUBLE_PINCH and toggles.get("double_click", True) and self.cooldown.allow("double_click"):
            self._emit(self.actions.double_click().message)

        elif g == Gesture.THREE_FINGER_PINCH and stable and toggles.get("right_click", True) and self.cooldown.allow("right_click"):
            self._emit(self.actions.right_click().message)

        elif g == Gesture.PINCH_HOLD and toggles.get("drag", True):
            self.actions.drag_relative(delta, True)

        else:
            if self.actions.dragging and g != Gesture.PINCH_HOLD:
                self.actions.drag_relative((0.0, 0.0), False)

        if not stable:
            return

        custom_action = self.config.custom_mappings.get(g.value) if self.config.custom_mappings else None
        if custom_action and g not in (Gesture.NONE, Gesture.INDEX_POINTER, Gesture.PINCH_HOLD):
            if self.cooldown.allow(f"custom_{g.value}"):
                self._emit(self.actions.execute_custom(custom_action).message)
                return

        if g == Gesture.FIST and toggles.get("play_pause", True) and self.cooldown.allow("play_pause"):
            self._emit(self.actions.play_pause().message)
        elif g == Gesture.THUMB_UP and toggles.get("mute", True) and self.cooldown.allow("mute"):
            self._emit(self.actions.mute().message)
        elif g == Gesture.SWIPE_RIGHT and toggles.get("swipe_right", True) and self.cooldown.allow("next"):
            self._emit(self.actions.next().message)
        elif g == Gesture.SWIPE_LEFT and toggles.get("swipe_left", True) and self.cooldown.allow("previous"):
            self._emit(self.actions.previous().message)
        elif g == Gesture.TWO_FINGER_SCROLL and toggles.get("scroll", True) and self.cooldown.allow("scroll"):
            self.actions.scroll(event.value)

    def step(self) -> Tuple[object, Optional[GestureEvent], float]:
        ok, frame, dt = self.camera.read()
        if not ok or frame is None:
            raise RuntimeError("Camera frame unavailable")
        if self.config.low_light_enhancement:
            frame = self._enhance_low_light(frame)
        hands = self.tracker.process(frame)
        self.last_hands = hands
        event = GestureEvent(Gesture.NONE, 0.0)
        if hands:
            event = self.recognizer.recognize(hands[0])
            custom = self.trainer.predict(hands[0].landmarks)
            if custom and self.cooldown.allow(custom.name):
                self._emit(self.actions.execute_custom(custom.action).message)
            self._execute(event, dt)
            if self.config.show_skeleton:
                frame = self.tracker.draw(frame, hands)
        else:
            if self.actions.dragging:
                self.actions.drag_relative((0.0, 0.0), False)
            self.cursor.reset()
        self.last_event = event
        frame = self._draw_hud(frame, event, self.camera.fps, bool(hands))
        return frame, event, self.camera.fps

    def _enhance_low_light(self, frame):
        # CLAHE on luminance gives better landmark stability in dim rooms.
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        y, cr, cb = cv2.split(ycrcb)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        y = clahe.apply(y)
        return cv2.cvtColor(cv2.merge((y, cr, cb)), cv2.COLOR_YCrCb2BGR)

    def _draw_hud(self, frame, event: GestureEvent, fps: float, hand_seen: bool):
        h, w = frame.shape[:2]
        active = self.config.gesture_mode_active or not self.config.activation_required
        color = (80, 220, 120) if active else (70, 90, 120)
        cv2.rectangle(frame, (14, 14), (430, 116), (10, 12, 18), -1)
        cv2.rectangle(frame, (14, 14), (430, 116), color, 2)
        cv2.putText(frame, "GestureOS", (28, 44), cv2.FONT_HERSHEY_SIMPLEX, 0.85, (245, 245, 245), 2, cv2.LINE_AA)
        cv2.putText(frame, f"Mode: {'ACTIVE' if active else 'STANDBY'}   Hand: {'LOCKED' if hand_seen else 'NO HAND'}", (28, 72), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"Gesture: {event.gesture.value}  Conf: {event.confidence:.2f}  FPS: {fps:.1f}", (28, 98), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (220, 230, 240), 1, cv2.LINE_AA)
        # Left-edge volume zone hint.
        cv2.rectangle(frame, (0, 0), (int(w * 0.18), h), (40, 80, 120), 1)
        cv2.putText(frame, "VOL", (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (120, 180, 255), 1, cv2.LINE_AA)
        return frame
