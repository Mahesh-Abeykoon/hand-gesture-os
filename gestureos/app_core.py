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
from gestureos.vision.hand_tracker import HandTracker, HandResult
from gestureos.vision.hand_stabilizer import OpticalFlowPointTracker
from gestureos.utils.math_utils import dist
import time


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
        self.last_cursor_delta = (0.0, 0.0)
        self.last_draw_active = False
        self.precision_pinching = False
        self.precision_pinch_started = 0.0
        self.precision_last_release = 0.0
        self.precision_click_movement = 0.0
        self.precision_pinch_frames = 0
        self.precision_release_frames = 0
        self.precision_pinch_grace = 0   # absorbs 1-2 frame landmark dropout during pinch
        self.precision_dragging = False
        self.precision_frozen_pos = (0.5, 0.5)
        self.optical_pointer = OpticalFlowPointTracker(max_age=0.40)

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


    def _raw_palm_size(self, lm) -> float:
        return max(0.035, (dist((lm[0][0], lm[0][1]), (lm[9][0], lm[9][1])) + dist((lm[5][0], lm[5][1]), (lm[17][0], lm[17][1]))) / 2.0)

    def _raw_pointer_state(self, hand: HandResult):
        """Very direct pointer/click state from landmarks, independent of gesture labels."""
        lm = hand.landmarks
        palm = self._raw_palm_size(lm)
        index = (lm[8][0], lm[8][1])
        thumb = (lm[4][0], lm[4][1])
        pinch_ratio = dist(index, thumb) / palm
        # Hysteresis: wider thresholds so the user doesn't need to press insanely hard.
        down_threshold = 0.42
        up_threshold = 0.62
        pinching = pinch_ratio < (up_threshold if self.precision_pinching else down_threshold)
        # Pointer can be considered available if index tip is not buried in palm.
        wrist = (lm[0][0], lm[0][1])
        index_pip = (lm[6][0], lm[6][1])
        ring_folded = lm[16][1] > lm[14][1] - 0.015 or dist((lm[16][0], lm[16][1]), wrist) < dist((lm[14][0], lm[14][1]), wrist) + 0.015
        pinky_folded = lm[20][1] > lm[18][1] - 0.015 or dist((lm[20][0], lm[20][1]), wrist) < dist((lm[18][0], lm[18][1]), wrist) + 0.015
        pointer_ok = dist(index, wrist) > dist(index_pip, wrist) - 0.02 and ring_folded and pinky_folded
        return index, pinching, pointer_ok, pinch_ratio

    def _execute_precision_pointer(self, hand: HandResult, event: GestureEvent, dt: float, frame=None) -> bool:
        """
        High-priority raw pointer engine.

        Critical design: cursor movement and click are COMPLETELY SEPARATED.
        - While NOT pinching: index finger moves cursor freely.
        - The MOMENT pinch starts: cursor FREEZES. No delta is computed.
          The internal cursor filter is also frozen so no drift accumulates.
        - On release: click fires at the frozen position.
        - For drag: only after holding 400ms does cursor movement resume.
        """
        if not self._active_gate(event):
            return True

        index, pinching, pointer_ok, pinch_ratio = self._raw_pointer_state(hand)
        whiteboard_mode = bool(getattr(self.config, "whiteboard_mode", False))
        if not pointer_ok and not whiteboard_mode:
            return False

        if frame is not None:
            self.optical_pointer.update(frame, index)

        # ── KEY FIX: Only compute cursor movement when NOT pinching ──
        # Previously, _pointer_pos/_pointer_delta were called every frame,
        # which kept advancing the filter even during pinch, causing drift.
        if not self.precision_pinching:
            pos = self._pointer_pos(GestureEvent(Gesture.INDEX_POINTER, 0.95, index), dt)
            delta = self._pointer_delta(GestureEvent(Gesture.INDEX_POINTER, 0.95, index), dt)

            # Magnetic friction: as thumb approaches, slow cursor
            down_threshold = 0.42
            if down_threshold < pinch_ratio < 1.2:
                slow_factor = max(0.08, (pinch_ratio - down_threshold) / (1.2 - down_threshold))
                delta = (delta[0] * slow_factor, delta[1] * slow_factor)

            self.last_cursor_pos = pos
            self.last_cursor_delta = delta
        else:
            # Pinching: cursor is frozen. Do NOT touch the filter.
            delta = (0.0, 0.0)
            self.last_cursor_delta = delta

        self.last_draw_active = pinching

        # Whiteboard/draw mode
        if whiteboard_mode:
            self.last_draw_active = True
            self.precision_pinching = False
            if self.actions.dragging:
                self.actions.mouse_up()
            return True

        toggles = self.config.gesture_toggles
        # Volume strip
        if pinching and toggles.get("volume", True) and index[0] < 0.18:
            if self.cooldown.allow("volume"):
                self._emit(self.actions.volume_by_position(index[1]).message)
            self.precision_pinching = pinching
            return True

        if toggles.get("pointer", True):
            now = time.perf_counter()

            if pinching:
                self.precision_release_frames = 0
                self.precision_pinch_grace = 0
                self.precision_pinch_frames += 1

                if not self.precision_pinching and self.precision_pinch_frames >= 1:
                    # ── Pinch just started: freeze cursor immediately ──
                    self.precision_pinching = True
                    self.precision_pinch_started = now
                    self.precision_click_movement = 0.0
                    self.precision_dragging = False
                    self.precision_frozen_pos = self.last_cursor_pos
                    # Freeze the filter state so no drift accumulates.
                    self.cursor.last_relative = None
                    return True

                if self.precision_pinching:
                    held = now - self.precision_pinch_started
                    if not self.precision_dragging:
                        # Only become drag after intentional hold (>400ms).
                        # Do NOT accumulate movement — cursor is frozen.
                        if held > 0.40:
                            self.actions.mouse_down()
                            self.precision_dragging = True
                            # Re-initialize cursor filter for drag movement.
                            self.cursor.last_relative = None
                    if self.precision_dragging:
                        # During drag, re-enable cursor movement.
                        drag_pos = self._pointer_pos(GestureEvent(Gesture.INDEX_POINTER, 0.95, index), dt)
                        drag_delta = self._pointer_delta(GestureEvent(Gesture.INDEX_POINTER, 0.95, index), dt)
                        self.actions.pointer_relative(drag_delta)
                        self.last_cursor_pos = drag_pos
                return True

            # ── Not pinching ──
            self.precision_pinch_frames = 0

            if self.precision_pinching:
                self.precision_pinch_grace += 1
                # Grace: absorb up to 3 frames of dropout during pinch.
                if self.precision_pinch_grace <= 3:
                    return True

                self.precision_release_frames += 1
                if self.precision_release_frames >= 1:
                    held = now - self.precision_pinch_started
                    if self.precision_dragging:
                        self.actions.mouse_up()
                        self._emit("Drag end")
                    else:
                        # Click: any pinch from 30ms to 2s.
                        if 0.030 <= held <= 2.0 and self.cooldown.allow("left_click"):
                            # Move cursor to frozen target, then click.
                            self.actions.pointer(self.precision_frozen_pos)
                            self.actions.left_click()
                            self._emit("Left click")
                    self.precision_pinching = False
                    self.precision_dragging = False
                    self.precision_pinch_grace = 0
                    # Reset filter for clean hover after click.
                    self.cursor.reset()
                return True

            # ── Normal pointer hover ──
            self.actions.pointer_relative(delta)
            return True
        return False

    def _execute_optical_fallback(self, frame, dt: float) -> bool:
        fb = self.optical_pointer.predict(frame)
        if fb is None:
            self.last_cursor_delta = (0.0, 0.0)
            self.last_draw_active = False
            return False
        if getattr(self.config, "whiteboard_mode", False):
            pos = self._pointer_pos(GestureEvent(Gesture.INDEX_POINTER, fb.confidence, fb.point), dt)
            delta = self._pointer_delta(GestureEvent(Gesture.INDEX_POINTER, fb.confidence, fb.point), dt)
            self.last_cursor_pos = pos
            self.last_cursor_delta = delta
            # In drawing mode the index finger is the pen; bridge short detector drops.
            self.last_draw_active = fb.age < 0.20
            return True
        if self.config.gesture_mode_active or not self.config.activation_required:
            delta = self._pointer_delta(GestureEvent(Gesture.INDEX_POINTER, fb.confidence, fb.point), dt)
            self.last_cursor_delta = delta
            if self.config.gesture_toggles.get("pointer", True):
                if self.precision_pinching and fb.age < 0.10:
                    self.actions.pointer_relative(delta)
                else:
                    if self.precision_pinching:
                        self.actions.mouse_up()
                        self.precision_pinching = False
                    self.actions.pointer_relative(delta)
            return True
        return False

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
        cursor_gestures = (Gesture.INDEX_POINTER, Gesture.PINCH, Gesture.PINCH_HOLD)
        if g in cursor_gestures:
            # Touchpad-style relative cursor movement is much more precise than
            # absolute webcam mapping. It also lets the user keep moving while
            # touching thumb to index for click/drag, like a real touchpad.
            pos = self._pointer_pos(event, dt)
            delta = self._pointer_delta(event, dt)
        elif g not in (Gesture.DOUBLE_PINCH, Gesture.THREE_FINGER_PINCH):
            # Reset the relative filter when the cursor gesture is not active;
            # this avoids a jump when the user returns to pointer mode.
            self.cursor.reset()
        self.last_cursor_pos = pos
        self.last_cursor_delta = delta
        self.last_draw_active = g in (Gesture.PINCH, Gesture.PINCH_HOLD) and phase != "release"

        if getattr(self.config, "whiteboard_mode", False):
            # Whiteboard mode consumes gestures in the UI and prevents OS mouse/keyboard actions.
            return

        # Volume control: intentional pinch/hold in the left control strip.
        # This prevents normal clicks from becoming volume changes.
        if g in (Gesture.PINCH, Gesture.PINCH_HOLD) and toggles.get("volume", True) and event.position[0] < 0.18:
            if self.cooldown.allow("volume"):
                self._emit(self.actions.volume_by_position(event.position[1]).message)
            return

        # Mouse movement is continuous and should not wait for debouncing.
        if g == Gesture.INDEX_POINTER and toggles.get("pointer", True):
            self.actions.pointer_relative(delta)

        elif g == Gesture.PINCH and phase in ("start", "hold") and toggles.get("pointer", True):
            # Thumb touching index keeps cursor movable. Release triggers the click.
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
            if custom and not getattr(self.config, "whiteboard_mode", False) and self.cooldown.allow(custom.name):
                self._emit(self.actions.execute_custom(custom.action).message)
            consumed = self._execute_precision_pointer(hands[0], event, dt, frame)
            if not consumed:
                self._execute(event, dt)
            if self.config.show_skeleton:
                frame = self.tracker.draw(frame, hands)
        else:
            bridged = self._execute_optical_fallback(frame, dt)
            if not bridged:
                if self.actions.dragging:
                    self.actions.drag_relative((0.0, 0.0), False)
                if self.precision_pinching:
                    self.actions.mouse_up()
                    self.precision_pinching = False
                self.cursor.reset()
                self.last_cursor_delta = (0.0, 0.0)
                self.last_draw_active = False
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
