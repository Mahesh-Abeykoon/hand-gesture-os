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
        self.tracker = HandTracker(
            max_hands=2 if config.dual_hand_mode else 1,
            min_detection_confidence=max(0.55, config.confidence_threshold),
            min_tracking_confidence=max(0.50, config.confidence_threshold - 0.05)
        )
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
        self.os_cursor_history: List[Tuple[int, int]] = []
        self.precision_frozen_os_pos = (0, 0)
        self.pre_pinching = False
        self.pre_pinch_frozen_os_pos = None
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

        now = time.perf_counter()
        toggles = self.config.gesture_toggles

        # ── Double-stage Tremor-proof Click Snap Logic ──
        # 1. Hover stage: pinch_ratio >= 0.65
        if pinch_ratio >= 0.65:
            self.pre_pinching = False
            self.pre_pinch_frozen_os_pos = None

            if not self.precision_pinching:
                # Normal hover relative tracking
                pos = self._pointer_pos(GestureEvent(Gesture.INDEX_POINTER, 0.95, index), dt)
                delta = self._pointer_delta(GestureEvent(Gesture.INDEX_POINTER, 0.95, index), dt)

                # Magnetic friction: slow cursor as fingers approach 0.65
                if 0.65 < pinch_ratio < 1.1:
                    slow_factor = (pinch_ratio - 0.65) / (1.1 - 0.65)
                    slow_factor = max(0.1, slow_factor)
                    delta = (delta[0] * slow_factor, delta[1] * slow_factor)

                self.last_cursor_pos = pos
                self.last_cursor_delta = delta

                # Move OS pointer relatively
                if toggles.get("pointer", True) and not whiteboard_mode:
                    self.actions.pointer_relative(delta)

                # Keep track of actual OS cursor positions in hover history
                cur_os = self.actions.get_cursor_position()
                self.os_cursor_history.append(cur_os)
                self.os_cursor_history[:] = self.os_cursor_history[-10:]
            else:
                # During active pinch hold, cursor is locked
                delta = (0.0, 0.0)
                self.last_cursor_delta = delta

        # 2. Pre-pinch / Freeze stage: pinch_ratio < 0.65
        else:
            delta = (0.0, 0.0)
            self.last_cursor_delta = delta

            if not self.pre_pinching:
                self.pre_pinching = True
                # Lock target at the OS cursor position from 5 frames ago to absorb pre-pinch drift
                if len(self.os_cursor_history) >= 5:
                    self.pre_pinch_frozen_os_pos = self.os_cursor_history[-5]
                elif self.os_cursor_history:
                    self.pre_pinch_frozen_os_pos = self.os_cursor_history[0]
                else:
                    self.pre_pinch_frozen_os_pos = self.actions.get_cursor_position()

                # Freeze the cursor filter
                self.cursor.last_relative = None

            # Actively hold the OS cursor rock-solid at the frozen target
            if self.pre_pinch_frozen_os_pos and not self.precision_dragging and not whiteboard_mode:
                self.actions.pointer_absolute_pixel(*self.pre_pinch_frozen_os_pos)

        self.last_draw_active = pinching

        # Whiteboard/draw mode: index-only drawing, pinch is paused
        if whiteboard_mode:
            self.last_draw_active = True
            self.precision_pinching = False
            self.pre_pinching = False
            self.pre_pinch_frozen_os_pos = None
            if self.actions.dragging:
                self.actions.mouse_up()
            return True

        # Volume strip (left edge)
        if pinching and toggles.get("volume", True) and index[0] < 0.18:
            if self.cooldown.allow("volume"):
                self._emit(self.actions.volume_by_position(index[1]).message)
            self.precision_pinching = pinching
            return True

        if toggles.get("pointer", True):
            if pinching:
                self.precision_release_frames = 0
                self.precision_pinch_grace = 0
                self.precision_pinch_frames += 1

                if not self.precision_pinching and self.precision_pinch_frames >= 1:
                    self.precision_pinching = True
                    self.precision_pinch_started = now
                    self.precision_dragging = False
                    self.precision_frozen_pos = self.last_cursor_pos

                if self.precision_pinching:
                    held = now - self.precision_pinch_started
                    if not self.precision_dragging:
                        # Keep locking the OS cursor
                        if self.pre_pinch_frozen_os_pos:
                            self.actions.pointer_absolute_pixel(*self.pre_pinch_frozen_os_pos)
                        
                        # Hold > 400ms starts dragging
                        if held > 0.40:
                            self.actions.mouse_down()
                            self.precision_dragging = True
                            self.cursor.last_relative = None
                    else:
                        # Dragging: move relatively
                        drag_pos = self._pointer_pos(GestureEvent(Gesture.INDEX_POINTER, 0.95, index), dt)
                        drag_delta = self._pointer_delta(GestureEvent(Gesture.INDEX_POINTER, 0.95, index), dt)
                        self.actions.pointer_relative(drag_delta)
                        self.last_cursor_pos = drag_pos
                return True

            # Not fully pinching
            self.precision_pinch_frames = 0

            if self.precision_pinching:
                self.precision_pinch_grace += 1
                # Absorb brief dropout frames
                if self.precision_pinch_grace <= 3:
                    if self.pre_pinch_frozen_os_pos and not self.precision_dragging:
                        self.actions.pointer_absolute_pixel(*self.pre_pinch_frozen_os_pos)
                    return True

                self.precision_release_frames += 1
                if self.precision_release_frames >= 1:
                    held = now - self.precision_pinch_started
                    if self.precision_dragging:
                        self.actions.mouse_up()
                        self._emit("Drag end")
                    else:
                        # Left click at frozen target
                        if 0.030 <= held <= 2.0 and self.cooldown.allow("left_click"):
                            if self.pre_pinch_frozen_os_pos:
                                self.actions.pointer_absolute_pixel(*self.pre_pinch_frozen_os_pos)
                            self.actions.left_click()
                            self._emit("Left click")
                    
                    self.precision_pinching = False
                    self.precision_dragging = False
                    self.precision_pinch_grace = 0
                    self.pre_pinching = False
                    self.pre_pinch_frozen_os_pos = None
                    self.cursor.reset()
                return True

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
        bracket_color = (255, 200, 80) if active else (130, 110, 95)  # Neon Cyan-Blue in BGR
        
        # 1. Sleek tactical corner brackets on the camera viewport edges
        bracket_len = 24
        bracket_thick = 2
        # Top Left
        cv2.line(frame, (8, 8), (8 + bracket_len, 8), bracket_color, bracket_thick)
        cv2.line(frame, (8, 8), (8, 8 + bracket_len), bracket_color, bracket_thick)
        # Top Right
        cv2.line(frame, (w - 8, 8), (w - 8 - bracket_len, 8), bracket_color, bracket_thick)
        cv2.line(frame, (w - 8, 8), (w - 8, 8 + bracket_len), bracket_color, bracket_thick)
        # Bottom Left
        cv2.line(frame, (8, h - 8), (8 + bracket_len, h - 8), bracket_color, bracket_thick)
        cv2.line(frame, (8, h - 8), (8, h - 8 - bracket_len), bracket_color, bracket_thick)
        # Bottom Right
        cv2.line(frame, (w - 8, h - 8), (w - 8 - bracket_len, h - 8), bracket_color, bracket_thick)
        cv2.line(frame, (w - 8, h - 8), (w - 8, h - 8 - bracket_len), bracket_color, bracket_thick)

        # 2. Glowing target locking crosshair at the fingertip
        if hand_seen and self.last_hands:
            hand = self.last_hands[0]
            lm = hand.landmarks
            px = int(lm[8][0] * w)
            py = int(lm[8][1] * h)
            
            # Glow circle
            cv2.circle(frame, (px, py), 12, (255, 230, 100), 1, cv2.LINE_AA)
            # Target center point
            cv2.circle(frame, (px, py), 2, (0, 165, 255), -1, cv2.LINE_AA)  # Orange reticle point
            
            # Crosshair segments
            cv2.line(frame, (px - 18, py), (px - 8, py), (255, 230, 100), 1)
            cv2.line(frame, (px + 8, py), (px + 18, py), (255, 230, 100), 1)
            cv2.line(frame, (px, py - 18), (px, py - 8), (255, 230, 100), 1)
            cv2.line(frame, (px, py + 8), (px, py + 18), (255, 230, 100), 1)
            
            # Glowing label
            cv2.putText(frame, "TARGET LOCK", (px + 14, py - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 165, 255), 1, cv2.LINE_AA)

        # 3. Sophisticated glassmorphism status telemetry block
        hud_x, hud_y, hud_w, hud_h = 16, 16, 440, 110
        overlay = frame.copy()
        cv2.rectangle(overlay, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (12, 16, 28), -1)
        cv2.addWeighted(overlay, 0.72, frame, 0.28, 0, frame)
        
        # Left neon vertical accent bar
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + 6, hud_y + hud_h), bracket_color, -1)
        # Grid border
        cv2.rectangle(frame, (hud_x, hud_y), (hud_x + hud_w, hud_y + hud_h), (45, 60, 95), 1, cv2.LINE_AA)
        
        # Telemetry Content
        dot_color = (80, 245, 120) if active else (100, 110, 130)
        cv2.circle(frame, (hud_x + 24, hud_y + 24), 5, dot_color, -1, cv2.LINE_AA)
        cv2.putText(frame, "GESTURE OS // TELEMETRY v2.5", (hud_x + 36, hud_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 240), 1, cv2.LINE_AA)
        
        # State title
        status_str = "ACTIVE CONTROL" if active else "SYSTEM STANDBY"
        status_color = (255, 230, 100) if active else (150, 160, 175)
        cv2.putText(frame, status_str, (hud_x + 20, hud_y + 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2, cv2.LINE_AA)
        
        # Detailed stats row
        hand_status = "LOCKED" if hand_seen else "SEARCHING..."
        hand_color = (80, 245, 120) if hand_seen else (80, 150, 255)
        cv2.putText(frame, f"TRACKING: {hand_status}", (hud_x + 20, hud_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, hand_color, 1, cv2.LINE_AA)
        cv2.putText(frame, f"GESTURE: {event.gesture.value.upper()}", (hud_x + 190, hud_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (240, 240, 245), 1, cv2.LINE_AA)
        cv2.putText(frame, f"CONF: {event.confidence:.2f}", (hud_x + 335, hud_y + 82), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (180, 200, 240), 1, cv2.LINE_AA)
        
        # Sleek upper-right FPS module
        cv2.rectangle(frame, (hud_x + hud_w - 75, hud_y + 12), (hud_x + hud_w - 12, hud_y + 36), (20, 28, 48), -1)
        cv2.rectangle(frame, (hud_x + hud_w - 75, hud_y + 12), (hud_x + hud_w - 12, hud_y + 36), (45, 60, 95), 1, cv2.LINE_AA)
        cv2.putText(frame, f"{fps:.1f} FPS", (hud_x + hud_w - 68, hud_y + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 220, 255), 1, cv2.LINE_AA)

        # 4. Premium Holographic Volume Zone scale (Left Edge)
        vol_w = int(w * 0.18)
        overlay_vol = frame.copy()
        cv2.rectangle(overlay_vol, (0, 0), (vol_w, h), (10, 15, 30), -1)
        cv2.addWeighted(overlay_vol, 0.35, frame, 0.65, 0, frame)
        
        # Audio divider grid
        cv2.line(frame, (vol_w, 0), (vol_w, h), (45, 60, 95), 1, cv2.LINE_AA)
        cv2.line(frame, (vol_w // 2, 40), (vol_w // 2, h - 40), (45, 60, 95), 1, cv2.LINE_AA)
        
        # graduated ruler ticks
        for tick in range(5):
            y_pos = int(40 + (h - 80) * (tick / 4.0))
            cv2.line(frame, (vol_w // 2 - 8, y_pos), (vol_w // 2 + 8, y_pos), (80, 120, 180), 1, cv2.LINE_AA)
            pct = 100 - tick * 25
            cv2.putText(frame, f"{pct}", (vol_w // 2 + 14, y_pos + 4), cv2.FONT_HERSHEY_SIMPLEX, 0.32, (120, 160, 200), 1, cv2.LINE_AA)
            
        # Draw dynamic slider indicator if index tip enters the volume zone
        if hand_seen and self.last_hands:
            lm = self.last_hands[0].landmarks
            idx_x = lm[8][0]
            if idx_x < 0.18:
                idx_y = lm[8][1]
                vol_y_px = int(idx_y * h)
                cv2.circle(frame, (vol_w // 2, vol_y_px), 7, (0, 165, 255), -1, cv2.LINE_AA)
                cv2.circle(frame, (vol_w // 2, vol_y_px), 11, (0, 165, 255), 1, cv2.LINE_AA)
                cv2.putText(frame, f"VOL: {int((1.0 - idx_y) * 100)}%", (vol_w // 2 - 32, vol_y_px - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 165, 255), 1, cv2.LINE_AA)
                
        cv2.putText(frame, "AUDIO ZONE", (vol_w // 2 - 32, h - 15), cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 180, 255), 1, cv2.LINE_AA)
        return frame
