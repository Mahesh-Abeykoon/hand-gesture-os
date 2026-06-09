"""
Gesture Drawing
===============
- Camera fills the ENTIRE window. You see your face + hand.
- Raise ONLY your index finger → draws on the camera like a real pen on paper.
- All other poses → pen lifts (no accidental marks).
- Buttons (Pen / Eraser / Clear) float at the top of the camera view.
- cv2.line every frame: lines are NEVER broken or dotted.
- EMA smoothing: strokes are smooth like a real pencil.

Run:
    pip install opencv-python mediapipe numpy PyQt6
    python gesture_drawing_pro.py

Keyboard: P = Pen · E = Eraser · C = Clear · Esc = Quit
"""

from __future__ import annotations

import math
import os
import sys
import time
from pathlib import Path
from typing import Optional, Tuple
from urllib.request import urlopen

import cv2
import numpy as np
from PyQt6.QtCore import QObject, Qt, QThread, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QCursor, QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QLabel, QMainWindow, QSizePolicy, QVBoxLayout, QWidget

# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe model download
# ──────────────────────────────────────────────────────────────────────────────

MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)
MODEL_PATH = Path(__file__).with_name("hand_landmarker.task")


def ensure_model() -> str:
    if MODEL_PATH.is_file():
        return str(MODEL_PATH)
    print(f"Downloading model → {MODEL_PATH} …")
    with urlopen(MODEL_URL, timeout=60) as r:
        MODEL_PATH.write_bytes(r.read())
    print("Done.")
    return str(MODEL_PATH)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


def bgr_to_qimage(bgr: np.ndarray) -> QImage:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w, ch = rgb.shape
    return QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()


# ──────────────────────────────────────────────────────────────────────────────
# EMA smoother — keeps strokes smooth without lag
# ──────────────────────────────────────────────────────────────────────────────

class EMA2D:
    """
    Exponential Moving Average for 2-D normalized coordinates.
    alpha=0.50  →  responsive & smooth (good for drawing).
    max_jump    →  reject one-frame teleport glitches from MediaPipe.
    """
    def __init__(self, alpha: float = 0.50, max_jump: float = 0.20):
        self.alpha    = alpha
        self.max_jump = max_jump
        self.val: Optional[Tuple[float, float]] = None

    def reset(self) -> None:
        self.val = None

    def update(self, x: float, y: float) -> Tuple[float, float]:
        x = clamp(x, 0.0, 1.0)
        y = clamp(y, 0.0, 1.0)
        if self.val is None:
            self.val = (x, y)
            return self.val
        lx, ly = self.val
        if math.hypot(x - lx, y - ly) > self.max_jump:
            return self.val          # ignore teleport, keep last good point
        sx = lx + self.alpha * (x - lx)
        sy = ly + self.alpha * (y - ly)
        self.val = (sx, sy)
        return self.val


# ──────────────────────────────────────────────────────────────────────────────
# Ink canvas — lives in memory, blended onto every camera frame
# ──────────────────────────────────────────────────────────────────────────────

class InkCanvas:
    INK_COLOR    = (20,  24,  32)    # near-black
    ERASER_COLOR = (255, 255, 255)   # white
    PEN_THICK    = 6
    ERASER_THICK = 40

    def __init__(self, w: int, h: int):
        self.w, self.h = w, h
        self.buf  = np.zeros((h, w, 4), dtype=np.uint8)  # BGRA, A=0 means transparent
        self._prev: Optional[Tuple[int, int]] = None
        self.erasing = False

    def clear(self) -> None:
        self.buf[:] = 0
        self._prev  = None

    def lift(self) -> None:
        """Lift the pen — next draw starts a fresh stroke."""
        self._prev = None

    def draw(self, nx: float, ny: float) -> None:
        """
        Draw from previous position to (nx, ny).
        nx, ny are normalized [0..1] relative to canvas size.
        Uses cv2.line every frame → NEVER broken or dotted.
        """
        px = int(clamp(nx, 0.0, 1.0) * (self.w - 1))
        py = int(clamp(ny, 0.0, 1.0) * (self.h - 1))
        cur = (px, py)

        color_bgr = self.ERASER_COLOR if self.erasing else self.INK_COLOR
        thick     = self.ERASER_THICK  if self.erasing else self.PEN_THICK

        if self._prev is None:
            # First point of stroke — draw a filled circle so single taps are visible
            cv2.circle(self.buf[:, :, :3], cur, thick // 2, color_bgr, -1, cv2.LINE_AA)
            if not self.erasing:
                cv2.circle(self.buf[:, :, 3], cur, thick // 2, 255, -1, cv2.LINE_AA)
            else:
                cv2.circle(self.buf[:, :, 3], cur, thick // 2, 0, -1, cv2.LINE_AA)
        else:
            # Connect prev → cur: THIS is what makes lines continuous
            cv2.line(self.buf[:, :, :3], self._prev, cur, color_bgr, thick, cv2.LINE_AA)
            alpha_val = 0 if self.erasing else 255
            cv2.line(self.buf[:, :, 3],  self._prev, cur, alpha_val,  thick, cv2.LINE_AA)

        self._prev = cur

    def blend_onto(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        Composite ink over the camera frame.
        Alpha channel of buf controls opacity: 255=opaque ink, 0=camera shows through.
        """
        fh, fw = frame_bgr.shape[:2]
        ink = cv2.resize(self.buf, (fw, fh), interpolation=cv2.INTER_LINEAR)

        b, g, r, a = cv2.split(ink)
        a_f   = a.astype(np.float32) / 255.0
        a_inv = 1.0 - a_f

        out = frame_bgr.copy().astype(np.float32)
        ink_bgr = cv2.merge([b, g, r]).astype(np.float32)

        for c in range(3):
            out[:, :, c] = out[:, :, c] * a_inv + ink_bgr[:, :, c] * a_f

        return out.astype(np.uint8)


# ──────────────────────────────────────────────────────────────────────────────
# Overlay button (rendered by OpenCV directly on the frame)
# ──────────────────────────────────────────────────────────────────────────────

class OvButton:
    """A button drawn on the frame by OpenCV. Coordinates are in frame pixels."""

    def __init__(self, label: str, action: str, x: int, y: int, w: int, h: int):
        self.label  = label
        self.action = action
        self.x, self.y, self.w, self.h = x, y, w, h

    def hit(self, fx: int, fy: int) -> bool:
        return self.x <= fx <= self.x + self.w and self.y <= fy <= self.y + self.h

    def draw(self, frame: np.ndarray, active: bool) -> None:
        x1, y1 = self.x, self.y
        x2, y2 = self.x + self.w, self.y + self.h

        bg_color = (25, 30, 45) if active else (15, 20, 30)
        cv2.rectangle(frame, (x1, y1), (x2, y2), bg_color, -1)

        border = (80, 200, 100) if active else (50, 65, 90)
        cv2.rectangle(frame, (x1, y1), (x2, y2), border, 2, cv2.LINE_AA)

        txt_color = (80, 220, 110) if active else (190, 200, 220)
        cv2.putText(frame, self.label,
                    (x1 + 12, y1 + int(self.h * 0.65)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.58, txt_color, 1, cv2.LINE_AA)


def make_buttons(frame_w: int, frame_h: int) -> list[OvButton]:
    """Create 3 buttons in the top-left corner of the frame."""
    bw, bh, gap, top = 120, 44, 10, 12
    defs = [("✏ Pen", "PEN"), ("◻ Eraser", "ERASER"), ("✕ Clear", "CLEAR")]
    btns = []
    for i, (lbl, act) in enumerate(defs):
        btns.append(OvButton(lbl, act, 14 + i * (bw + gap), top, bw, bh))
    return btns


# ──────────────────────────────────────────────────────────────────────────────
# MediaPipe hand tracker
# ──────────────────────────────────────────────────────────────────────────────

class HandTracker:
    def __init__(self):
        try:
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
            from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
            from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
        except ImportError as e:
            raise RuntimeError("pip install mediapipe") from e

        self._Image  = Image
        self._IFmt   = ImageFormat
        self._t0     = time.perf_counter()
        self._last_ts = 0

        opts = HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=ensure_model()),
            running_mode=VisionTaskRunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.55,
            min_hand_presence_confidence=0.55,
            min_tracking_confidence=0.55,
        )
        self._det = HandLandmarker.create_from_options(opts)

    def _ts(self) -> int:
        ts = int((time.perf_counter() - self._t0) * 1000)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts
        return ts

    def process(self, bgr: np.ndarray):
        """
        Returns (tip_x, tip_y, index_only) normalized coords + gesture flag.
        Returns (None, None, False) when no hand detected.

        index_only = True when ONLY index finger is raised (middle/ring folded).
        This is the most natural "pointing/writing" gesture.
        """
        rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        res = self._det.detect_for_video(self._Image(self._IFmt.SRGB, rgb), self._ts())

        if not res.hand_landmarks:
            return None, None, False

        h = res.hand_landmarks[0]
        lm = [(float(p.x), float(p.y)) for p in h]

        # Tip above PIP = finger is extended
        idx_up    = lm[8][1]  < lm[6][1]   # index
        mid_up    = lm[12][1] < lm[10][1]  # middle
        ring_up   = lm[16][1] < lm[14][1]  # ring
        # pinky allowed to be free — it's often semi-extended while writing

        index_only = idx_up and not mid_up and not ring_up

        return lm[8][0], lm[8][1], index_only


# ──────────────────────────────────────────────────────────────────────────────
# Vision worker — runs in its own thread
# ──────────────────────────────────────────────────────────────────────────────

class Worker(QObject):
    frameReady = pyqtSignal(QImage)
    errorReady = pyqtSignal(str)

    sigTool  = pyqtSignal(str)
    sigClear = pyqtSignal()

    CAM_W = 1280
    CAM_H = 720

    def __init__(self, cam_idx: int = 0):
        super().__init__()
        self.cam_idx  = cam_idx
        self.running  = False
        self.tool     = "PEN"   # "PEN" | "ERASER"
        self.ink      = InkCanvas(self.CAM_W, self.CAM_H)
        self.ema      = EMA2D(alpha=0.50, max_jump=0.20)
        self.buttons: list[OvButton] = []   # built once we know frame size
        self._fps     = 0.0
        self._lt      = time.perf_counter()
        self._was_drawing = False

        self.sigTool.connect(self._set_tool)
        self.sigClear.connect(self._clear)

    @pyqtSlot(str)
    def _set_tool(self, t: str) -> None:
        self.tool = t
        self.ink.erasing = (t == "ERASER")
        self.ink.lift()

    @pyqtSlot()
    def _clear(self) -> None:
        self.ink.clear()

    def _hud(self, frame: np.ndarray, tip_px: Optional[Tuple[int, int]], drawing: bool) -> None:
        """Minimal professional HUD."""
        fh, fw = frame.shape[:2]

        for btn in self.buttons:
            btn.draw(frame, btn.action == self.tool)

        if tip_px is not None:
            col = (60, 230, 100) if drawing else (180, 180, 180)
            cv2.circle(frame, tip_px, 8, col, -1, cv2.LINE_AA)
            cv2.circle(frame, tip_px, 8, (255, 255, 255), 1, cv2.LINE_AA)

        label = f"{'DRAWING' if drawing else 'HOVER'}  {self._fps:.0f} FPS"
        cv2.rectangle(frame, (0, fh - 32), (fw, fh), (8, 10, 18), -1)
        cv2.putText(frame, label, (12, fh - 9),
                    cv2.FONT_HERSHEY_DUPLEX, 0.45, (180, 200, 230), 1, cv2.LINE_AA)

    @pyqtSlot()
    def run(self) -> None:
        try:
            tracker = HandTracker()
            backend = cv2.CAP_DSHOW if os.name == "nt" else 0
            cap = cv2.VideoCapture(self.cam_idx, backend)
            if not cap.isOpened():
                raise RuntimeError(f"Cannot open camera {self.cam_idx}")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH,  self.CAM_W)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAM_H)
            cap.set(cv2.CAP_PROP_FPS, 60)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

            # Build buttons now that we know the frame size
            actual_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.buttons = make_buttons(actual_w, actual_h)
            self.ink = InkCanvas(actual_w, actual_h)
            self.ink.erasing = (self.tool == "ERASER")

            self.running = True
            while self.running:
                ok, frame = cap.read()
                if not ok or frame is None:
                    QThread.msleep(8)
                    continue

                frame = cv2.flip(frame, 1)   # mirror so it feels natural
                fh, fw = frame.shape[:2]

                # FPS
                now = time.perf_counter()
                self._fps = 0.85 * self._fps + 0.15 / max(1e-6, now - self._lt)
                self._lt  = now

                # Hand detection
                nx, ny, index_only = tracker.process(frame)

                tip_px: Optional[Tuple[int, int]] = None
                drawing = False

                if nx is not None:
                    sx, sy = self.ema.update(nx, ny)
                    tip_px = (int(sx * (fw - 1)), int(sy * (fh - 1)))

                    if index_only:
                        # Check if tip is in button area (top 70px of frame)
                        if tip_px[1] < 70:
                            # Hover over button zone — check hit & lift pen
                            for btn in self.buttons:
                                if btn.hit(*tip_px):
                                    if btn.action == "CLEAR":
                                        self._clear()
                                    else:
                                        self._set_tool(btn.action)
                                    break
                            self.ink.lift()
                        else:
                            # ── AUTO DRAW ──
                            # Lift on the very first frame of a new stroke to avoid
                            # connecting to wherever the finger was last
                            if not self._was_drawing:
                                self.ink.lift()

                            # Draw: normalized coords so canvas stays in sync
                            self.ink.draw(sx, sy)
                            drawing = True
                    else:
                        self.ink.lift()
                else:
                    self.ema.reset()
                    self.ink.lift()

                self._was_drawing = drawing

                # Composite ink over camera frame
                out = self.ink.blend_onto(frame)

                # HUD on top
                self._hud(out, tip_px, drawing)

                self.frameReady.emit(bgr_to_qimage(out))
                QThread.msleep(1)

            cap.release()
        except Exception as exc:
            self.errorReady.emit(str(exc))

    @pyqtSlot()
    def stop(self) -> None:
        self.running = False


# ──────────────────────────────────────────────────────────────────────────────
# Main window — camera fills the ENTIRE window, nothing else
# ──────────────────────────────────────────────────────────────────────────────

class App(QMainWindow):
    sigTool  = pyqtSignal(str)
    sigClear = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Gesture Drawing")
        self.setStyleSheet("QMainWindow, QWidget { background: #000; }")
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.BlankCursor))

        # Full-window camera label
        container = QWidget()
        self.setCentralWidget(container)
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = QLabel()
        self.view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.view.setStyleSheet("background: #000;")
        layout.addWidget(self.view)

        # Worker
        self.worker = Worker(cam_idx=0)
        self.thread = QThread(self)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)

        self.sigTool.connect(self.worker.sigTool)
        self.sigClear.connect(self.worker.sigClear)
        self.worker.frameReady.connect(self._on_frame)
        self.worker.errorReady.connect(self._on_error)

        self.thread.start()
        self.showFullScreen()

    @pyqtSlot(QImage)
    def _on_frame(self, img: QImage) -> None:
        pix = QPixmap.fromImage(img).scaled(
            self.view.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.view.setPixmap(pix)

    @pyqtSlot(str)
    def _on_error(self, msg: str) -> None:
        self.view.setText(f"<span style='color:red;font-size:18px'>{msg}</span>")

    def keyPressEvent(self, e) -> None:
        k = e.key()
        if   k == Qt.Key.Key_Escape: self.close()
        elif k == Qt.Key.Key_P:      self.sigTool.emit("PEN")
        elif k == Qt.Key.Key_E:      self.sigTool.emit("ERASER")
        elif k == Qt.Key.Key_C:      self.sigClear.emit()

    def closeEvent(self, e) -> None:
        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass
        self.worker.stop()
        self.thread.quit()
        self.thread.wait(3000)
        super().closeEvent(e)


# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    app = QApplication(sys.argv)
    app.setApplicationName("Gesture Drawing")
    win = App()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
