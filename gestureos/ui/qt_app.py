from __future__ import annotations

import math
import sys
import time
from collections import deque
from typing import Deque, Dict, Optional

import cv2
import numpy as np

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QCursor, QFont, QImage, QPainter, QPainterPath, QPen, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QStackedWidget,
    QFileDialog,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gestureos.app_core import GestureEngine
from gestureos.gestures.types import Gesture
from gestureos.settings.config import AppConfig
from gestureos.utils.logging import get_logger
from gestureos.ui.ocr_engine import AdvancedOCREngine


STYLE = """
QMainWindow, QWidget { background: #0a1128; color: #f1f5f9; font-family: Segoe UI, Inter, Arial; }
QFrame#Card, QGroupBox { background: #131b33; border: 1px solid #1e2947; border-radius: 16px; }
QGroupBox { margin-top: 18px; padding: 12px; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 8px; color: #93c5fd; }
QLabel#Title { font-size: 30px; font-weight: 900; color: white; }
QLabel#Subtitle { color: #94a3b8; font-size: 13px; }
QLabel#MetricValue { font-size: 24px; font-weight: 800; color: white; }
QLabel#MetricName { color: #94a3b8; font-size: 12px; }
QPushButton { background: #1e2947; border: 1px solid #2d3c63; border-radius: 10px; padding: 9px 12px; font-weight: 700; }
QPushButton:hover { background: #2a3857; }
QPushButton#Primary { background: #3b82f6; border-color: #2563eb; color: white; }
QPushButton#Danger { background: #b91c1c; border-color: #991b1b; color: white; }
QPushButton#Success { background: #15803d; border-color: #166534; color: white; }
QTabWidget::pane { border: 1px solid #1e2947; border-radius: 14px; top: -1px; }
QTabBar::tab { background: #131b33; color: #cbd5e1; padding: 10px 16px; border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 2px; }
QTabBar::tab:selected { background: #1e2947; color: white; }
QSlider::groove:horizontal { height: 7px; background: #2d3c63; border-radius: 4px; }
QSlider::handle:horizontal { background: #60a5fa; width: 18px; margin: -6px 0; border-radius: 9px; }
QCheckBox { spacing: 8px; color: #f1f5f9; }
QTableWidget, QListWidget, QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #0d152a; border: 1px solid #2d3c63; border-radius: 8px; color: #f1f5f9; padding: 6px; }
QHeaderView::section { background: #1e2947; color: #cbd5e1; padding: 7px; border: none; }
"""


class MetricCard(QFrame):
    def __init__(self, name: str, value: str = "--"):
        super().__init__()
        self.setObjectName("Card")
        layout = QVBoxLayout(self)
        self.value = QLabel(value)
        self.value.setObjectName("MetricValue")
        self.name = QLabel(name)
        self.name.setObjectName("MetricName")
        layout.addWidget(self.value)
        layout.addWidget(self.name)

    def set_value(self, value: str) -> None:
        self.value.setText(value)


class Sparkline(QWidget):
    def __init__(self, title: str, maxlen: int = 120):
        super().__init__()
        self.title = title
        self.values: Deque[float] = deque(maxlen=maxlen)
        self.setMinimumHeight(120)

    def add(self, value: float) -> None:
        self.values.append(float(value))
        self.update()

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(10, 10, -10, -10)
        painter.fillRect(self.rect(), QColor("#0e1420"))
        painter.setPen(QPen(QColor("#29364a"), 1))
        painter.drawRoundedRect(rect, 10, 10)
        painter.setPen(QColor("#9fb0c8"))
        painter.drawText(rect.adjusted(10, 4, -10, -4), Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft, self.title)
        if len(self.values) < 2:
            return
        vals = list(self.values)
        lo, hi = min(vals), max(vals)
        if abs(hi - lo) < 1e-6:
            hi = lo + 1.0
        graph = rect.adjusted(14, 28, -14, -12)
        painter.setPen(QPen(QColor("#60a5fa"), 2))
        prev = None
        for i, v in enumerate(vals):
            x = graph.left() + i * graph.width() / max(1, len(vals) - 1)
            y = graph.bottom() - (v - lo) / (hi - lo) * graph.height()
            if prev is not None:
                painter.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
            prev = (x, y)


class WhiteboardRecognizer:
    """Compatibility wrapper around the advanced multi-backend OCR engine."""

    def __init__(self):
        self.engine = AdvancedOCREngine()

    def recognize(self, ink_binary: np.ndarray):
        return self.engine.best(ink_binary)


class WhiteboardCanvas(QWidget):
    """
    Natural drawing canvas.

    Gesture input is noisy and arrives as small deltas. This widget turns that
    into human-like ink by applying:
    - low-pass motion filtering
    - per-frame speed limiting
    - stroke start separation, so pen-down does not draw a jump line
    - quadratic Bézier stroke interpolation
    - hidden cursor by default so the drawing itself is not disturbed
    """

    def __init__(self, show_cursor: bool = False):
        super().__init__()
        self.setMinimumHeight(560)
        self.canvas = QImage(1920, 1080, QImage.Format.Format_RGB32)
        self.canvas.fill(QColor("white"))
        self.pen_color = QColor("#111827")
        self.pen_width = 7
        self.tool = "pen"  # pen | eraser | hover
        self.cursor_x = self.canvas.width() / 2
        self.cursor_y = self.canvas.height() / 2
        self.smooth_x = self.cursor_x
        self.smooth_y = self.cursor_y
        self.show_cursor = show_cursor
        self.last_draw = False
        self.mouse_down = False
        self.stroke_points: list[tuple[float, float]] = []
        self.recognizer = WhiteboardRecognizer()
        self.motion_gain = 0.72
        self.max_step_px = 22.0
        self.motion_alpha = 0.30

    def clear(self):
        self.canvas.fill(QColor("white"))
        self.stroke_points.clear()
        self.last_draw = False
        self.update()

    def set_pen_width(self, width: int):
        self.pen_width = int(width)

    def set_tool(self, tool: str):
        self.tool = tool if tool in {"pen", "eraser", "hover"} else "pen"
        if self.tool == "hover":
            self.stroke_points.clear()
            self.last_draw = False
        self.update()

    def set_cursor_visible(self, visible: bool):
        self.show_cursor = bool(visible)
        self.update()

    def _limited_target(self, dx: float, dy: float) -> tuple[float, float]:
        # Convert normalized touchpad delta to canvas pixels. The multiplier is
        # intentionally conservative; writing should feel like paper, not a laser.
        tx = self.cursor_x + dx * self.canvas.width() * self.motion_gain
        ty = self.cursor_y + dy * self.canvas.height() * self.motion_gain
        step_x, step_y = tx - self.cursor_x, ty - self.cursor_y
        mag = (step_x * step_x + step_y * step_y) ** 0.5
        if mag > self.max_step_px:
            scale = self.max_step_px / max(mag, 1e-6)
            tx = self.cursor_x + step_x * scale
            ty = self.cursor_y + step_y * scale
        tx = max(0, min(self.canvas.width() - 1, tx))
        ty = max(0, min(self.canvas.height() - 1, ty))
        # EMA after speed limiting removes micro zig-zags.
        self.smooth_x = self.smooth_x + self.motion_alpha * (tx - self.smooth_x)
        self.smooth_y = self.smooth_y + self.motion_alpha * (ty - self.smooth_y)
        return self.smooth_x, self.smooth_y

    def update_from_gesture(self, dx: float, dy: float, drawing: bool):
        drawing = bool(drawing and self.tool != "hover")
        nx, ny = self._limited_target(dx, dy)
        if drawing:
            if not self.last_draw:
                # Start a fresh stroke at the current location. Do NOT draw from
                # previous hover location to pen-down location.
                self.stroke_points = [(nx, ny)]
            else:
                self._draw_smooth_segment(nx, ny)
        else:
            self.stroke_points.clear()
        self.cursor_x, self.cursor_y = nx, ny
        self.last_draw = drawing
        self.update()

    def update_from_absolute(self, x_norm: float, y_norm: float, drawing: bool):
        """Draw using absolute camera-to-board mapping.

        The index fingertip position maps directly to a canvas point.
        alpha=0.55 keeps lines smooth but NEVER clamps movement so far that
        a gap appears between two consecutive drawn points.
        max_step is intentionally large (80px) — we want responsiveness, not
        clamping that causes dotted / broken strokes.
        """
        drawing = bool(drawing and self.tool != "hover")
        target_x = max(0, min(self.canvas.width() - 1, x_norm * self.canvas.width()))
        target_y = max(0, min(self.canvas.height() - 1, y_norm * self.canvas.height()))

        # Reject genuine teleports only (detector landmark swap across the frame).
        jump = ((target_x - self.cursor_x) ** 2 + (target_y - self.cursor_y) ** 2) ** 0.5
        if jump > max(self.canvas.width(), self.canvas.height()) * 0.20:
            self.stroke_points.clear()
            self.last_draw = False
            self.cursor_x = self.smooth_x = target_x
            self.cursor_y = self.smooth_y = target_y
            self.update()
            return

        # EMA smoothing — high alpha so the pen follows the finger closely.
        alpha = 0.55
        nx = self.smooth_x + alpha * (target_x - self.smooth_x)
        ny = self.smooth_y + alpha * (target_y - self.smooth_y)

        # Only clamp genuinely huge steps (80 px) to prevent one bad frame
        # leaving a long scar.  Normal fast writing should NOT be clamped.
        max_step = 80.0
        step = ((nx - self.cursor_x) ** 2 + (ny - self.cursor_y) ** 2) ** 0.5
        if step > max_step:
            scale = max_step / max(step, 1e-6)
            nx = self.cursor_x + (nx - self.cursor_x) * scale
            ny = self.cursor_y + (ny - self.cursor_y) * scale

        if drawing:
            if not self.last_draw:
                # Fresh stroke — plant the first point without connecting to the
                # previous hover position.
                self.stroke_points = [(nx, ny)]
            else:
                self._draw_smooth_segment(nx, ny)
        else:
            self.stroke_points.clear()
        self.cursor_x = self.smooth_x = nx
        self.cursor_y = self.smooth_y = ny
        self.last_draw = drawing
        self.update()

    def _draw_smooth_segment(self, x: float, y: float):
        """Connect the last known point to (x, y) — always draws a LINE, never leaves a gap."""
        if self.stroke_points:
            last_x, last_y = self.stroke_points[-1]
            # Only skip if the movement is truly sub-pixel (< 0.8 px).
            if ((x - last_x) ** 2 + (y - last_y) ** 2) ** 0.5 < 0.8:
                return
        self.stroke_points.append((x, y))
        # Keep a small window for Bézier smoothing.
        if len(self.stroke_points) > 6:
            self.stroke_points = self.stroke_points[-6:]

        painter = QPainter(self.canvas)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        color = QColor("#ffffff") if self.tool == "eraser" else self.pen_color
        width = int(self.pen_width * 3.5) if self.tool == "eraser" else self.pen_width
        painter.setPen(QPen(color, width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))

        pts = self.stroke_points
        if len(pts) == 1:
            # Single tap — draw a dot so it's visible.
            painter.drawEllipse(int(x) - width // 2, int(y) - width // 2, width, width)
        elif len(pts) == 2:
            # Two points — draw a straight line; guaranteed no gap.
            painter.drawLine(int(pts[-2][0]), int(pts[-2][1]), int(pts[-1][0]), int(pts[-1][1]))
        else:
            # Three+ points — quadratic Bézier through midpoints for smooth curves.
            p0 = pts[-3]
            p1 = pts[-2]
            p2 = pts[-1]
            m1 = ((p0[0] + p1[0]) / 2.0, (p0[1] + p1[1]) / 2.0)
            m2 = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0)
            path = QPainterPath()
            path.moveTo(m1[0], m1[1])
            path.quadTo(p1[0], p1[1], m2[0], m2[1])
            painter.drawPath(path)
        painter.end()

    def _ink_binary(self) -> np.ndarray:
        img = self.canvas.convertToFormat(QImage.Format.Format_RGB32)
        ptr = img.bits()
        ptr.setsize(img.bytesPerLine() * img.height())
        arr = np.frombuffer(ptr, np.uint8).reshape((img.height(), img.bytesPerLine() // 4, 4))[:, :img.width(), :3]
        gray = cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
        return (gray < 245).astype(np.uint8) * 255

    def recognize(self) -> tuple[str, str]:
        ink = self._ink_binary()
        if cv2.countNonZero(ink) < 40:
            return "Nothing drawn", ""
        best, results = self.recognizer.recognize(ink)
        primary = f"Best: {best.text!r}  via {best.backend}  confidence={best.confidence:.2f}"
        lines = []
        for r in results:
            if r.text:
                lines.append(f"{r.backend}: {r.text!r}  conf={r.confidence:.2f}  {r.details}")
        return primary, "\n".join(lines[:8])

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        # Stretch canvas to fill widget with zero padding.
        pix = QPixmap.fromImage(self.canvas).scaled(
            self.rect().size(),
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        painter.drawPixmap(0, 0, pix)
        if self.show_cursor:
            sx = pix.width() / self.canvas.width()
            sy = pix.height() / self.canvas.height()
            painter.setPen(QPen(QColor("#2563eb"), 2))
            painter.drawEllipse(int(self.cursor_x * sx - 7), int(self.cursor_y * sy - 7), 14, 14)

    def mousePressEvent(self, event):  # noqa: N802
        self.mouse_down = True
        self._mouse_draw(event.position().x(), event.position().y(), False)

    def mouseMoveEvent(self, event):  # noqa: N802
        self._mouse_draw(event.position().x(), event.position().y(), self.mouse_down)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self.mouse_down = False
        self.stroke_points.clear()

    def _mouse_draw(self, x, y, drawing):
        target = self.rect().adjusted(0, 0, 0, 0)
        scale = min(target.width() / self.canvas.width(), target.height() / self.canvas.height())
        ox = target.x() + (target.width() - self.canvas.width() * scale) / 2
        oy = target.y() + (target.height() - self.canvas.height() * scale) / 2
        nx = max(0, min(self.canvas.width() - 1, (x - ox) / scale))
        ny = max(0, min(self.canvas.height() - 1, (y - oy) / scale))
        if drawing:
            if not self.last_draw:
                self.stroke_points = [(nx, ny)]
            else:
                self._draw_smooth_segment(nx, ny)
        else:
            self.stroke_points.clear()
        self.cursor_x = self.smooth_x = nx
        self.cursor_y = self.smooth_y = ny
        self.last_draw = drawing
        self.update()


class FullScreenDrawWindow(QMainWindow):
    """
    Full-screen drawing window.
    Camera fills 100% of the window. Ink is a numpy array blended onto each
    camera frame with cv2.line — guaranteed continuous, never broken.
    """
    closed = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GestureOS Full Screen Draw")
        self.setStyleSheet("QMainWindow,QWidget{background:#000;}")

        central = QWidget()
        self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(0, 0, 0, 0)
        root.setSpacing(0)

        # Camera fills 100% of the window
        self.camera_view = QLabel()
        self.camera_view.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_view.setStyleSheet("background:#000;")
        self.camera_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        root.addWidget(self.camera_view, 1)

        # Toolbar — solid dark background so buttons are always visible
        toolbar = QWidget()
        toolbar.setFixedHeight(54)
        toolbar.setAutoFillBackground(True)
        p = toolbar.palette(); p.setColor(toolbar.backgroundRole(), QColor("#0a0e1a")); toolbar.setPalette(p)
        tb = QHBoxLayout(toolbar)
        tb.setContentsMargins(14, 6, 14, 6)
        tb.setSpacing(8)

        BTN_STYLE = (
            "QPushButton{background:#1a2540;color:#c8d8f0;border:1px solid #2b3e56;"
            "border-radius:8px;padding:0 14px;font-weight:700;font-size:13px;min-height:36px;}"
            "QPushButton:hover{background:#22304e;}"
        )

        def _btn(label):
            b = QPushButton(label)
            b.setStyleSheet(BTN_STYLE)
            return b

        self._btn_pen    = _btn("Pen")
        self._btn_eraser = _btn("Eraser")
        self._btn_hover  = _btn("Hover")
        self._btn_clear  = _btn("Clear")
        self._btn_close  = _btn("Close")
        self._status     = QLabel("Index finger up = draw")
        self._status.setStyleSheet("color:#5a7a9e;font-size:12px;")

        self._btn_pen.clicked.connect(lambda: self._set_tool("pen"))
        self._btn_eraser.clicked.connect(lambda: self._set_tool("eraser"))
        self._btn_hover.clicked.connect(lambda: self._set_tool("hover"))
        self._btn_clear.clicked.connect(self._clear)
        self._btn_close.clicked.connect(self.close)

        for w in [self._btn_pen, self._btn_eraser, self._btn_hover, self._btn_clear]:
            tb.addWidget(w)
        tb.addStretch(1)
        tb.addWidget(self._status)
        tb.addStretch(1)
        tb.addWidget(self._btn_close)
        root.addWidget(toolbar)

        # ── Ink state (pure numpy — cv2.line draws directly, never breaks) ──
        self._ink: Optional[np.ndarray] = None   # white BGR canvas, camera size
        self._ink_w = 0
        self._ink_h = 0
        self._tool = "pen"
        self._pen_w = 7
        self._eraser_w = 36
        self._prev_pt: Optional[tuple] = None
        self._prev_drawing = False
        self._sx = 0.5   # smoothed normalized x
        self._sy = 0.5   # smoothed normalized y
        self._sinit = False

    def _set_tool(self, t: str):
        self._tool = t
        self._prev_pt = None   # lift pen on tool change

    def _clear(self):
        if self._ink is not None:
            self._ink[:] = 255
        self._prev_pt = None

    # ------------------------------------------------------------------
    def update_camera(self, img: QImage):
        """Blend numpy ink over camera frame and display full-window."""
        img_rgb = img.convertToFormat(QImage.Format.Format_RGB888)
        ptr = img_rgb.bits()
        ptr.setsize(img_rgb.bytesPerLine() * img_rgb.height())
        frame_rgb = np.frombuffer(ptr, np.uint8).reshape(
            (img_rgb.height(), img_rgb.width(), 3)).copy()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        fh, fw = frame_bgr.shape[:2]

        # (Re-)create ink buffer if size changed
        if self._ink is None or self._ink_w != fw or self._ink_h != fh:
            self._ink = np.full((fh, fw, 3), 255, dtype=np.uint8)
            self._ink_w, self._ink_h = fw, fh
            self._prev_pt = None

        # Blend: only paint where ink pixel is NOT white
        mask = np.any(self._ink < 245, axis=2)
        out = frame_bgr.copy()
        if mask.any():
            out[mask] = (
                frame_bgr[mask].astype(np.float32) * 0.15
                + self._ink[mask].astype(np.float32) * 0.85
            ).astype(np.uint8)

        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        h, w, ch = out_rgb.shape
        qi = QImage(out_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
        pix = QPixmap.fromImage(qi).scaled(
            self.camera_view.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.FastTransformation,
        )
        self.camera_view.setPixmap(pix)

    def update_from_absolute(self, cx: float, cy: float, drawing: bool):
        """Draw with cv2.line — always connected, never broken."""
        if self._ink is None or self._tool == "hover":
            self._prev_pt = None
            self._prev_drawing = False
            return

        # EMA smoothing
        alpha = 0.55
        if not self._sinit:
            self._sx, self._sy, self._sinit = cx, cy, True
        else:
            if ((cx - self._sx) ** 2 + (cy - self._sy) ** 2) ** 0.5 < 0.20:
                self._sx += alpha * (cx - self._sx)
                self._sy += alpha * (cy - self._sy)

        px = int(max(0, min(self._ink_w - 1, self._sx * self._ink_w)))
        py = int(max(0, min(self._ink_h - 1, self._sy * self._ink_h)))
        cur = (px, py)

        if drawing:
            color = (255, 255, 255) if self._tool == "eraser" else (20, 24, 32)
            thick = self._eraser_w if self._tool == "eraser" else self._pen_w
            if not self._prev_drawing or self._prev_pt is None:
                cv2.circle(self._ink, cur, thick // 2, color, -1, cv2.LINE_AA)
            else:
                cv2.line(self._ink, self._prev_pt, cur, color, thick, cv2.LINE_AA)
            self._prev_pt = cur
        else:
            self._prev_pt = None

        self._prev_drawing = bool(drawing)
        tool_name = self._tool.upper()
        state = "DRAWING" if drawing else "HOVER"
        self._status.setText(f"{state} | {tool_name}")

    def update_from_gesture(self, dx: float, dy: float, drawing: bool):
        pass   # not used in full-screen mode

    def keyPressEvent(self, event):  # noqa: N802
        k = event.key()
        if   k == Qt.Key.Key_Escape: self.close()
        elif k == Qt.Key.Key_P: self._set_tool("pen")
        elif k == Qt.Key.Key_E: self._set_tool("eraser")
        elif k == Qt.Key.Key_C: self._clear()

    def closeEvent(self, event):  # noqa: N802
        self.closed.emit()
        super().closeEvent(event)


class EngineWorker(QObject):
    frameReady = pyqtSignal(QImage, str, float, float, bool, bool, float, float, float, float, bool)
    logReady = pyqtSignal(str)
    errorReady = pyqtSignal(str)
    startedReady = pyqtSignal()

    def __init__(self, config: AppConfig):
        super().__init__()
        self.config = config
        self.engine: Optional[GestureEngine] = None
        self.running = False
        self.paused = False

    @pyqtSlot()
    def start(self):
        try:
            self.engine = GestureEngine(self.config, self.logReady.emit)
            self.running = True
            self.startedReady.emit()
            while self.running:
                if self.paused:
                    QThread.msleep(30)
                    continue
                frame, event, fps = self.engine.step()
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                h, w, ch = rgb.shape
                qimg = QImage(rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
                active = self.config.gesture_mode_active or not self.config.activation_required
                hand_seen = bool(getattr(self.engine, "last_hands", []))
                cx, cy = getattr(self.engine, "last_cursor_pos", (0.5, 0.5))
                dx, dy = getattr(self.engine, "last_cursor_delta", (0.0, 0.0))
                draw_active = bool(getattr(self.engine, "last_draw_active", False))
                self.frameReady.emit(qimg, event.gesture.value, float(event.confidence), float(fps), active, hand_seen, float(cx), float(cy), float(dx), float(dy), draw_active)
                QThread.msleep(1)
        except Exception as exc:
            self.errorReady.emit(str(exc))
        finally:
            if self.engine:
                self.engine.close()

    @pyqtSlot()
    def stop(self):
        self.running = False

    @pyqtSlot(bool)
    def set_paused(self, paused: bool):
        self.paused = paused

    @pyqtSlot()
    def apply_config(self):
        if self.engine:
            self.engine.update_config()

    @pyqtSlot(bool)
    def set_active(self, active: bool):
        self.config.gesture_mode_active = active

    @pyqtSlot(bool)
    def set_whiteboard_mode(self, enabled: bool):
        setattr(self.config, "whiteboard_mode", bool(enabled))
        self.logReady.emit("Whiteboard gesture mode enabled" if enabled else "Whiteboard gesture mode disabled")

    @pyqtSlot(str, str)
    def record_custom(self, name: str, action: str):
        try:
            if not self.engine or not getattr(self.engine, "last_hands", []):
                self.logReady.emit("Custom gesture failed: no hand locked")
                return
            self.engine.trainer.add_sample(name, action, self.engine.last_hands[0].landmarks)
            self.logReady.emit(f"Recorded custom gesture '{name}' -> {action}")
        except Exception as exc:
            self.errorReady.emit(str(exc))


class CalibrationDialog(QDialog):
    def __init__(self, config: AppConfig, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GestureOS Calibration Wizard")
        self.config = config
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        title = QLabel("Calibration Wizard")
        title.setObjectName("Title")
        subtitle = QLabel("Tune camera, safety, smoothing, and gesture thresholds for your room.")
        subtitle.setObjectName("Subtitle")
        layout.addWidget(title)
        layout.addWidget(subtitle)
        form = QFormLayout()
        self.width = QSpinBox(); self.width.setRange(320, 1920); self.width.setValue(config.camera_width)
        self.height = QSpinBox(); self.height.setRange(240, 1080); self.height.setValue(config.camera_height)
        self.fps = QSpinBox(); self.fps.setRange(15, 120); self.fps.setValue(config.target_fps)
        self.sensitivity = QDoubleSpinBox(); self.sensitivity.setRange(0.20, 0.95); self.sensitivity.setSingleStep(0.01); self.sensitivity.setValue(config.sensitivity)
        self.confidence = QDoubleSpinBox(); self.confidence.setRange(0.20, 0.95); self.confidence.setSingleStep(0.01); self.confidence.setValue(config.confidence_threshold)
        self.smoothing = QDoubleSpinBox(); self.smoothing.setRange(0.05, 0.95); self.smoothing.setSingleStep(0.01); self.smoothing.setValue(config.cursor_smoothing)
        form.addRow("Camera width", self.width)
        form.addRow("Camera height", self.height)
        form.addRow("Target FPS", self.fps)
        form.addRow("Gesture sensitivity", self.sensitivity)
        form.addRow("Confidence threshold", self.confidence)
        form.addRow("Cursor smoothing", self.smoothing)
        layout.addLayout(form)
        guide = QTextEdit()
        guide.setReadOnly(True)
        guide.setText(
            "Recommended physical setup:\n"
            "• Sit 45–75 cm from the camera.\n"
            "• Use front lighting, avoid backlight.\n"
            "• Keep the full hand visible; hand should occupy 20–45% of preview height.\n"
            "• Use 640x480/30 FPS if your laptop struggles.\n"
            "• Start smoothing at 0.55, then adjust while moving the pointer."
        )
        layout.addWidget(guide)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def accept(self):  # noqa: A003
        self.config.camera_width = self.width.value()
        self.config.camera_height = self.height.value()
        self.config.target_fps = self.fps.value()
        self.config.sensitivity = self.sensitivity.value()
        self.config.confidence_threshold = self.confidence.value()
        self.config.cursor_smoothing = self.smoothing.value()
        self.config.save()
        super().accept()


class CustomGestureDialog(QDialog):
    requested = pyqtSignal(str, str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Record Custom Gesture")
        layout = QVBoxLayout(self)
        layout.addWidget(QLabel("Hold the custom gesture in the camera preview, then save."))
        form = QFormLayout()
        self.name = QLineEdit()
        self.action = QLineEdit()
        self.action.setPlaceholderText("hotkey:ctrl+l or press:space")
        form.addRow("Gesture name", self.name)
        form.addRow("Action", self.action)
        layout.addLayout(form)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Save | QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self._save)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _save(self):
        name = self.name.text().strip()
        action = self.action.text().strip()
        if not name or not action:
            QMessageBox.warning(self, "Missing data", "Enter both a name and an action.")
            return
        self.requested.emit(name, action)
        self.accept()


class GestureOSQtApp(QMainWindow):
    applyConfigRequested = pyqtSignal()
    setActiveRequested = pyqtSignal(bool)
    recordCustomRequested = pyqtSignal(str, str)
    whiteboardModeRequested = pyqtSignal(bool)
    pauseRequested = pyqtSignal(bool)
    stopRequested = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.config_data = AppConfig.load()
        self.logger = get_logger("GestureOS.QtUI")
        self.setWindowTitle("GestureOS Pro — Real-Time Hand Gesture Control")
        self.resize(1440, 900)
        self.setMinimumSize(1180, 760)
        self.thread: Optional[QThread] = None
        self.worker: Optional[EngineWorker] = None
        self.last_hand_seen = False
        self.fullscreen_draw = None
        self._build_ui()
        self._start_worker()

    def _build_ui(self):
        self.setStyleSheet(STYLE)
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(18, 18, 18, 18)

        header = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("GestureOS Pro")
        title.setObjectName("Title")
        subtitle = QLabel("Low-latency webcam gesture control for productivity, media, and mouse workflows")
        subtitle.setObjectName("Subtitle")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        header.addLayout(title_box)
        header.addStretch(1)
        self.active_btn = QPushButton("Activate")
        self.active_btn.setObjectName("Success")
        self.active_btn.setCheckable(True)
        self.active_btn.setChecked(self.config_data.gesture_mode_active)
        self.active_btn.clicked.connect(self._toggle_active)
        self.pause_btn = QPushButton("Pause Camera")
        self.pause_btn.setCheckable(True)
        self.pause_btn.clicked.connect(lambda v: self.pauseRequested.emit(bool(v)))
        header.addWidget(self.pause_btn)
        header.addWidget(self.active_btn)
        outer.addLayout(header)

        main = QHBoxLayout()
        outer.addLayout(main, 1)

        left = QVBoxLayout()
        main.addLayout(left, 3)

        self.video_card = QFrame(); self.video_card.setObjectName("Card")
        video_layout = QVBoxLayout(self.video_card)
        self.video = QLabel("Starting camera...")
        self.video.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.video.setMinimumHeight(560)
        self.video.setStyleSheet("background:#0a1128; border-radius:12px; color:#94a3b8;")
        video_layout.addWidget(self.video, 1)

        # ── Draw toolbar below camera ───────────────────────────────────
        self.draw_bar_widget = QWidget()
        self.draw_bar_widget.setStyleSheet(
            "QWidget { background: #131b33; border-radius: 12px; border: 1px solid #1e2947; }"
            "QPushButton { background: #1e2947; color: #f1f5f9; border: none; border-radius: 8px; padding: 8px 16px; font-weight: 600; font-size: 13px; }"
            "QPushButton:hover { background: #2a3857; }"
            "QPushButton#Primary { background: #3b82f6; color: white; }"
            "QPushButton#Primary:hover { background: #2563eb; }"
            "QPushButton#Danger { background: #b91c1c; color: white; }"
            "QPushButton#Danger:hover { background: #991b1b; }"
            "QPushButton#Magic { background: #8b5cf6; color: white; }"
            "QPushButton#Magic:hover { background: #7c3aed; }"
            "QLabel { color: #cbd5e1; font-size: 13px; font-weight: 500; border: none; }"
        )
        self.draw_bar_widget.hide()  # Hidden until Whiteboard tab is selected
        draw_bar = QHBoxLayout(self.draw_bar_widget)
        draw_bar.setContentsMargins(12, 12, 12, 12)
        draw_bar.setSpacing(10)
        
        self._dpen = QPushButton("✏ Pen")
        self._dpen.setObjectName("Primary")
        self._dera = QPushButton("◻ Eraser")
        self._dclr = QPushButton("✕ Clear")
        self._dclr.setObjectName("Danger")
        self._dfix = QPushButton("✨ Auto-Fix")
        self._dfix.setObjectName("Magic")
        self._dsave = QPushButton("💾 Save Image")
        self._dtool = "pen"   # current draw tool for camera ink
        
        self._dpen.clicked.connect(lambda: self._set_draw_tool("pen"))
        self._dera.clicked.connect(lambda: self._set_draw_tool("eraser"))
        self._dclr.clicked.connect(self._clear_ink)
        self._dsave.clicked.connect(self._save_ink)
        self._dfix.clicked.connect(self._auto_fix_ink)
        
        self._dstatus = QLabel("Raise index finger to draw. Pinch to pause.")
        
        draw_bar.addWidget(self._dpen)
        draw_bar.addWidget(self._dera)
        draw_bar.addWidget(self._dfix)
        draw_bar.addWidget(self._dsave)
        draw_bar.addWidget(self._dclr)
        draw_bar.addStretch(1)
        draw_bar.addWidget(self._dstatus)
        video_layout.addWidget(self.draw_bar_widget)

        # Ink buffer state (initialized on first frame)
        self._ink: Optional[np.ndarray] = None
        self._ink_w = self._ink_h = 0
        self._ink_prev: Optional[tuple] = None
        self._ink_was_drawing = False
        self._ink_sx = self._ink_sy = 0.5
        self._ink_sinit = False

        left.addWidget(self.video_card, 1)

        metrics = QGridLayout()
        self.fps_card = MetricCard("FPS", "--")
        self.gesture_card = MetricCard("Gesture", "none")
        self.conf_card = MetricCard("Confidence", "0.00")
        self.hand_card = MetricCard("Hand Tracking", "NO HAND")
        metrics.addWidget(self.fps_card, 0, 0)
        metrics.addWidget(self.gesture_card, 0, 1)
        metrics.addWidget(self.conf_card, 0, 2)
        metrics.addWidget(self.hand_card, 0, 3)
        left.addLayout(metrics)

        self.right_tabs = QTabWidget()
        self.right_tabs.setMinimumWidth(410)
        main.addWidget(self.right_tabs, 1)
        self.right_tabs.addTab(self._whiteboard_tab(), "Whiteboard")
        self.whiteboard_tab_widget = self.right_tabs.widget(0)
        self.right_tabs.addTab(self._controls_tab(), "Control")
        self.right_tabs.addTab(self._gestures_tab(), "Gestures")
        self.right_tabs.addTab(self._instructions_tab(), "Instructions")
        self.right_tabs.addTab(self._diagnostics_tab(), "Diagnostics")
        self.right_tabs.addTab(self._trainer_tab(), "Trainer")
        self.right_tabs.currentChanged.connect(self._tab_changed)
        self._tab_changed(0)

    def _controls_tab(self) -> QWidget:
        tab = QWidget(); layout = QVBoxLayout(tab)
        safety = QGroupBox("Safety Layer")
        f = QVBoxLayout(safety)
        self.activation_required = QCheckBox("Require open-palm activation")
        self.activation_required.setChecked(self.config_data.activation_required)
        self.skeleton = QCheckBox("Visual skeleton overlay")
        self.skeleton.setChecked(self.config_data.show_skeleton)
        self.sound = QCheckBox("Sound feedback")
        self.sound.setChecked(self.config_data.sound_feedback)
        self.low_light = QCheckBox("Low-light enhancement")
        self.low_light.setChecked(self.config_data.low_light_enhancement)
        for w in [self.activation_required, self.skeleton, self.sound, self.low_light]:
            w.stateChanged.connect(self._save_controls)
            f.addWidget(w)
        layout.addWidget(safety)

        tuning = QGroupBox("Live Tuning")
        tf = QFormLayout(tuning)
        self.sensitivity_slider = self._slider(self.config_data.sensitivity)
        self.confidence_slider = self._slider(self.config_data.confidence_threshold)
        self.smoothing_slider = self._slider(self.config_data.cursor_smoothing)
        self.sensitivity_slider.valueChanged.connect(lambda _: self._slider_changed())
        self.confidence_slider.valueChanged.connect(lambda _: self._slider_changed())
        self.smoothing_slider.valueChanged.connect(lambda _: self._slider_changed())
        tf.addRow("Sensitivity", self.sensitivity_slider)
        tf.addRow("Confidence", self.confidence_slider)
        tf.addRow("Cursor smoothing", self.smoothing_slider)
        layout.addWidget(tuning)

        buttons = QGroupBox("System")
        bl = QVBoxLayout(buttons)
        calibrate = QPushButton("Calibration Wizard")
        calibrate.setObjectName("Primary")
        calibrate.clicked.connect(self._calibrate)
        save = QPushButton("Save Settings")
        save.clicked.connect(self._save_all)
        restart = QPushButton("Restart App After Camera Changes")
        restart.clicked.connect(lambda: QMessageBox.information(self, "Restart", "Close and reopen GestureOS to apply camera resolution/FPS changes."))
        bl.addWidget(calibrate); bl.addWidget(save); bl.addWidget(restart)
        layout.addWidget(buttons)
        layout.addStretch(1)
        return tab

    def _gestures_tab(self) -> QWidget:
        tab = QWidget(); layout = QVBoxLayout(tab)
        self.gesture_table = QTableWidget(0, 3)
        self.gesture_table.setHorizontalHeaderLabels(["Enabled", "Gesture", "Custom action override"])
        self.gesture_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        self.gesture_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        self.gesture_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        rows = [
            ("pointer", "index_pointer"), ("left_click", "pinch"), ("double_click", "double_pinch"),
            ("right_click", "three_finger_pinch"), ("drag", "pinch_hold"), ("scroll", "two_finger_scroll"),
            ("volume", "pinch in left zone"), ("play_pause", "fist"), ("mute", "thumb_up"),
            ("swipe_left", "swipe_left"), ("swipe_right", "swipe_right"),
        ]
        self.toggle_checks: Dict[str, QCheckBox] = {}
        self.mapping_edits: Dict[str, QLineEdit] = {}
        for toggle_key, gesture_name in rows:
            r = self.gesture_table.rowCount(); self.gesture_table.insertRow(r)
            cb = QCheckBox(); cb.setChecked(self.config_data.gesture_toggles.get(toggle_key, True)); cb.stateChanged.connect(self._save_gesture_table)
            self.toggle_checks[toggle_key] = cb
            self.gesture_table.setCellWidget(r, 0, cb)
            self.gesture_table.setItem(r, 1, QTableWidgetItem(gesture_name))
            edit = QLineEdit(self.config_data.custom_mappings.get(gesture_name, ""))
            edit.setPlaceholderText("optional: hotkey:ctrl+tab or press:space")
            edit.editingFinished.connect(self._save_gesture_table)
            self.mapping_edits[gesture_name] = edit
            self.gesture_table.setCellWidget(r, 2, edit)
        layout.addWidget(self.gesture_table)
        note = QLabel("Custom action override runs instead of the default action after debounce. Leave blank for default behavior.")
        note.setObjectName("Subtitle")
        layout.addWidget(note)
        return tab


    def _instructions_tab(self) -> QWidget:
        tab = QWidget(); layout = QVBoxLayout(tab)
        title = QLabel("How GestureOS Works")
        title.setObjectName("MetricValue")
        layout.addWidget(title)
        text = QTextEdit()
        text.setReadOnly(True)
        text.setText(
            "NORMAL DESKTOP MODE\n"
            "• Open palm activates GestureOS.\n"
            "• Index finger moves the desktop cursor.\n"
            "• Thumb + index quick touch/release = click.\n"
            "• Thumb + index hold + move = drag.\n"
            "• Three-finger pinch = right click.\n\n"
            "DRAWING / WHITEBOARD MODE\n"
            "• Open the Whiteboard tab or Full Screen Draw.\n"
            "• Desktop cursor and desktop actions are locked.\n"
            "• NO PINCH is required for drawing.\n"
            "• Your index fingertip is the pen. Keep the index finger visible.\n"
            "• Move the index finger to write on the board.\n"
            "• Use Hover / Pause to move without drawing.\n"
            "• Use Eraser to erase with the same index movement.\n"
            "• The pen cursor is hidden so it does not disturb the drawing.\n"
            "• Full Screen Draw includes a right-side live camera preview and tools.\n\n"
            "WHY DRAW MODE DOES NOT USE PINCH\n"
            "Pinch makes thumb and index landmarks dance/occlude each other, which creates false strokes. "
            "For smooth handwriting, GestureOS uses index-only drawing in draw mode. Pinch is only for clicking in desktop mode.\n\n"
            "BEST RESULTS\n"
            "• Use 640x480 or 960x540 at 60 FPS.\n"
            "• Use strong front lighting.\n"
            "• Keep hand 45–75 cm from webcam.\n"
            "• Keep index finger clearly visible.\n"
            "• Write slowly like on a real board.\n"
            "• Use Full Screen Draw for teaching/presentations."
        )
        layout.addWidget(text, 1)
        return tab

    def _diagnostics_tab(self) -> QWidget:
        tab = QWidget(); layout = QVBoxLayout(tab)
        self.fps_graph = Sparkline("FPS")
        self.conf_graph = Sparkline("Gesture Confidence")
        layout.addWidget(self.fps_graph)
        layout.addWidget(self.conf_graph)
        self.log_list = QListWidget()
        layout.addWidget(QLabel("Event History"))
        layout.addWidget(self.log_list, 1)
        clear = QPushButton("Clear History")
        clear.clicked.connect(self.log_list.clear)
        layout.addWidget(clear)
        return tab


    def _whiteboard_tab(self) -> QWidget:
        tab = QWidget()
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        
        # Small clean preview card (white drawing paper style)
        preview_card = QFrame()
        preview_card.setObjectName("Card")
        preview_layout = QVBoxLayout(preview_card)
        preview_layout.setContentsMargins(8, 8, 8, 8)
        
        self.whiteboard_preview = QLabel("Drawing Preview")
        self.whiteboard_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.whiteboard_preview.setMinimumHeight(240)
        self.whiteboard_preview.setStyleSheet("background:#ffffff; border-radius:12px; border: 1px solid #1e2947;")
        
        preview_layout.addWidget(self.whiteboard_preview)
        layout.addWidget(preview_card, 1)
        
        self.mini_status = QLabel("Clean Canvas Preview")
        self.mini_status.setObjectName("Subtitle")
        self.mini_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.mini_status)
        return tab

    def _trainer_tab(self) -> QWidget:
        tab = QWidget(); layout = QVBoxLayout(tab)
        info = QTextEdit()
        info.setReadOnly(True)
        info.setText(
            "Custom Gesture Trainer\n\n"
            "1. Activate GestureOS.\n"
            "2. Hold your custom hand pose clearly in front of the camera.\n"
            "3. Click Record Custom Gesture.\n"
            "4. Assign an action such as:\n"
            "   • hotkey:ctrl+l\n"
            "   • hotkey:ctrl+tab\n"
            "   • press:space\n\n"
            "For best results, record 3–5 samples of the same gesture under the same name."
        )
        layout.addWidget(info)
        btn = QPushButton("Record Custom Gesture")
        btn.setObjectName("Primary")
        btn.clicked.connect(self._record_custom_dialog)
        layout.addWidget(btn)
        layout.addStretch(1)
        return tab

    def _slider(self, value: float) -> QSlider:
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(5, 95)
        s.setValue(int(value * 100))
        return s

    def _start_worker(self):
        self.thread = QThread(self)
        self.worker = EngineWorker(self.config_data)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.start)
        self.worker.frameReady.connect(self._on_frame)
        self.worker.logReady.connect(self._log)
        self.worker.errorReady.connect(self._error)
        self.worker.startedReady.connect(lambda: self._log("Engine started"))
        self.applyConfigRequested.connect(self.worker.apply_config)
        self.setActiveRequested.connect(self.worker.set_active)
        self.recordCustomRequested.connect(self.worker.record_custom)
        self.whiteboardModeRequested.connect(self.worker.set_whiteboard_mode)
        self.pauseRequested.connect(self.worker.set_paused)
        self.stopRequested.connect(self.worker.stop)
        self.thread.start()

    def _set_draw_tool(self, t: str):
        self._dtool = t
        self._ink_prev = None
        self._dstatus.setText(f"Tool: {t.upper()}")

    def _clear_ink(self):
        if self._ink is not None:
            self._ink[:] = 255
        self._ink_prev = None
        self._dstatus.setText("Cleared")

    def _save_ink(self):
        if self._ink is None:
            return
        filename, _ = QFileDialog.getSaveFileName(self, "Save Drawing", "", "Images (*.png *.jpg)")
        if filename:
            cv2.imwrite(filename, self._ink)
            self._log(f"Drawing saved to {filename}")
            self._dstatus.setText("Saved")

    def _auto_fix_ink(self):
        if self._ink is None: return
        ink_gray = cv2.cvtColor(self._ink, cv2.COLOR_BGR2GRAY)
        binary = (ink_gray < 245).astype(np.uint8) * 255
        if cv2.countNonZero(binary) < 20: return
        
        coords = cv2.findNonZero(binary)
        if coords is None: return
        x, y, w, h = cv2.boundingRect(coords)
        
        # 1. Try to recognize text (digits/letters)
        best, _ = self.recognizer.recognize(binary)
        if best and best.confidence > 0.5 and len(best.text) > 0:
            self._ink[:] = 255
            font = cv2.FONT_HERSHEY_SIMPLEX
            fs = max(1.0, h / 30.0)
            th = max(2, int(fs * 2))
            ts, _ = cv2.getTextSize(best.text, font, fs, th)
            tx, ty = x + (w - ts[0]) // 2, y + (h + ts[1]) // 2
            cv2.putText(self._ink, best.text, (tx, ty), font, fs, (20, 24, 32), th, cv2.LINE_AA)
            self._log(f"Auto-fixed to text: {best.text}")
            self._dstatus.setText(f"Text: {best.text}")
            return
            
        # 2. Try shape detection
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            if area > 100:
                peri = cv2.arcLength(c, True)
                approx = cv2.approxPolyDP(c, 0.04 * peri, True)
                if len(approx) == 4:
                    self._ink[:] = 255
                    cv2.drawContours(self._ink, [approx], -1, (20, 24, 32), max(3, int(w/50)))
                    self._log("Auto-fixed to Rectangle")
                    self._dstatus.setText("Fixed to Rectangle")
                else:
                    (cx, cy), radius = cv2.minEnclosingCircle(c)
                    circle_area = np.pi * (radius**2)
                    if circle_area > 0 and area / circle_area > 0.70:
                        self._ink[:] = 255
                        cv2.circle(self._ink, (int(cx), int(cy)), int(radius), (20, 24, 32), max(3, int(w/50)), cv2.LINE_AA)
                        self._log("Auto-fixed to Circle")
                        self._dstatus.setText("Fixed to Circle")

    @pyqtSlot(QImage, str, float, float, bool, bool, float, float, float, float, bool)
    def _on_frame(self, img: QImage, gesture: str, confidence: float, fps: float,
                  active: bool, hand_seen: bool, cx: float, cy: float,
                  dx: float, dy: float, draw_active: bool):

        # ── Convert camera QImage → numpy BGR ──────────────────────────────
        img_rgb = img.convertToFormat(QImage.Format.Format_RGB888)
        ptr = img_rgb.bits()
        ptr.setsize(img_rgb.bytesPerLine() * img_rgb.height())
        frame_rgb = np.frombuffer(ptr, np.uint8).reshape(
            (img_rgb.height(), img_rgb.width(), 3)).copy()
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        fh, fw = frame_bgr.shape[:2]

        # ── Init / resize ink buffer ────────────────────────────────────
        if self._ink is None or self._ink_w != fw or self._ink_h != fh:
            self._ink = np.full((fh, fw, 3), 255, dtype=np.uint8)
            self._ink_w, self._ink_h = fw, fh
            self._ink_prev = None

        is_whiteboard_tab = (hasattr(self, "whiteboard_tab_widget") and self.right_tabs.currentWidget() is self.whiteboard_tab_widget)
        
        # ── Draw into ink buffer when hand is seen & active ───────────────
        drawing = bool(hand_seen and active and self._dtool != "eraser_noop" and is_whiteboard_tab and gesture != "pinch")
        if drawing:
            # EMA smooth (reject teleports > 20% of frame)
            if not self._ink_sinit:
                self._ink_sx, self._ink_sy = cx, cy
                self._ink_sinit = True
            else:
                dist_px = math.hypot(cx - self._ink_sx, cy - self._ink_sy)
                if dist_px < 0.20:
                    a = 0.55
                    self._ink_sx += a * (cx - self._ink_sx)
                    self._ink_sy += a * (cy - self._ink_sy)

            px = int(max(0, min(fw - 1, self._ink_sx * fw)))
            py = int(max(0, min(fh - 1, self._ink_sy * fh)))
            cur = (px, py)

            if self._dtool == "eraser":
                cv2.circle(self._ink, cur, 20, (255, 255, 255), -1, cv2.LINE_AA)
            else:
                # PEN: connect prev → cur every frame (never broken)
                if not self._ink_was_drawing or self._ink_prev is None:
                    cv2.circle(self._ink, cur, 3, (20, 24, 32), -1, cv2.LINE_AA)
                else:
                    cv2.line(self._ink, self._ink_prev, cur, (20, 24, 32), 5, cv2.LINE_AA)
            self._ink_prev = cur
        else:
            self._ink_sinit = False
            self._ink_prev = None
        self._ink_was_drawing = bool(hand_seen and active)

        # ── Blend ink over camera frame ────────────────────────────────
        mask = False
        if is_whiteboard_tab:
            mask = np.any(self._ink < 245, axis=2)
            
        out = frame_bgr.copy()
        if is_whiteboard_tab and mask.any():
            out[mask] = (
                frame_bgr[mask].astype(np.float32) * 0.15
                + self._ink[mask].astype(np.float32) * 0.85
            ).astype(np.uint8)
            
        out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
        h, w, ch = out_rgb.shape
        display_img = QImage(out_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()

        # ── Display on main camera label ───────────────────────────────
        pix = QPixmap.fromImage(display_img).scaled(
            self.video.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        self.video.setPixmap(pix)
        
        # ── Display clean drawing preview on the right tab if active ──
        if is_whiteboard_tab:
            ink_rgb = cv2.cvtColor(self._ink, cv2.COLOR_BGR2RGB)
            ink_img = QImage(ink_rgb.data, w, h, ch * w, QImage.Format.Format_RGB888).copy()
            pix_ink = QPixmap.fromImage(ink_img).scaled(
                self.whiteboard_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            self.whiteboard_preview.setPixmap(pix_ink)

        # ── Metrics & other panels ──────────────────────────────────
        self.fps_card.set_value(f"{fps:.1f}")
        self.gesture_card.set_value(gesture)
        self.conf_card.set_value(f"{confidence:.2f}")
        self.hand_card.set_value("LOCKED" if hand_seen else "NO HAND")
        self.hand_card.value.setStyleSheet("color:#22c55e;" if hand_seen else "color:#f97316;")
        self.active_btn.setChecked(active)
        self.active_btn.setText("Active" if active else "Activate")
        self.fps_graph.add(fps)
        self.conf_graph.add(confidence)
        state = "DRAWING" if (hand_seen and active) else "HOVER"
        self._dstatus.setText(f"{state} | {self._dtool.upper()}")
        if getattr(self, "fullscreen_draw", None) is not None and self.fullscreen_draw.isVisible():
            self.fullscreen_draw.update_camera(img)
            self.fullscreen_draw.update_from_absolute(cx, cy, hand_seen and active)
        if hand_seen != self.last_hand_seen:
            self._log("Hand locked" if hand_seen else "Hand lost")
            self.last_hand_seen = hand_seen


    def _tab_changed(self, index: int):
        enabled = hasattr(self, "whiteboard_tab_widget") and self.right_tabs.currentWidget() is self.whiteboard_tab_widget
        if hasattr(self, "draw_bar_widget"):
            self.draw_bar_widget.setVisible(enabled)
        
        if enabled:
            self.config_data.gesture_mode_active = True
            self.setActiveRequested.emit(True)
            self._log("Whiteboard mode: OS actions paused; gestures draw on camera screen")
            
        self.whiteboardModeRequested.emit(bool(enabled))


    def _open_fullscreen_draw(self):
        self.fullscreen_draw = FullScreenDrawWindow(self)
        self.fullscreen_draw.closed.connect(self._fullscreen_closed)
        QApplication.setOverrideCursor(QCursor(Qt.CursorShape.BlankCursor))
        self.config_data.gesture_mode_active = True
        self.setActiveRequested.emit(True)
        self.whiteboardModeRequested.emit(True)
        self._log("Full screen draw opened: desktop cursor/actions locked")
        self.fullscreen_draw.showFullScreen()

    def _fullscreen_closed(self):
        # Keep whiteboard mode if the normal Whiteboard tab is still active; otherwise unlock desktop actions.
        try:
            QApplication.restoreOverrideCursor()
        except Exception:
            pass
        enabled = hasattr(self, "whiteboard_tab_widget") and self.right_tabs.currentWidget() is self.whiteboard_tab_widget
        self.whiteboardModeRequested.emit(bool(enabled))
        self._log("Full screen draw closed")

    def _toggle_active(self, checked: bool):
        self.config_data.gesture_mode_active = bool(checked)
        self.setActiveRequested.emit(bool(checked))
        self._save_all()

    def _save_controls(self):
        self.config_data.activation_required = self.activation_required.isChecked()
        self.config_data.show_skeleton = self.skeleton.isChecked()
        self.config_data.sound_feedback = self.sound.isChecked()
        self.config_data.low_light_enhancement = self.low_light.isChecked()
        self._save_all()

    def _slider_changed(self):
        self.config_data.sensitivity = self.sensitivity_slider.value() / 100.0
        self.config_data.confidence_threshold = self.confidence_slider.value() / 100.0
        self.config_data.cursor_smoothing = self.smoothing_slider.value() / 100.0
        self.config_data.save()
        self.applyConfigRequested.emit()

    def _save_gesture_table(self):
        for k, cb in self.toggle_checks.items():
            self.config_data.gesture_toggles[k] = cb.isChecked()
        for gesture, edit in self.mapping_edits.items():
            val = edit.text().strip()
            if val:
                self.config_data.custom_mappings[gesture] = val
            else:
                self.config_data.custom_mappings.pop(gesture, None)
        self._save_all()

    def _save_all(self):
        self.config_data.save()
        self.applyConfigRequested.emit()

    def _calibrate(self):
        dlg = CalibrationDialog(self.config_data, self)
        if dlg.exec():
            self._sync_controls_from_config()
            self.applyConfigRequested.emit()
            QMessageBox.information(self, "Calibration saved", "Live thresholds updated. Restart app if you changed camera resolution or FPS.")

    def _sync_controls_from_config(self):
        self.activation_required.setChecked(self.config_data.activation_required)
        self.skeleton.setChecked(self.config_data.show_skeleton)
        self.sound.setChecked(self.config_data.sound_feedback)
        self.low_light.setChecked(self.config_data.low_light_enhancement)
        self.sensitivity_slider.setValue(int(self.config_data.sensitivity * 100))
        self.confidence_slider.setValue(int(self.config_data.confidence_threshold * 100))
        self.smoothing_slider.setValue(int(self.config_data.cursor_smoothing * 100))

    def _record_custom_dialog(self):
        dlg = CustomGestureDialog(self)
        dlg.requested.connect(self.recordCustomRequested.emit)
        dlg.exec()

    def _log(self, msg: str):
        stamp = time.strftime("%H:%M:%S")
        self.log_list.insertItem(0, f"{stamp}  {msg}")
        while self.log_list.count() > 300:
            self.log_list.takeItem(self.log_list.count() - 1)

    def _error(self, msg: str):
        self._log(f"ERROR: {msg}")
        QMessageBox.critical(self, "GestureOS error", msg)

    def closeEvent(self, event):  # noqa: N802
        try:
            self.config_data.save()
            self.stopRequested.emit()
            if self.thread:
                self.thread.quit()
                self.thread.wait(2500)
        finally:
            super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("GestureOS Pro")
    app.setStyleSheet(STYLE)
    win = GestureOSQtApp()
    win.show()
    sys.exit(app.exec())
