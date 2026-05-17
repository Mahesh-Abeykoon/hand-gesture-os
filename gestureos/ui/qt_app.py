from __future__ import annotations

import sys
import time
from collections import deque
from typing import Deque, Dict, Optional

import cv2
import numpy as np

from PyQt6.QtCore import QObject, Qt, QThread, QTimer, pyqtSignal, pyqtSlot
from PyQt6.QtGui import QColor, QFont, QImage, QPainter, QPen, QPixmap
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
    QSlider,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from gestureos.app_core import GestureEngine
from gestureos.gestures.types import Gesture
from gestureos.settings.config import AppConfig
from gestureos.utils.logging import get_logger


STYLE = """
QMainWindow, QWidget { background: #0b0f17; color: #e8eef7; font-family: Segoe UI, Inter, Arial; }
QFrame#Card, QGroupBox { background: #121824; border: 1px solid #202a3a; border-radius: 16px; }
QGroupBox { margin-top: 18px; padding: 12px; font-weight: 700; }
QGroupBox::title { subcontrol-origin: margin; left: 14px; padding: 0 8px; color: #a9c7ff; }
QLabel#Title { font-size: 30px; font-weight: 900; color: white; }
QLabel#Subtitle { color: #8fa4c1; font-size: 13px; }
QLabel#MetricValue { font-size: 24px; font-weight: 800; color: white; }
QLabel#MetricName { color: #8fa4c1; font-size: 12px; }
QPushButton { background: #1b2535; border: 1px solid #2f3e55; border-radius: 10px; padding: 9px 12px; font-weight: 700; }
QPushButton:hover { background: #25334a; }
QPushButton#Primary { background: #2563eb; border-color: #3b82f6; color: white; }
QPushButton#Danger { background: #7f1d1d; border-color: #ef4444; color: white; }
QPushButton#Success { background: #166534; border-color: #22c55e; color: white; }
QTabWidget::pane { border: 1px solid #202a3a; border-radius: 14px; top: -1px; }
QTabBar::tab { background: #121824; color: #9fb0c8; padding: 10px 16px; border-top-left-radius: 10px; border-top-right-radius: 10px; margin-right: 2px; }
QTabBar::tab:selected { background: #1b2535; color: white; }
QSlider::groove:horizontal { height: 7px; background: #263142; border-radius: 4px; }
QSlider::handle:horizontal { background: #60a5fa; width: 18px; margin: -6px 0; border-radius: 9px; }
QCheckBox { spacing: 8px; color: #dbe7f6; }
QTableWidget, QListWidget, QTextEdit, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #0e1420; border: 1px solid #263142; border-radius: 8px; color: #e8eef7; padding: 6px; }
QHeaderView::section { background: #182132; color: #b9c7dc; padding: 7px; border: none; }
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
    """Lightweight template OCR for demo-ready digit/character recognition."""

    def __init__(self):
        self.templates = self._build_templates()

    def _build_templates(self):
        templates = {}
        chars = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ"
        fonts = [cv2.FONT_HERSHEY_SIMPLEX, cv2.FONT_HERSHEY_DUPLEX, cv2.FONT_HERSHEY_COMPLEX]
        for ch in chars:
            variants = []
            for font in fonts:
                for scale in (1.5, 1.8, 2.1):
                    img = np.zeros((96, 96), dtype=np.uint8)
                    (tw, th), _ = cv2.getTextSize(ch, font, scale, 4)
                    x = max(1, (96 - tw) // 2)
                    y = max(th + 2, (96 + th) // 2)
                    cv2.putText(img, ch, (x, y), font, scale, 255, 4, cv2.LINE_AA)
                    variants.append(self._normalize(img))
            templates[ch] = variants
        return templates

    def _normalize(self, img: np.ndarray) -> np.ndarray:
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        pts = cv2.findNonZero(img)
        if pts is None:
            return np.zeros((64, 64), dtype=np.float32)
        x, y, w, h = cv2.boundingRect(pts)
        crop = img[y:y+h, x:x+w]
        side = max(w, h) + 18
        square = np.zeros((side, side), dtype=np.uint8)
        ox, oy = (side - w) // 2, (side - h) // 2
        square[oy:oy+h, ox:ox+w] = crop
        norm = cv2.resize(square, (64, 64), interpolation=cv2.INTER_AREA)
        return (norm > 80).astype(np.float32)

    def recognize_symbol(self, ink_binary: np.ndarray) -> tuple[str, float]:
        x = self._normalize(ink_binary)
        best_ch, best_score = "?", 1e9
        for ch, variants in self.templates.items():
            score = min(float(np.mean((x - t) ** 2)) for t in variants)
            if score < best_score:
                best_ch, best_score = ch, score
        confidence = max(0.0, min(1.0, 1.0 - best_score * 3.0))
        return best_ch, confidence

    def recognize_components(self, ink_binary: np.ndarray) -> str:
        kernel = np.ones((5, 5), np.uint8)
        merged = cv2.dilate(ink_binary, kernel, iterations=1)
        n, labels, stats, _ = cv2.connectedComponentsWithStats((merged > 0).astype(np.uint8), 8)
        boxes = []
        for i in range(1, n):
            x, y, w, h, area = stats[i]
            if area > 25 and w > 4 and h > 8:
                boxes.append((x, y, w, h))
        boxes.sort(key=lambda b: b[0])
        if not boxes:
            return ""
        chars = []
        prev_end = None
        for x, y, w, h in boxes[:16]:
            if prev_end is not None and x - prev_end > max(18, w * 0.8):
                chars.append(" ")
            pad = 8
            crop = ink_binary[max(0, y-pad):min(ink_binary.shape[0], y+h+pad), max(0, x-pad):min(ink_binary.shape[1], x+w+pad)]
            ch, _ = self.recognize_symbol(crop)
            chars.append(ch)
            prev_end = x + w
        return "".join(chars)


class WhiteboardCanvas(QWidget):
    def __init__(self):
        super().__init__()
        self.setMinimumHeight(560)
        self.canvas = QImage(1200, 720, QImage.Format.Format_RGB32)
        self.canvas.fill(QColor("white"))
        self.pen_color = QColor("#111827")
        self.pen_width = 7
        self.cursor_x = self.canvas.width() / 2
        self.cursor_y = self.canvas.height() / 2
        self.last_draw = False
        self.mouse_down = False
        self.recognizer = WhiteboardRecognizer()

    def clear(self):
        self.canvas.fill(QColor("white"))
        self.update()

    def set_pen_width(self, width: int):
        self.pen_width = int(width)

    def update_from_gesture(self, dx: float, dy: float, drawing: bool):
        # Gesture deltas are normalized; amplify a little for canvas-space drawing.
        nx = max(0, min(self.canvas.width() - 1, self.cursor_x + dx * self.canvas.width() * 1.55))
        ny = max(0, min(self.canvas.height() - 1, self.cursor_y + dy * self.canvas.height() * 1.55))
        if drawing:
            painter = QPainter(self.canvas)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawLine(int(self.cursor_x), int(self.cursor_y), int(nx), int(ny))
            painter.end()
        self.cursor_x, self.cursor_y = nx, ny
        self.last_draw = drawing
        self.update()

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
        ch, conf = self.recognizer.recognize_symbol(ink)
        seq = self.recognizer.recognize_components(ink)
        return f"Best single symbol: {ch}  ({conf:.2f})", f"Components / word guess: {seq}"

    def paintEvent(self, event):  # noqa: N802
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#0e1420"))
        target = self.rect().adjusted(10, 10, -10, -10)
        pix = QPixmap.fromImage(self.canvas).scaled(target.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        x = target.x() + (target.width() - pix.width()) // 2
        y = target.y() + (target.height() - pix.height()) // 2
        painter.drawPixmap(x, y, pix)
        sx, sy = pix.width() / self.canvas.width(), pix.height() / self.canvas.height()
        cx, cy = x + self.cursor_x * sx, y + self.cursor_y * sy
        painter.setPen(QPen(QColor("#2563eb"), 2))
        painter.drawEllipse(int(cx - 8), int(cy - 8), 16, 16)
        if self.last_draw:
            painter.setPen(QPen(QColor("#ef4444"), 2))
            painter.drawEllipse(int(cx - 13), int(cy - 13), 26, 26)

    def mousePressEvent(self, event):  # noqa: N802
        self.mouse_down = True
        self._mouse_draw(event.position().x(), event.position().y(), False)

    def mouseMoveEvent(self, event):  # noqa: N802
        self._mouse_draw(event.position().x(), event.position().y(), self.mouse_down)

    def mouseReleaseEvent(self, event):  # noqa: N802
        self.mouse_down = False

    def _mouse_draw(self, x, y, drawing):
        # Test drawing with real mouse too.
        target = self.rect().adjusted(10, 10, -10, -10)
        scale = min(target.width() / self.canvas.width(), target.height() / self.canvas.height())
        ox = target.x() + (target.width() - self.canvas.width() * scale) / 2
        oy = target.y() + (target.height() - self.canvas.height() * scale) / 2
        nx = max(0, min(self.canvas.width() - 1, (x - ox) / scale))
        ny = max(0, min(self.canvas.height() - 1, (y - oy) / scale))
        if drawing:
            painter = QPainter(self.canvas)
            painter.setRenderHint(QPainter.RenderHint.Antialiasing)
            painter.setPen(QPen(self.pen_color, self.pen_width, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin))
            painter.drawLine(int(self.cursor_x), int(self.cursor_y), int(nx), int(ny))
            painter.end()
        self.cursor_x, self.cursor_y = nx, ny
        self.update()


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
        self.video.setStyleSheet("background:#05070b; border-radius:12px; color:#789;")
        video_layout.addWidget(self.video, 1)
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
        self.right_tabs.addTab(self._controls_tab(), "Control")
        self.right_tabs.addTab(self._gestures_tab(), "Gestures")
        self.right_tabs.addTab(self._diagnostics_tab(), "Diagnostics")
        self.whiteboard_tab_widget = self._whiteboard_tab()
        self.right_tabs.addTab(self.whiteboard_tab_widget, "Whiteboard")
        self.right_tabs.addTab(self._trainer_tab(), "Trainer")
        self.right_tabs.currentChanged.connect(self._tab_changed)

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
        tab = QWidget(); layout = QVBoxLayout(tab)
        title = QLabel("Gesture Whiteboard")
        title.setObjectName("MetricValue")
        subtitle = QLabel("Move with index finger. Touch thumb to index and move to draw. Release thumb to stop drawing. Recognition is lightweight template OCR for digits/letters.")
        subtitle.setObjectName("Subtitle")
        subtitle.setWordWrap(True)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        self.whiteboard = WhiteboardCanvas()
        layout.addWidget(self.whiteboard, 1)
        controls = QHBoxLayout()
        clear_btn = QPushButton("Clear")
        clear_btn.setObjectName("Danger")
        clear_btn.clicked.connect(lambda: (self.whiteboard.clear(), self.whiteboard_result.setText("Draw a number/letter, then click Recognize.")))
        recog_btn = QPushButton("Recognize Number / Character / Word")
        recog_btn.setObjectName("Primary")
        recog_btn.clicked.connect(self._recognize_whiteboard)
        self.pen_slider = QSlider(Qt.Orientation.Horizontal)
        self.pen_slider.setRange(2, 18)
        self.pen_slider.setValue(7)
        self.pen_slider.valueChanged.connect(self.whiteboard.set_pen_width)
        controls.addWidget(clear_btn)
        controls.addWidget(recog_btn)
        controls.addWidget(QLabel("Pen"))
        controls.addWidget(self.pen_slider)
        layout.addLayout(controls)
        self.whiteboard_result = QLabel("Draw a number/letter, then click Recognize.")
        self.whiteboard_result.setObjectName("Subtitle")
        self.whiteboard_result.setWordWrap(True)
        layout.addWidget(self.whiteboard_result)
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

    @pyqtSlot(QImage, str, float, float, bool, bool, float, float, float, float, bool)
    def _on_frame(self, img: QImage, gesture: str, confidence: float, fps: float, active: bool, hand_seen: bool, cx: float, cy: float, dx: float, dy: float, draw_active: bool):
        pix = QPixmap.fromImage(img).scaled(self.video.size(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.video.setPixmap(pix)
        self.fps_card.set_value(f"{fps:.1f}")
        self.gesture_card.set_value(gesture)
        self.conf_card.set_value(f"{confidence:.2f}")
        self.hand_card.set_value("LOCKED" if hand_seen else "NO HAND")
        self.hand_card.value.setStyleSheet("color:#22c55e;" if hand_seen else "color:#f97316;")
        self.active_btn.setChecked(active)
        self.active_btn.setText("Active" if active else "Activate")
        self.fps_graph.add(fps)
        self.conf_graph.add(confidence)
        if hasattr(self, "whiteboard") and self.right_tabs.currentWidget() is self.whiteboard_tab_widget:
            self.whiteboard.update_from_gesture(dx, dy, draw_active and hand_seen and active)
        if hand_seen != self.last_hand_seen:
            self._log("Hand locked" if hand_seen else "Hand lost")
            self.last_hand_seen = hand_seen


    def _tab_changed(self, index: int):
        enabled = hasattr(self, "whiteboard_tab_widget") and self.right_tabs.currentWidget() is self.whiteboard_tab_widget
        self.whiteboardModeRequested.emit(bool(enabled))
        if enabled:
            self._log("Whiteboard mode: OS actions paused; gestures draw on canvas")

    def _recognize_whiteboard(self):
        if not hasattr(self, "whiteboard"):
            return
        single, sequence = self.whiteboard.recognize()
        self.whiteboard_result.setText(single + (f"\n{sequence}" if sequence else ""))

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
