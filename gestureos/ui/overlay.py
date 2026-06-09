from __future__ import annotations

import math
import random
import time
from typing import List, Tuple, Optional

from PyQt6.QtCore import Qt, QTimer, QPoint, QRectF, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QScreen, QLinearGradient, QCursor
from PyQt6.QtWidgets import QWidget, QApplication


class Particle:
    """A floating neon particle in the cursor trail."""
    def __init__(self, x: float, y: float, color: QColor, size: float):
        self.x = x
        self.y = y
        self.vx = random.uniform(-0.6, 0.6)
        self.vy = random.uniform(-0.6, 0.6)
        self.size = size
        self.initial_size = size
        self.color = color
        self.age = 0.0
        self.max_age = random.uniform(0.35, 0.55)

    def update(self, dt: float) -> bool:
        """Update physics. Return False if the particle has died."""
        self.x += self.vx
        self.y += self.vy
        self.age += dt
        ratio = self.age / self.max_age
        self.size = self.initial_size * (1.0 - ratio)
        return ratio < 1.0

    def draw(self, painter: QPainter):
        opacity = int(255 * (1.0 - (self.age / self.max_age)))
        color = QColor(self.color)
        color.setAlpha(opacity)
        painter.setBrush(QBrush(color))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawEllipse(QPointF(self.x, self.y), self.size, self.size)


class ClickRipple:
    """An expanding concentric circle animation for click events."""
    def __init__(self, x: float, y: float, color: QColor, button: str = "left"):
        self.x = x
        self.y = y
        self.color = color
        self.button = button
        self.radius = 5.0
        self.max_radius = 65.0 if button == "double" else 45.0
        self.speed = 135.0  # pixels per second
        self.opacity = 1.0

    def update(self, dt: float) -> bool:
        """Expand and fade. Return False if animation is finished."""
        self.radius += self.speed * dt
        ratio = self.radius / self.max_radius
        self.opacity = 1.0 - ratio
        return ratio < 1.0

    def draw(self, painter: QPainter):
        alpha = int(255 * self.opacity)
        color = QColor(self.color)
        color.setAlpha(alpha)
        
        # Thinner line as it expands
        width = max(1.0, 3.5 * self.opacity)
        pen = QPen(color, width)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QPointF(self.x, self.y), self.radius, self.radius)
        
        if self.button == "double":
            # Inner delayed secondary ring for double clicks
            inner_radius = self.radius - 18.0
            if inner_radius > 0:
                inner_opacity = 1.0 - (inner_radius / self.max_radius)
                if inner_opacity > 0:
                    color_inner = QColor(self.color)
                    color_inner.setAlpha(int(255 * inner_opacity))
                    pen_inner = QPen(color_inner, max(0.5, 2.0 * inner_opacity))
                    painter.setPen(pen_inner)
                    painter.drawEllipse(QPointF(self.x, self.y), inner_radius, inner_radius)


class VolumeOSD:
    """A floating, glassmorphic top-center volume slider OSD."""
    def __init__(self):
        self.value = 50
        self.opacity = 0.0
        self.state = "hidden"  # hidden, fade_in, active, fade_out
        self.last_activity = 0.0
        self.fade_duration = 0.18
        self.active_duration = 1.8

    def trigger(self, value: int):
        self.value = value
        self.last_activity = time.perf_counter()
        if self.state in ("hidden", "fade_out"):
            self.state = "fade_in"

    def update(self, dt: float):
        now = time.perf_counter()
        if self.state == "fade_in":
            self.opacity = min(1.0, self.opacity + dt / self.fade_duration)
            if self.opacity >= 1.0:
                self.state = "active"
        elif self.state == "active":
            self.opacity = 1.0
            if now - self.last_activity > self.active_duration:
                self.state = "fade_out"
        elif self.state == "fade_out":
            self.opacity = max(0.0, self.opacity - dt / (self.fade_duration * 2.0))
            if self.opacity <= 0.0:
                self.state = "hidden"

    def draw(self, painter: QPainter, screen_w: int):
        if self.state == "hidden" or self.opacity <= 0:
            return

        osd_w = 280
        osd_h = 52
        osd_x = (screen_w - osd_w) // 2
        osd_y = 45

        # Render with floating/fade opacity
        alpha_bg = int(220 * self.opacity)
        alpha_border = int(60 * self.opacity)
        alpha_text = int(240 * self.opacity)

        # 1. Glassmorphism card background (dark slate-blue, highly transparent)
        painter.setBrush(QBrush(QColor(13, 18, 30, alpha_bg)))
        border_pen = QPen(QColor(56, 189, 248, alpha_border), 1.5)  # Cyan border
        painter.setPen(border_pen)
        
        rect = QRectF(osd_x, osd_y, osd_w, osd_h)
        painter.drawRoundedRect(rect, 14, 14)

        # 2. Text label (e.g. "VOLUME 75%")
        painter.setPen(QColor(230, 242, 255, alpha_text))
        font = painter.font()
        font.setFamily("Outfit")
        font.setPixelSize(12)
        font.setBold(True)
        painter.setFont(font)
        
        label_text = f"🔊 VOLUME  {self.value}%"
        painter.drawText(QRectF(osd_x + 18, osd_y + 8, osd_w - 36, 18), Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter, label_text)

        # 3. Sleek neon progress bar container
        bar_x = osd_x + 18
        bar_y = osd_y + 30
        bar_w = osd_w - 36
        bar_h = 6
        
        # Background bar track
        painter.setBrush(QBrush(QColor(30, 41, 59, int(150 * self.opacity))))
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 3, 3)

        # Filled neon cyan progress bar
        fill_w = int(bar_w * (self.value / 100.0))
        if fill_w > 0:
            gradient = QLinearGradient(QPointF(bar_x, bar_y), QPointF(bar_x + fill_w, bar_y))
            cyan = QColor(14, 165, 233, alpha_text)   # Neon Cyan
            blue = QColor(99, 102, 241, alpha_text)   # Indgo-blue accent
            gradient.setColorAt(0, cyan)
            gradient.setColorAt(1, blue)
            painter.setBrush(QBrush(gradient))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, fill_w, bar_h), 3, 3)


class GlobalHUDOverlay(QWidget):
    """
    Transparent, stays-on-top, click-through desktop overlay.
    Renders visual feedback on top of other applications.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("GestureOS HUD Overlay")
        
        # Frameless, transparent, stays-on-top, click-through overlay window
        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.WindowStaysOnTopHint |
            Qt.WindowType.WindowTransparentForInput |
            Qt.WindowType.Tool
        )
        self.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self.setAttribute(Qt.WidgetAttribute.WA_NoSystemBackground, True)

        self.particles: List[Particle] = []
        self.ripples: List[ClickRipple] = []
        self.volume_osd = VolumeOSD()
        self.last_cursor_pos: Optional[QPoint] = None
        self.last_update_time = time.perf_counter()
        
        # Neon theme colors
        self.theme_cyan = QColor(0, 225, 255)
        self.theme_green = QColor(80, 245, 120)
        self.theme_amber = QColor(251, 146, 60)

        # Position to cover the primary screen geometry
        self.resize_to_screen()

        # Update and repaint loop (60 FPS)
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.tick)
        self.timer.start(16)

    def resize_to_screen(self):
        screen = QApplication.primaryScreen()
        if screen:
            rect = screen.geometry()
            self.setGeometry(rect)
            self.last_update_time = time.perf_counter()

    def trigger_click_ripple(self, button: str):
        """Spawn a ripple animation at the current mouse position."""
        pos = QCursor.pos()
        local_pos = self.mapFromGlobal(pos)
        
        color = self.theme_cyan
        if button == "double":
            color = self.theme_green
        elif button == "right":
            color = self.theme_amber

        self.ripples.append(ClickRipple(local_pos.x(), local_pos.y(), color, button))

    def set_volume(self, value: int):
        """Show and update the Volume OSD."""
        self.volume_osd.trigger(value)

    def tick(self):
        """Update logic and request redraw."""
        now = time.perf_counter()
        dt = now - self.last_update_time
        self.last_update_time = now

        # 1. Update existing animations
        self.particles = [p for p in self.particles if p.update(dt)]
        self.ripples = [r for r in self.ripples if r.update(dt)]
        self.volume_osd.update(dt)

        # 2. Track cursor and spawn trail particles
        pos = QCursor.pos()
        local_pos = self.mapFromGlobal(pos)
        
        if self.last_cursor_pos is not None:
            dx = local_pos.x() - self.last_cursor_pos.x()
            dy = local_pos.y() - self.last_cursor_pos.y()
            dist = math.hypot(dx, dy)
            
            # Spawn trail particles relative to speed
            if dist > 2.0:
                count = min(3, int(dist / 6.0) + 1)
                for i in range(count):
                    # Linearly interpolate spawn points to avoid dotted gaps when cursor moves fast
                    t = (i + 1) / (count + 1)
                    px = self.last_cursor_pos.x() + dx * t
                    py = self.last_cursor_pos.y() + dy * t
                    
                    size = random.uniform(1.8, 3.2)
                    # Interpolate color slightly between cyan and indigo-blue
                    color = QColor(self.theme_cyan)
                    if random.random() > 0.5:
                        color = QColor(99, 102, 241) # Indigo accent
                        
                    self.particles.append(Particle(px, py, color, size))

        # Only remember coordinates if they lie within our window geometry
        if self.rect().contains(local_pos):
            self.last_cursor_pos = local_pos
        else:
            self.last_cursor_pos = None

        # Redraw
        self.update()

    def paintEvent(self, event):
        """Render HUD graphics using QPainter."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # 1. Draw particle trails
        for p in self.particles:
            p.draw(painter)

        # 2. Draw click ripples
        for r in self.ripples:
            r.draw(painter)

        # 3. Draw Volume OSD
        self.volume_osd.draw(painter, self.width())

        # 4. Draw Cursor Halo around actual mouse position
        pos = QCursor.pos()
        local_pos = self.mapFromGlobal(pos)
        if self.rect().contains(local_pos):
            # Pulsating halo circle
            pulse = 14.5 + 1.8 * math.sin(time.perf_counter() * 7.5)
            
            # Faint background brush ring
            halo_bg = QColor(self.theme_cyan)
            halo_bg.setAlpha(12)
            painter.setBrush(QBrush(halo_bg))
            
            # Bright neon stroke
            halo_stroke = QColor(self.theme_cyan)
            halo_stroke.setAlpha(170)
            painter.setPen(QPen(halo_stroke, 1.25))
            
            painter.drawEllipse(QPointF(local_pos.x(), local_pos.y()), pulse, pulse)

            # Tiny glowing center core dot
            painter.setBrush(QBrush(self.theme_cyan))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(QPointF(local_pos.x(), local_pos.y()), 2.0, 2.0)
            
        painter.end()


# Helper to easily test the overlay as a standalone script
if __name__ == "__main__":
    import sys
    app = QApplication(sys.argv)
    overlay = GlobalHUDOverlay()
    overlay.show()
    
    # Standalone demo triggers: left clicks spawn ripples automatically
    class ClickFilter(QWidget):
        def __init__(self, overlay):
            super().__init__()
            self.overlay = overlay
            self.startTimer(1500) # Trigger test ripple every 1.5 seconds
        def timerEvent(self, event):
            self.overlay.trigger_click_ripple("left" if random.random() > 0.3 else "double")
            self.overlay.set_volume(random.randint(10, 100))
            
    test_filter = ClickFilter(overlay)
    sys.exit(app.exec())
