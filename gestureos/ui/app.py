from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox, simpledialog
from typing import Optional

import cv2
from PIL import Image, ImageTk

from gestureos.app_core import GestureEngine
from gestureos.settings.config import AppConfig
from gestureos.utils.logging import get_logger


class GestureOSApp(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("GestureOS — Real-Time Gesture Control")
        self.geometry("1180x760")
        self.minsize(1040, 680)
        self.protocol("WM_DELETE_WINDOW", self.on_close)
        self.config_data = AppConfig.load()
        self.logger = get_logger("GestureOS.UI")
        self.engine: Optional[GestureEngine] = None
        self.running = False
        self.photo = None
        self._build_style()
        self._build_ui()
        self.after(150, self.start_engine)

    def _build_style(self):
        self.configure(bg="#0f1117")
        style = ttk.Style(self)
        style.theme_use("clam")
        style.configure("TFrame", background="#0f1117")
        style.configure("Card.TFrame", background="#171a23", relief="flat")
        style.configure("TLabel", background="#0f1117", foreground="#f2f5f8", font=("Segoe UI", 10))
        style.configure("Card.TLabel", background="#171a23", foreground="#f2f5f8", font=("Segoe UI", 10))
        style.configure("Title.TLabel", background="#0f1117", foreground="#ffffff", font=("Segoe UI", 22, "bold"))
        style.configure("Accent.TButton", font=("Segoe UI", 10, "bold"), padding=8)
        style.configure("TCheckbutton", background="#171a23", foreground="#f2f5f8")

    def _build_ui(self):
        root = ttk.Frame(self, padding=16)
        root.pack(fill="both", expand=True)
        header = ttk.Frame(root)
        header.pack(fill="x")
        ttk.Label(header, text="GestureOS", style="Title.TLabel").pack(side="left")
        self.status_var = tk.StringVar(value="Starting camera...")
        ttk.Label(header, textvariable=self.status_var).pack(side="right")

        body = ttk.Frame(root)
        body.pack(fill="both", expand=True, pady=(14, 0))
        left = ttk.Frame(body, style="Card.TFrame", padding=12)
        left.pack(side="left", fill="both", expand=True, padx=(0, 12))
        right = ttk.Frame(body, style="Card.TFrame", padding=12, width=330)
        right.pack(side="right", fill="y")
        right.pack_propagate(False)

        self.preview = tk.Label(left, bg="#05060a")
        self.preview.pack(fill="both", expand=True)
        self.overlay_var = tk.StringVar(value="Gesture: none | FPS: --")
        ttk.Label(left, textvariable=self.overlay_var, style="Card.TLabel").pack(anchor="w", pady=(8, 0))

        # Controls
        ttk.Label(right, text="Control Center", style="Card.TLabel", font=("Segoe UI", 14, "bold")).pack(anchor="w")
        self.active_var = tk.BooleanVar(value=self.config_data.gesture_mode_active)
        ttk.Checkbutton(right, text="Gesture mode active", variable=self.active_var, command=self.toggle_active).pack(anchor="w", pady=8)
        self.activation_required_var = tk.BooleanVar(value=self.config_data.activation_required)
        ttk.Checkbutton(right, text="Require open-palm activation", variable=self.activation_required_var, command=self.save_ui_settings).pack(anchor="w")
        self.skeleton_var = tk.BooleanVar(value=self.config_data.show_skeleton)
        ttk.Checkbutton(right, text="Visual hand skeleton overlay", variable=self.skeleton_var, command=self.save_ui_settings).pack(anchor="w")
        self.sound_var = tk.BooleanVar(value=self.config_data.sound_feedback)
        ttk.Checkbutton(right, text="Sound feedback", variable=self.sound_var, command=self.save_ui_settings).pack(anchor="w")

        self._slider(right, "Gesture sensitivity", self.config_data.sensitivity, self.on_sensitivity)
        self._slider(right, "Confidence threshold", self.config_data.confidence_threshold, self.on_confidence)
        self._slider(right, "Cursor smoothing", self.config_data.cursor_smoothing, self.on_smoothing)

        ttk.Separator(right).pack(fill="x", pady=12)
        ttk.Label(right, text="Gesture Toggles", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.toggle_vars = {}
        for key in ["pointer", "left_click", "double_click", "right_click", "drag", "scroll", "volume", "play_pause", "mute", "swipe_left", "swipe_right"]:
            var = tk.BooleanVar(value=self.config_data.gesture_toggles.get(key, True))
            self.toggle_vars[key] = var
            ttk.Checkbutton(right, text=key.replace("_", " ").title(), variable=var, command=self.save_ui_settings).pack(anchor="w")

        ttk.Separator(right).pack(fill="x", pady=12)
        ttk.Button(right, text="Calibration Wizard", style="Accent.TButton", command=self.calibration).pack(fill="x", pady=4)
        ttk.Button(right, text="Record Custom Gesture", style="Accent.TButton", command=self.record_custom_gesture).pack(fill="x", pady=4)
        ttk.Button(right, text="Save Settings", command=self.save_ui_settings).pack(fill="x", pady=4)

        log_frame = ttk.Frame(right, style="Card.TFrame")
        log_frame.pack(fill="both", expand=True, pady=(12, 0))
        ttk.Label(log_frame, text="Gesture History", style="Card.TLabel", font=("Segoe UI", 12, "bold")).pack(anchor="w")
        self.log_box = tk.Listbox(log_frame, bg="#0b0d12", fg="#d9e2ec", height=8, borderwidth=0, highlightthickness=0)
        self.log_box.pack(fill="both", expand=True, pady=(6, 0))

    def _slider(self, parent, label, value, command):
        ttk.Label(parent, text=label, style="Card.TLabel").pack(anchor="w", pady=(12, 0))
        var = tk.DoubleVar(value=value)
        scale = ttk.Scale(parent, from_=0.05, to=0.95, value=value, command=lambda _v: command(var.get()))
        scale.configure(variable=var)
        scale.pack(fill="x")
        return var

    def start_engine(self):
        try:
            self.engine = GestureEngine(self.config_data, self.add_log)
            self.running = True
            self.status_var.set("Running — show open palm to activate")
            self.loop()
        except Exception as exc:
            self.logger.exception("Startup failed")
            messagebox.showerror("GestureOS startup failed", str(exc))
            self.status_var.set("Startup failed")

    def loop(self):
        if not self.running or self.engine is None:
            return
        try:
            frame, event, fps = self.engine.step()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img = Image.fromarray(frame)
            img.thumbnail((820, 560))
            self.photo = ImageTk.PhotoImage(img)
            self.preview.configure(image=self.photo)
            self.overlay_var.set(f"Gesture: {event.gesture.value} ({event.confidence:.2f}) | FPS: {fps:.1f} | Active: {self.config_data.gesture_mode_active}")
            self.active_var.set(self.config_data.gesture_mode_active)
        except Exception as exc:
            self.add_log(f"Error: {exc}")
        self.after(max(1, int(1000 / max(15, self.config_data.target_fps))), self.loop)

    def add_log(self, msg: str):
        stamp = time.strftime("%H:%M:%S")
        self.log_box.insert(0, f"{stamp}  {msg}")
        while self.log_box.size() > 120:
            self.log_box.delete(tk.END)

    def toggle_active(self):
        self.config_data.gesture_mode_active = bool(self.active_var.get())
        self.save_ui_settings()

    def save_ui_settings(self):
        self.config_data.activation_required = bool(self.activation_required_var.get())
        self.config_data.show_skeleton = bool(self.skeleton_var.get())
        self.config_data.sound_feedback = bool(self.sound_var.get())
        for k, v in self.toggle_vars.items():
            self.config_data.gesture_toggles[k] = bool(v.get())
        self.config_data.save()
        if self.engine:
            self.engine.update_config()
        self.add_log("Settings saved")

    def on_sensitivity(self, value):
        self.config_data.sensitivity = float(value)
        if self.engine:
            self.engine.update_config()

    def on_confidence(self, value):
        self.config_data.confidence_threshold = float(value)
        if self.engine:
            self.engine.update_config()

    def on_smoothing(self, value):
        self.config_data.cursor_smoothing = float(value)
        if self.engine:
            self.engine.update_config()

    def calibration(self):
        messagebox.showinfo(
            "Calibration Wizard",
            "1. Sit 45–75 cm from camera.\n2. Keep your hand inside the preview.\n3. Open palm for 1 second to activate.\n4. Adjust sensitivity if false triggers occur.\n5. Use good front lighting for best tracking.",
        )
        self.add_log("Calibration wizard completed")

    def record_custom_gesture(self):
        if not self.engine or not self.engine.last_event:
            return
        name = simpledialog.askstring("Custom Gesture", "Gesture name:")
        if not name:
            return
        action = simpledialog.askstring("Custom Gesture", "Action, e.g. hotkey:ctrl+l or press:space")
        if not action:
            return
        # Capture from current camera frame hand if possible.
        try:
            ok, frame, _ = self.engine.camera.read()
            hands = self.engine.tracker.process(frame) if ok else []
            if not hands:
                messagebox.showwarning("No hand", "No hand detected. Hold the gesture in view and try again.")
                return
            self.engine.trainer.add_sample(name, action, hands[0].landmarks)
            self.add_log(f"Recorded custom gesture '{name}' -> {action}")
        except Exception as exc:
            messagebox.showerror("Recording failed", str(exc))

    def on_close(self):
        self.running = False
        self.save_ui_settings()
        if self.engine:
            self.engine.close()
        self.destroy()


def main():
    app = GestureOSApp()
    app.mainloop()
