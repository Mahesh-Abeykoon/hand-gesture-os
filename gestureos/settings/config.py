from __future__ import annotations

import json
import shutil
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Any

ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = ROOT / "config.json"
DEFAULT_CONFIG_PATH = Path(__file__).with_name("default_config.json")


@dataclass
class AppConfig:
    camera_index: int = 0
    camera_width: int = 1280
    camera_height: int = 720
    target_fps: int = 60
    dual_hand_mode: bool = False
    activation_required: bool = True
    gesture_mode_active: bool = False
    sensitivity: float = 0.72
    confidence_threshold: float = 0.72
    cursor_smoothing: float = 0.32
    low_light_enhancement: bool = True
    sound_feedback: bool = True
    dark_mode: bool = True
    show_skeleton: bool = True
    multi_monitor: bool = True
    cooldowns: Dict[str, float] = field(default_factory=dict)
    gesture_toggles: Dict[str, bool] = field(default_factory=dict)
    custom_mappings: Dict[str, str] = field(default_factory=dict)
    profiles_path: str = "profiles/custom_gestures.json"

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "AppConfig":
        if not path.exists():
            shutil.copyfile(DEFAULT_CONFIG_PATH, path)
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self), f, indent=2)

    def update(self, **kwargs: Any) -> None:
        for k, v in kwargs.items():
            if hasattr(self, k):
                setattr(self, k, v)
