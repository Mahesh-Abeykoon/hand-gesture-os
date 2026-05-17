from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np


@dataclass
class CustomGestureProfile:
    name: str
    action: str
    samples: List[List[float]]


class GestureTrainer:
    """Template-based custom gesture recorder using normalized landmark vectors."""

    def __init__(self, path: str = "profiles/custom_gestures.json"):
        self.path = Path(path)
        self.profiles: Dict[str, CustomGestureProfile] = {}
        self.load()

    def feature_vector(self, landmarks: List[Tuple[float, float, float]]) -> List[float]:
        arr = np.array(landmarks, dtype=np.float32)
        wrist = arr[0, :2].copy()
        arr[:, :2] -= wrist
        scale = np.linalg.norm(arr[9, :2]) + 1e-6
        arr[:, :2] /= scale
        return arr[:, :2].flatten().tolist()

    def add_sample(self, name: str, action: str, landmarks: List[Tuple[float, float, float]]) -> None:
        profile = self.profiles.get(name, CustomGestureProfile(name, action, []))
        profile.action = action
        profile.samples.append(self.feature_vector(landmarks))
        self.profiles[name] = profile
        self.save()

    def predict(self, landmarks: List[Tuple[float, float, float]], max_distance: float = 2.25):
        if not self.profiles:
            return None
        x = np.array(self.feature_vector(landmarks), dtype=np.float32)
        best = None
        for profile in self.profiles.values():
            if not profile.samples:
                continue
            samples = np.array(profile.samples, dtype=np.float32)
            d = np.linalg.norm(samples - x, axis=1).min()
            if best is None or d < best[0]:
                best = (float(d), profile)
        if best and best[0] <= max_distance:
            return best[1]
        return None

    def load(self) -> None:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        self.profiles = {k: CustomGestureProfile(**v) for k, v in raw.items()}

    def save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({k: asdict(v) for k, v in self.profiles.items()}, f, indent=2)
