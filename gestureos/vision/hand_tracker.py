from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple
from urllib.request import urlopen

import cv2
import numpy as np

HAND_LANDMARK_COLOR = (0, 255, 0)
HAND_CONNECTION_COLOR = (255, 255, 255)
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
)


@dataclass
class HandResult:
    landmarks: List[Tuple[float, float, float]]
    handedness: str
    confidence: float
    bbox: Tuple[int, int, int, int]


def ensure_model(path: str) -> str:
    """Download the MediaPipe Tasks hand landmarker model once."""
    if os.path.isfile(path):
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"Downloading hand_landmarker model to {path} ...")
    with urlopen(MODEL_URL, timeout=30) as resp:
        data = resp.read()
    with open(path, "wb") as f:
        f.write(data)
    print("Download complete.")
    return path


class HandTracker:
    """
    MediaPipe Tasks HandLandmarker backend.

    This replaces the older `mediapipe.solutions.hands` API because many modern
    Windows installs expose Tasks correctly while `mediapipe.solutions` is
    missing or broken. The tracker uses VIDEO mode when available, which keeps
    temporal state and is faster/more stable than calling static image detection
    on every frame.
    """

    def __init__(self, max_hands: int = 1, min_detection_confidence: float = 0.55, min_tracking_confidence: float = 0.55):
        try:
            from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarker, HandLandmarkerOptions
            from mediapipe.tasks.python.vision.hand_landmarker import HandLandmarksConnections
            from mediapipe.tasks.python.core.base_options import BaseOptions
            from mediapipe.tasks.python.vision.core.image import Image, ImageFormat
            try:
                from mediapipe.tasks.python.vision.core.vision_task_running_mode import VisionTaskRunningMode
            except Exception:  # older mediapipe builds
                VisionTaskRunningMode = None
        except ImportError as exc:
            raise RuntimeError(
                "MediaPipe Tasks Hands is required but could not be imported.\n\n"
                "Try reinstalling inside your active virtual environment:\n"
                "  pip uninstall -y mediapipe\n"
                "  pip install mediapipe==0.10.14\n\n"
                "Also make sure your project does not contain a file or folder named 'mediapipe'."
            ) from exc

        self._Image = Image
        self._ImageFormat = ImageFormat
        self._connections = HandLandmarksConnections.HAND_CONNECTIONS
        self._video_mode = VisionTaskRunningMode is not None
        self._start_time = time.perf_counter()
        self._last_timestamp_ms = 0

        model_path = ensure_model(str(Path(__file__).parent / "hand_landmarker.task"))

        kwargs = dict(
            base_options=BaseOptions(model_asset_path=model_path),
            num_hands=max_hands,
            min_hand_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        # Some versions support presence confidence; add it if available.
        try:
            kwargs["min_hand_presence_confidence"] = float(min_tracking_confidence)
            if self._video_mode:
                kwargs["running_mode"] = VisionTaskRunningMode.VIDEO
            options = HandLandmarkerOptions(**kwargs)
        except TypeError:
            kwargs.pop("min_hand_presence_confidence", None)
            if self._video_mode:
                kwargs["running_mode"] = VisionTaskRunningMode.VIDEO
            options = HandLandmarkerOptions(**kwargs)

        self.hands = HandLandmarker.create_from_options(options)

    def _timestamp_ms(self) -> int:
        ts = int((time.perf_counter() - self._start_time) * 1000)
        if ts <= self._last_timestamp_ms:
            ts = self._last_timestamp_ms + 1
        self._last_timestamp_ms = ts
        return ts

    def process(self, frame_bgr: np.ndarray) -> List[HandResult]:
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        # Tasks Image does not copy in some versions; ensure contiguous memory.
        rgb = np.ascontiguousarray(rgb)
        image = self._Image(self._ImageFormat.SRGB, rgb)
        if self._video_mode and hasattr(self.hands, "detect_for_video"):
            results = self.hands.detect_for_video(image, self._timestamp_ms())
        else:
            results = self.hands.detect(image)

        hands: List[HandResult] = []
        if not results.hand_landmarks:
            return hands

        h, w = frame_bgr.shape[:2]
        for i, hand_lms in enumerate(results.hand_landmarks):
            lm = [(float(p.x), float(p.y), float(p.z)) for p in hand_lms]
            xs, ys = [p[0] for p in lm], [p[1] for p in lm]
            bbox = (
                int(max(0.0, min(xs)) * w),
                int(max(0.0, min(ys)) * h),
                int(min(1.0, max(xs)) * w),
                int(min(1.0, max(ys)) * h),
            )
            label = "Unknown"
            score = 1.0
            if results.handedness and i < len(results.handedness):
                cls = results.handedness[i][0]
                label, score = cls.category_name, cls.score
            hands.append(HandResult(lm, label, float(score), bbox))
        return hands

    def draw(self, frame_bgr: np.ndarray, hands: List[HandResult]) -> np.ndarray:
        h, w = frame_bgr.shape[:2]
        for hand in hands:
            pts = [(int(p[0] * w), int(p[1] * h)) for p in hand.landmarks]
            for conn in self._connections:
                # Different MediaPipe versions expose either tuple-like or object-like connections.
                start = getattr(conn, "start", conn[0] if isinstance(conn, tuple) else 0)
                end = getattr(conn, "end", conn[1] if isinstance(conn, tuple) else 0)
                if start < len(pts) and end < len(pts):
                    cv2.line(frame_bgr, pts[start], pts[end], HAND_CONNECTION_COLOR, 2, cv2.LINE_AA)
            for idx, (x, y) in enumerate(pts):
                radius = 6 if idx in (4, 8, 12) else 4
                cv2.circle(frame_bgr, (x, y), radius, HAND_LANDMARK_COLOR, -1, cv2.LINE_AA)
            x0, y0, x1, y1 = hand.bbox
            cv2.rectangle(frame_bgr, (x0, y0), (x1, y1), (80, 180, 255), 2, cv2.LINE_AA)
            cv2.putText(frame_bgr, f"{hand.handedness} {hand.confidence:.2f}", (x0, max(20, y0 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (80, 180, 255), 1, cv2.LINE_AA)
        return frame_bgr
