# GestureOS

GestureOS is a production-oriented Python desktop application for real-time webcam hand-gesture control. It combines OpenCV, MediaPipe Hands, PyAutoGUI, NumPy smoothing/Kalman filtering, optional Windows `pycaw` system-volume control, a safety layer, and a polished Tkinter dashboard.

## Features

- Real-time single-hand tracking, optional dual-hand mode in config.
- Stable cursor movement with Kalman + adaptive smoothing.
- Activation safety: open palm activates, closed palm deactivates.
- Debounce + confidence threshold + per-action cooldowns.
- Media controls: pinch volume, fist/closed palm play-pause, thumb-up mute.
- Productivity controls: swipes for next/previous, two-finger vertical scroll.
- Mouse controls: index pointer, pinch click, double pinch double-click, three-finger right-click, pinch-hold drag.
- Modern dark dashboard with webcam preview, sliders, toggles, gesture history, skeleton overlay.
- Custom gesture trainer with persisted profiles.
- Startup calibration guidance and low-light enhancement.

## Folder structure

```text
GestureOS/
├─ main.py
├─ requirements.txt
├─ README.md
├─ config.json                         # created automatically from defaults
├─ profiles/custom_gestures.json        # created when training custom gestures
└─ gestureos/
   ├─ app_core.py                       # orchestration engine
   ├─ vision/                           # camera, MediaPipe tracker, filters
   ├─ gestures/                         # recognizer, safety, trainer, gesture types
   ├─ actions/                          # mouse, keyboard, media, volume actions
   ├─ ui/                               # Tkinter dashboard
   ├─ settings/                         # configuration dataclass/defaults
   └─ utils/                            # logging, math, sound, monitor helpers
```

## Setup

> Recommended: Python 3.10–3.13. MediaPipe 0.10.14+ supports Python 3.13.

```bash
cd GestureOS
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
python main.py
```

### Windows permissions

- Run normally first. If global hotkeys or media keys are blocked, run your terminal as Administrator.
- `pycaw` is only installed on Windows and gives precise system-volume control. Other OSes fall back to media keys.

### macOS permissions

Grant Accessibility and Camera permissions to Terminal/Python in System Settings.

## Gesture guide

| Gesture | Action |
|---|---|
| Open palm | Activate gesture mode |
| Closed palm/fist | Deactivate gesture mode; play/pause while active after debounce |
| Thumb up | Mute/unmute |
| Index finger | Move mouse cursor |
| Thumb + index pinch | Left click; if hand is near left edge, controls volume by vertical position |
| Double pinch | Double click |
| Thumb + index + middle/ring pinch | Right click |
| Pinch hold | Drag and drop |
| Two fingers vertical movement | Scroll |
| Swipe right/left | Next/previous slide or browser-like navigation |

## Custom gesture actions

Use **Record Custom Gesture** in the UI and assign an action string:

- `hotkey:ctrl+l`
- `hotkey:ctrl+tab`
- `press:space`
- `press:f5`

Samples are stored in `profiles/custom_gestures.json`.

## Tuning for production use

1. Use front lighting; avoid strong backlight.
2. Start with sensitivity `0.70–0.80` and confidence threshold `0.72`.
3. Increase smoothing if cursor jitters; decrease smoothing if cursor feels delayed.
4. Keep activation mode enabled for safety.
5. Disable gestures you do not use in the dashboard.
6. For presentations, keep PowerPoint/Google Slides focused and use swipe left/right.

## Implementation notes

- `GestureEngine.step()` is the central low-latency loop: camera → low-light enhancement → MediaPipe → gesture recognition → safety gate → action execution → overlay frame.
- `GestureRecognizer` uses MediaPipe landmarks, finger-state heuristics, pinch distances, and movement history.
- `CooldownGate` and `GestureDebouncer` prevent accidental rapid triggers.
- Cursor motion is filtered by `AdaptiveSmoother`, combining a constant-velocity Kalman filter with EMA.
- The UI persists settings to `config.json`.

## Troubleshooting

- **Camera cannot open**: set `camera_index` in `config.json` to `1` or `2`.
- **Low FPS**: reduce `camera_width`/`camera_height` to `640x480`, disable skeleton overlay, or set `target_fps` to `30`.
- **Clicks trigger unexpectedly**: increase confidence threshold or disable click gestures until calibrated.
- **No volume control**: on non-Windows systems GestureOS uses keyboard volume keys as fallback.

## Run without UI for testing

The core engine is reusable from scripts:

```python
from gestureos.settings.config import AppConfig
from gestureos.app_core import GestureEngine

engine = GestureEngine(AppConfig.load())
while True:
    frame, event, fps = engine.step()
    print(event, fps)
```

## Advanced PyQt6 dashboard

GestureOS now includes a professional PyQt6 dashboard in addition to the legacy Tkinter UI.

Run the advanced dashboard:

```bash
python main_qt.py
```

Run the legacy dashboard:

```bash
python main.py

### PyQt6 dashboard features

- Large low-latency camera preview with GestureOS HUD.
- Live FPS, gesture, confidence, and hand-lock metric cards.
- Real-time FPS and confidence sparklines.
- Professional dark UI with tabs.
- Safety controls: activation mode, skeleton overlay, sound, low-light enhancement.
- Live tuning sliders for sensitivity, confidence, and cursor smoothing.
- Gesture enable/disable table.
- Per-gesture custom action override editor.
- Custom gesture trainer dialog.
- Calibration wizard for camera resolution, FPS, thresholds, and smoothing.

### Recommended PyQt install command

```bash
pip install PyQt6
```

If PyQt6 installation fails, you can still run:

```bash
python main_tk.py
```

## Cursor quality v2 notes

GestureOS now uses the MediaPipe Tasks `HandLandmarker` backend instead of the older `mediapipe.solutions` backend. The model is downloaded automatically to:

```text
gestureos/vision/hand_landmarker.task
```

The cursor now defaults to touchpad-style relative movement internally. This is more accurate than absolute webcam-to-screen mapping because consumer webcams produce jittery fingertip coordinates and perspective distortion near frame edges.

For the best cursor feel:

- Use 60 FPS if your camera supports it.
- Use 640x480 or 960x540 before trying 1280x720.
- Keep only the index finger raised for pointer mode.
- Keep your hand 45–75 cm from the webcam.
- Use strong front lighting.
- Set cursor smoothing between 0.45 and 0.65.
- Lower smoothing if it feels laggy; raise smoothing if it shakes.

## Click + draw interaction upgrade

GestureOS now supports touchpad-style click behavior while moving the pointer:

- **Move cursor:** raise index finger and move your hand.
- **Left click:** touch thumb to index finger, then release quickly.
- **Move while thumb is touching:** cursor continues moving while pinched.
- **Drag:** keep thumb touching index finger for a longer hold, then move.
- **Release drag:** separate thumb and index.
- **Right click:** three-finger pinch.

This makes the interaction closer to a laptop touchpad: index finger moves, thumb contact acts as click/press.

## Whiteboard + handwriting recognition

The PyQt6 dashboard includes a **Whiteboard** tab.

How to use it:

1. Open the **Whiteboard** tab.
2. GestureOS automatically pauses OS mouse/keyboard actions while this tab is active.
3. Move the canvas cursor with your index finger.
4. Touch thumb to index and move to draw.
5. Release thumb to stop drawing.
6. Click **Recognize Number / Character / Word**.

The recognizer is a lightweight template OCR engine. It works best for clearly drawn single digits and uppercase letters. It can also make a basic connected-component word/sequence guess, but it is not a full neural OCR model.