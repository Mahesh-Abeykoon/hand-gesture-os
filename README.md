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

> Recommended: Python 3.10 or 3.11. MediaPipe support can lag on the newest Python versions.

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
python main_tk.py
```

`python main.py` attempts to start the PyQt6 dashboard first and falls back to Tkinter if PyQt6 is unavailable.

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


## Advanced OCR / handwriting recognition

The Whiteboard tab now uses a multi-backend OCR engine:

1. **EasyOCR** if installed — local neural OCR, good for block handwriting and text.
2. **Tesseract** via `pytesseract` if installed — local OCR engine.
3. **OCR.space online API** if `OCR_SPACE_API_KEY` is set.
4. **OpenCV template fallback** — always available for digits and uppercase A-Z.

### Install stronger OCR

Base app:

```bash
pip install -r requirements.txt
```

Advanced OCR add-ons:

```bash
pip install -r requirements-ocr.txt
```

EasyOCR may download its recognition model on first use and can take time to start.

### Optional Tesseract setup on Windows

`pytesseract` is only the Python wrapper. Install the Tesseract executable too:

- Download installer from: https://github.com/UB-Mannheim/tesseract/wiki
- Add Tesseract to PATH, or set its location in Python if you customize the app.

### Optional online OCR.space

If you want online OCR fallback, set an API key before starting GestureOS:

```powershell
$env:OCR_SPACE_API_KEY="your_api_key_here"
python main_qt.py
```

If no optional OCR packages are installed, GestureOS still uses the built-in OpenCV template recognizer.

### Whiteboard recognition tips

- Draw large, clear symbols.
- Digits and uppercase block letters are most reliable.
- For words, leave space between letters.
- Press **Recognize Number / Character / Word** after finishing the drawing.
- Use Clear before drawing a new symbol or word.

## Precision pointer / pen engine upgrade

GestureOS now has a high-priority raw pointer engine that bypasses symbolic gesture labels for mouse and drawing. This is important because symbolic labels can flicker while the hand moves.

Behavior:

- Index finger moves cursor/pen.
- Thumb touching index immediately presses mouse/pen down.
- Releasing thumb releases mouse/pen.
- Quick press/release becomes a normal click.
- Press + move becomes drag/draw/select.
- Short MediaPipe dropouts are bridged using optical flow so strokes do not break as easily.

This engine is used for both the desktop cursor and the Whiteboard tab.

## OpenAI Vision OCR option

For the strongest handwritten-note recognition, set an OpenAI API key before launching GestureOS:

```powershell
$env:OPENAI_API_KEY="your_api_key_here"
$env:OPENAI_VISION_MODEL="gpt-4o-mini"
python main_qt.py
```

When enabled, the Whiteboard recognizer tries OpenAI Vision in addition to EasyOCR, Tesseract, OCR.space, and the built-in template recognizer. This is the most accurate option for messy handwritten notes and words.


## Smart anti-jump cursor upgrade

GestureOS now uses a more conservative smart cursor pipeline:

- raw landmark jump rejection
- sudden movement clamping
- warm-up frames after hand reacquisition
- lower acceleration defaults
- vector-level max speed clamp
- fractional pixel accumulation for smoother small movement

This prevents the cursor from quickly jumping here and there when MediaPipe briefly swaps or misplaces the fingertip landmark.

Recommended for maximum stability:

```json
"cursor_smoothing": 0.65,
"confidence_threshold": 0.60,
"camera_width": 640,
"camera_height": 480,
"target_fps": 60
```

If the cursor feels too slow, reduce cursor smoothing gradually to `0.50`.

## Full Screen Draw mode

The Whiteboard tab now includes **Open Full Screen Draw**.

When opened:

- the app switches to draw-lock mode
- desktop cursor/actions are blocked
- the drawing canvas fills the screen
- a control panel appears on the right side
- index finger moves the pen
- thumb + index draws
- releasing thumb stops drawing
- recognition is available from the right-side panel

This is intended for presentations, teaching, note writing, and full-screen gesture drawing.

## Natural handwriting stroke smoothing

The drawing canvas now uses a paper-like stroke engine instead of direct frame-to-frame lines:

- hidden pen cursor by default, so the cursor indicator does not disturb the drawing
- conservative canvas motion gain
- per-frame speed limit
- low-pass smoothing for pen motion
- no jump line when thumb first touches index
- quadratic Bézier interpolation for smooth handwriting curves
- full-screen canvas fills the drawing area while the right-side control panel remains available

In Full Screen Draw mode, the user sees only the white board and the right-side tools; the moving pointer indicator is hidden.

## Draw mode behavior change: index-only drawing

Whiteboard and Full Screen Draw now intentionally ignore pinch for drawing.

Reason: thumb-index pinch causes landmark occlusion and fingertip jitter, which created false strokes and broken lines. In drawing mode GestureOS now behaves like this:

- Open Whiteboard / Full Screen Draw = desktop actions locked.
- Index fingertip = pen.
- Index movement draws directly on the board.
- No visible pen cursor is shown.
- Pinch is reserved for desktop click mode only, not drawing mode.
- Move hand out of camera view to stop drawing.

The board now uses absolute camera-to-board mapping rather than mouse-style relative deltas, which makes handwriting easier and less broken.

An **Instructions** tab was also added to the PyQt dashboard.

## Precision click and drawing redesign

GestureOS now separates **desktop clicking** from **drawing** more carefully.

### Desktop click mode

- Index finger moves cursor.
- Thumb + index no longer instantly drags/clicks.
- When thumb/index touch, the cursor freezes on the target.
- Quick stable release triggers one click at the frozen position.
- Hold or deliberate movement becomes drag.

This prevents tiny hand shake during a pinch from moving the cursor away from the intended button.

### Drawing mode tools

Whiteboard and Full Screen Draw now include tools:

- **Pen**: index finger draws.
- **Hover / Pause**: index finger moves without drawing.
- **Eraser**: index finger erases.
- **Clear**: clears the board.
- **Recognize Notes**: OCR recognition.

Full Screen Draw now also shows a right-side live camera preview, so the user can see the tracked hand while drawing. The drawing cursor indicator remains hidden to avoid disturbing the handwriting.
