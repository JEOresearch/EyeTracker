**Eye Tracking and Monitor Control**

This is a 3D eye tracker that works with your webcam. This is only in the protoype stage, and I still need to optimize heavily and implement a multi-point calibration. 

To help support this software and other open-source projects, please consider subscribing to my YouTube channel:
https://www.youtube.com/@jeoresearch

Or join for $1 per month:
https://www.youtube.com/@jeoresearch/join

Recommended webcam (used in testing, ~$37):
https://amzn.to/43of401

## Project Structure

The codebase has been refactored from a single file (`MonitorTracking.py`) into a modular Python package under `src/monitor_tracking/` for better organization, testing, and extensibility:

- **config/**: Settings and initial state (e.g., `Settings` dataclass for globals like filter_length, screen dimensions).
- **core/**: logic
  - `detection.py`: MediaPipe face mesh, landmark extraction, head pose via PCA on nose landmarks.
  - `gaze.py`: Eye sphere locking, combined gaze smoothing, screen coordinate mapping (`GazeTracker` class).
  - `monitor.py`: 3D plane creation, gaze intersection, markers (`MonitorPlane` class).
- **utils/**: Reusable helpers
  - `geometry.py`: Math functions (rotations, normalization, scale).
  - `visualization.py`: Drawing (gaze rays, cubes, 3D orbit debug view).
  - `controls.py`: Keyboard input, mouse threading, file I/O.
- **main.py**: Entry point orchestrating detection, gaze, monitor, and loop.
- **__init__.py** files: Package exports for easy imports (e.g., `from monitor_tracking import main`).

Backward compatibility: Original `MonitorTracking.py` now imports and runs the new main.

## Setup

1. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

2. (Optional) Use uv for faster installs:
   ```
   uv sync
   ```

## Usage

Connect a webcam (default index 0; change in main.py if needed).

Run the tracker:
```
python -m src.monitor_tracking
```
or
```
uv run -m src.monitor_tracking
```
or
```
python MonitorTracking.py  # Legacy entry point
```

Windows will open:
- **Integrated Eye Tracking**: Live video with landmarks, gaze rays, calibration overlays.
- **Head/Eye Debug**: 3D orbit view with head, gaze, virtual monitor, markers.

### Controls
- **c** = Calibrate eye spheres (look straight ahead at screen center).
- **s** = Calibrate screen mapping (look at center after eye calib).
- **F7** = Toggle mouse control (disabled by default).
- **j/l** = Orbit yaw left/right in debug view.
- **i/k** = Orbit pitch up/down.
- **[ / ]** = Zoom out/in.
- **r** = Reset orbit view.
- **x** = Add green marker where gaze hits monitor (after calib).
- **q** = Quit.

#### Notes
- Make sure you look at screen center when pressing c. The debug view won't render until you do this. 
- Markers (x key) allow quick tests of where the system thinks you are looking.

### Features
- Gaze smoothing with deque buffer.
- Scale-aware eye spheres (adjusts with head distance).
- 3D monitor plane at 50cm, 60x40cm.
- Screen position output to `./screen_position.txt`.
- Mouse control via pyautogui.

## Development

- **Tuning**: Edit `config/settings.py` (e.g., `FILTER_LENGTH = 10`).
- **Testing**: Run unit tests:
  ```
  python -m pytest tests/
  ```
- **Extending**: Add to core modules (e.g., multi-face in detection.py). Use classes for state.

## Troubleshooting
- Jittery gaze: Increase `filter_length` in settings.
- If the wrong camera opens: Change `cv2.VideoCapture(0)` to another index in main.py.
- Import errors: Run from project root with `python -m src.monitor_tracking`.
- Debug view blank: Complete 'c' calibration first.
- For better accuracy, use consistent lighting and this webcam: https://amzn.to/43of401.

