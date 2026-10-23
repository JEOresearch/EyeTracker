# HeadTracker Class

An object-oriented Python interface for head tracking and optional mouse control using
MediaPipe and OpenCV. This refactor wraps the original procedural implementation from
[JEOresearch/EyeTracker](https://github.com/JEOresearch/EyeTracker) into a reusable
`HeadTracker` class that u can import into your project.

## Requirements & install

Primary dependencies are included in `requirements.txt`. From the folder run:

```bash
pip install -r requirements.txt
```

## Setup

1. Install the required dependencies:

```bash
pip install opencv-python mediapipe numpy pyautogui keyboard
```

2. Setup:

   - Choose how the webcam output looks

   ```python
        view = img # default webcam view
        view = landmarks_frame # EDM vibes
        view = main_frame # cool 3d cuboid wiyth dots and lines on your face
   ```

3. Run the application:

   ```
   python main.py
   ```

4. Controls:

   - press `c` to calibrate
   - press `q` to exit
   - press `t` to enable mouse control

5. Copy or import the `HeadTracker` class from this folder.

Run the included test script to try it out:

```bash
python head_tracker_test.py
```

Controls inside the display window:

- press `c` → calibrate (center your head and look at screen center before pressing)
- press `t` → toggle mouse control ON/OFF
- press `q` → quit

## Config

- Sensitivity: `set_sensitivity(yaw_degrees, pitch_degrees)` controls how much angular
  range maps to the full screen. Smaller values make the cursor more sensitive.
- Camera index: set `camera_index` to 0/1 depending on your system camera.
- Smoothing: `filter_length` controls the length of the averaging deque used to
  smooth the computed ray origin/direction (default 8).

## Troubleshooting

- If no face is detected, the windows will show empty landmark overlays — ensure good
  lighting and that the face is in frame.
- For OpenCV camera errors, try different `camera_index` values or check that your
  camera is not used by another application.

## Acknowledgements

Original project: [JEOresearch/HeadTracker](https://github.com/JEOresearch/EyeTracker/tree/main/HeadTracker)

## Current Implementations

[[github] Phone Control via Head Motion](https://github.com/findpiyush/handsfree-medassist)
