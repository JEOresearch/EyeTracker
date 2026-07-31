HEAD TRACKING MOUSE CONTROL
====================================

This script uses your webcam and MediaPipe's Face Mesh to estimate your head pose and control the mouse cursor using head orientation.

To help support this software and other open-source projects, please consider subscribing to my YouTube channel: https://www.youtube.com/@jeoresearch, or joining for $1 per month: https://www.youtube.com/@jeoresearch/join. 

The webcam used for tracking can be found here ($35): https://amzn.to/43of401

Output should look like this video: https://youtu.be/hImmJDTgXjw

REQUIREMENTS
------------
- Python 3.x
- OpenCV
- MediaPipe (`>=0.10.5,<0.10.15` — needed for the `solutions.face_mesh` API used here)
- NumPy
- PyAutoGUI
- PyQt5 (optional, for CursorCircle.py)

To install the required packages:
```bash
    pip install -r requirements.txt
```

USAGE
-----
1. Ensure a webcam is connected. The script defaults to camera index 1. Change to 0 if needed.
2. Run the script:
```bash
    python MonitorTracking.py
```
3. Run the cursor visualization (optional) in a separate terminal:
```bash
    python CursorCircle.py
```
4. Two windows will appear:
    - "Head-Aligned Cube": shows your face with overlayed bounding box and tracking vector.
    - "Facial Landmarks": shows the detected facial landmarks.

FEATURES
--------
- Real-time head pose estimation.
- Virtual wireframe cube aligned to your head orientation.
- Mouse control mapped to yaw and pitch angle from your face.
- Press **m** (or **f** / **7**) to toggle mouse control on/off.
- Press 'c' to calibrate when looking straight at your monitor center.
- Press 'q' to quit the application.

Note: Hotkeys use OpenCV window focus (`cv2.waitKey`), so no root/`keyboard` package is required on Linux or macOS.

NOTES
-----
- Calibration centers the screen mapping to your current head pose.
- The program calculates yaw and pitch from the forward-facing vector derived from facial landmarks.
- Movement is smoothed using a running average of recent pose estimates.

TROUBLESHOOTING
---------------
- If tracking is jittery, increase the `filter_length` value.
- If the wrong camera is used, change `cv2.VideoCapture(1)` to another index.
