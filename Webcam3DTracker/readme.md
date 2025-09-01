Eye Tracking and Monitor Control

====================================

This is a 3D eye tracker that works with your webcam. This is only in the protoype stage, and I still need to optimize heavily and implement a multi-point calibration. 

To help support this software and other open-source projects, please consider subscribing to my YouTube channel:
https://www.youtube.com/@jeoresearch

Or join for $1 per month:
https://www.youtube.com/@jeoresearch/join

Recommended webcam (used in testing, ~$37):
https://amzn.to/43of401


Usage

Connect a webcam. By default, camera index 0 is used. Change in code if needed.

Run the tracker:

python MonitorTracking.py



Windows will open showing:

Integrated Eye Tracking: live video with eye landmarks, gaze rays, and calibration overlays.

Head/Eye Debug: a 3D orbit-view with the head, gaze vectors, and the calibrated virtual monitor.


Interactive controls:

c = calibrate (screen center)

F7 = toggle mouse control (disabled by default)

j/l = orbit yaw left/right

i/k = orbit pitch up/down

[ / ] = zoom orbit view out/in

r = reset orbit view

x = stamp a green marker on the monitor where your gaze hits (2% width circle)

q = quit

Notes

Calibration aligns the screen plane with your current gaze at center.

Gaze mapping uses both eye positions and iris centers, smoothed by a rolling buffer.

The debug view lets you orbit around the virtual setup to confirm calibration accuracy.

Markers (x key) allow quick tests of where the system thinks you are looking.

Troubleshooting

If gaze appears jittery, increase filter_length.

If the wrong camera opens, change cv2.VideoCapture(0) to another index.

For better accuracy, use consistent lighting and a high-resolution webcam.
