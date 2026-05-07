# Unity VR Gaze Visualizer and Calibration Script

This Unity C# script visualizes a 3D gaze ray produced by my external Python eye tracker. The Python tracker continuously writes a gaze origin and gaze direction to a text file, and this Unity script reads that file in real time to display the gaze direction inside a VR headset.

To help support this software and other open-source projects, please consider subscribing to my YouTube channel: https://www.youtube.com/@jeoresearch, or joining for $1 per month: https://www.youtube.com/@jeoresearch/join.

## Features

- Reads live gaze data from a text file written by Python
- Parses six values:
  - `origin_x, origin_y, origin_z`
  - `direction_x, direction_y, direction_z`
- Displays the gaze direction as a small sphere in front of the headset
- Parents the live gaze sphere to the headset transform
- Supports a multi-point calibration via the "c" key
- Supports vertical up/down calibration using ±10 degree targets
- Requires no prefabs; all objects are created by script

## Expected Gaze File Format

The script expects a text file containing six comma-separated or whitespace-separated floating point values:

```text
origin_x,origin_y,origin_z,direction_x,direction_y,direction_z
