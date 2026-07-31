This project provides a real-time 3D eye tracking system using a near-eye infrared camera, OpenCV, and optional OpenGL visualization. It detects the pupil in each frame, fits an ellipse to estimate eye orientation, and projects a 3D gaze direction vector from the user's eye center through the pupil.

Use Orlosky3DEyeTracker.py for single-eye tracking or Orlosky3DEyeTrackerStereo.py for simultaneous left- and right-eye tracking. The optional gl_sphere.py module provides a render of the 3D sphere. In the GUI that runs with the application, you’ll be prompted to select a camera stream or video file. The main display shows the detected pupil and 3D origin and direction vector. If gl_sphere is available, a 3D model will be rendered in a separate OpenGL window. A video with DIY tracking glasses and sample output can be found here: https://youtu.be/zuoOvywtwtA

An inexpensive eye tracking camera and extension cables for testing can be found here: 
- GC0308 Eye Tracking Camera ($17): https://amzn.to/41x8p2W
- USB extension cables ($10): https://amzn.to/43SznVl

To help support this software and other open-source projects, please consider subscribing to my YouTube channel: https://www.youtube.com/@jeoresearch, or joining for $1 per month: https://www.youtube.com/@jeoresearch/join. 

Requirements
- Python 3 or above
- OpenCV (opencv-python)
- NumPy
- Tkinter (usually included with Python; on Linux install `python3-tk` if needed)
- (Optional) PyOpenGL and gl_sphere.py for 3D visualization

To install dependencies via terminal: 
```bash
pip install opencv-python numpy
```

Camera backends
- On Windows, the trackers try MSMF then DirectShow.
- On macOS/Linux, they use OpenCV's default (`CAP_ANY`) backend.
- Stereo tracking supports independent left/right camera selection and flip.

Stereo Tracking

Run the stereo tracker from the project directory:

```bash
python Orlosky3DEyeTrackerStereo.py
```

The input window contains separate controls for the left- and right-eye cameras:

1. Select the camera index for each eye. When two cameras are available, the initial selections default to the first camera for the left eye and the second camera for the right eye.
2. Use **Start Left Camera** and **Start Right Camera** to begin tracking. Either eye can also be started or stopped independently.
3. Use each camera's **flip image** checkbox if its image is mounted upside down.
4. Move each eye through several gaze directions while the tracker estimates the eye sphere. Press **F**, or select **fixed eye sphere**, once the estimated sphere is stable. Press F again or clear the checkbox to resume automatic adjustment.

The stereo camera windows support these keys:

- **Q**: stop both cameras.
- **Space**: pause or resume both camera streams.
- **F**: toggle automatic eye-sphere adjustment.
- **E**: toggle capture of persistent pupil ellipses for inspection.
- **C**: clear the captured ellipses.

The **Browse Video** option processes one recorded video with the currently active eye state; simultaneous stereo tracking is intended for two live camera inputs.

Stereo Output

Orlosky3DEyeTrackerStereo.py continuously writes one comma-separated line containing 12 values to `gaze_vector.txt`, in this order:

```text
left_origin_x,left_origin_y,left_origin_z,left_direction_x,left_direction_y,left_direction_z,right_origin_x,right_origin_y,right_origin_z,right_direction_x,right_direction_y,right_direction_z
```

An eye that has not produced a valid gaze result remains represented by six zeroes. Use `EyeTrackerStereo.cs` in Unity to read this 12-value format; update its `GazeFilePath` constant to the absolute location of `gaze_vector.txt` on your system.

Output
- gaze_vector.txt: Continuously updated with the current origin and direction vector. You can read this into Unity using the GazeFollower.cs script. It just reads the file constantly and updates the position and direction of the object it's attached to. 
- Gaze vectors are also shown in the bottom-left corner of the OpenCV display.

Notes
- Press Q to quit or Space to pause a frame.
