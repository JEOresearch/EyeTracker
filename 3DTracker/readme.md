This project provides a real-time 3D eye tracking system using a near-eye infrared camera, OpenCV, and optional OpenGL visualization. It detects the pupil in each frame, fits an ellipse to estimate eye orientation, and projects a 3D gaze direction vector from the user's eye center through the pupil.

The only file you need to do the eyetracking is Orlosky3DEyeTracker.py, though gl_sphere.py can be included to provide a render of the 3D sphere. In the GUI that runs with the application, you’ll be prompted to select a camera stream or video file. The main display shows the detected pupil and 3D origin and direction vector. If gl_sphere is available, a 3D model will be rendered in a separate OpenGL window. A video with DIY tracking glasses and sample output can be found here: https://youtu.be/zuoOvywtwtA

An inexpensive eye tracking camera and extension cables for testing can be found here: 
- GC0308 Eye Tracking Camera ($17): https://amzn.to/41x8p2W
- USB extension cables ($10): https://amzn.to/43SznVl

To help support this software and other open-source projects, please consider subscribing to my YouTube channel: https://www.youtube.com/@jeoresearch, or joining for $1 per month: https://www.youtube.com/@jeoresearch/join. 

Requirements
- Python 3 or above
- OpenCV (opencv-python)
- NumPy
- (Optional) PyOpenGL and gl_sphere.py for 3D visualization

To install dependencies via terminal: 
```bash
pip install opencv-python numpy tkinter
```

Output
- gaze_vector.txt: Continuously updated with the current origin and direction vector. You can read this into Unity using the GazeFollower.cs script. It just reads the file constantly and updates the position and direction of the object it's attached to. 
- Gaze vectors are also shown in the bottom-left corner of the OpenCV display.

Notes
- Press Q to quit or Space to pause a frame.
