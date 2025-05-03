This project provides a real-time 3D eye tracking system using a webcam, OpenCV, and optional OpenGL visualization. It detects the pupil in each frame, fits an ellipse to estimate eye orientation, and projects a 3D gaze direction vector from the user's eye center through the pupil.

Requirements
Python 3 or above
OpenCV (opencv-python)
NumPy
(Optional) PyOpenGL and gl_sphere.py for 3D visualization

In the GUI that runs with the application, you’ll be prompted to select a webcam or video file. The main display shows the detected pupil and 3D direction. If gl_sphere is available, a 3D model will be rendered in a separate OpenGL window.

Output
gaze_vector.txt: Continuously updated with the current origin and direction vector. You can read this into Unity using the GazeFollower.cs script. It just reads the file constantly and updates the position and direction of the object it's attached to. 

Gaze vectors are also shown in the bottom-left corner of the OpenCV display.

Notes
If gl_sphere.py is not found, the script still runs without OpenGL output.

Press Q to quit or Space to pause a frame.
