"""
A Python module for eye/gaze tracking using MediaPipe face mesh and OpenCV.
Built on the foundation put by MonitorTracking.py 
This implementation refactors the original procedural code into a reusable class structure so u can import it in ur project using simple functions.

Original work by JEOresearch: https://github.com/JEOresearch/EyeTracker/tree/main/HeadTracker
OOP refactoring by: findpiyush

Example:
    Basic usage:
        #   tracker = HeadTracker(camera_index=1)
        #   tracker.run_loop()
    
    Advanced usage:
        #   tracker = HeadTracker()
        #   tracker.start()
        #   while True:
                frame, landmarks, clean = tracker.process_frame()
                Process frames...
        #   tracker.stop()
"""


import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pyautogui
import math
import threading
import time
import keyboard

class HeadTracker:
    # Face mesh landmark indices
    FACE_OUTLINE_INDICES = [
        10, 338, 297, 332, 284, 251, 389, 356,
        454, 323, 361, 288, 397, 365, 379, 378,
        400, 377, 152, 148, 176, 149, 150, 136,
        172, 58, 132, 93, 234, 127, 162, 21,
        54, 103, 67, 109
    ]
    
    # Landmark indices for bounding box
    LANDMARKS = {
        "left": 234,
        "right": 454,
        "top": 10,
        "bottom": 152,
        "front": 1,
    }
    
    def __init__(self, camera_index=1, filter_length=8):
        # Initialize screen dimensions
        self.MONITOR_WIDTH, self.MONITOR_HEIGHT = pyautogui.size()
        self.CENTER_X = self.MONITOR_WIDTH // 2
        self.CENTER_Y = self.MONITOR_HEIGHT // 2
        
        # Control variables
        self.mouse_control_enabled = False
        self.filter_length = filter_length
        self.calibration_offset_yaw = 0
        self.calibration_offset_pitch = 0
        
        # Tracking degrees
        self.yawDegrees = 20  # x degrees left or right
        self.pitchDegrees = 10  # x degrees up or down
        
        # Mouse tracking state
        self.mouse_target = [self.CENTER_X, self.CENTER_Y]
        self.mouse_lock = threading.Lock()
        
        # Smoothing buffers to reduce jitter
        self.ray_origins = deque(maxlen=self.filter_length)
        self.ray_directions = deque(maxlen=self.filter_length)
        
        # Initialize MediaPipe Face Mesh
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
        
        # initialize camera
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.mouse_thread = None
        
        # Frame data
        self.frame = None
        self.landmarks_frame = None
        self.raw_yaw_deg = 180
        self.raw_pitch_deg = 180
    
    def landmark_to_np(self, landmark, w, h):
        """Convert MediaPipe landmark to numpy array"""
        return np.array([landmark.x * w, landmark.y * h, landmark.z * w])
    
    def mouse_mover(self):
        """Thread function to move the mouse based on gaze direction"""
        while self.is_running:
            if self.mouse_control_enabled:
                with self.mouse_lock:
                    x, y = self.mouse_target
                pyautogui.moveTo(x, y)
            time.sleep(0.01)  # adjust for responsiveness
    
    def start(self):
        """Start the eye tracking and optionally mouse control"""
        if self.is_running:
            return
            
        # Initialize camera
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open camera at index {self.camera_index}")
            
        self.is_running = True
        
        # Start the mouse mover thread
        self.mouse_thread = threading.Thread(target=self.mouse_mover, daemon=True)
        self.mouse_thread.start()
        return self.cap

    
    def stop(self):
        """Stop eye tracking and release resources"""
        self.is_running = False
        if self.mouse_thread:
            self.mouse_thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            
        cv2.destroyAllWindows()
    
    def toggle_mouse_control(self):
        """Toggle mouse control on/off"""
        self.mouse_control_enabled = not self.mouse_control_enabled
        return self.mouse_control_enabled
    
    def calibrate(self):
        """Calibrate the eye tracker to center position"""
        self.calibration_offset_yaw = 180 - self.raw_yaw_deg
        self.calibration_offset_pitch = 180 - self.raw_pitch_deg
        return self.calibration_offset_yaw, self.calibration_offset_pitch
    
    def process_frame(self):
        """Process a single frame from the camera"""
        if not self.is_running or not self.cap:
            return None, None, None, False
            
        ret, frame = self.cap.read()
        if not ret:
            return None, None, None, False

        img = frame.copy()
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb)
        
        landmarks_frame = np.zeros_like(frame)
        
        if not results.multi_face_landmarks:
            self.frame = frame
            self.landmarks_frame = landmarks_frame
            self.img = frame.copy()
            return frame.copy(), landmarks_frame, frame, False
            
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        # Draw all landmarks as single white pixels
        for i, landmark in enumerate(face_landmarks):
            pt = self.landmark_to_np(landmark, w, h)
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                color = (155, 155, 155) if i in self.FACE_OUTLINE_INDICES else (255, 25, 10)
                cv2.circle(landmarks_frame, (x, y), 3, color, -1)
                frame[y, x] = (255, 255, 255)  # optional: also update main frame if needed
        
        # Highlight bounding landmarks in pink
        key_points = {}
        for name, idx in self.LANDMARKS.items():
            pt = self.landmark_to_np(face_landmarks[idx], w, h)
            key_points[name] = pt
            x, y = int(pt[0]), int(pt[1])
            cv2.circle(frame, (x, y), 4, (0, 0, 0), -1)
        
        # Extract points
        left = key_points["left"]
        right = key_points["right"]
        top = key_points["top"]
        bottom = key_points["bottom"]
        front = key_points["front"]
        
        # Oriented axes based on head geometry
        right_axis = (right - left)
        right_axis /= np.linalg.norm(right_axis)
        
        up_axis = (top - bottom)
        up_axis /= np.linalg.norm(up_axis)
        
        forward_axis = np.cross(right_axis, up_axis)
        forward_axis /= np.linalg.norm(forward_axis)
        
        # Flip to ensure forward vector comes out of the face
        forward_axis = -forward_axis
        
        # Compute center of the head
        center = (left + right + top + bottom + front) / 5
        
        # Half-sizes (width, height, depth)
        half_width = np.linalg.norm(right - left) / 2
        half_height = np.linalg.norm(top - bottom) / 2
        half_depth = 80  # can be tuned or calculated if you have a back landmark
        
        # Generate cube corners in face-aligned space
        def corner(x_sign, y_sign, z_sign):
            return (center
                    + x_sign * half_width * right_axis
                    + y_sign * half_height * up_axis
                    + z_sign * half_depth * forward_axis)
        
        cube_corners = [
            corner(-1, 1, -1),   # top-left-front
            corner(1, 1, -1),    # top-right-front
            corner(1, -1, -1),   # bottom-right-front
            corner(-1, -1, -1),  # bottom-left-front
            corner(-1, 1, 1),    # top-left-back
            corner(1, 1, 1),     # top-right-back
            corner(1, -1, 1),    # bottom-right-back
            corner(-1, -1, 1)    # bottom-left-back
        ]
        
        # Projection function
        def project(pt3d):
            return int(pt3d[0]), int(pt3d[1])
        
        # Draw wireframe cube
        cube_corners_2d = [project(pt) for pt in cube_corners]
        edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),  # front face
            (4, 5), (5, 6), (6, 7), (7, 4),  # back face
            (0, 4), (1, 5), (2, 6), (3, 7)   # sides
        ]
        for i, j in edges:
            cv2.line(frame, cube_corners_2d[i], cube_corners_2d[j], (255, 125, 35), 2)
        
        # Update smoothing buffers
        self.ray_origins.append(center)
        self.ray_directions.append(forward_axis)
        
        # Compute averaged ray origin and direction
        avg_origin = np.mean(self.ray_origins, axis=0)
        avg_direction = np.mean(self.ray_directions, axis=0)
        avg_direction /= np.linalg.norm(avg_direction)  # normalize
        
        # Reference forward direction (camera looking straight ahead)
        reference_forward = np.array([0, 0, -1])  # Z-axis into the screen
        
        # Horizontal (yaw) angle from reference (project onto XZ plane)
        xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
        xz_proj /= np.linalg.norm(xz_proj)
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad  # left is negative
        
        # Vertical (pitch) angle from reference (project onto YZ plane)
        yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
        yz_proj /= np.linalg.norm(yz_proj)
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad  # up is positive
        
        # Convert to degrees and re-center around 0
        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)
        
        # convert left rotations to 0-180
        if yaw_deg < 0:
            yaw_deg = abs(yaw_deg)
        elif yaw_deg < 180:
            yaw_deg = 360 - yaw_deg
        
        if pitch_deg < 0:
            pitch_deg = 360 + pitch_deg
        
        self.raw_yaw_deg = yaw_deg
        self.raw_pitch_deg = pitch_deg
        
        # Apply calibration offsets
        yaw_deg += self.calibration_offset_yaw
        pitch_deg += self.calibration_offset_pitch
        
        # Map to full screen resolution
        screen_x = int(((yaw_deg - (180 - self.yawDegrees)) / (2 * self.yawDegrees)) * self.MONITOR_WIDTH)
        screen_y = int(((180 + self.pitchDegrees - pitch_deg) / (2 * self.pitchDegrees)) * self.MONITOR_HEIGHT)
        
        # Clamp screen position to monitor bounds
        screen_x = max(10, min(screen_x, self.MONITOR_WIDTH - 10))
        screen_y = max(10, min(screen_y, self.MONITOR_HEIGHT - 10))
        
        # Update mouse target
        if self.mouse_control_enabled:
            with self.mouse_lock:
                self.mouse_target[0] = screen_x
                self.mouse_target[1] = screen_y
        
        # Draw smoothed ray
        ray_length = 2.5 * half_depth
        ray_end = avg_origin - avg_direction * ray_length
        
        # Draw the ray
        cv2.line(frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)
        cv2.line(landmarks_frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)
        cv2.line(img, project(avg_origin), project(ray_end), (15, 255, 0), 3)
        self.frame = frame
        self.landmarks_frame = landmarks_frame
        self.img = img
        
        return img, landmarks_frame, frame, ret
    
    def run_loop(self):
        """Run the eye tracker in a continuous loop until 'q' is pressed"""
        self.start()
        
        try:
            while self.is_running:
                img, landmarks_frame, frame, ret = self.process_frame()
                if frame is None:
                    break
                    
                cv2.imshow("Head-Aligned Cube", frame) 
                cv2.imshow("Facial Landmarks", landmarks_frame) #EDM
                cv2.imshow("Clean", img)
                
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('c'):
                    self.calibrate()
                    print(f"[Calibrated] Offset Yaw: {self.calibration_offset_yaw}, Offset Pitch: {self.calibration_offset_pitch}")
                elif key == ord('t'):
                    enabled = self.toggle_mouse_control()
                    print(f"[Mouse Control] {'Enabled' if enabled else 'Disabled'}")
                    time.sleep(0.3)  # debounce to prevent rapid toggling
        finally:
            self.stop()
    
    def get_current_position(self):
        """Get the current calculated screen position"""
        with self.mouse_lock:
            return self.mouse_target[0], self.mouse_target[1]
    
    def set_sensitivity(self, yaw_degrees=None, pitch_degrees=None):
        """Set the sensitivity of the eye tracker"""
        if yaw_degrees is not None:
            self.yawDegrees = yaw_degrees
        if pitch_degrees is not None:
            self.pitchDegrees = pitch_degrees


# Example usage if this file is run directly
if __name__ == "__main__":
    try:
        tracker = HeadTracker(camera_index=1)
        print("Press 'c' to calibrate")
        print("Press 't' to toggle mouse control")
        print("Press 'q' to quit")
        tracker.run_loop()
    except Exception as e:
        print(f"Error: {e}")
