import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import pyautogui
import math
import threading
import time
import subprocess
import keyboard

MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
mouse_control_enabled = True
filter_length = 8


FACE_OUTLINE_INDICES = [
    10, 338, 297, 332, 284, 251, 389, 356,
    454, 323, 361, 288, 397, 365, 379, 378,
    400, 377, 152, 148, 176, 149, 150, 136,
    172, 58, 132, 93, 234, 127, 162, 21,
    54, 103, 67, 109
]


# Shared mouse target position
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

calibration_offset_yaw = 0
calibration_offset_pitch = 0

# Buffers to store recent ray data
ray_origins = deque(maxlen=filter_length)
ray_directions = deque(maxlen=filter_length)

# Initialize MediaPipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False,
                                  max_num_faces=1,
                                  refine_landmarks=True,
                                  min_detection_confidence=0.5,
                                  min_tracking_confidence=0.5)

# Open camera
cap = cv2.VideoCapture(1)

# Landmark indices for bounding box
LANDMARKS = {
    "left": 234,
    "right": 454,
    "top": 10,
    "bottom": 152,
    "front": 1,
}

def mouse_mover():
    while True:
        if mouse_control_enabled == True:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y)
        time.sleep(0.01)  # adjust for responsiveness

def landmark_to_np(landmark, w, h):
    return np.array([landmark.x * w, landmark.y * h, landmark.z * w])

threading.Thread(target=mouse_mover, daemon=True).start()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        landmarks_frame = np.zeros_like(frame)  # Blank black frame

        outline_pts = []
        # Draw all landmarks as single white pixels
        for i, landmark in enumerate(face_landmarks):
            pt = landmark_to_np(landmark, w, h)
            x, y = int(pt[0]), int(pt[1])
            if 0 <= x < w and 0 <= y < h:
                color = (155, 155, 155) if i in FACE_OUTLINE_INDICES else (255, 25, 10)
                cv2.circle(landmarks_frame, (x, y), 3, color, -1)
                frame[y, x] = (255, 255, 255)  # optional: also update main frame if needed

        

        # Highlight bounding landmarks in pink
        key_points = {}
        for name, idx in LANDMARKS.items():
            pt = landmark_to_np(face_landmarks[idx], w, h)
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
        ray_origins.append(center)
        ray_directions.append(forward_axis)

        # Compute averaged ray origin and direction
        avg_origin = np.mean(ray_origins, axis=0)
        avg_direction = np.mean(ray_directions, axis=0)
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

        #Specify 

        # Convert to degrees and re-center around 0
        yaw_deg = np.degrees(yaw_rad)
        pitch_deg = np.degrees(pitch_rad)

        #this results in the center being 180, +10 left = -170, +10 right = +170

        #convert left rotations to 0-180
        if yaw_deg < 0:
            yaw_deg = abs(yaw_deg)
        elif yaw_deg < 180:
            yaw_deg = 360 - yaw_deg

        if pitch_deg < 0:
            pitch_deg = 360 + pitch_deg

        raw_yaw_deg = yaw_deg
        raw_pitch_deg = pitch_deg
        
        #yaw is now converted to 90 (looking directly left) to 270 (looking directly right), wrt camera
        #pitch is now converted to 90 (looking straight down) and 270 (looking straight up), wrt camera
        #print(f"Angles: yaw={yaw_deg}, pitch={pitch_deg}")

        #specify degrees at which screen border will be reached
        yawDegrees = 20 # x degrees left or right
        pitchDegrees = 10 # x degrees up or down
        
        # leftmost pixel position must correspond to 180 - yaw degrees
        # rightmost pixel position must correspond to 180 + yaw degrees
        # topmost pixel position must correspond to 180 + pitch degrees
        # bottommost pixel position must correspond to 180 - pitch degrees

        # Apply calibration offsets
        yaw_deg += calibration_offset_yaw
        pitch_deg += calibration_offset_pitch

        # Map to full screen resolution
        screen_x = int(((yaw_deg - (180 - yawDegrees)) / (2 * yawDegrees)) * MONITOR_WIDTH)
        screen_y = int(((180 + pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)

        # Clamp screen position to monitor bounds
        if(screen_x < 10):
            screen_x = 10
        if(screen_y < 10):
            screen_y = 10
        if(screen_x > MONITOR_WIDTH - 10):
            screen_x = MONITOR_WIDTH - 10
        if(screen_y > MONITOR_HEIGHT - 10):
            screen_y = MONITOR_HEIGHT - 10

        print(f"Screen position: x={screen_x}, y={screen_y}")

        if mouse_control_enabled:
            with mouse_lock:
                mouse_target[0] = screen_x
                mouse_target[1] = screen_y

        # Draw smoothed ray
        ray_length = 2.5 * half_depth
        ray_end = avg_origin - avg_direction * ray_length
        
        # Draw the ray
        cv2.line(frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)
        cv2.line(landmarks_frame, project(avg_origin), project(ray_end), (15, 255, 0), 3)

    cv2.imshow("Head-Aligned Cube", frame)
    cv2.imshow("Facial Landmarks", landmarks_frame)

    if keyboard.is_pressed('f7'):
        mouse_control_enabled = not mouse_control_enabled
        print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
        time.sleep(0.3)  # debounce to prevent rapid toggling


    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibration_offset_yaw = 180 - raw_yaw_deg
        calibration_offset_pitch = 180 - raw_pitch_deg
        print(f"[Calibrated] Offset Yaw: {calibration_offset_yaw}, Offset Pitch: {calibration_offset_pitch}")


cap.release()
cv2.destroyAllWindows()
