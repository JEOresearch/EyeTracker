"""Main entry point for the modular eye tracking application."""

import sys
import cv2
import numpy as np
import threading
import time
import keyboard
from .config import Settings
from .core import Detection, GazeTracker, MonitorPlane
from .utils import visualization, controls
from .utils.geometry import normalize


def main():
    settings = Settings()

    # Webcam setup
    cap = cv2.VideoCapture(0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # Optional resize for performance
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    w = 640
    h = 480

    # Initialize components
    detector = Detection(settings)
    gaze_tracker = GazeTracker(settings)
    monitor = MonitorPlane(settings)

    # Mouse setup
    mouse_target = [settings.center_x, settings.center_y]
    mouse_lock = threading.Lock()
    controls.start_mouse_thread(settings, mouse_target, mouse_lock)

    # Left and right iris indices
    left_iris_idx = 468
    right_iris_idx = 473

    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        combined_dir = None

        # Detection
        head_center, R_final, nose_points_3d, face_landmarks = detector.process_frame(frame, w, h)

        if face_landmarks:
            # Iris positions
            left_iris = face_landmarks[left_iris_idx]
            right_iris = face_landmarks[right_iris_idx]
            x_iris_l = int(left_iris.x * w)
            y_iris_l = int(left_iris.y * h)
            x_iris_r = int(right_iris.x * w)
            y_iris_r = int(right_iris.y * h)

            iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
            iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])

            # Update eye positions
            sphere_world_l, scaled_radius_l, sphere_world_r, scaled_radius_r = gaze_tracker.update_eye_positions(
                head_center, R_final, iris_3d_left, iris_3d_right, nose_points_3d, w, h
            )

            # Draw eyes
            if not gaze_tracker.left_sphere_locked:
                cv2.circle(frame, (x_iris_l, y_iris_l), 10, (255, 25, 25), 2)
            else:
                cv2.circle(frame, (int(sphere_world_l[0]), int(sphere_world_l[1])), int(scaled_radius_l), (255, 255, 25), 2)

            if not gaze_tracker.right_sphere_locked:
                cv2.circle(frame, (x_iris_r, y_iris_r), 10, (25, 255, 25), 2)
            else:
                cv2.circle(frame, (int(sphere_world_r[0]), int(sphere_world_r[1])), int(scaled_radius_r), (25, 255, 255), 2)

            if gaze_tracker.left_sphere_locked and gaze_tracker.right_sphere_locked:
                # Draw individual gazes
                visualization.draw_gaze(frame, sphere_world_l, iris_3d_left, scaled_radius_l, (55, 255, 0), 130)
                visualization.draw_gaze(frame, sphere_world_r, iris_3d_right, scaled_radius_r, (55, 255, 0), 130)

                # Compute combined gaze
                combined_dir = gaze_tracker.compute_combined_gaze(sphere_world_l, scaled_radius_l, iris_3d_left,
                                                                 sphere_world_r, scaled_radius_r, iris_3d_right)

                if combined_dir is not None:
                    # Screen mapping
                    screen_x, screen_y, raw_yaw, raw_pitch = gaze_tracker.convert_gaze_to_screen_coordinates(combined_dir)

                    # Update mouse
                    if settings.mouse_control_enabled:
                        with mouse_lock:
                            mouse_target[0] = screen_x
                            mouse_target[1] = screen_y

                    # Write position
                    controls.write_screen_position(screen_x, screen_y)

                    # Draw combined gaze
                    combined_origin = (sphere_world_l + sphere_world_r) / 2
                    combined_target = combined_origin + combined_dir * settings.gaze_length
                    cv2.line(frame, (int(combined_origin[0]), int(combined_origin[1])),
                             (int(combined_target[0]), int(combined_target[1])), (255, 255, 10), 3)

                    # Text overlay
                    texts = [f"Screen: ({screen_x}, {screen_y})"]
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.7
                    thickness = 2
                    for i, text in enumerate(texts):
                        (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        center_x = (w - text_width) // 2
                        cv2.putText(frame, text, (center_x, 30 + i * 30), font, font_scale, (0, 255, 0), thickness)

            # Draw all landmarks
            for lm in face_landmarks:
                x, y = int(lm.x * w), int(lm.y * h)
                cv2.circle(frame, (x, y), 0, (255, 255, 255), -1)

            # Draw head pose
            if head_center is not None:
                draw_head_pose(frame, head_center, R_final, (0, 255, 0), 80, w, h, face_landmarks, settings.nose_indices)

            # Update orbit controls
            controls.update_orbit_from_keys(settings)

            # Prepare debug view
            landmarks3d = np.array([[lm.x * w, lm.y * h, lm.z * w] for lm in face_landmarks], dtype=float)
            visualization.render_debug_view_orbit(
                h, w,
                head_center3d=head_center,
                sphere_world_l=sphere_world_l,
                scaled_radius_l=scaled_radius_l,
                sphere_world_r=sphere_world_r,
                scaled_radius_r=scaled_radius_r,
                iris3d_l=iris_3d_left,
                iris3d_r=iris_3d_right,
                left_locked=gaze_tracker.left_sphere_locked,
                right_locked=gaze_tracker.right_sphere_locked,
                landmarks3d=landmarks3d,
                combined_dir=combined_dir,
                gaze_len=5230,
                monitor_corners=monitor.corners,
                monitor_center=monitor.center_w,
                monitor_normal=monitor.normal_w,
                gaze_markers=monitor.get_markers(),
                orbit_yaw=settings.orbit_yaw,
                orbit_pitch=settings.orbit_pitch,
                orbit_radius=settings.orbit_radius,
                orbit_fov_deg=settings.orbit_fov_deg,
                debug_world_frozen=settings.debug_world_frozen,
                orbit_pivot_frozen=settings.orbit_pivot_frozen,
                units_per_cm=monitor.units_per_cm
            )

        cv2.imshow("Integrated Eye Tracking", frame)

        # Keyboard handling
        if keyboard.is_pressed('f7'):
            settings.mouse_control_enabled = not settings.mouse_control_enabled
            print(f"[Mouse Control] {'Enabled' if settings.mouse_control_enabled else 'Disabled'}")
            time.sleep(0.3)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c') and not (gaze_tracker.left_sphere_locked and gaze_tracker.right_sphere_locked):
            if head_center is not None and face_landmarks is not None:
                gaze_tracker.lock_spheres(head_center, R_final, iris_3d_left, iris_3d_right, nose_points_3d)

                # Create monitor plane
                sphere_world_l_calib = head_center + R_final @ gaze_tracker.left_sphere_local_offset
                sphere_world_r_calib = head_center + R_final @ gaze_tracker.right_sphere_local_offset
                left_dir = iris_3d_left - sphere_world_l_calib
                right_dir = iris_3d_right - sphere_world_r_calib
                if np.linalg.norm(left_dir) > 1e-9:
                    left_dir = normalize(left_dir)
                if np.linalg.norm(right_dir) > 1e-9:
                    right_dir = normalize(right_dir)
                forward_hint = (left_dir + right_dir) * 0.5
                if np.linalg.norm(forward_hint) > 1e-9:
                    forward_hint = normalize(forward_hint)
                gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
                gaze_dir = forward_hint
                monitor.create_from_calibration(head_center, R_final, face_landmarks, w, h, forward_hint, gaze_origin, gaze_dir)
        elif key == ord('s') and gaze_tracker.left_sphere_locked and gaze_tracker.right_sphere_locked:
            if combined_dir is not None:
                gaze_tracker.calibrate_screen(combined_dir)
        elif key == ord('x') and monitor.corners is not None and gaze_tracker.left_sphere_locked and gaze_tracker.right_sphere_locked:
            if combined_dir is not None and sphere_world_l is not None and sphere_world_r is not None:
                hit = monitor.compute_gaze_hit(combined_dir, sphere_world_l, sphere_world_r)
                if hit:
                    monitor.add_marker(*hit)

    cap.release()
    cv2.destroyAllWindows()


def draw_head_pose(frame: np.ndarray, center: np.ndarray, R_final: np.ndarray, color: tuple, size: float, w: int, h: int, 
                   face_landmarks: list, nose_indices: list):
    """Draw head pose visualization (landmarks, cube, axes)."""
    # Draw nose landmarks
    for i in nose_indices:
        x, y = int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)
        cv2.circle(frame, (x, y), 3, color, -1)

    # Draw cube
    visualization.draw_wireframe_cube(frame, center, R_final, size)

    # Draw axes
    axis_length = size * 1.2
    axis_dirs = [R_final[:, 0], -R_final[:, 1], -R_final[:, 2]]
    axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    for i in range(3):
        end_pt = center + axis_dirs[i] * axis_length
        cv2.line(frame, (int(center[0]), int(center[1])), (int(end_pt[0]), int(end_pt[1])), axis_colors[i], 2)


if __name__ == "__main__":
    main()