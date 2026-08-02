"""Core gaze tracking module for eye sphere locking, gaze direction computation, and screen mapping."""

import math
import numpy as np
from typing import Optional, Tuple
from ..config.settings import Settings
from ..utils.geometry import normalize, compute_scale


class GazeTracker:
    """Handles eye sphere tracking, gaze direction smoothing, and screen coordinate conversion."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.base_radius = settings.base_radius
        self.left_sphere_locked = False
        self.left_sphere_local_offset = None
        self.left_calibration_nose_scale = None
        self.right_sphere_locked = False
        self.right_sphere_local_offset = None
        self.right_calibration_nose_scale = None

    def lock_spheres(self, head_center: np.ndarray, R_final: np.ndarray, iris_3d_left: np.ndarray, 
                     iris_3d_right: np.ndarray, nose_points_3d: np.ndarray):
        """Lock left and right eye spheres during calibration ('c' key)."""
        current_nose_scale = compute_scale(nose_points_3d)

        # Lock left
        self.left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
        camera_dir_world = np.array([0, 0, 1])
        camera_dir_local = R_final.T @ camera_dir_world
        self.left_sphere_local_offset += self.base_radius * camera_dir_local
        self.left_calibration_nose_scale = current_nose_scale
        self.left_sphere_locked = True

        # Lock right
        self.right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
        self.right_sphere_local_offset += self.base_radius * camera_dir_local
        self.right_calibration_nose_scale = current_nose_scale
        self.right_sphere_locked = True

        print("[Both Spheres Locked] Eye sphere calibration complete.")

    def update_eye_positions(self, head_center: np.ndarray, R_final: np.ndarray, 
                             iris_3d_left: np.ndarray, iris_3d_right: np.ndarray, 
                             nose_points_3d: np.ndarray, w: int, h: int) -> Tuple[Optional[np.ndarray], Optional[float], Optional[np.ndarray], Optional[float]]:
        """Update left and right eye sphere positions and radii based on current scale."""
        current_nose_scale = compute_scale(nose_points_3d)
        scale_ratio = current_nose_scale / self.left_calibration_nose_scale if self.left_calibration_nose_scale else 1.0

        if self.left_sphere_locked:
            scaled_offset = self.left_sphere_local_offset * scale_ratio
            sphere_world_l = head_center + R_final @ scaled_offset
            scaled_radius_l = self.base_radius * scale_ratio
        else:
            sphere_world_l = None
            scaled_radius_l = None

        scale_ratio_r = current_nose_scale / self.right_calibration_nose_scale if self.right_calibration_nose_scale else 1.0
        if self.right_sphere_locked:
            scaled_offset_r = self.right_sphere_local_offset * scale_ratio_r
            sphere_world_r = head_center + R_final @ scaled_offset_r
            scaled_radius_r = self.base_radius * scale_ratio_r
        else:
            sphere_world_r = None
            scaled_radius_r = None

        return sphere_world_l, scaled_radius_l, sphere_world_r, scaled_radius_r

    def compute_combined_gaze(self, sphere_world_l: np.ndarray, scaled_radius_l: float, iris_3d_left: np.ndarray,
                              sphere_world_r: np.ndarray, scaled_radius_r: float, iris_3d_right: np.ndarray) -> Optional[np.ndarray]:
        """Compute and smooth combined left/right gaze direction."""
        if not (self.left_sphere_locked and self.right_sphere_locked):
            return None

        # Individual gaze directions
        left_gaze_dir = iris_3d_left - sphere_world_l
        left_gaze_dir = normalize(left_gaze_dir)

        right_gaze_dir = iris_3d_right - sphere_world_r
        right_gaze_dir = normalize(right_gaze_dir)

        # Raw combined
        raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
        raw_combined_direction = normalize(raw_combined_direction)

        # Update buffer for smoothing
        self.settings.combined_gaze_directions.append(raw_combined_direction)

        # Smoothed
        if len(self.settings.combined_gaze_directions) > 0:
            avg_combined_direction = np.mean(self.settings.combined_gaze_directions, axis=0)
            avg_combined_direction = normalize(avg_combined_direction)
            return avg_combined_direction

        return None

    def convert_gaze_to_screen_coordinates(self, combined_gaze_direction: np.ndarray) -> Tuple[int, int, float, float]:
        """Convert 3D gaze direction to 2D screen coordinates with calibration offsets."""
        reference_forward = np.array([0, 0, -1])

        avg_direction = normalize(combined_gaze_direction)

        # Yaw (horizontal)
        xz_proj = np.array([avg_direction[0], 0, avg_direction[2]])
        xz_proj = normalize(xz_proj)
        yaw_rad = math.acos(np.clip(np.dot(reference_forward, xz_proj), -1.0, 1.0))
        if avg_direction[0] < 0:
            yaw_rad = -yaw_rad

        # Pitch (vertical)
        yz_proj = np.array([0, avg_direction[1], avg_direction[2]])
        yz_proj = normalize(yz_proj)
        pitch_rad = math.acos(np.clip(np.dot(reference_forward, yz_proj), -1.0, 1.0))
        if avg_direction[1] > 0:
            pitch_rad = -pitch_rad

        yaw_deg = math.degrees(yaw_rad)
        pitch_deg = math.degrees(pitch_rad)

        # Adjust signs for screen mapping
        if yaw_deg < 0:
            yaw_deg = -yaw_deg
        elif yaw_deg > 0:
            yaw_deg = -yaw_deg

        raw_yaw_deg = yaw_deg
        raw_pitch_deg = pitch_deg

        yaw_degrees = 5 * 3  # Horizontal range
        pitch_degrees = 2.0 * 2.5  # Vertical range

        # Apply calibration
        yaw_deg += self.settings.calibration_offset_yaw
        pitch_deg += self.settings.calibration_offset_pitch

        # Map to screen
        screen_x = int(((yaw_deg + yaw_degrees) / (2 * yaw_degrees)) * self.settings.monitor_width)
        screen_y = int(((pitch_degrees - pitch_deg) / (2 * pitch_degrees)) * self.settings.monitor_height)

        # Clamp
        screen_x = max(10, min(screen_x, self.settings.monitor_width - 10))
        screen_y = max(10, min(screen_y, self.settings.monitor_height - 10))

        return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

    def calibrate_screen(self, combined_direction: np.ndarray):
        """Calibrate screen offsets during 's' key press (look at center)."""
        _, _, raw_yaw, raw_pitch = self.convert_gaze_to_screen_coordinates(combined_direction)
        self.settings.calibration_offset_yaw = 0 - raw_yaw
        self.settings.calibration_offset_pitch = 0 - raw_pitch
        print(f"[Screen Calibrated] Offset Yaw: {self.settings.calibration_offset_yaw:.2f}, Offset Pitch: {self.settings.calibration_offset_pitch:.2f}")