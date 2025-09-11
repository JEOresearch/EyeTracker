"""Core monitor module for 3D plane estimation, gaze intersection, and marker placement."""

import math
import numpy as np
from typing import Optional, Tuple, List
from ..config.settings import Settings
from ..utils.geometry import normalize


class MonitorPlane:
    """Represents the 3D monitor plane, handling creation, gaze intersection, and markers."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.corners: Optional[List[np.ndarray]] = None
        self.center_w: Optional[np.ndarray] = None
        self.normal_w: Optional[np.ndarray] = None
        self.units_per_cm: Optional[float] = None
        self.gaze_markers: List[Tuple[float, float]] = settings.gaze_markers

    def create_from_calibration(self, head_center: np.ndarray, R_final: np.ndarray, face_landmarks: List, 
                                w: int, h: int, forward_hint: Optional[np.ndarray] = None, 
                                gaze_origin: Optional[np.ndarray] = None, gaze_dir: Optional[np.ndarray] = None):
        """Create the 3D monitor plane during eye calibration ('c' key)."""
        # Estimate scale from chin-forehead
        try:
            lm_chin = face_landmarks[152]
            lm_fore = face_landmarks[10]
            chin_w = np.array([lm_chin.x * w, lm_chin.y * h, lm_chin.z * w], dtype=float)
            fore_w = np.array([lm_fore.x * w, lm_fore.y * h, lm_fore.z * w], dtype=float)
            face_h_units = np.linalg.norm(fore_w - chin_w)
            upc = face_h_units / 15.0  # units per cm
        except Exception:
            upc = 5.0

        # Monitor geometry
        dist_cm = 50.0
        mon_w_cm, mon_h_cm = 60.0, 40.0
        half_w = (mon_w_cm * 0.5) * upc
        half_h = (mon_h_cm * 0.5) * upc

        # Head forward
        head_forward = -R_final[:, 2]
        if forward_hint is not None:
            head_forward = normalize(forward_hint)

        # Place center using gaze ray if available
        if gaze_origin is not None and gaze_dir is not None:
            gaze_dir = normalize(gaze_dir)
            plane_point = head_center + head_forward * (50.0 * upc)
            plane_normal = head_forward
            denom = np.dot(plane_normal, gaze_dir)
            if abs(denom) > 1e-6:
                t = np.dot(plane_normal, plane_point - gaze_origin) / denom
                center_w = gaze_origin + t * gaze_dir
            else:
                center_w = head_center + head_forward * (50.0 * upc)
        else:
            center_w = head_center + head_forward * (50.0 * upc)

        # Local axes
        world_up = np.array([0, -1, 0], dtype=float)
        head_right = np.cross(world_up, head_forward)
        head_right = normalize(head_right)
        head_up = np.cross(head_forward, head_right)
        head_up = normalize(head_up)

        # Corners
        p0 = center_w - head_right * half_w - head_up * half_h
        p1 = center_w + head_right * half_w - head_up * half_h
        p2 = center_w + head_right * half_w + head_up * half_h
        p3 = center_w - head_right * half_w + head_up * half_h

        normal_w = normalize(head_forward)

        self.corners = [p0, p1, p2, p3]
        self.center_w = center_w
        self.normal_w = normal_w
        self.units_per_cm = upc

        # Freeze debug pivot
        self.settings.debug_world_frozen = True
        self.settings.orbit_pivot_frozen = center_w.copy()
        print(f"[Monitor] units_per_cm={upc:.3f}, center={center_w}, normal={normal_w}")

    def compute_gaze_hit(self, combined_dir: np.ndarray, sphere_world_l: np.ndarray, sphere_world_r: np.ndarray) -> Optional[Tuple[float, float]]:
        """Compute gaze ray intersection with plane, return (a, b) in plane coords if on monitor."""
        if self.corners is None or self.center_w is None or self.normal_w is None:
            return None

        O = (np.asarray(sphere_world_l, dtype=float) + np.asarray(sphere_world_r, dtype=float)) * 0.5
        D = normalize(np.asarray(combined_dir, dtype=float))
        C = np.asarray(self.center_w, dtype=float)
        N = normalize(np.asarray(self.normal_w, dtype=float))

        denom = float(np.dot(N, D))
        if abs(denom) > 1e-6:
            t = float(np.dot(N, (C - O)) / denom)
            if t > 0.0:
                P = O + t * D

                # Barycentric coords (a, b)
                p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in self.corners]
                u = p1 - p0
                v = p3 - p0
                wv = P - p0

                u_len2 = float(np.dot(u, u))
                v_len2 = float(np.dot(v, v))
                if u_len2 > 1e-9 and v_len2 > 1e-9:
                    a = float(np.dot(wv, u) / u_len2)
                    b = float(np.dot(wv, v) / v_len2)

                    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                        return a, b

        return None

    def add_marker(self, a: float, b: float):
        """Add a gaze marker at (a, b) on the plane if inside bounds."""
        if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
            self.gaze_markers.append((a, b))
            print(f"[Marker] Added at a={a:.3f}, b={b:.3f}")
        else:
            print("[Marker] Gaze not on monitor; no marker.")

    def get_markers(self) -> List[Tuple[float, float]]:
        """Get stored gaze markers."""
        return self.gaze_markers