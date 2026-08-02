"""Visualization utilities for drawing gaze, coordinate boxes, and 3D debug views."""

import cv2
import numpy as np
import math
from scipy.spatial.transform import Rotation as Rscipy
from typing import Optional, List, Tuple
from .geometry import normalize, focal_px, rot_x, rot_y


def draw_gaze(frame: np.ndarray, eye_center: np.ndarray, iris_center: np.ndarray, 
              eye_radius: float, color: Tuple[int, int, int], gaze_length: int):
    """Draw gaze vector and segments on the frame."""
    gaze_direction = iris_center - eye_center
    gaze_direction = normalize(gaze_direction)
    gaze_endpoint = eye_center + gaze_direction * gaze_length

    cv2.line(frame, tuple(int(v) for v in eye_center[:2]), tuple(int(v) for v in gaze_endpoint[:2]), color, 2)

    iris_offset = eye_center + gaze_direction * (1.2 * eye_radius)

    # Back segment
    cv2.line(frame, (int(eye_center[0]), int(eye_center[1])), (int(iris_offset[0]), int(iris_offset[1])), color, 1)

    # Iris ellipse (original code had incomplete ellipse; simplified to skip for now)
    # up_dir = np.array([0, -1, 0])
    # right_dir = np.cross(gaze_direction, up_dir)
    # ... (ellipse logic omitted for brevity, can add cv2.ellipse if needed)

    # Front segment
    cv2.line(frame, (int(iris_offset[0]), int(iris_offset[1])), (int(gaze_endpoint[0]), int(gaze_endpoint[1])), color, 1)


def draw_wireframe_cube(frame: np.ndarray, center: np.ndarray, R: np.ndarray, size: float = 80):
    """Draw a wireframe cube aligned to the given rotation."""
    right = R[:, 0]
    up = -R[:, 1]
    forward = -R[:, 2]

    hw, hh, hd = size, size, size

    def corner(x_sign, y_sign, z_sign):
        return (center +
                x_sign * hw * right +
                y_sign * hh * up +
                z_sign * hd * forward)

    corners = [corner(x, y, z) for x in [-1, 1] for y in [1, -1] for z in [-1, 1]]
    projected = [(int(pt[0]), int(pt[1])) for pt in corners]

    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)


def compute_and_draw_coordinate_box(frame: np.ndarray, face_landmarks: List, indices: List[int], 
                                    ref_matrix_container: List[Optional[np.ndarray]], color: Tuple[int, int, int] = (0, 255, 0), 
                                    size: float = 80, w: int = 640, h: int = 480) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute PCA-based orientation and draw coordinate box for face landmarks."""
    # Extract 3D positions
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])

    center = np.mean(points_3d, axis=0)

    # Draw raw landmarks
    for i in indices:
        x, y = int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)
        cv2.circle(frame, (x, y), 3, color, -1)

    # PCA orientation
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]

    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    yaw *= 1
    roll *= 1
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    # Stabilize with reference
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    # Draw cube
    draw_wireframe_cube(frame, center, R_final, size)

    # Draw axes
    axis_length = size * 1.2
    axis_dirs = [R_final[:, 0], -R_final[:, 1], -R_final[:, 2]]
    axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    for i in range(3):
        end_pt = center + axis_dirs[i] * axis_length
        cv2.line(frame, (int(center[0]), int(center[1])), (int(end_pt[0]), int(end_pt[1])), axis_colors[i], 2)

    return center, R_final, points_3d


def render_debug_view_orbit(h: int, w: int, head_center3d: Optional[np.ndarray] = None,
                            sphere_world_l: Optional[np.ndarray] = None, scaled_radius_l: Optional[float] = None,
                            sphere_world_r: Optional[np.ndarray] = None, scaled_radius_r: Optional[float] = None,
                            iris3d_l: Optional[np.ndarray] = None, iris3d_r: Optional[np.ndarray] = None,
                            left_locked: bool = False, right_locked: bool = False,
                            landmarks3d: Optional[np.ndarray] = None,
                            combined_dir: Optional[np.ndarray] = None,
                            gaze_len: int = 430,
                            monitor_corners: Optional[List[np.ndarray]] = None,
                            monitor_center: Optional[np.ndarray] = None,
                            monitor_normal: Optional[np.ndarray] = None,
                            gaze_markers: Optional[List[Tuple[float, float]]] = None,
                            orbit_yaw: float = -151.0, orbit_pitch: float = 0.0, orbit_radius: float = 1500.0,
                            orbit_fov_deg: float = 50.0, debug_world_frozen: bool = False, orbit_pivot_frozen: Optional[np.ndarray] = None,
                            units_per_cm: Optional[float] = None) -> np.ndarray:
    """Render 3D orbit debug view of head, eyes, gaze, and monitor."""
    if head_center3d is None:
        return np.zeros((h, w, 3), dtype=np.uint8)

    debug = np.zeros((h, w, 3), dtype=np.uint8)

    # Choose pivot
    head_w = np.asarray(head_center3d, dtype=float)
    if debug_world_frozen and orbit_pivot_frozen is not None:
        pivot_w = np.asarray(orbit_pivot_frozen, dtype=float)
    else:
        if monitor_center is not None:
            pivot_w = (head_w + np.asarray(monitor_center, dtype=float)) * 0.5
        else:
            pivot_w = head_w

    # Camera pose
    f_px = focal_px(w, orbit_fov_deg)
    cam_offset = rot_y(orbit_yaw) @ (rot_x(orbit_pitch) @ np.array([0.0, 0.0, orbit_radius]))
    cam_pos = pivot_w + cam_offset

    up_world = np.array([0.0, -1.0, 0.0])
    fwd = normalize(pivot_w - cam_pos)
    right = normalize(np.cross(fwd, up_world))
    up = normalize(np.cross(right, fwd))
    V = np.stack([right, up, fwd], axis=0)

    def project_point(P: np.ndarray) -> Optional[Tuple[Tuple[int, int], float]]:
        Pw = np.asarray(P, dtype=float)
        Pc = V @ (Pw - cam_pos)
        if Pc[2] <= 1e-3:
            return None
        x = f_px * (Pc[0] / Pc[2]) + w * 0.5
        y = -f_px * (Pc[1] / Pc[2]) + h * 0.5
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (int(x), int(y)), Pc[2]

    # Draw landmarks
    if landmarks3d is not None:
        for P in landmarks3d:
            res = project_point(P)
            if res is not None:
                cv2.circle(debug, res[0], 0, (200, 200, 200), -1)

    # Head center cross
    res = project_point(head_w)
    if res is not None:
        (x, y), _ = res
        cv2.line(debug, (x - 12, y), (x + 12, y), (255, 0, 255), 2)
        cv2.line(debug, (x, y - 12), (x, y + 12), (255, 0, 255), 2)
        cv2.putText(debug, "Head Center", (x + 12, y - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # Pivot cross
    res = project_point(pivot_w)
    if res is not None:
        (x, y), _ = res
        cv2.line(debug, (x - 8, y), (x + 8, y), (180, 120, 255), 2)
        cv2.line(debug, (x, y - 8), (x, y + 8), (180, 120, 255), 2)

    # Lines to head and monitor
    if monitor_center is not None:
        hc2d = project_point(head_w)
        mc2d = project_point(monitor_center)
        pv2d = project_point(pivot_w)
        if hc2d is not None and mc2d is not None and pv2d is not None:
            cv2.line(debug, pv2d[0], hc2d[0], (160, 100, 255), 1)
            cv2.line(debug, pv2d[0], mc2d[0], (160, 100, 255), 1)

    # Left eye
    if left_locked and sphere_world_l is not None:
        res = project_point(sphere_world_l)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_l or 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (255, 255, 25), 1)
            if iris3d_l is not None:
                left_dir = np.asarray(iris3d_l) - np.asarray(sphere_world_l)
                p1 = project_point(np.asarray(sphere_world_l) + normalize(left_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (155, 155, 25), 1)
    elif iris3d_l is not None:
        res = project_point(iris3d_l)
        if res is not None:
            cv2.circle(debug, res[0], 2, (255, 255, 25), 1)

    # Right eye
    if right_locked and sphere_world_r is not None:
        res = project_point(sphere_world_r)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_r or 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (25, 255, 255), 1)
            if iris3d_r is not None:
                right_dir = np.asarray(iris3d_r) - np.asarray(sphere_world_r)
                p1 = project_point(np.asarray(sphere_world_r) + normalize(right_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (25, 155, 155), 1)
    elif iris3d_r is not None:
        res = project_point(iris3d_r)
        if res is not None:
            cv2.circle(debug, res[0], 2, (25, 255, 255), 1)

    # Combined gaze
    if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
        origin_mid = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) / 2.0
        if combined_dir is None:
            parts = []
            # Assume left_dir and right_dir from eye sections
            # For now, skip if not defined; in main, pass if available
            if len(parts) > 0:
                combined_dir = normalize(np.mean(parts, axis=0))
        if combined_dir is not None:
            p0 = project_point(origin_mid)
            p1 = project_point(origin_mid + normalize(combined_dir) * (gaze_len * 1.2))
            if p0 is not None and p1 is not None:
                cv2.line(debug, p0[0], p1[0], (155, 200, 10), 2)

    # Monitor plane
    if monitor_corners is not None:
        def draw_poly(pts: List[np.ndarray], color: Tuple[int, int, int], thickness: int = 2):
            projs = [project_point(p) for p in pts]
            if any(p is None for p in projs):
                return
            p2 = [p[0] for p in projs]
            for a, b in zip(p2, p2[1:] + [p2[0]]):
                cv2.line(debug, a, b, color, thickness)

        draw_poly(monitor_corners, (0, 200, 255), 2)
        draw_poly([monitor_corners[0], monitor_corners[2]], (0, 150, 210), 1)
        draw_poly([monitor_corners[1], monitor_corners[3]], (0, 150, 210), 1)
        if monitor_center is not None:
            res = project_point(monitor_center)
            if res is not None:
                (x, y), _ = res
                cv2.line(debug, (x - 8, y), (x + 8, y), (0, 200, 255), 2)
                cv2.line(debug, (x, y - 8), (x, y + 8), (0, 200, 255), 2)
            if monitor_normal is not None:
                tip = np.asarray(monitor_center) + np.asarray(monitor_normal) * (20.0 * (units_per_cm or 1.0))
                p0 = project_point(monitor_center)
                p1 = project_point(tip)
                if p0 is not None and p1 is not None:
                    cv2.line(debug, p0[0], p1[0], (0, 220, 255), 2)

    # Gaze markers
    if gaze_markers and monitor_corners is not None:
        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
        u = p1 - p0
        v = p3 - p0
        width_world = float(np.linalg.norm(u))
        if width_world > 1e-9:
            u_hat = u / width_world
            r_world = 0.01 * width_world
            for a, b in gaze_markers:
                Pm = p0 + a * u + b * v
                projP = project_point(Pm)
                projR = project_point(Pm + u_hat * r_world)
                if projP is not None and projR is not None:
                    center_px = projP[0]
                    r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                    cv2.circle(debug, center_px, r_px, (0, 255, 0), 1, lineType=cv2.LINE_AA)

    # Gaze hit on monitor
    if (monitor_corners is not None and monitor_center is not None and monitor_normal is not None
        and combined_dir is not None and sphere_world_l is not None and sphere_world_r is not None):
        O = (np.asarray(sphere_world_l, dtype=float) + np.asarray(sphere_world_r, dtype=float)) * 0.5
        D = normalize(np.asarray(combined_dir, dtype=float))
        C = np.asarray(monitor_center, dtype=float)
        N = normalize(np.asarray(monitor_normal, dtype=float))
        denom = float(np.dot(N, D))
        if abs(denom) > 1e-6:
            t = float(np.dot(N, (C - O)) / denom)
            if t > 0.0:
                P = O + t * D
                p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                u = p1 - p0
                v = p3 - p0
                wv = P - p0
                u_len2 = float(np.dot(u, u))
                v_len2 = float(np.dot(v, v))
                if u_len2 > 1e-9 and v_len2 > 1e-9:
                    a = float(np.dot(wv, u) / u_len2)
                    b = float(np.dot(wv, v) / v_len2)
                    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                        projP = project_point(P)
                        if projP is not None:
                            center_px = projP[0]
                            width_world = math.sqrt(u_len2)
                            r_world = 0.05 * width_world
                            u_hat = u / max(width_world, 1e-9)
                            projR = project_point(P + u_hat * r_world)
                            if projR is not None:
                                r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                                cv2.circle(debug, center_px, r_px, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    # Help text
    help_text = [
        "C = calibrate screen center",
        "J = yaw left",
        "L = yaw right",
        "I = pitch up",
        "K = pitch down",
        "[ = zoom out",
        "] = zoom in",
        "R = reset view",
        "X = add marker",
        "q = quit",
        "F7 = toggle mouse control"
    ]
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    thickness = 1
    line_height = 18
    y0 = h - (len(help_text) * line_height) - 10
    x0 = 10
    for i, text in enumerate(help_text):
        y = y0 + i * line_height
        cv2.putText(debug, text, (x0, y), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)

    cv2.imshow("Head/Eye Debug", debug)
    return debug