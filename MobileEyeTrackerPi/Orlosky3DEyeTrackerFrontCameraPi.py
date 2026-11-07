"""Pi eye/scene viewer with the desktop tracker's five-point calibration.

Standalone script requiring only Python 3, NumPy, and GUI-enabled OpenCV.
Run: python3 Orlosky3DEyeTrackerFrontCameraPi.py --eye-camera 0 --front-camera 2
Use v4l2-ctl --list-devices to find two distinct physical cameras.
Move your eye around until the model is ready. At each red scene target,
align your finger and gaze, then press C: CENTER, TOP, BOTTOM, LEFT, RIGHT.
The first sample freezes the eye model. C restarts completed calibration;
F freezes/resumes model updates (resuming clears calibration); X resets all.
R starts/stops paired 30 FPS AVI recordings in the recordings folder.
E toggles the eye image vertically and resets its model/calibration.
1 cycles the eye camera; 2 cycles the front camera. With only two cameras,
either key swaps the eye/front assignments.
Space pauses; Q/Esc or closing either window quits. No Tkinter/OpenGL needed.
Calibration maps directions to scene pixels; it is not a physical 3D calibration.
"""

import argparse
import os
import math
import random
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

# Complete pupil detection, ellipse fitting, and 3D eye model.
ray_lines = []

model_centers = []

stored_intersections = []

min_model_centers = 30

max_rays = 100

prev_model_center_avg = (320, 240)

max_observed_distance = 0

last_sphere_radius_ellipse = None

eye_sphere_adjustment_enabled = True

pupil_confidence_threshold = 0.85

pupil_confidence_threshold_sphere = 0.65

intersection_ray_count = 4

minimum_intersection_angle_degrees = 8

last_tracking_result = None

def crop_to_aspect_ratio(image, width=640, height=480):
    current_height, current_width = image.shape[:2]
    desired_ratio = width / height
    current_ratio = current_width / current_height

    if current_ratio > desired_ratio:
        # Current image is too wide
        new_width = int(desired_ratio * current_height)
        offset = (current_width - new_width) // 2
        cropped_img = image[:, offset:offset + new_width]
    else:
        # Current image is too tall
        new_height = int(current_width / desired_ratio)
        offset = (current_height - new_height) // 2
        cropped_img = image[offset:offset + new_height, :]

    return cv2.resize(cropped_img, (width, height))

def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    threshold = min(255, max(0, int(darkestPixelValue) + addedThreshold))
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

def fixed_search_sphere():
    """Restrict centers only once a valid eye sphere has been frozen."""
    if not eye_sphere_adjustment_enabled and max_observed_distance > 0:
        return prev_model_center_avg, max_observed_distance
    return None


def center_in_sphere(point, sphere):
    if sphere is None:
        return True
    (cx, cy), radius = sphere
    return (point[0] - cx) ** 2 + (point[1] - cy) ** 2 <= radius ** 2


def get_darkest_area(image, search_sphere=None):
    ignoreBounds = 20
    imageSkipSize = 10
    searchArea = 20
    internalSkipSize = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest_point = None

    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            # Gate the candidate center, not its full sampling square. Pixels
            # outside the sphere can still contribute to a boundary candidate.
            candidate_center = (x + searchArea // 2, y + searchArea // 2)
            if not center_in_sphere(candidate_center, search_sphere):
                continue
            current_sum = 0
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += int(gray[y + dy][x + dx])
                    num_pixels += 1

            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)

    return darkest_point

def mask_outside_square(image, center, size):
    x, y = center
    half_size = size // 2

    mask = np.zeros_like(image)
    top_left_x = max(0, x - half_size)
    top_left_y = max(0, y - half_size)
    bottom_right_x = min(image.shape[1], x + half_size)
    bottom_right_y = min(image.shape[0], y + half_size)
    mask[top_left_y:bottom_right_y, top_left_x:bottom_right_x] = 255
    return cv2.bitwise_and(image, mask)

def filter_contours_by_area_and_return_largest(contours, pixel_thresh, ratio_thresh):
    max_area = 0
    largest_contour = None

    for contour in contours:
        area = cv2.contourArea(contour)
        if area >= pixel_thresh:
            x, y, w, h = cv2.boundingRect(contour)
            length_to_width_ratio = max(w / h, h / w)
            if length_to_width_ratio <= ratio_thresh:
                if area > max_area:
                    max_area = area
                    largest_contour = contour

    return [largest_contour] if largest_contour is not None else []

def distance_to_pupil_outer_edge(eye_center, pupil_ellipse):
    pupil_center, axes, angle_degrees = pupil_ellipse
    direction_x = pupil_center[0] - eye_center[0]
    direction_y = pupil_center[1] - eye_center[1]
    center_distance = math.hypot(direction_x, direction_y)

    semi_axis_x = axes[0] / 2
    semi_axis_y = axes[1] / 2
    if center_distance == 0 or semi_axis_x <= 0 or semi_axis_y <= 0:
        return None

    unit_x = direction_x / center_distance
    unit_y = direction_y / center_distance
    angle_radians = math.radians(angle_degrees)
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)

    local_x = cosine * unit_x + sine * unit_y
    local_y = -sine * unit_x + cosine * unit_y
    edge_offset = 1 / math.sqrt(
        (local_x / semi_axis_x) ** 2
        + (local_y / semi_axis_y) ** 2
    )

    return center_distance + edge_offset

def update_eye_sphere_radius(eye_center, current_pupil_ellipse, current_pupil_confidence):
    global max_observed_distance
    global last_sphere_radius_ellipse

    if last_sphere_radius_ellipse is not None:
        anchored_distance = distance_to_pupil_outer_edge(
            eye_center,
            last_sphere_radius_ellipse,
        )
        if anchored_distance is not None:
            max_observed_distance = anchored_distance

    if (
        current_pupil_ellipse is not None
        and current_pupil_confidence >= pupil_confidence_threshold_sphere
        and len(model_centers) >= min_model_centers
    ):
        current_distance = distance_to_pupil_outer_edge(
            eye_center,
            current_pupil_ellipse,
        )
        if (
            current_distance is not None
            and (
                last_sphere_radius_ellipse is None
                or current_distance > max_observed_distance
            )
        ):
            max_observed_distance = current_distance
            last_sphere_radius_ellipse = current_pupil_ellipse

def toggle_eye_sphere_adjustment():
    global eye_sphere_adjustment_enabled

    eye_sphere_adjustment_enabled = not eye_sphere_adjustment_enabled
    state_text = "automatic" if eye_sphere_adjustment_enabled else "affixed"
    print(f"Eye sphere adjustment: {state_text}")

def update_and_average_point(point_list, new_point, N):
    """
    Adds a new point to the list, keeps only the last N points, 
    and returns the average of those points.
    
    Parameters:
    - point_list: Global list storing past points [(x1, y1), (x2, y2), ...]
    - new_point: Tuple (x, y) representing the new point to add.
    - N: Maximum number of points to keep in the list.
    
    Returns:
    - (avg_x, avg_y): The average point as a tuple of integers.
    - None if the list is empty.
    """
    point_list.append(new_point)  # Add new point

    if len(point_list) > N:
        point_list.pop(0)  # Remove the oldest point to maintain size N

    if not point_list:
        return None  # No points available

    avg_x = int(np.mean([p[0] for p in point_list]))
    avg_y = int(np.mean([p[1] for p in point_list]))

    return (avg_x, avg_y)

def compute_average_intersection(frame, ray_lines, number_lines, total_lines, minimum_angle_degrees):
    """
    Selects `number_lines` random lines from the list, computes their intersections,
    conditionally stores them (only if they agree within `pixel_limit`), and prunes
    stored intersections when exceeding `total_lines`.
    """
    pixel_limit = 30
    angle_threshold = 5  # degrees

    global stored_intersections

    if len(ray_lines) < 2 or number_lines < 2:
        return (0, 0)

    height, width = frame.shape[:2]

    selected_lines = random.sample(ray_lines, min(number_lines, len(ray_lines)))

    intersections = []
    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]
        line2 = selected_lines[i + 1]

        angle1 = line1[2]
        angle2 = line2[2]

        if abs((angle1 - angle2 + 90) % 180 - 90) >= minimum_angle_degrees:
            intersection = find_line_intersection(line1, line2)
            if intersection and (0 <= intersection[0] < width) and (0 <= intersection[1] < height):
                intersections.append(intersection)

    # Nothing usable this frame
    if not intersections:
        return (0, 0)

    # Check mutual agreement within pixel_limit
    accept = True
    if len(intersections) >= 2:
        for i in range(len(intersections)):
            for j in range(i + 1, len(intersections)):
                # --- distance check ---
                dx = intersections[i][0] - intersections[j][0]
                dy = intersections[i][1] - intersections[j][1]
                if (dx * dx + dy * dy) ** 0.5 > pixel_limit:
                    accept = False
                    break

                # --- angle check ---
                angle_i = selected_lines[i][2]
                angle_j = selected_lines[j][2]
                if abs((angle_i - angle_j + 90) % 180 - 90) < angle_threshold:
                    accept = False
                    break
            if not accept:
                break

    if accept:
        stored_intersections.extend(intersections)

    # Prune stored intersections
    if len(stored_intersections) > total_lines:
        stored_intersections = prune_intersections(stored_intersections, total_lines)

    if not stored_intersections:
        return (0, 0)

    avg_x = np.mean([pt[0] for pt in stored_intersections])
    avg_y = np.mean([pt[1] for pt in stored_intersections])

    if np.isnan(avg_x) or np.isnan(avg_y):
        return (0, 0)

    return (int(avg_x), int(avg_y))

def prune_intersections(intersections, maximum_intersections):

    if len(intersections) <= maximum_intersections:
        return intersections  # No need to prune if within the limit

    # Keep only the last M intersections
    pruned_intersections = intersections[-maximum_intersections:]

    return pruned_intersections

def find_line_intersection(ellipse1, ellipse2):
    """
    Computes the intersection of two lines that are orthogonal to the surface of given ellipses.
    
    Parameters:
    - ellipse1, ellipse2: Ellipse tuples ((cx, cy), (major_axis, minor_axis), angle).
    
    Returns:
    - (x, y): Intersection point of the two lines, or None if parallel.
    """

    (cx1, cy1), (_, minor_axis1), angle1 = ellipse1
    (cx2, cy2), (_, minor_axis2), angle2 = ellipse2

    # Convert angles to radians
    angle1_rad = np.deg2rad(angle1)
    angle2_rad = np.deg2rad(angle2)

    # Compute direction vectors for the two lines
    dx1, dy1 = (minor_axis1 / 2) * np.cos(angle1_rad), (minor_axis1 / 2) * np.sin(angle1_rad)
    dx2, dy2 = (minor_axis2 / 2) * np.cos(angle2_rad), (minor_axis2 / 2) * np.sin(angle2_rad)

    # Line equations in parametric form:
    # (x1, y1) + t1 * (dx1, dy1) = (x2, y2) + t2 * (dx2, dy2)
    A = np.array([[dx1, -dx2], [dy1, -dy2]])
    B = np.array([cx2 - cx1, cy2 - cy1])

    # Solve for t1, t2 using linear algebra (if the determinant is nonzero)
    if abs(np.linalg.det(A)) < 1e-8:
        return None  # Lines are parallel and do not intersect

    t1, t2 = np.linalg.solve(A, B)

    # Compute intersection point
    intersection_x = cx1 + t1 * dx1
    intersection_y = cy1 + t1 * dy1

    return (int(intersection_x), int(intersection_y))

def compute_gaze_vector(x, y, center_x, center_y, screen_width=640, screen_height=480):
    """Compute 3D gaze direction from pupil and sphere center screen coordinates.
    Returns:
        sphere_center (np.ndarray): 3D position of the sphere center in world space
        gaze_direction (np.ndarray): Normalized 3D direction vector from sphere center
    """

    # Get viewport dimensions
    viewport_width = screen_width
    viewport_height = screen_height

    # Define camera and projection settings
    fov_y_deg = 45.0
    aspect_ratio = viewport_width / viewport_height
    far_clip = 100.0

    # Camera position is fixed at z = 3
    camera_position = np.array([0.0, 0.0, 3.0])

    # Compute size of far plane in world units
    fov_y_rad = np.radians(fov_y_deg)
    half_height_far = np.tan(fov_y_rad / 2) * far_clip
    half_width_far = half_height_far * aspect_ratio

    # Convert screen (x, y) to normalized device coordinates [-1, 1]
    ndc_x = (2.0 * x) / viewport_width - 1.0
    ndc_y = 1.0 - (2.0 * y) / viewport_height

    # Project pupil center to far plane coordinates in world space
    far_x = ndc_x * half_width_far
    far_y = ndc_y * half_height_far
    far_z = camera_position[2] - far_clip
    far_point = np.array([far_x, far_y, far_z])

    # Compute ray direction from camera to far plane point
    ray_origin = camera_position
    ray_direction = far_point - camera_position
    ray_direction /= np.linalg.norm(ray_direction)
    ray_direction = -ray_direction

    # Sphere radius and center offset
    inner_radius = 1.0 / 1.05
    sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
    sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
    sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

    # Compute intersection with sphere
    origin = ray_origin
    direction = -ray_direction
    L = origin - sphere_center

    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_radius**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        # Compute the closest point to the sphere (tangent point approximation)
        t = -np.dot(direction, L) / np.dot(direction, direction)
        intersection_point = origin + t * direction
        intersection_local = intersection_point - sphere_center
        target_direction = intersection_local / np.linalg.norm(intersection_local)
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        t = None
        if t1 > 0 and t2 > 0:
            t = min(t1, t2)
        elif t1 > 0:
            t = t1
        elif t2 > 0:
            t = t2
        if t is None:
            return None, None

    # Final intersection point
    intersection_point = origin + t * direction
    intersection_local = intersection_point - sphere_center
    target_direction = intersection_local / np.linalg.norm(intersection_local)

    # Local green ring direction
    circle_local_center = np.array([0.0, 0.0, inner_radius])
    circle_local_center /= np.linalg.norm(circle_local_center)

    # Compute rotation to align local +Z to target
    rotation_axis = np.cross(circle_local_center, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        return sphere_center, circle_local_center

    rotation_axis /= rotation_axis_norm
    dot = np.dot(circle_local_center, target_direction)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)

    # Rotation matrix from axis-angle
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t_ = 1 - c
    x_, y_, z_ = rotation_axis

    rotation_matrix = np.array([
        [t_*x_*x_ + c, t_*x_*y_ - s*z_, t_*x_*z_ + s*y_],
        [t_*x_*y_ + s*z_, t_*y_*y_ + c, t_*y_*z_ - s*x_],
        [t_*x_*z_ - s*y_, t_*y_*z_ + s*x_, t_*z_*z_ + c]
    ])

    # Rotate +Z vector to get gaze direction
    gaze_local = np.array([0.0, 0.0, inner_radius])
    gaze_rotated = rotation_matrix @ gaze_local
    gaze_rotated /= np.linalg.norm(gaze_rotated)

    return sphere_center, gaze_rotated

def reset_tracking_state():
    global prev_model_center_avg, max_observed_distance
    global last_sphere_radius_ellipse, eye_sphere_adjustment_enabled
    global last_tracking_result
    ray_lines.clear()
    model_centers.clear()
    stored_intersections.clear()
    prev_model_center_avg = (320, 240)
    max_observed_distance = 0
    last_sphere_radius_ellipse = None
    eye_sphere_adjustment_enabled = True
    last_tracking_result = None

def get_last_tracking_result():
    """Latest valid pupil/model data, or None when no pupil was detected."""
    return last_tracking_result

def detect_pupil(frame, threshold=15):
    """Lite's single threshold, dilation, largest contour and ellipse fit.

    The additional confidence is the desktop model's fraction of contour
    border pixels within a four-pixel ellipse border; it gates model updates
    without introducing the desktop detector's multi-threshold search.
    """
    search_sphere = fixed_search_sphere()
    darkest_point = get_darkest_area(frame, search_sphere)
    if darkest_point is None:
        return None, 0.0
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    binary = apply_binary_threshold(gray, gray[darkest_point[1], darkest_point[0]], threshold)
    binary = mask_outside_square(binary, darkest_point, 250)
    dilated = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=2)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if search_sphere is not None:
        # Reject outside fitted centers before selecting the largest contour,
        # so an eyelash outside the sphere cannot displace a valid pupil.
        candidates = []
        for contour in contours:
            if len(contour) > 5:
                fitted = cv2.fitEllipse(contour)
                if center_in_sphere(fitted[0], search_sphere):
                    candidates.append(contour)
        contours = candidates
    contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)
    if not contours or len(contours[0]) <= 5:
        return None, 0.0
    ellipse = cv2.fitEllipse(contours[0])
    values = (*ellipse[0], *ellipse[1], ellipse[2])
    if not np.isfinite(values).all() or min(ellipse[1]) <= 0:
        return None, 0.0
    if not (0 <= ellipse[0][0] < frame.shape[1] and 0 <= ellipse[0][1] < frame.shape[0]):
        return None, 0.0
    if not center_in_sphere(ellipse[0], search_sphere):
        return None, 0.0
    border = np.zeros_like(gray)
    ellipse_border = np.zeros_like(gray)
    cv2.drawContours(border, contours, -1, 255, 1)
    cv2.ellipse(ellipse_border, ellipse, 255, 4)
    count = cv2.countNonZero(border)
    confidence = cv2.countNonZero(cv2.bitwise_and(border, ellipse_border)) / count if count else 0.0
    return ellipse, confidence

def process_frame(frame, threshold=15, render_cv_window=False, update_model=True):
    """Process one BGR frame; return its ellipse and retain model/gaze data.

    Like the Lite detector, ellipse coordinates use a 640x480 crop. No GUI
    calls occur unless render_cv_window is True. State belongs to one 
    """
    global prev_model_center_avg, last_tracking_result
    frame = crop_to_aspect_ratio(frame)
    ellipse, confidence = detect_pupil(frame, threshold)
    last_tracking_result = None
    if ellipse is not None:
        if eye_sphere_adjustment_enabled and update_model:
            if confidence >= pupil_confidence_threshold:
                ray_lines.append(ellipse)
                del ray_lines[:-max_rays]
                center = compute_average_intersection(
                    frame, ray_lines, intersection_ray_count, 1500,
                    minimum_intersection_angle_degrees,
                )
                if center != (0, 0):
                    prev_model_center_avg = update_and_average_point(model_centers, center, 200)
            update_eye_sphere_radius(prev_model_center_avg, ellipse, confidence)
        ready = len(model_centers) >= min_model_centers and max_observed_distance > 0
        origin, direction = (None, None)
        if ready:
            origin, direction = compute_gaze_vector(*ellipse[0], *prev_model_center_avg)
        last_tracking_result = {
            "pupil_ellipse": {"center": list(ellipse[0]), "axes": list(ellipse[1]),
                              "angle_degrees": float(ellipse[2])},
            "confidence": float(confidence),
            "eye_center": list(prev_model_center_avg),
            "sphere_radius": float(max_observed_distance),
            "model_ready": ready,
            "gaze_origin": origin.tolist() if origin is not None else None,
            "gaze_direction": direction.tolist() if direction is not None else None,
        }
    if render_cv_window:
        draw_tracking(frame, ellipse, confidence)
        cv2.imshow("Orlosky Pi 3D Eye Tracker", frame)
    return ellipse

def draw_tracking(frame, ellipse, confidence):
    if max_observed_distance > 0:
        cv2.circle(frame, prev_model_center_avg, int(max_observed_distance), (255, 50, 50), 2)
    if model_centers:
        cv2.circle(frame, prev_model_center_avg, 5, (255, 255, 0), -1)
    if ellipse is not None:
        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
        pupil = tuple(map(int, ellipse[0]))
        cv2.circle(frame, pupil, 3, (255, 255, 0), -1)
        if model_centers:
            cv2.line(frame, prev_model_center_avg, pupil, (200, 255, 0), 2)
    result = last_tracking_result
    state = "automatic" if eye_sphere_adjustment_enabled else "frozen"
    status = "No pupil" if ellipse is None else f"Pupil confidence: {confidence:.0%}"
    lines = [status, f"Model: {state}"]
    if result and result["gaze_direction"] is not None:
        lines.append("Direction:" + ",".join(f"{v:.3f}" for v in result["gaze_direction"]))
    elif ellipse is not None:
        lines.append("Move eye around to initialize model")
    lines.append("Q quit | F freeze model | R reset | Space pause")
    for index, line in enumerate(lines):
        position = (10, 25 + index * 23)
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, line, position, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

EXT_WIDTH, EXT_HEIGHT = 640, 480
EXT_CX, EXT_CY = EXT_WIDTH // 2, EXT_HEIGHT // 2
calibration_target_margin = 0.15


# Calibration mathematics retained from Orlosky3DEyeTrackerFrontCamera.py.
def calibration_targets():
    margin_x = int(EXT_WIDTH * calibration_target_margin)
    margin_y = int(EXT_HEIGHT * calibration_target_margin)
    return [
        ("CENTER", (EXT_CX, EXT_CY)),
        ("TOP", (EXT_CX, margin_y)),
        ("BOTTOM", (EXT_CX, EXT_HEIGHT - 1 - margin_y)),
        ("LEFT", (margin_x, EXT_CY)),
        ("RIGHT", (EXT_WIDTH - 1 - margin_x, EXT_CY)),
    ]


def project_gaze_to_tangent(direction, basis):
    direction = np.asarray(direction, dtype=float)
    if direction.shape != (3,) or not np.isfinite(direction).all():
        raise ValueError("No valid gaze direction.")
    norm = np.linalg.norm(direction)
    if norm < 1e-8:
        raise ValueError("Gaze direction has zero length.")
    direction = direction / norm
    forward, right, up = basis
    depth = np.dot(direction, forward)
    if depth <= 1e-4:
        raise ValueError("Gaze points away from the calibration plane.")
    return np.array([np.dot(direction, right), np.dot(direction, up)]) / depth


def build_camera_calibration(directions):
    """Fit orientation and a 2x2 tangent-to-pixel map, as in EyeTrackerStereo.cs."""
    directions = np.asarray(directions, dtype=float)
    if directions.shape != (5, 3) or not np.isfinite(directions).all():
        raise ValueError("Five valid gaze samples are required.")
    norms = np.linalg.norm(directions, axis=1)
    if np.any(norms < 1e-8):
        raise ValueError("A gaze sample has zero length.")
    directions = directions / norms[:, None]
    forward = directions[0]
    up = directions[1] - np.dot(directions[1], forward) * forward
    if np.linalg.norm(up) < 1e-3:
        raise ValueError("CENTER and TOP rays are too similar; repeat calibration.")
    up /= np.linalg.norm(up)
    right = np.cross(up, forward)
    right /= np.linalg.norm(right)
    basis = np.array([forward, right, up])
    measured = np.array([project_gaze_to_tangent(d, basis) for d in directions[1:]])
    singular_values = np.linalg.svd(measured, compute_uv=False)
    if singular_values[-1] < 1e-3 or singular_values[0] / singular_values[-1] > 100:
        raise ValueError("Outer gaze samples do not span two reliable axes; repeat calibration.")
    targets = np.array([point for _, point in calibration_targets()[1:]], dtype=float)
    targets -= np.array([EXT_CX, EXT_CY])
    coefficients, _, _, _ = np.linalg.lstsq(measured, targets, rcond=None)
    matrix = coefficients.T
    if np.linalg.matrix_rank(matrix) < 2 or np.linalg.cond(matrix) > 100:
        raise ValueError("Target mapping is degenerate; repeat calibration.")
    errors = np.linalg.norm(measured @ coefficients - targets, axis=1)
    return basis, matrix, float(np.sqrt(np.mean(errors ** 2)))


class Calibration:
    def __init__(self):
        self.reset()

    def reset(self):
        self.samples = []
        self.basis = None
        self.matrix = None
        self.message = "Build eye model; align finger and gaze with CENTER; press C."

    def capture(self, result):
        global eye_sphere_adjustment_enabled
        if self.matrix is not None:
            self.reset()
            return
        direction = None if result is None else result.get("gaze_direction")
        if direction is None:
            self.message = "No current gaze: initialize model or reacquire pupil; retry C."
            return
        direction = np.asarray(direction, dtype=float)
        if direction.shape != (3,) or not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
            self.message = "Invalid gaze sample; retry C."
            return
        # All five samples must use the same eye center and sphere radius.
        eye_sphere_adjustment_enabled = False
        self.samples.append(direction / np.linalg.norm(direction))
        if len(self.samples) < 5:
            name, _ = calibration_targets()[len(self.samples)]
            self.message = f"Captured. Align finger and gaze with {name}; press C."
            return
        try:
            self.basis, self.matrix, error = build_camera_calibration(self.samples)
        except ValueError as exc:
            self.reset()
            self.message = str(exc)
            return
        self.message = f"Fit RMS: {error:.1f} px. C restarts calibration."
        print(self.message)

    def point(self, result):
        if self.matrix is None or result is None or result.get("gaze_direction") is None:
            return None
        try:
            measured = project_gaze_to_tangent(result["gaze_direction"], self.basis)
        except ValueError:
            return None
        point = np.array([EXT_CX, EXT_CY]) + self.matrix @ measured
        return point if np.isfinite(point).all() else None

    def draw(self, frame, result):
        if self.matrix is None:
            name, point = calibration_targets()[len(self.samples)]
            text = f"{len(self.samples) + 1}/5 {name}: align finger and gaze, press C"
        else:
            point = self.point(result)
            text = "Calibrated | C restart | F unlock sphere"
        if point is not None:
            # Clip for display only; the mapping retains the original coordinates.
            pixel = tuple(np.rint(np.clip(point, [0, 0], [EXT_WIDTH - 1, EXT_HEIGHT - 1])).astype(int))
            cv2.circle(frame, pixel, 8, (0, 0, 255), -1)
        for y, line in ((24, text), (46, self.message), (EXT_HEIGHT - 38, "Q quit | Space pause | C calibrate | F freeze | X reset"),
                        (EXT_HEIGHT - 15, "R record | E flip eye | 1 eye cam | 2 front cam")):
            draw_text(frame, line, (10, y), scale=0.516)


def camera_source(value):
    # Normalize numeric indices and equivalent /dev/videoN paths.
    if value.isdecimal():
        return f"/dev/video{int(value)}" if sys.platform.startswith("linux") else int(value)
    return os.path.realpath(value)


def available_camera_sources(current_sources):
    """Enumerate Linux capture devices without interrupting running streams.

    QUERYCAP filters out metadata nodes. Grouping by USB bus information avoids
    offering multiple video nodes belonging to the same physical camera.
    """
    if not sys.platform.startswith("linux"):
        return list(current_sources)
    import fcntl
    groups = {}
    paths = sorted(Path("/dev").glob("video[0-9]*"), key=lambda path: int(path.name[5:]))
    for path in paths:
        fd = None
        try:
            fd = os.open(str(path), os.O_RDONLY | os.O_NONBLOCK)
            capability = bytearray(104)
            fcntl.ioctl(fd, 0x80685600, capability, True)  # VIDIOC_QUERYCAP
            flags = int.from_bytes(capability[84:88], sys.byteorder)
            if flags & 0x80000000:  # V4L2_CAP_DEVICE_CAPS
                flags = int.from_bytes(capability[88:92], sys.byteorder)
            if not flags & (0x1 | 0x1000):  # VIDEO_CAPTURE / VIDEO_CAPTURE_MPLANE
                continue
            source = os.path.realpath(path)
            bus = bytes(capability[48:80]).split(b"\0", 1)[0] or source
            if bus not in groups or source in current_sources:
                groups[bus] = source
        except OSError:
            continue
        finally:
            if fd is not None:
                os.close(fd)
    return list(groups.values())


def next_camera_source(sources, current, other):
    candidates = [source for source in sources if source != other]
    if not candidates:
        return current
    if current not in candidates:
        return candidates[0]
    return candidates[(candidates.index(current) + 1) % len(candidates)]


def open_camera(source, fps):
    backend = cv2.CAP_V4L2 if sys.platform.startswith("linux") else cv2.CAP_ANY
    cap = cv2.VideoCapture(source, backend)
    if not cap.isOpened():
        cap.release()
        raise RuntimeError(f"Could not open camera {source}")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    return cap


def flip_frame(frame, mode):
    return frame if mode == "none" else cv2.flip(frame, {"vertical": 0, "horizontal": 1, "both": -1}[mode])


def process_eye_frame(raw_frame, threshold=15, flip="none", fresh=True):
    """Run the complete original Pi pipeline on the native camera image.

    Includes the original Pi tracker detection steps:
    darkest-area search, threshold, square mask, dilation, contour filtering,
    ellipse fitting/validation, confidence, eye model, and gaze calculation.
    Only repeated cached images suppress model updates, never pupil detection.
    """
    frame = flip_frame(raw_frame, flip)
    process_frame(frame, threshold=threshold, render_cv_window=False, update_model=fresh)
    return crop_to_aspect_ratio(frame), get_last_tracking_result()


def draw_text(frame, text, position, scale=0.6, color=(0, 255, 255), thickness=2, shadow=True):
    """Draw yellow explanation text with a one-pixel black drop shadow."""
    if shadow:
        x, y = position
        cv2.putText(frame, text, (x + 1, y + 1), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness)
    cv2.putText(frame, text, position, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thickness)


def draw_eye_tracking(frame, ellipse, result):
    if max_observed_distance > 0:
        cv2.circle(frame, prev_model_center_avg, int(max_observed_distance), (255, 50, 50), 2)
    if model_centers:
        cv2.circle(frame, prev_model_center_avg, 5, (255, 255, 0), -1)
    if ellipse is not None:
        cv2.ellipse(frame, ellipse, (0, 255, 0), 2)
        pupil = tuple(map(int, ellipse[0]))
        cv2.circle(frame, pupil, 3, (255, 255, 0), -1)
        if model_centers:
            cv2.line(frame, prev_model_center_avg, pupil, (200, 255, 0), 2)
    state = "automatic" if eye_sphere_adjustment_enabled else "frozen"
    lines = ["No pupil" if result is None else f"Pupil confidence: {result['confidence']:.0%}",
             f"Model: {state}"]
    if result is not None and result["gaze_direction"] is not None:
        lines.append("Direction:" + ",".join(f"{v:.3f}" for v in result["gaze_direction"]))
    elif ellipse is not None:
        lines.append("Move eye around to initialize model")
    for index, line in enumerate(lines):
        draw_text(frame, line, (10, 25 + index * 23))
    draw_text(frame, "Q quit | F freeze | X reset | Space pause", (10, 120), scale=0.54)
    draw_text(frame, "R record | E flip eye | 1 eye cam | 2 front cam", (10, 146), scale=0.54)


class RecoveringCamera:
    """Capture independently so USB reads/reopens cannot stall the preview."""

    def __init__(self, source, fps):
        self.source = source
        self.fps = fps
        self.frame = None
        self.sequence = 0
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()

    def snapshot(self):
        with self.lock:
            # Published frames are never changed by the capture thread.
            return self.frame, self.sequence

    def run(self):
        while not self.stop.is_set():
            cap = None
            try:
                cap = open_camera(self.source, self.fps)
                while not self.stop.is_set():
                    ok, frame = cap.read()
                    if not ok or frame is None or frame.size == 0:
                        raise RuntimeError(f"Lost camera {self.source}")
                    with self.lock:
                        self.frame = frame
                        self.sequence += 1
            except (RuntimeError, cv2.error) as exc:
                print(f"{exc}; retrying camera in 1 second.", file=sys.stderr)
            finally:
                if cap is not None:
                    cap.release()
            if not self.stop.is_set():
                self.stop.wait(1.0)

    def close(self):
        self.stop.set()
        # Some USB drivers block inside read(). Keep window shutdown responsive;
        # the worker owns release() and will clean up when that call returns.
        self.thread.join(timeout=0.2)


class PairedRecorder:
    """One clock and frame index for both fixed-rate video files.

    The render loop publishes immutable pairs. Recording never requests camera
    frames or waits for tracking. Slow writes catch up using the cached pair.
    """

    FPS = 30

    def __init__(self, directory, frames):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        self.paths = [directory / f"{stamp}_{role}.avi" for role in ("eye", "front")]
        self.writers = []
        self.frames = tuple(frame.copy() for frame in frames)
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.end = float("inf")
        self.error = None
        self.count = 0
        try:
            for path, frame in zip(self.paths, self.frames):
                writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), self.FPS,
                                         (frame.shape[1], frame.shape[0]))
                self.writers.append(writer)
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create recording {path}")
        except Exception:
            for writer in self.writers:
                writer.release()
            raise
        self.start = time.perf_counter()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        print("Recording:", *(str(path) for path in self.paths))

    def publish(self, frames):
        pair = tuple(frame.copy() for frame in frames)
        with self.lock:
            self.frames = pair

    def run(self):
        try:
            while True:
                deadline = self.start + self.count / self.FPS
                with self.lock:
                    end = self.end
                if deadline >= end:
                    break
                remaining = deadline - time.perf_counter()
                if remaining > 0:
                    self.stop.wait(remaining)
                    continue
                with self.lock:
                    pair = self.frames
                for writer, frame in zip(self.writers, pair):
                    writer.write(frame)
                self.count += 1
        except Exception as exc:
            self.error = str(exc)
            print(f"Recording stopped: {exc}", file=sys.stderr)
        finally:
            for writer in self.writers:
                writer.release()

    def close(self):
        with self.lock:
            self.end = time.perf_counter()
        self.stop.set()
        self.thread.join()
        print(f"Saved {self.count} paired frames at {self.FPS} FPS.")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--eye-camera", default="0", help="Eye camera index or device path")
    parser.add_argument("--front-camera", default="2", help="Separate scene camera index or device path")
    parser.add_argument("--fps", type=float, default=30)
    parser.add_argument("--threshold", type=int, default=15, help="Pi pupil detector threshold offset")
    parser.add_argument("--record-dir", default=str(Path(__file__).resolve().parent / "recordings"))
    for role in ("eye", "front"):
        parser.add_argument(f"--{role}-flip", choices=("none", "vertical", "horizontal", "both"), default="none")
    args = parser.parse_args(argv)
    if not np.isfinite(args.fps) or args.fps <= 0 or not 0 <= args.threshold <= 255:
        parser.error("fps must be positive and finite; threshold must be 0..255")
    eye_source, front_source = camera_source(args.eye_camera), camera_source(args.front_camera)
    if eye_source == front_source:
        parser.error("Eye and front cameras must be distinct physical cameras")
    captures = []
    windows = ("Pi Eye Tracker", "Pi Front Camera")
    calibration = Calibration()
    reset_tracking_state()
    paused = False
    last_eye_sequence = 0
    eye_frame = np.zeros((480, 640, 3), dtype=np.uint8)
    front_frame = eye_frame.copy()
    result = None
    recorder = None
    processing_ms = None
    try:
        for source in (eye_source, front_source):
            captures.append(RecoveringCamera(source, args.fps))
        for window in windows:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        print("Move eye around to build model, then press C at each scene target.")
        while True:
            processing_start = None
            if not paused:
                raw_eye, sequence = captures[0].snapshot()
                raw_front, _ = captures[1].snapshot()
                if raw_front is not None:
                    front_frame = crop_to_aspect_ratio(flip_frame(raw_front, args.front_flip))
                if raw_eye is not None:
                    processing_start = time.perf_counter()
                    eye_frame, result = process_eye_frame(
                        raw_eye, args.threshold, args.eye_flip,
                        fresh=sequence != last_eye_sequence,
                    )
                    last_eye_sequence = sequence
            eye_view, front_view = eye_frame.copy(), front_frame.copy()
            ellipse = None if result is None else (
                tuple(result["pupil_ellipse"]["center"]),
                tuple(result["pupil_ellipse"]["axes"]),
                result["pupil_ellipse"]["angle_degrees"],
            )
            draw_eye_tracking(eye_view, ellipse, result)
            if processing_start is not None:
                processing_ms = (time.perf_counter() - processing_start) * 1000
            if processing_ms is not None:
                label = f"{1000 / max(processing_ms, 1e-6):.1f} FPS"
                draw_text(eye_view, label, (10, 460), scale=1.3, color=(0, 255, 0), thickness=2, shadow=False)
            calibration.draw(front_view, result)
            if paused:
                cv2.putText(front_view, "PAUSED", (10, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if recorder is not None:
                if recorder.error is not None:
                    recorder.close()
                    recorder = None
                else:
                    recorder.publish((eye_view, front_view))
            # Recording receives clean renders; the red indicator is preview-only.
            preview_eye, preview_front = eye_view.copy(), front_view.copy()
            if recorder is not None:
                for view in (preview_eye, preview_front):
                    cv2.putText(view, "REC 30 FPS", (480, 75), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            cv2.imshow(windows[0], preview_eye)
            cv2.imshow(windows[1], preview_front)
            key = cv2.waitKey(20 if paused else 10) & 0xFF
            if key in (27, ord("q"), ord("Q")) or any(cv2.getWindowProperty(w, cv2.WND_PROP_VISIBLE) < 1 for w in windows):
                break
            if key == ord(" "):
                paused = not paused
            elif key in (ord("1"), ord("2")):
                index = 0 if key == ord("1") else 1
                sources = [capture.source for capture in captures]
                available = available_camera_sources(sources)
                selected = next_camera_source(available, sources[index], sources[1 - index])
                if selected == sources[index] and len(available) == 2 and all(source in available for source in sources):
                    # With two physical cameras, cycling requires exchanging roles.
                    for capture in captures:
                        capture.close()
                    captures = [RecoveringCamera(source, args.fps) for source in reversed(sources)]
                    reset_tracking_state()
                    calibration.reset()
                    result = None
                    processing_ms = None
                    last_eye_sequence = 0
                    print(f"Swapped cameras: eye {sources[1]}, front {sources[0]}")
                elif selected == sources[index]:
                    print("No other available camera for this view.")
                else:
                    captures[index].close()
                    captures[index] = RecoveringCamera(selected, args.fps)
                    calibration.reset()
                    if index == 0:
                        reset_tracking_state()
                        result = None
                        processing_ms = None
                        last_eye_sequence = 0
                    print(f"{'Eye' if index == 0 else 'Front'} camera: {selected}")
            elif key in (ord("c"), ord("C")) and not paused:
                calibration.capture(result)
            elif key in (ord("f"), ord("F")):
                toggle_eye_sphere_adjustment()
                if eye_sphere_adjustment_enabled:
                    calibration.reset()
            elif key in (ord("e"), ord("E")):
                args.eye_flip = {"none": "vertical", "vertical": "none", "horizontal": "both", "both": "horizontal"}[args.eye_flip]
                eye_frame = cv2.flip(eye_frame, 0)
                reset_tracking_state()
                calibration.reset()
                result = None
                processing_ms = None
                last_eye_sequence = 0
            elif key in (ord("r"), ord("R")):
                if recorder is None:
                    try:
                        recorder = PairedRecorder(args.record_dir, (eye_view, front_view))
                    except (OSError, RuntimeError, cv2.error) as exc:
                        print(f"Could not start recording: {exc}", file=sys.stderr)
                else:
                    recorder.close()
                    recorder = None
            elif key in (ord("x"), ord("X")):
                reset_tracking_state()
                calibration.reset()
                result = None
    except (RuntimeError, cv2.error) as exc:
        print(f"Camera error: {exc}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        pass
    finally:
        if recorder is not None:
            recorder.close()
        for cap in captures:
            cap.close()
        cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
