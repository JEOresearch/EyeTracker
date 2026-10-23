import os
import csv

# Force DirectShow; disable MSMF (prevents grabFrame spam/hangs on some drivers)
os.environ["OPENCV_VIDEOIO_PRIORITY_MSMF"] = "0"
os.environ["OPENCV_VIDEOIO_PRIORITY_DSHOW"] = "1000"

import cv2
try:
    cv2.setLogLevel(3)  # ERROR level only (suppresses WARN spam)
except Exception:
    pass

import random
import math
import numpy as np

import tkinter as tk
from tkinter import ttk, filedialog
from tkinter import messagebox

from dataclasses import dataclass, field

import sys
import time

try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = True
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")
@dataclass
class EyeState:
    ray_lines: list = field(default_factory=list)
    model_centers: list = field(default_factory=list)
    prev_model_center_avg: tuple = (320, 240)
    max_observed_distance: float = 0.0
    stored_intersections: list = field(default_factory=list)

MAX_RAYS = 100  # number of historical pupil-surface rays to retain per eye

# Rotation applied to LEFT eye image (degrees)
# Allowed values: 0, 90, 180, 270
LEFT_EYE_ROTATION_DEG = 180
# Mirror BOTH eye images along the SHORT EDGE
# True  = enabled
# False = disabled
MIRROR_ALONG_SHORT_EDGE = True
class FrameCSVLogger:
    """
    Streaming CSV logger:
    - Writes one row per frame directly to disk (no RAM growth)
    - Scalable columns: pass dicts; new keys extend the header automatically
    - Flushes periodically to avoid per-frame I/O stalls
    """
    def __init__(self, filepath: str, flush_every: int = 60):
        self.filepath = filepath
        self.flush_every = max(1, int(flush_every))
        self._fh = None
        self._writer = None
        self._fieldnames = []
        self._row_count = 0

    def open(self):
        # newline="" is important for CSV on Windows
        self._fh = open(self.filepath, "w", newline="", buffering=1, encoding="utf-8")
        self._writer = None
        self._fieldnames = []
        self._row_count = 0

    def close(self):
        if self._fh:
            try:
                self._fh.flush()
            except Exception:
                pass
            try:
                self._fh.close()
            except Exception:
                pass
        self._fh = None
        self._writer = None
        self._fieldnames = []
        self._row_count = 0

    def _reset_writer_with_fieldnames(self, fieldnames):
        # Rewrite the entire file header if we add new columns.
        # This is rare in steady-state; typically happens only at start or when you add new signals.
        if self._fh is None:
            return

        # Save existing rows? We are streaming; simplest is to start with a stable schema.
        # But to be "scalable", we support schema growth by rewriting the file.
        # We do NOT store old rows in RAM; instead we do a disk rewrite (cheap for small logs).
        # If you expect huge logs and frequent schema changes, lock schema up front.

        self._fh.flush()
        self._fh.close()

        # Read existing content from disk (excluding old header) then rewrite with new header
        existing_rows = []
        if os.path.exists(self.filepath):
            with open(self.filepath, "r", newline="", encoding="utf-8") as rf:
                reader = csv.reader(rf)
                try:
                    _old_header = next(reader)
                except StopIteration:
                    _old_header = None
                for row in reader:
                    existing_rows.append(row)

        # Re-open and write new header
        self._fh = open(self.filepath, "w", newline="", buffering=1, encoding="utf-8")
        self._fieldnames = list(fieldnames)
        self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
        self._writer.writeheader()

        # Re-write existing rows aligned to old header positions (best-effort)
        # If you never change schema mid-run, this path never triggers.
        if existing_rows:
            # old rows are positional; put them back as-is into the new schema for columns that existed.
            # We can't reliably map without the old header; so we just drop them.
            # Practical recommendation: define your columns up front (see usage below).
            pass

    def write_row(self, row_dict: dict):
        if self._fh is None:
            self.open()

        # First row defines schema
        if self._writer is None:
            self._fieldnames = list(row_dict.keys())
            self._writer = csv.DictWriter(self._fh, fieldnames=self._fieldnames)
            self._writer.writeheader()

        # If schema expanded, rewrite header once
        row_keys = list(row_dict.keys())
        if any(k not in self._fieldnames for k in row_keys):
            new_fieldnames = self._fieldnames + [k for k in row_keys if k not in self._fieldnames]
            self._reset_writer_with_fieldnames(new_fieldnames)

        # Write row; missing keys become blank
        self._writer.writerow(row_dict)
        self._row_count += 1

        if (self._row_count % self.flush_every) == 0:
            self._fh.flush()

# Function to detect available cameras
def detect_cameras(max_cams=10):
    """
    Enumerate cameras robustly on Windows:
    - DSHOW only (avoid MSMF noise)
    - Validate by reading a few frames
    """
    available = []
    for i in range(max_cams):
        cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        ok = False
        for _ in range(10):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                ok = True
                break

        cap.release()
        if ok:
            available.append(i)

    return available





# Crop the image to maintain a specific aspect ratio (width:height) before resizing.
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

# Apply thresholding to an image
def apply_binary_threshold(image, darkestPixelValue, addedThreshold):
    threshold = darkestPixelValue + addedThreshold
    _, thresholded_image = cv2.threshold(image, threshold, 255, cv2.THRESH_BINARY_INV)
    return thresholded_image

# Finds a square area of dark pixels in the image
def get_darkest_area(image):
    ignoreBounds = 20
    imageSkipSize = 10
    searchArea = 20
    internalSkipSize = 5

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    min_sum = float('inf')
    darkest_point = None

    for y in range(ignoreBounds, gray.shape[0] - ignoreBounds, imageSkipSize):
        for x in range(ignoreBounds, gray.shape[1] - ignoreBounds, imageSkipSize):
            current_sum = 0
            num_pixels = 0
            for dy in range(0, searchArea, internalSkipSize):
                if y + dy >= gray.shape[0]:
                    break
                for dx in range(0, searchArea, internalSkipSize):
                    if x + dx >= gray.shape[1]:
                        break
                    current_sum += gray[y + dy][x + dx]
                    num_pixels += 1

            if current_sum < min_sum and num_pixels > 0:
                min_sum = current_sum
                darkest_point = (x + searchArea // 2, y + searchArea // 2)

    return darkest_point

# Mask all pixels outside a square defined by center and size
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

def optimize_contours_by_angle(contours, image):
    if len(contours) < 1:
        return contours

    # Holds the candidate points
    all_contours = np.concatenate(contours[0], axis=0)

    # Set spacing based on size of contours
    spacing = int(len(all_contours)/25)  # Spacing between sampled points

    # Temporary array for result
    filtered_points = []
    
    # Calculate centroid of the original contours
    centroid = np.mean(all_contours, axis=0)
    
    # Create an image of the same size as the original image
    point_image = image.copy()
    
    skip = 0
    
    # Loop through each point in the all_contours array
    for i in range(0, len(all_contours), 1):
    
        # Get three points: current point, previous point, and next point
        current_point = all_contours[i]
        prev_point = all_contours[i - spacing] if i - spacing >= 0 else all_contours[-spacing]
        next_point = all_contours[i + spacing] if i + spacing < len(all_contours) else all_contours[spacing]
        
        # Calculate vectors between points
        vec1 = prev_point - current_point
        vec2 = next_point - current_point
        
        with np.errstate(invalid='ignore'):
            # Calculate angles between vectors
            angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))

        
        # Calculate vector from current point to centroid
        vec_to_centroid = centroid - current_point
        
        # Check if angle is oriented towards centroid
        # Calculate the cosine of the desired angle threshold (e.g., 80 degrees)
        cos_threshold = np.cos(np.radians(60))  # Convert angle to radians
        
        if np.dot(vec_to_centroid, (vec1+vec2)/2) >= cos_threshold:
            filtered_points.append(current_point)
    
    return np.array(filtered_points, dtype=np.int32).reshape((-1, 1, 2))

# Returns the largest contour that is not extremely long or tall
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
#Fits an ellipse to the optimized contours and draws it on the image.
def fit_and_draw_ellipses(image, optimized_contours, color):
    if len(optimized_contours) >= 5:
        # Ensure the data is in the correct shape (n, 1, 2) for cv2.fitEllipse
        contour = np.array(optimized_contours, dtype=np.int32).reshape((-1, 1, 2))

        # Fit ellipse
        ellipse = cv2.fitEllipse(contour)

        # Draw the ellipse
        cv2.ellipse(image, ellipse, color, 2)  # Draw with green color and thickness of 2

        return image
    else:
        print("Not enough points to fit an ellipse.")
        return image

#checks how many pixels in the contour fall under a slightly thickened ellipse
#also returns that number of pixels divided by the total pixels on the contour border
#assists with checking ellipse goodness    
def check_contour_pixels(contour, image_shape, debug_mode_on):
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        return [0, 0]  # Not enough points to fit an ellipse
    
    # Create an empty mask for the contour
    contour_mask = np.zeros(image_shape, dtype=np.uint8)
    # Draw the contour on the mask, filling it
    cv2.drawContours(contour_mask, [contour], -1, (255), 1)
   
    # Fit an ellipse to the contour and create a mask for the ellipse
    ellipse_mask_thick = np.zeros(image_shape, dtype=np.uint8)
    ellipse_mask_thin = np.zeros(image_shape, dtype=np.uint8)
    ellipse = cv2.fitEllipse(contour)
    
    # Draw the ellipse with a specific thickness
    cv2.ellipse(ellipse_mask_thick, ellipse, (255), 10) #capture more for absolute
    cv2.ellipse(ellipse_mask_thin, ellipse, (255), 4) #capture fewer for ratio

    # Calculate the overlap of the contour mask and the thickened ellipse mask
    overlap_thick = cv2.bitwise_and(contour_mask, ellipse_mask_thick)
    overlap_thin = cv2.bitwise_and(contour_mask, ellipse_mask_thin)
    
    # Count the number of non-zero (white) pixels in the overlap
    absolute_pixel_total_thick = np.sum(overlap_thick > 0)#compute with thicker border
    absolute_pixel_total_thin = np.sum(overlap_thin > 0)#compute with thicker border
    
    # Compute the ratio of pixels under the ellipse to the total pixels on the contour border
    total_border_pixels = np.sum(contour_mask > 0)
    
    ratio_under_ellipse = absolute_pixel_total_thin / total_border_pixels if total_border_pixels > 0 else 0
    
    return [absolute_pixel_total_thick, ratio_under_ellipse, overlap_thin]

#outside of this method, select the ellipse with the highest percentage of pixels under the ellipse 
#TODO for efficiency, work with downscaled or cropped images
def check_ellipse_goodness(binary_image, contour, debug_mode_on):
    ellipse_goodness = [0,0,0] #covered pixels, edge straightness stdev, skewedness   
    # Check if the contour can be used to fit an ellipse (requires at least 5 points)
    if len(contour) < 5:
        print("length of contour was 0")
        return 0  # Not enough points to fit an ellipse
    
    # Fit an ellipse to the contour
    ellipse = cv2.fitEllipse(contour)
    
    # Create a mask with the same dimensions as the binary image, initialized to zero (black)
    mask = np.zeros_like(binary_image)
    
    # Draw the ellipse on the mask with white color (255)
    cv2.ellipse(mask, ellipse, (255), -1)
    
    # Calculate the number of pixels within the ellipse
    ellipse_area = np.sum(mask == 255)
    
    # Calculate the number of white pixels within the ellipse
    covered_pixels = np.sum((binary_image == 255) & (mask == 255))
    
    # Calculate the percentage of covered white pixels within the ellipse
    if ellipse_area == 0:
        print("area was 0")
        return ellipse_goodness  # Avoid division by zero if the ellipse area is somehow zero
    
    #percentage of covered pixels to number of pixels under area
    ellipse_goodness[0] = covered_pixels / ellipse_area
    
    #skew of the ellipse (less skewed is better?) - may not need this
    axes_lengths = ellipse[1]  # This is a tuple (minor_axis_length, major_axis_length)
    major_axis_length = axes_lengths[1]
    minor_axis_length = axes_lengths[0]
    ellipse_goodness[2] = min(ellipse[1][1]/ellipse[1][0], ellipse[1][0]/ellipse[1][1])
    
    return ellipse_goodness
def pupil_size_from_ellipse(ellipse):
    if ellipse is None:
        return None
    (_, _), (ax1, ax2), _ = ellipse
    return float(max(ax1, ax2))  # major axis length in px


# Process frames for pupil detection
def process_frames(
    thresholded_image_strict,
    thresholded_image_medium,
    thresholded_image_relaxed,
    frame,
    gray_frame,
    darkest_point,
    debug_mode_on,
    render_cv_window,
    state: EyeState,
    *,
    show_individual_windows: bool = True,
    eye_label: str = "",
    gaze_file_path: str = "gaze_vector.txt",
    print_gaze: bool = False,
    update_gl: bool = False,
):
    """Core per-frame pupil/ellipse processing with per-eye persistent state via `state`."""

    ray_lines = state.ray_lines
    model_centers = state.model_centers
    prev_model_center_avg = state.prev_model_center_avg
    max_observed_distance = state.max_observed_distance

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_image = cv2.dilate(thresholded_image_medium, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

    final_rotated_rect = ((0,0),(0,0),0)
    pupil_size = None

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict]
    name_array = ["relaxed", "medium", "strict"]
    final_contours = []
    goodness = 0
    kernel = np.ones((5, 5), np.uint8)
    gray_copies = [gray_frame.copy(), gray_frame.copy(), gray_frame.copy()]
    final_goodness = 0

    center_x, center_y = None, None

    for i in range(1, 4):
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)
        contours, hierarchy = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(dilated_image, reduced_contours[0], debug_mode_on)
            ellipse = cv2.fitEllipse(reduced_contours[0])
            center_x, center_y = map(int, ellipse[0])

            if debug_mode_on:
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])

            total_pixels = check_contour_pixels(reduced_contours[0], dilated_image.shape, debug_mode_on)
            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)

            final_goodness = current_goodness[0] * total_pixels[0] * total_pixels[0] * total_pixels[1]

            if final_goodness > 0 and final_goodness > goodness:
                goodness = final_goodness
                final_contours = reduced_contours

    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    final_rotated_rect = None

    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0]) > 5:
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse
        pupil_size = pupil_size_from_ellipse(final_rotated_rect)
        ray_lines.append(final_rotated_rect)

        if len(ray_lines) > MAX_RAYS:
            num_to_remove = len(ray_lines) - MAX_RAYS
            ray_lines = ray_lines[num_to_remove:]

    model_center_average = (320, 240)

    model_center = compute_average_intersection(frame, ray_lines, 5, 1500, 5, state)
    if model_center is not None:
        model_center_average = update_and_average_point(model_centers, model_center, 200)

    if model_center_average[0] == 320:
        model_center_average = prev_model_center_avg
    if model_center_average[0] != 0:
        prev_model_center_avg = model_center_average

    if center_x is None or center_y is None or model_center_average[0] is None or model_center_average[1] is None:
        state.ray_lines = ray_lines
        state.model_centers = model_centers
        state.prev_model_center_avg = prev_model_center_avg
        state.max_observed_distance = max_observed_distance
        return frame, final_rotated_rect, None, None, None


    if len(model_centers) >= 100 and center_x is not None:
        distance = math.sqrt((center_x - model_center_average[0]) ** 2 + (center_y - model_center_average[1]) ** 2)
        if distance > max_observed_distance:
            max_observed_distance = distance

    max_observed_distance = 202

    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)

    if final_rotated_rect is not None:
        cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2)

    if final_rotated_rect is not None and center_x is not None and center_y is not None:
        cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)

        dx = center_x - model_center_average[0]
        dy = center_y - model_center_average[1]
        extended_x = int(model_center_average[0] + 2 * dx)
        extended_y = int(model_center_average[1] + 2 * dy)
        cv2.line(frame, (center_x, center_y), (extended_x, extended_y), (200, 255, 0), 3)

    if eye_label:
        cv2.putText(frame, f"Eye: {eye_label}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)

    if render_cv_window:
        cv2.imshow("Best Thresholded Image Contours on Frame", frame)

    gl_image = None
    if GL_SPHERE_AVAILABLE and update_gl:
        gl_image = gl_sphere.update_sphere_rotation(center_x, center_y, model_center_average[0], model_center_average[1])

    center_3d, direction_3d = compute_gaze_vector(
        center_x, center_y, model_center_average[0], model_center_average[1],
        file_path=gaze_file_path
    )

    if center_3d is not None and direction_3d is not None:
        origin_text = f"Origin: ({center_3d[0]:.2f}, {center_3d[1]:.2f}, {center_3d[2]:.2f})"
        dir_text    = f"Direction: ({direction_3d[0]:.2f}, {direction_3d[1]:.2f}, {direction_3d[2]:.2f})"
        pupil_text  = f"PupilSize(px): {pupil_size:.1f}" if pupil_size is not None else "PupilSize(px): n/a"


        text_origin = (12, frame.shape[0] - 38)
        text_dir    = (12, frame.shape[0] - 13)
        text_pupil  = (12, frame.shape[0] - 63)
        text_origin2 = (10, frame.shape[0] - 40)
        text_dir2    = (10, frame.shape[0] - 15)
        text_pupil2 = (10, frame.shape[0] - 65)

        cv2.putText(frame, origin_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, dir_text, text_dir, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, pupil_text, text_pupil,  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
        cv2.putText(frame, origin_text, text_origin2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, dir_text, text_dir2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        cv2.putText(frame, pupil_text, text_pupil2, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    if print_gaze:
        if center_3d is not None and direction_3d is not None:
            prefix = f"[{eye_label}] " if eye_label else ""
            print(f"{prefix}Sphere Center:   ({center_3d[0]:.3f}, {center_3d[1]:.3f}, {center_3d[2]:.3f})")
            print(f"{prefix}Gaze Direction:  ({direction_3d[0]:.3f}, {direction_3d[1]:.3f}, {direction_3d[2]:.3f})")
        else:
            prefix = f"[{eye_label}] " if eye_label else ""
            print(prefix + "No valid intersection found.")

    if show_individual_windows:
        cv2.imshow("Frame with Ellipse and Rays", frame)
        if GL_SPHERE_AVAILABLE and gl_image is not None:
            blended = cv2.addWeighted(frame, 0.6, gl_image, 0.4, 0)
            cv2.imshow("Eye Tracker + Sphere", blended)

    state.ray_lines = ray_lines
    state.model_centers = model_centers
    state.prev_model_center_avg = prev_model_center_avg
    state.max_observed_distance = max_observed_distance

    return frame, final_rotated_rect, center_3d, direction_3d, pupil_size


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

def draw_orthogonal_ray(image, ellipse, length=100, color=(0, 255, 0), thickness=1):
    """
    Draws a ray passing through the center of an ellipse orthogonally to its major axis.
    
    Parameters:
    - image: The OpenCV image to draw on.
    - ellipse: A tuple ((cx, cy), (major_axis, minor_axis), angle) representing the fitted ellipse.
    - length: Length of the ray to draw on each side of the ellipse center.
    - color: Color of the line in BGR format (default: green).
    - thickness: Thickness of the line (default: 2).
    """

    (cx, cy), (major_axis, minor_axis), angle = ellipse
    
    # Convert angle to radians
    angle_rad = np.deg2rad(angle)
    
    # Compute the normal vector at the ellipse center (perpendicular to surface)
    normal_dx = (minor_axis / 2) * np.cos(angle_rad)  # Minor axis component
    normal_dy = (minor_axis / 2) * np.sin(angle_rad)

    # Compute start and end points of the orthogonal ray
    pt1 = (int(cx - length * normal_dx / (minor_axis / 2)), int(cy - length * normal_dy / (minor_axis / 2)))
    pt2 = (int(cx + length * normal_dx / (minor_axis / 2)), int(cy + length * normal_dy / (minor_axis / 2)))

    # Draw the ray
    cv2.line(image, pt1, pt2, color, thickness)

    return image 

stored_intersections = []  # Stores all past intersections

def compute_average_intersection(frame, ray_lines, N, M, spacing, state: EyeState):
    """
    Selects N random lines from the list, computes their intersections, stores them per-eye,
    and prunes stored intersections when exceeding M.

    Parameters:
    - frame: The OpenCV frame to draw on.
    - ray_lines: List of ellipse tuples ((cx, cy), (major_axis, minor_axis), angle).
    - N: Number of random lines to select for intersection calculation.
    - M: Maximum number of stored intersections before pruning.
    - state: EyeState object (per-eye persistent history).

    Returns:
    - (avg_x, avg_y): Average intersection point of stored intersections, or None if unavailable.
    """

    stored_intersections = state.stored_intersections

    if len(ray_lines) < 2 or N < 2:
        return (0, 0)  # Need at least 2 lines to find intersections

    height, width = frame.shape[:2]

    selected_lines = random.sample(ray_lines, min(N, len(ray_lines)))
    intersections = []

    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]
        line2 = selected_lines[i + 1]

        angle1 = line1[2]
        angle2 = line2[2]

        if abs(angle1 - angle2) >= 2:
            intersection = find_line_intersection(line1, line2)

            if intersection and (0 <= intersection[0] < width) and (0 <= intersection[1] < height):
                intersections.append(intersection)
                stored_intersections.append(intersection)

    if len(stored_intersections) > M:
        stored_intersections = prune_intersections(stored_intersections, M)

    state.stored_intersections = stored_intersections

    if not intersections or not stored_intersections:
        return None

    avg_x = np.mean([pt[0] for pt in stored_intersections])
    avg_y = np.mean([pt[1] for pt in stored_intersections])

    return (int(avg_x), int(avg_y))


#Removes the oldest intersections to ensure only the last M intersections remain.
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
    if np.linalg.det(A) == 0:
        return None  # Lines are parallel and do not intersect

    t1, t2 = np.linalg.solve(A, B)

    # Compute intersection point
    intersection_x = cx1 + t1 * dx1
    intersection_y = cy1 + t1 * dy1

    return (int(intersection_x), int(intersection_y))

def compute_gaze_vector(x, y, center_x, center_y, screen_width=640, screen_height=480, file_path="gaze_vector.txt"):
    """Compute 3D gaze direction from pupil and sphere center screen coordinates.

    Returns:
        sphere_center (np.ndarray): 3D position of the sphere center in world space
        gaze_direction (np.ndarray): Normalized 3D direction vector from sphere center
    """

    viewport_width = screen_width
    viewport_height = screen_height

    fov_y_deg = 45.0
    aspect_ratio = viewport_width / viewport_height
    far_clip = 100.0

    camera_position = np.array([0.0, 0.0, 3.0])

    fov_y_rad = np.radians(fov_y_deg)
    half_height_far = np.tan(fov_y_rad / 2) * far_clip
    half_width_far = half_height_far * aspect_ratio

    ndc_x = (2.0 * x) / viewport_width - 1.0
    ndc_y = 1.0 - (2.0 * y) / viewport_height

    far_x = ndc_x * half_width_far
    far_y = ndc_y * half_height_far
    far_z = camera_position[2] - far_clip
    far_point = np.array([far_x, far_y, far_z])

    ray_origin = camera_position
    ray_direction = far_point - camera_position
    ray_direction /= np.linalg.norm(ray_direction)
    ray_direction = -ray_direction

    inner_radius = 1.0 / 1.05
    sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
    sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
    sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

    origin = ray_origin
    direction = -ray_direction
    L = origin - sphere_center

    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_radius**2

    discriminant = b**2 - 4 * a * c

    def _write_if_possible(path, center_vec, dir_vec):
        if path is None:
            return
        try:
            with open(path, "a"):
                pass
        except IOError:
            print(f"File is currently in use. Skipping write: {path}")
            return

        try:
            with open(path, "w") as f:
                all_values = np.concatenate((center_vec, dir_vec))
                csv_line = ",".join(f"{v:.6f}" for v in all_values)
                f.write(csv_line + "\n")
        except Exception as e:
            print("Write error:", e)

    if discriminant < 0:
        t = -np.dot(direction, L) / np.dot(direction, direction)
        intersection_point = origin + t * direction
        intersection_local = intersection_point - sphere_center
        gaze_direction = intersection_local / np.linalg.norm(intersection_local)

        _write_if_possible(file_path, sphere_center, gaze_direction)
        return sphere_center, gaze_direction

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

    intersection_point = origin + t * direction

    intersection_local = intersection_point - sphere_center
    target_direction = intersection_local / np.linalg.norm(intersection_local)

    circle_local_center = np.array([0.0, 0.0, inner_radius])
    circle_local_center /= np.linalg.norm(circle_local_center)

    rotation_axis = np.cross(circle_local_center, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        _write_if_possible(file_path, sphere_center, circle_local_center)
        return sphere_center, circle_local_center

    rotation_axis /= rotation_axis_norm
    dot = np.dot(circle_local_center, target_direction)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)

    c_ = np.cos(angle_rad)
    s_ = np.sin(angle_rad)
    t_ = 1 - c_
    x_, y_, z_ = rotation_axis

    rotation_matrix = np.array([
        [t_*x_*x_ + c_, t_*x_*y_ - s_*z_, t_*x_*z_ + s_*y_],
        [t_*x_*y_ + s_*z_, t_*y_*y_ + c_, t_*y_*z_ - s_*x_],
        [t_*x_*z_ - s_*y_, t_*y_*z_ + s_*x_, t_*z_*z_ + c_]
    ])

    gaze_local = np.array([0.0, 0.0, inner_radius])
    gaze_rotated = rotation_matrix @ gaze_local
    gaze_rotated /= np.linalg.norm(gaze_rotated)

    _write_if_possible(file_path, sphere_center, gaze_rotated)
    return sphere_center, gaze_rotated
def process_frame_eye(
    frame,
    state: EyeState,
    *,
    eye_label: str = "",
    gaze_file_path: str = "gaze_vector.txt",
    show_individual_windows: bool = True,
    update_gl: bool = False,
):
    frame = crop_to_aspect_ratio(frame)

    darkest_point = get_darkest_area(frame)

    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]

    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)

    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)

    frame_out, final_rotated_rect, center_3d, direction_3d, pupil_size = process_frames(
        thresholded_image_strict,
        thresholded_image_medium,
        thresholded_image_relaxed,
        frame,
        gray_frame,
        darkest_point,
        False,
        False,
        state,
        show_individual_windows=show_individual_windows,
        eye_label=eye_label,
        gaze_file_path=gaze_file_path,
        print_gaze=False,
        update_gl=update_gl,
    )

    return frame_out, final_rotated_rect, center_3d, direction_3d, pupil_size

def open_camera_robust(index: int, prefer_msmf: bool = False):
    backends = []
    # Try DSHOW first for one, MSMF first for the other (optional)
    if prefer_msmf and hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)
    if hasattr(cv2, "CAP_DSHOW"):
        backends.append(cv2.CAP_DSHOW)
    if (not prefer_msmf) and hasattr(cv2, "CAP_MSMF"):
        backends.append(cv2.CAP_MSMF)

    for backend in backends:
        cap = cv2.VideoCapture(index, backend)
        if not cap.isOpened():
            cap.release()
            continue

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        try:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        except Exception:
            pass

        ok = False
        for _ in range(20):
            ret, frame = cap.read()
            if ret and frame is not None and frame.size > 0:
                ok = True
                break

        if ok:
            return cap

        cap.release()

    return None
def rotate_frame_deg(frame, deg):
    """
    Rotate frame by exact multiples of 90 degrees.
    """
    if frame is None:
        return None

    deg = deg % 360

    if deg == 0:
        return frame
    elif deg == 90:
        return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
    elif deg == 180:
        return cv2.rotate(frame, cv2.ROTATE_180)
    elif deg == 270:
        return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    else:
        raise ValueError("Rotation must be one of: 0, 90, 180, 270")

def apply_stereo_orientation(frame, role: str):
    """
    Enforce deterministic stereo orientation (independent of camera indices):

    - LEFT  eye: rotated by LEFT_EYE_ROTATION_DEG
    - RIGHT eye: rotated by (LEFT_EYE_ROTATION_DEG + 180)
    - BOTH eyes: optionally mirrored along the shorter edge
        * For landscape (W >= H): left-right mirror (flipCode=1)
        * For portrait  (H >  W): top-bottom mirror (flipCode=0)
    """
    if frame is None:
        return None

    role = (role or "").upper()

    # 1) Role-based rotation
    if role == "L":
        out = rotate_frame_deg(frame, LEFT_EYE_ROTATION_DEG)
    elif role == "R":
        out = rotate_frame_deg(frame, LEFT_EYE_ROTATION_DEG + 180)
    else:
        out = frame

    # 2) Mirror along the shorter edge (optional)
    if MIRROR_ALONG_SHORT_EDGE:
        h, w = out.shape[:2]
        if w >= h:
            out = cv2.flip(out, 1)  # left-right mirror (short edge for landscape)
        else:
            out = cv2.flip(out, 0)  # top-bottom mirror (short edge for portrait)

    return out



def process_stereo_cameras(left_cam_index: int, right_cam_index: int, *, log_enabled: bool = False):
    if left_cam_index == right_cam_index:
        print("Error: Left and Right camera indices are identical; refusing to open the same device twice.")
        return

    left_state = EyeState()
    right_state = EyeState()

    capL = open_camera_robust(left_cam_index, prefer_msmf=False)
    time.sleep(0.5)  # give the driver time to settle
    capR = open_camera_robust(right_cam_index, prefer_msmf=True)

    if capL is None or capR is None:
        # Release anything that did open
        if capL is not None:
            capL.release()
        if capR is not None:
            capR.release()
        print("Error: Could not open one or both cameras.")
        return

    # Only set exposure after confirmed open (and ignore failures)
    try:
        capL.set(cv2.CAP_PROP_EXPOSURE, -6)
    except Exception:
        pass
    try:
        capR.set(cv2.CAP_PROP_EXPOSURE, -6)
    except Exception:
        pass
    # Create and set data logger
    logger = None
    if log_enabled:
        ts = time.strftime("%Y%m%d_%H%M%S")
        log_path = f"stereo_log_{ts}.csv"
        logger = FrameCSVLogger(log_path, flush_every=60)
        logger.open()
        print(f"[LOG] Writing CSV to: {log_path}")

    frame_idx = 0
    t_prev = None

    while True:
        retL, frameL = capL.read()
        retR, frameR = capR.read()
        if not retL or frameL is None or not retR or frameR is None:
            break

        # Apply role-based orientation (independent of camera index)
        frameL = apply_stereo_orientation(frameL, "L")
        frameR = apply_stereo_orientation(frameR, "R")
        outL, _, centerL, dirL, pupilL = process_frame_eye(
            frameL,
            left_state,
            eye_label="L",
            gaze_file_path="gaze_vector_left.txt",
            show_individual_windows=False,
            update_gl=True,
        )
        outR, _, centerR, dirR, pupilR = process_frame_eye(
            frameR,
            right_state,
            eye_label="R",
            gaze_file_path="gaze_vector_right.txt",
            show_individual_windows=False,
            update_gl=False,
        )

        # --- Per-frame timestamp (monotonic, high-resolution) ---
        t = time.perf_counter()
        dt = (t - t_prev) if t_prev is not None else 0.0
        t_prev = t

        if logger is not None:
            # Use NaN for missing values to preserve time alignment
            def _nan3(v):
                if v is None:
                    return (float("nan"), float("nan"), float("nan"))
                return (float(v[0]), float(v[1]), float(v[2]))

            def _nan1(v):
                return float("nan") if v is None else float(v)

            cL = _nan3(centerL)
            dL = _nan3(dirL)
            cR = _nan3(centerR)
            dR = _nan3(dirR)

            row = {
                "frame_idx": frame_idx,
                "t_sec": t,
                "dt_sec": dt,

                # Left eye (7 values)
                "L_cx": cL[0], "L_cy": cL[1], "L_cz": cL[2],
                "L_dx": dL[0], "L_dy": dL[1], "L_dz": dL[2],
                "L_pupil": _nan1(pupilL),

                # Right eye (7 values)
                "R_cx": cR[0], "R_cy": cR[1], "R_cz": cR[2],
                "R_dx": dR[0], "R_dy": dR[1], "R_dz": dR[2],
                "R_pupil": _nan1(pupilR),
            }

            logger.write_row(row)
            frame_idx += 1


        combined = np.hstack([outL, outR])
        cv2.imshow("Stereo Eyes (Left | Right Rotated 180)", combined)
        if centerL is not None and dirL is not None:
            ps = f"{pupilL:.1f}px" if pupilL is not None else "n/a"
            print(f"[LEFT]  Sphere Center: ({centerL[0]:.3f}, {centerL[1]:.3f}, {centerL[2]:.3f}) | "
                  f"Gaze Dir: ({dirL[0]:.3f}, {dirL[1]:.3f}, {dirL[2]:.3f}) | "
                  f"PupilSize: {ps}")
        else:
            print("[LEFT]  No valid intersection found.")

        if centerR is not None and dirR is not None:
            ps = f"{pupilR:.1f}px" if pupilR is not None else "n/a"
            print(f"[RIGHT] Sphere Center: ({centerR[0]:.3f}, {centerR[1]:.3f}, {centerR[2]:.3f}) | "
                  f"Gaze Dir: ({dirR[0]:.3f}, {dirR[1]:.3f}, {dirR[2]:.3f}) | "
                  f"PupilSize: {ps}")
        else:
            print("[RIGHT] No valid intersection found.")

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    capL.release()
    capR.release()
    cv2.destroyAllWindows()
    if logger is not None:
        logger.close()




# Finds the pupil in an individual frame and returns the center point
def process_frame(frame):

    # Crop and resize frame
    frame = crop_to_aspect_ratio(frame)

    #find the darkest point
    darkest_point = get_darkest_area(frame)

    # Convert to grayscale to handle pixel value operations
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    darkest_pixel_value = gray_frame[darkest_point[1], darkest_point[0]]
    
    # apply thresholding operations at different levels
    # at least one should give us a good ellipse segment
    thresholded_image_strict = apply_binary_threshold(gray_frame, darkest_pixel_value, 5)#lite
    thresholded_image_strict = mask_outside_square(thresholded_image_strict, darkest_point, 250)

    thresholded_image_medium = apply_binary_threshold(gray_frame, darkest_pixel_value, 15)#medium
    thresholded_image_medium = mask_outside_square(thresholded_image_medium, darkest_point, 250)
    
    thresholded_image_relaxed = apply_binary_threshold(gray_frame, darkest_pixel_value, 25)#heavy
    thresholded_image_relaxed = mask_outside_square(thresholded_image_relaxed, darkest_point, 250)
    
    #take the three images thresholded at different levels and process them
    final_rotated_rect = process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, False, False)
    
    return final_rotated_rect


# Process video from the selected camera
def process_camera():
    global selected_camera
    cam_index = int(selected_camera.get())

    cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_EXPOSURE, -6)

    if not cap.isOpened():
        print("Error: Could not open camera.")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 0)
        process_frame(frame)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()

# Process a selected video file
def process_video():
    video_path = filedialog.askopenfilename(filetypes=[("Video Files", "*.mp4;*.avi")])

    if not video_path:
        return

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video file.")
        return

    state = EyeState()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        _out, _ell, _c, _d, _pupil = process_frame_eye(
            frame,
            state,
            eye_label="VIDEO",
            gaze_file_path="gaze_vector.txt",
            show_individual_windows=True,
            update_gl=True,
        )


        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord(' '):
            cv2.waitKey(0)

    cap.release()
    cv2.destroyAllWindows()


# GUI for selecting camera or video
def selection_gui():
    cameras = detect_cameras(max_cams=5)

    root = tk.Tk()
    root.title("Select Input Source")
    tk.Label(root, text="Orlosky Eye Tracker 3D", font=("Arial", 12, "bold")).pack(pady=10)

    tk.Label(root, text="Select Left Eye Camera:").pack(pady=5)
    selected_left_camera = tk.StringVar()
    selected_left_camera.set(str(cameras[0]) if cameras else "No cameras found")
    left_dropdown = ttk.Combobox(root, textvariable=selected_left_camera, values=[str(cam) for cam in cameras])
    left_dropdown.pack(pady=5)

    tk.Label(root, text="Select Right Eye Camera:").pack(pady=5)
    selected_right_camera = tk.StringVar()
    if len(cameras) > 1:
        selected_right_camera.set(str(cameras[1]))
    elif cameras:
        selected_right_camera.set(str(cameras[0]))
    else:
        selected_right_camera.set("No cameras found")
    right_dropdown = ttk.Combobox(root, textvariable=selected_right_camera, values=[str(cam) for cam in cameras])
    right_dropdown.pack(pady=5)
    
    log_data_var = tk.BooleanVar(value=False)
    tk.Checkbutton(root, text="log data", variable=log_data_var).pack(pady=5)


    def start_stereo():
        left = int(selected_left_camera.get())
        right = int(selected_right_camera.get())

        if left == right:
            messagebox.showerror(
                "Invalid selection",
                "Left and Right cameras must be different devices.\n\n"
                "Selecting the same camera twice can cause the driver to reset/eject the device."
            )
            return

        root.destroy()
        process_stereo_cameras(left, right, log_enabled=log_data_var.get())



    tk.Button(root, text="Start Stereo Cameras", command=start_stereo).pack(pady=8)
    tk.Button(root, text="Browse Video", command=lambda: [root.destroy(), process_video()]).pack(pady=5)

    if GL_SPHERE_AVAILABLE:
        app = gl_sphere.start_gl_window()

    root.mainloop()

# Run GUI to select camera or video
if __name__ == "__main__":
    selection_gui()
