"""Front-camera tracker: modern tracking/UI with five-point camera calibration.

Select distinct eye and forward cameras. Press C at center, top, bottom, left and right targets; F affixes/resumes sphere localization (resuming clears calibration).
"""

import cv2
import random
import math
import numpy as np
import os
import tkinter as tk
from tkinter import ttk, filedialog
import sys
import time

# Allow launching this script directly from its tracker subfolder.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import gl_sphere
    GL_SPHERE_AVAILABLE = True
except ImportError:
    GL_SPHERE_AVAILABLE = False
    print("gl_sphere module not found. OpenGL rendering will be disabled.")

ray_lines = []  # Store recent pupil ellipse rays
model_centers = []  # Store recent estimated eye centers
min_model_centers = 100  # Minimum model centers required before updating sphere radius
max_rays = 100  # Limit the number of stored pupil rays
prev_model_center_avg = (320,240)  # Preserve the last valid eye center
max_observed_distance = 0  # Initialize adaptive radius
last_sphere_radius_ellipse = None  # Last pupil ellipse used to expand the eye sphere radius
last_gaze_vector_text = None  # Preserve vector text during temporary detection dropouts
eye_sphere_adjustment_enabled = True  # Toggle automatic eye center/radius updates with F
render_intersections = False  # Release view omits intersection debugging overlays
pupil_confidence_threshold = 0.85  # Minimum pupil confidence for storing a ray
pupil_confidence_threshold_sphere = 0.90  # Minimum pupil confidence for determining eye sphere radius
intersection_ray_count = 4  # Rays sampled for each intersection estimate
minimum_intersection_angle_degrees = 8  # Minimum angle between sampled rays
last_tracking_result = None  # Store the latest tracker output

capture_stuck_ellipses = False     # toggled with 'e'
stuck_ellipses = []                # saved pupil ellipses
capture_frame_counter = 0          # counts processed frames while capture is enabled
ellipse_capture_interval = 5      # save one ellipse every 10 frames
max_stuck_ellipses = 20           # safety cap

CAMERA_CAPTURE_MODES = (
    (
        ("MSMF", cv2.CAP_MSMF, None),
        ("DirectShow + mp4v", cv2.CAP_DSHOW, "XVID"),
    )
    if sys.platform == "win32"
    else (("Auto", cv2.CAP_ANY, None),)
)



# --- Gaze → external camera projection globals ---
last_sphere_center = None
last_gaze_dir = None

calibrated = False
calibration_target_margin = 0.15  # Targets sit 15% inward from each image edge.
calibration_samples = []
calibration_basis = None
calibration_matrix = None
calibration_message = "Align your finger and gaze with CENTER; press C."
calibrated_sphere_center = None 

sphere_center_locked_2d = False
locked_model_center_avg = prev_model_center_avg

# External camera / screen params (for 640x480)
EXT_WIDTH = 640
EXT_HEIGHT = 480
EXT_CX = EXT_WIDTH // 2
EXT_CY = EXT_HEIGHT // 2

# Approximate focal length in pixels (simple pinhole model)
EXT_FX = 600.0
EXT_FY = 600.0

# Current red circle position on external camera
circle_x = EXT_CX
circle_y = EXT_CY

ACCURACY_GRID_SIZE = 5
FRONT_DIAGONAL_FOV_DEGREES = 60.0
accuracy_grid_visible = False
accuracy_grid_index = -1
accuracy_grid_samples = []
accuracy_grid_message = "Press T to start at the upper-left target."


def open_camera_capture(cam_index, start_mode=0):
    for mode_index in range(start_mode, len(CAMERA_CAPTURE_MODES)):
        mode_name, backend, fourcc = CAMERA_CAPTURE_MODES[mode_index]
        cap = cv2.VideoCapture(cam_index, backend)
        if fourcc is not None:
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        cap.set(cv2.CAP_PROP_FPS, 30)
        ret, frame = cap.read() if cap.isOpened() else (False, None)

        if ret:
            return cap, frame, mode_index

        cap.release()
        print(f"Could not grab a frame from camera {cam_index} with {mode_name}.")

    return None, None, None


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
            print(f"Radius-increasing ellipse confidence: {current_pupil_confidence * 100:.2f}%")


def get_eye_sphere_adjustment_prompt():
    if eye_sphere_adjustment_enabled and not sphere_center_locked_2d:
        return "Press F to affix eyeball sphere"
    return "Press F again to engage automatic eye sphere adjustment"


def toggle_eye_sphere_adjustment():
    global eye_sphere_adjustment_enabled
    if sphere_center_locked_2d:
        reset_front_calibration()
        eye_sphere_adjustment_enabled = True
        print("Calibration cleared; automatic eye sphere adjustment resumed.")
    else:
        eye_sphere_adjustment_enabled = not eye_sphere_adjustment_enabled
        print(f"Eye sphere adjustment: {'automatic' if eye_sphere_adjustment_enabled else 'affixed'}")


def reset_tracking_state():
    global ray_lines
    global model_centers
    global prev_model_center_avg
    global max_observed_distance
    global last_sphere_radius_ellipse
    global eye_sphere_adjustment_enabled
    global stored_intersections
    global capture_frame_counter
    global stuck_ellipses
    global last_tracking_result
    global last_gaze_vector_text

    ray_lines = []
    model_centers = []
    prev_model_center_avg = (320, 240)
    max_observed_distance = 0
    last_sphere_radius_ellipse = None
    eye_sphere_adjustment_enabled = True
    stored_intersections = []
    capture_frame_counter = 0
    stuck_ellipses = []
    last_tracking_result = None
    last_gaze_vector_text = None
    reset_front_calibration()


def draw_persistent_overlays(frame):
    prompt_text = get_eye_sphere_adjustment_prompt()
    prompt_shadow = (12, frame.shape[0] - 63)
    prompt_origin = (10, frame.shape[0] - 65)
    cv2.putText(frame, prompt_text, prompt_shadow, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(frame, prompt_text, prompt_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    if last_gaze_vector_text is None:
        return

    origin_text, dir_text = last_gaze_vector_text
    text_origin = (12, frame.shape[0] - 38)
    text_dir = (12, frame.shape[0] - 13)
    text_origin_green = (10, frame.shape[0] - 40)
    text_dir_green = (10, frame.shape[0] - 15)

    cv2.putText(frame, origin_text, text_origin, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(frame, dir_text, text_dir, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 3)
    cv2.putText(frame, origin_text, text_origin_green, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, dir_text, text_dir_green, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


def get_last_tracking_result():
    return last_tracking_result


def draw_stuck_ellipses(frame):
    global stuck_ellipses

    for ellipse in stuck_ellipses:
        if ellipse is not None:
            cv2.ellipse(frame, ellipse,(0, 255, 255), 2)

# Function to detect available cameras
def detect_cameras(max_cams=10):
    available_cameras = []

    for i in range(max_cams):
        cap, _, _ = open_camera_capture(i)
        if cap is not None:
            available_cameras.append(i)
            cap.release()

    return available_cameras

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

# Process frames for pupil detection
def process_frames(thresholded_image_strict, thresholded_image_medium, thresholded_image_relaxed, frame, gray_frame, darkest_point, debug_mode_on, render_cv_window):
    global ray_lines
    global max_rays
    global prev_model_center_avg
    global max_observed_distance
    global last_sphere_radius_ellipse
    global last_tracking_result
    global last_gaze_vector_text
    global last_gaze_dir
    global capture_stuck_ellipses
    global stuck_ellipses
    global capture_frame_counter
    global ellipse_capture_interval
    global max_stuck_ellipses

    kernel_size = 5
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    dilated_image = cv2.dilate(thresholded_image_medium, kernel, iterations=2)
    contours, _ = cv2.findContours(dilated_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

    final_rotated_rect = ((0,0),(0,0),0)

    image_array = [thresholded_image_relaxed, thresholded_image_medium, thresholded_image_strict] #holds images
    name_array = ["relaxed", "medium", "strict"] #for naming windows
    final_image = image_array[0] #holds return array
    final_contours = [] #holds final contours
    ellipse_reduced_contours = [] #holds an array of the best contour points from the fitting process
    goodness = 0 #goodness value for best ellipse
    best_array = 0 
    kernel_size = 5  # Size of the kernel (5x5)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)
    gray_copy1 = gray_frame.copy()
    gray_copy2 = gray_frame.copy()
    gray_copy3 = gray_frame.copy()
    gray_copies = [gray_copy1, gray_copy2, gray_copy3]

    final_goodness = 0
    best_ratio_under_ellipse = 0
    best_center_x, best_center_y = None, None

    # iterate through binary images and see which fits the ellipse best
    for i in range(1,4):
        dilated_image = cv2.dilate(image_array[i-1], kernel, iterations=2)

        contours, hierarchy = cv2.findContours(
            dilated_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        contour_img2 = np.zeros_like(dilated_image)
        reduced_contours = filter_contours_by_area_and_return_largest(contours, 1000, 3)

        center_x, center_y = None, None

        if len(reduced_contours) > 0 and len(reduced_contours[0]) > 5:
            current_goodness = check_ellipse_goodness(
                dilated_image,
                reduced_contours[0],
                debug_mode_on
            )

            ellipse = cv2.fitEllipse(reduced_contours[0])
            center_x, center_y = map(int, ellipse[0])

            if debug_mode_on:
                cv2.imshow(name_array[i-1] + " threshold", gray_copies[i-1])

            total_pixels = check_contour_pixels(
                reduced_contours[0],
                dilated_image.shape,
                debug_mode_on
            )

            cv2.ellipse(gray_copies[i-1], ellipse, (255, 0, 0), 2)

            final_goodness = (
                current_goodness[0]
                * total_pixels[0]
                * total_pixels[0]
                * total_pixels[1]
            )

            if final_goodness > 0 and final_goodness > goodness:
                goodness = final_goodness
                best_ratio_under_ellipse = total_pixels[1]
                ellipse_reduced_contours = total_pixels[2]
                best_image = image_array[i-1]
                final_contours = reduced_contours
                final_image = dilated_image

                # Critical fix:
                # Keep the pupil center associated with the chosen contour.
                best_center_x = center_x
                best_center_y = center_y

    # After threshold selection, use the center from the chosen/best contour.
    center_x = best_center_x
    center_y = best_center_y

    test_frame = frame.copy()
    
    final_contours = [optimize_contours_by_angle(final_contours, gray_frame)]
    
    final_rotated_rect = None

    if final_contours and not isinstance(final_contours[0], list) and len(final_contours[0]) >= 5:
        ellipse = cv2.fitEllipse(final_contours[0])
        final_rotated_rect = ellipse
        center_x, center_y = map(int, ellipse[0])

        if (
            eye_sphere_adjustment_enabled
            and not sphere_center_locked_2d
            and best_ratio_under_ellipse >= pupil_confidence_threshold
        ):
            ray_lines.append(final_rotated_rect)
            if len(ray_lines) > max_rays:
                num_to_remove = len(ray_lines) - max_rays
                ray_lines = ray_lines[num_to_remove:]  # Keep only the last `max_rays` elements

    global locked_model_center_avg
    model_center_average = locked_model_center_avg if sphere_center_locked_2d else prev_model_center_avg

    if eye_sphere_adjustment_enabled and not sphere_center_locked_2d:
        model_center = compute_average_intersection(
            frame,
            ray_lines,
            intersection_ray_count,
            1500,
            minimum_intersection_angle_degrees,
        )
        if model_center is not None and model_center != (0, 0):
            model_center_average = update_and_average_point(model_centers, model_center, 200)

    if model_center_average[0] == 320:
        model_center_average = prev_model_center_avg
    if model_center_average[0] != 0:
        prev_model_center_avg = model_center_average
        if not sphere_center_locked_2d:
            locked_model_center_avg = model_center_average
    
    # Example safety check
    if final_rotated_rect is None or center_x is None or center_y is None or model_center_average[0] is None or model_center_average[1] is None:
        last_tracking_result = None
        last_gaze_dir = None
        draw_persistent_overlays(frame)
        cv2.imshow("Frame with Ellipse and Rays", frame)
        return  # or skip this frame

    if eye_sphere_adjustment_enabled and not sphere_center_locked_2d:
        update_eye_sphere_radius(
            model_center_average,
            final_rotated_rect,
            best_ratio_under_ellipse,
        )

    last_tracking_result = {
        "pupil_ellipse": {
            "center": [float(final_rotated_rect[0][0]), float(final_rotated_rect[0][1])] if final_rotated_rect is not None else None,
            "axes": [float(final_rotated_rect[1][0]), float(final_rotated_rect[1][1])] if final_rotated_rect is not None else None,
            "angle_degrees": float(final_rotated_rect[2]) if final_rotated_rect is not None else None,
        } if final_rotated_rect is not None else None,
        "eye_center": [int(model_center_average[0]), int(model_center_average[1])],
        "sphere_radius": float(max_observed_distance),
    }

    # Draw reference lines/ellipses
    cv2.circle(frame, model_center_average, int(max_observed_distance), (255, 50, 50), 2)  # Draw eye sphere (circle)
    cv2.circle(frame, model_center_average, 8, (255, 255, 0), -1)  # Draw eye center



    if final_rotated_rect is not None and center_x is not None and center_y is not None:
        cv2.line(frame, model_center_average, (center_x, center_y), (255, 150, 50), 2)  # # Draw line from eye center to ellipse center
        
    if final_rotated_rect is not None:
        cv2.ellipse(frame, final_rotated_rect, (20, 255, 255), 2)  # draw current ellipse

        # If capture mode is on, save one ellipse every N frames
        if capture_stuck_ellipses:
            capture_frame_counter += 1
            if capture_frame_counter % ellipse_capture_interval == 0:
                stuck_ellipses.append(final_rotated_rect)

                # safety cap so list does not grow forever
                if len(stuck_ellipses) > max_stuck_ellipses:
                    stuck_ellipses = stuck_ellipses[-max_stuck_ellipses:]

    # Draw previously saved ellipses so they stay "stuck" on screen
        draw_stuck_ellipses(frame)

    # Calculate the extended endpoint of gaze line
    if final_rotated_rect is not None and center_x is not None and center_y is not None:
        # Compute the vector from model_center_average to center_x, center_y
        dx = center_x - model_center_average[0]
        dy = center_y - model_center_average[1]

        # Scale the vector by 1.2x
        extended_x = int(model_center_average[0] + 2 * dx)
        extended_y = int(model_center_average[1] + 2 * dy)

        # Draw the extended gaze line
        cv2.line(frame, (center_x, center_y), (extended_x, extended_y), (200, 255, 0), 3) 




    if render_cv_window:
        cv2.imshow("Best Thresholded Image Contours on Frame", frame)


    if GL_SPHERE_AVAILABLE:
        gl_image = gl_sphere.update_sphere_rotation(center_x, center_y, model_center_average[0], model_center_average[1])
    #cv2.circle(frame, (center_x, center_y), 22, (255, 255, 0), -1)  # Draw intersection center

    # Call the function
    center, direction = compute_gaze_vector(center_x, center_y, model_center_average[0], model_center_average[1])

    if center is not None and direction is not None:
        origin_text = f"Origin: ({center[0]:.2f}, {center[1]:.2f}, {center[2]:.2f})"
        dir_text    = f"Direction: ({direction[0]:.2f}, {direction[1]:.2f}, {direction[2]:.2f})"
        last_gaze_vector_text = (origin_text, dir_text)

    ratio_text = f"{best_ratio_under_ellipse * 100:.2f}%"
    cv2.putText(frame, ratio_text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 4)
    cv2.putText(frame, ratio_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    draw_persistent_overlays(frame)
    cv2.imshow("Frame with Ellipse and Rays", frame)

    if GL_SPHERE_AVAILABLE:
        if gl_image is not None:
            blended = cv2.addWeighted(frame, 0.6, gl_image, 0.4, 0)
            cv2.imshow("Eye Tracker + Sphere", blended)

    return final_rotated_rect

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

def compute_average_intersection(frame, ray_lines, number_lines, total_lines, minimum_angle_degrees):
    """
    Selects `number_lines` random lines from the list, highlights them in red on the frame,
    computes their intersections, conditionally stores them (only if they agree within
    `pixel_limit`), and prunes stored intersections when exceeding `total_lines`.
    """
    pixel_limit = 30
    angle_threshold = 5  # degrees

    global stored_intersections

    if len(ray_lines) < 2 or number_lines < 2:
        return (0, 0)

    height, width = frame.shape[:2]

    selected_lines = random.sample(ray_lines, min(number_lines, len(ray_lines)))

    # Highlight selected rays
    if render_intersections:
        for ray in selected_lines:
            draw_orthogonal_ray(frame, ray, color=(0, 0, 255), thickness=2)

    intersections = []
    for i in range(len(selected_lines) - 1):
        line1 = selected_lines[i]
        line2 = selected_lines[i + 1]

        angle1 = line1[2]
        angle2 = line2[2]

        if abs(angle1 - angle2) >= minimum_angle_degrees:
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
                if abs(angle_i - angle_j) < angle_threshold:
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

    # Draw stored intersections
    if render_intersections:
        for pt in stored_intersections:
            cv2.circle(frame, pt, 3, (255, 255, 255), -1)

    avg_x = np.mean([pt[0] for pt in stored_intersections])
    avg_y = np.mean([pt[1] for pt in stored_intersections])

    if np.isnan(avg_x) or np.isnan(avg_y):
        return (0, 0)

    return (int(avg_x), int(avg_y))

#Removes the oldest intersections to ensure only the last M intersections remain.
def prune_intersections(intersections, maximum_intersections):

    if len(intersections) <= maximum_intersections:
        return intersections  # No need to prune if within the limit

    # Keep only the last M intersections
    pruned_intersections = intersections[-maximum_intersections:]

    return pruned_intersections

def rotation_from_a_to_b(a, b):
    """
    Compute rotation matrix R such that R @ a = b
    using Rodrigues' rotation formula.
    """
    a = a / np.linalg.norm(a)
    b = b / np.linalg.norm(b)

    v = np.cross(a, b)
    c = np.dot(a, b)

    if np.linalg.norm(v) < 1e-6:
        # Vectors are parallel or nearly so
        if c > 0:
            return np.eye(3, dtype=np.float32)
        else:
            # 180-degree flip: choose any axis orthogonal to a
            axis = np.array([1.0, 0.0, 0.0])
            if abs(a[0]) > 0.9:
                axis = np.array([0.0, 1.0, 0.0])
            v = np.cross(a, axis)
            v = v / np.linalg.norm(v)
            s = np.linalg.norm(v)
    else:
        s = np.linalg.norm(v)
        v = v / s

    # Skew-symmetric cross-product matrix
    vx, vy, vz = v
    K = np.array([
        [0,    -vz,  vy],
        [vz,    0,  -vx],
        [-vy,  vx,   0 ]
    ], dtype=np.float32)

    R = np.eye(3, dtype=np.float32) + K * s + (K @ K) * ((1 - c) / (s ** 2))
    return R

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


def current_gaze_camera_point():
    """Return the unclipped calibrated pixel position; never score a clamped dot."""
    if not calibrated or last_gaze_dir is None:
        return None
    try:
        measured = project_gaze_to_tangent(last_gaze_dir, calibration_basis)
    except ValueError:
        return None
    point = np.array([EXT_CX, EXT_CY]) + calibration_matrix @ measured
    return point if np.isfinite(point).all() else None


def update_gaze_circle_from_current_gaze():
    global circle_x, circle_y
    point = current_gaze_camera_point()
    if point is None:
        return False
    circle_x = int(np.clip(round(point[0]), 0, EXT_WIDTH - 1))
    circle_y = int(np.clip(round(point[1]), 0, EXT_HEIGHT - 1))
    return True


def accuracy_grid_targets():
    xs = np.linspace(EXT_WIDTH * 0.15, EXT_WIDTH * 0.85, ACCURACY_GRID_SIZE)
    ys = np.linspace(EXT_HEIGHT * 0.15, EXT_HEIGHT * 0.85, ACCURACY_GRID_SIZE)
    return [(int(round(x)), int(round(y))) for y in ys for x in xs]


def camera_pixel_ray(point):
    # Square pixels, centered principal point, 4:3 image and 60 degree diagonal FOV.
    focal_pixels = math.hypot(EXT_WIDTH, EXT_HEIGHT) / (
        2.0 * math.tan(math.radians(FRONT_DIAGONAL_FOV_DEGREES) / 2.0)
    )
    ray = np.array([(point[0] - EXT_CX) / focal_pixels,
                    (EXT_CY - point[1]) / focal_pixels, 1.0], dtype=float)
    return ray / np.linalg.norm(ray)


def camera_angular_error_degrees(target, gaze):
    target_ray, gaze_ray = camera_pixel_ray(target), camera_pixel_ray(gaze)
    # atan2 stays accurate for very small errors and identical samples.
    return math.degrees(math.atan2(np.linalg.norm(np.cross(target_ray, gaze_ray)),
                                   np.dot(target_ray, gaze_ray)))


def reset_accuracy_grid():
    global accuracy_grid_visible, accuracy_grid_index, accuracy_grid_samples
    global accuracy_grid_message
    accuracy_grid_visible = False
    accuracy_grid_index = -1
    accuracy_grid_samples = []
    accuracy_grid_message = "Press T to start at the upper-left target."


def toggle_accuracy_grid():
    global accuracy_grid_visible
    was_visible = accuracy_grid_visible
    reset_accuracy_grid()
    if not was_visible:
        accuracy_grid_visible = True
    print("Accuracy grid shown." if accuracy_grid_visible else "Accuracy grid hidden and reset.")


def advance_accuracy_grid_test():
    global accuracy_grid_index, accuracy_grid_message
    if not accuracy_grid_visible:
        print("Show the accuracy grid with G before pressing T.")
        return False
    if accuracy_grid_index >= len(accuracy_grid_targets()):
        return False
    if accuracy_grid_index < 0:
        accuracy_grid_index = 0
        accuracy_grid_message = "Look at target 1/25; press T to record."
        return True
    gaze = current_gaze_camera_point()
    if gaze is None:
        accuracy_grid_message = "No calibrated gaze: calibrate or reacquire eye, then retry T."
        return False
    target = accuracy_grid_targets()[accuracy_grid_index]
    error = camera_angular_error_degrees(target, gaze)
    accuracy_grid_samples.append({"target": target, "gaze": tuple(gaze), "error_degrees": error})
    accuracy_grid_index += 1
    print(f"Grid target {accuracy_grid_index}/25 angular error: {error:.2f} degrees.")
    if accuracy_grid_index == len(accuracy_grid_targets()):
        mean_error = np.mean([sample["error_degrees"] for sample in accuracy_grid_samples])
        accuracy_grid_message = f"Complete. Mean error: {mean_error:.2f} deg | G: hide/reset"
        print(accuracy_grid_message)
    else:
        accuracy_grid_message = f"Look at target {accuracy_grid_index + 1}/25; press T to record."
    return True


def draw_accuracy_billboard(frame, text, center, scale=0.4):
    (width, height), baseline = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 1)
    x = int(np.clip(center[0] - width // 2, 3, frame.shape[1] - width - 4))
    y = int(np.clip(center[1], height + 4, frame.shape[0] - baseline - 4))
    cv2.rectangle(frame, (x - 3, y - height - 3), (x + width + 3, y + baseline + 3), (255, 255, 255), -1)
    cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 1, cv2.LINE_AA)


def draw_accuracy_grid(frame):
    for index, target in enumerate(accuracy_grid_targets()):
        cv2.circle(frame, target, 6, (0, 255, 255), -1)
        if index == accuracy_grid_index:
            cv2.circle(frame, target, 11, (0, 0, 255), 2)
    # Record at the actual gaze location, not at the intended target.
    for sample in accuracy_grid_samples:
        x, y = sample["gaze"]
        if 0 <= x < EXT_WIDTH and 0 <= y < EXT_HEIGHT:
            cv2.circle(frame, (int(round(x)), int(round(y))), 7, (0, 0, 0), -1)
    for sample in accuracy_grid_samples:
        target = sample["target"]
        draw_accuracy_billboard(frame, f"{sample['error_degrees']:.2f} deg", (target[0], target[1] - 15))
    draw_accuracy_billboard(frame, accuracy_grid_message, (EXT_CX, 24), scale=0.42)


def draw_front_calibration_overlay(frame):
    if calibrated:
        if update_gaze_circle_from_current_gaze():
            cv2.circle(frame, (circle_x, circle_y), 8, (0, 0, 255), -1)
        text = "Calibrated | C: restart | F: unlock sphere"
    else:
        name, point = calibration_targets()[len(calibration_samples)]
        cv2.circle(frame, point, 8, (0, 0, 255), -1)
        text = f"{len(calibration_samples) + 1}/5 {name}: align finger and gaze, press C"
    if accuracy_grid_visible:
        draw_accuracy_grid(frame)
        return
    for y, line in ((24, text), (46, calibration_message)):
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 0, 0), 3)
        cv2.putText(frame, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)



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
    else:
        sqrt_disc = np.sqrt(discriminant)
        t1 = (-b - sqrt_disc) / (2 * a)
        t2 = (-b + sqrt_disc) / (2 * a)

        positive_intersections = [value for value in (t1, t2) if value > 0]
        if positive_intersections:
            t = min(positive_intersections)
        else:
            t = -np.dot(direction, L) / np.dot(direction, direction)

    # Final intersection point
    intersection_point = origin + t * direction

    # Convert to local space relative to sphere center
    intersection_local = intersection_point - sphere_center
    intersection_norm = np.linalg.norm(intersection_local)
    if intersection_norm < 1e-6:
        target_direction = np.array([0.0, 0.0, 1.0])
    else:
        target_direction = intersection_local / intersection_norm

    # Local green ring direction
    circle_local_center = np.array([0.0, 0.0, inner_radius])
    circle_local_center /= np.linalg.norm(circle_local_center)

    # Compute rotation to align local +Z to target
    rotation_axis = np.cross(circle_local_center, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        rotation_axis = np.array([1.0, 0.0, 0.0])
        rotation_axis_norm = 1.0

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

    global last_sphere_center, last_gaze_dir
    last_sphere_center = sphere_center.copy()
    last_gaze_dir = gaze_rotated.copy()
    sphere_center_out = calibrated_sphere_center if calibrated_sphere_center is not None else sphere_center

    # Write to file (overwrite every frame)
    file_path = "gaze_vector.txt"

    def is_file_available(path):
        try:
            with open(path, "a"):
                return True
        except IOError:
            return False

    if is_file_available(file_path):
        try:
            with open(file_path, "w") as f:
                all_values = np.concatenate((sphere_center_out, gaze_rotated))
                csv_line = ",".join(f"{v:.6f}" for v in all_values)
                f.write(csv_line + "\n")
        except Exception as e:
            print("Write error:", e)
    else:
        print("File is currently in use. Skipping write.")

    return sphere_center_out, gaze_rotated



def calibrate_gaze_to_external():
    global calibrated, calibrated_sphere_center, sphere_center_locked_2d
    global locked_model_center_avg, calibration_basis, calibration_matrix
    global calibration_message, calibration_samples
    if last_gaze_dir is None or last_sphere_center is None:
        calibration_message = "No current gaze detected; keep looking at target and retry C."
        print(calibration_message)
        return False
    direction = np.asarray(last_gaze_dir, dtype=float)
    if not np.isfinite(direction).all() or np.linalg.norm(direction) < 1e-8:
        calibration_message = "Invalid gaze sample; retry C."
        return False
    direction = direction / np.linalg.norm(direction)
    if calibrated:
        reset_accuracy_grid()
        # Preserve the current eye model and start a new center sample.
        calibration_samples = []
        calibration_basis = calibration_matrix = None
        calibrated = False
    if not calibration_samples:
        calibrated_sphere_center = last_sphere_center.copy()
        locked_model_center_avg = prev_model_center_avg
        sphere_center_locked_2d = True
    calibration_samples.append(direction.copy())
    if len(calibration_samples) < 5:
        name, _ = calibration_targets()[len(calibration_samples)]
        calibration_message = f"Captured. Move finger to {name}, look at it, press C."
        print(calibration_message)
        return True
    try:
        basis, matrix, error = build_camera_calibration(calibration_samples)
    except ValueError as exc:
        calibration_samples = []
        calibration_message = str(exc)
        print("Calibration failed:", calibration_message)
        return False
    calibration_basis, calibration_matrix = basis, matrix
    calibrated = True
    calibration_message = f"Fit RMS: {error:.1f} pixels. C starts a new center capture."
    print("Five-point calibration complete.", calibration_message)
    print("Tangent-to-pixel matrix:", matrix)
    return True







# Finds the pupil in an individual frame and returns the center point
def process_frame(frame, return_rendered_frame=False):


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
    
    if return_rendered_frame:
        return final_rotated_rect, frame

    return final_rotated_rect


# Process video from the selected eye camera + external camera preview
def reset_front_calibration():
    reset_accuracy_grid()
    global calibrated, calibrated_sphere_center
    global last_sphere_center, last_gaze_dir, sphere_center_locked_2d
    global locked_model_center_avg, circle_x, circle_y
    global calibration_samples, calibration_basis, calibration_matrix, calibration_message
    calibrated = False
    calibrated_sphere_center = None
    last_sphere_center = None
    last_gaze_dir = None
    sphere_center_locked_2d = False
    locked_model_center_avg = prev_model_center_avg
    circle_x, circle_y = EXT_CX, EXT_CY
    calibration_samples = []
    calibration_basis = calibration_matrix = None
    calibration_message = "Align your finger and gaze with CENTER; press C."




# Process a selected video file
def process_video(parent=None):
    while True:
        video_path = filedialog.askopenfilename(
            parent=parent,
            filetypes=[("Video Files", "*.mp4;*.avi")],
        )

        if not video_path:
            cv2.destroyAllWindows()
            return  # User canceled selection

        reset_tracking_state()
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print("Error: Could not open video file.")
            continue

        video_stem, video_extension = os.path.splitext(video_path)
        output_path = f"{video_stem}_tracked{video_extension}"
        output_codec = "XVID" if video_extension.lower() == ".avi" else "mp4v"
        writer = cv2.VideoWriter(
            output_path,
            cv2.VideoWriter_fourcc(*output_codec),
            60,
            (640, 480),
        )
        if not writer.isOpened():
            print(f"Error: Could not create tracked video: {output_path}")
            cap.release()
            continue

        quit_requested = False

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                _, tracked_frame = process_frame(frame, return_rendered_frame=True)
                writer.write(tracked_frame)

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    quit_requested = True
                    break
                elif key == ord(' '):
                    cv2.waitKey(0)
                elif key in (ord('f'), ord('F')):
                    toggle_eye_sphere_adjustment()
        finally:
            cap.release()
            writer.release()
            cv2.destroyAllWindows()

        print(f"Tracked video saved to: {output_path}")

        if quit_requested:
            return
    

# GUI for selecting camera or video
def selection_gui():
    global capture_stuck_ellipses
    global capture_frame_counter
    global stuck_ellipses

    camera_roles = ("eye", "front")
    cameras = detect_cameras()
    active_captures = {eye_id: None for eye_id in camera_roles}
    active_camera_indexes = {eye_id: None for eye_id in camera_roles}
    active_capture_mode_indexes = {eye_id: None for eye_id in camera_roles}
    camera_paused = False

    # Create Tkinter window
    root = tk.Tk()
    root.title("Orlosky 3D Front Camera")
    tk.Label(root, text="Orlosky Eye Tracker 3D", font=("Arial", 12, "bold")).pack(pady=10)

    camera_dropdowns = {}
    flip_image_vars = {}

    def any_camera_active():
        return any(capture is not None for capture in active_captures.values())

    def stop_camera(eye_id):
        if active_captures[eye_id] is not None:
            active_captures[eye_id].release()
            active_captures[eye_id] = None

        reset_front_calibration()
        active_camera_indexes[eye_id] = None
        active_capture_mode_indexes[eye_id] = None

        if not any_camera_active():
            cv2.destroyAllWindows()

    def stop_all_cameras():
        for eye_id in camera_roles:
            stop_camera(eye_id)

    def start_camera(eye_id):
        nonlocal camera_paused

        selected_camera = camera_dropdowns[eye_id].get()
        if not selected_camera.isdigit():
            return

        cam_index = int(selected_camera)
        other_role = "front" if eye_id == "eye" else "eye"
        if active_camera_indexes[other_role] == cam_index:
            print("Eye and forward-facing cameras must use different camera indexes.")
            return
        stop_camera(eye_id)
        new_capture, _, mode_index = open_camera_capture(cam_index)

        if new_capture is None:
            print(f"Error: Could not open camera {cam_index}.")
            return

        if eye_id == "eye":
            reset_tracking_state()
            cv2.namedWindow("Frame with Ellipse and Rays")
        else:
            reset_front_calibration()
        sync_fixed_sphere_checkbox()
        active_captures[eye_id] = new_capture
        active_camera_indexes[eye_id] = cam_index
        active_capture_mode_indexes[eye_id] = mode_index
        camera_paused = False
        print(
            f"Streaming {eye_id} camera {cam_index} "
            f"with {CAMERA_CAPTURE_MODES[mode_index][0]}."
        )

    def switch_active_camera(eye_id):
        if active_captures[eye_id] is not None:
            start_camera(eye_id)

    camera_controls_frame = ttk.Frame(root)
    camera_controls_frame.pack(pady=5)
    ttk.Separator(camera_controls_frame, orient="vertical").grid(row=0, column=1, sticky="ns", padx=8)

    for column_index, eye_id in enumerate(camera_roles):
        frame = ttk.Frame(camera_controls_frame)
        frame.grid(row=0, column=column_index * 2, padx=5, pady=5, sticky="n")

        camera_label = "Eye Camera" if eye_id == "eye" else "Forward-facing Camera"
        tk.Label(frame, text=f"{camera_label}:").grid(row=0, column=0, columnspan=2, pady=2)

        dropdown = ttk.Combobox(frame, values=[str(index) for index in cameras], state="readonly", width=12)
        if cameras:
            default_index = min(camera_roles.index(eye_id), len(cameras) - 1)
            dropdown.current(default_index)
        else:
            dropdown.set("No cameras found")
        dropdown.grid(row=1, column=0, columnspan=2, pady=2)
        dropdown.bind("<<ComboboxSelected>>", lambda event, selected_eye=eye_id: switch_active_camera(selected_eye))
        camera_dropdowns[eye_id] = dropdown

        tk.Button(frame, text=f"Start {camera_label}", command=lambda selected_eye=eye_id: start_camera(selected_eye)).grid(row=2, column=0, pady=2)
        tk.Button(frame, text=f"Stop {camera_label}", command=lambda selected_eye=eye_id: stop_camera(selected_eye)).grid(row=2, column=1, pady=2)

        flip_var = tk.BooleanVar(value=eye_id == "eye")
        tk.Checkbutton(frame, text="flip image", variable=flip_var, command=reset_front_calibration).grid(row=3, column=0, columnspan=2, pady=2)
        flip_image_vars[eye_id] = flip_var

    fixed_sphere_var = tk.BooleanVar(value=sphere_center_locked_2d or not eye_sphere_adjustment_enabled)

    def sync_fixed_sphere_checkbox():
        fixed_sphere_var.set(sphere_center_locked_2d or not eye_sphere_adjustment_enabled)

    def set_fixed_sphere_from_checkbox():
        desired_fixed = fixed_sphere_var.get()
        currently_fixed = sphere_center_locked_2d or not eye_sphere_adjustment_enabled
        if desired_fixed != currently_fixed:
            toggle_eye_sphere_adjustment()
        sync_fixed_sphere_checkbox()

    def process_camera_frame():
        nonlocal camera_paused
        global capture_stuck_ellipses
        global capture_frame_counter
        global stuck_ellipses

        if not camera_paused:
            for eye_id in camera_roles:
                active_capture = active_captures[eye_id]
                if active_capture is None:
                    continue

                ret, frame = active_capture.read()
                if not ret:
                    failed_capture = active_captures[eye_id]
                    active_captures[eye_id] = None
                    failed_capture.release()

                    retry_mode = active_capture_mode_indexes[eye_id] + 1
                    new_capture, frame, mode_index = open_camera_capture(
                        active_camera_indexes[eye_id],
                        retry_mode,
                    )
                    if new_capture is None:
                        print(f"Error: Could not read from {eye_id} eye camera {active_camera_indexes[eye_id]}.")
                        stop_camera(eye_id)
                        continue

                    active_captures[eye_id] = new_capture
                    active_capture_mode_indexes[eye_id] = mode_index
                    print(
                        f"Switched {eye_id} eye camera {active_camera_indexes[eye_id]} to "
                        f"{CAMERA_CAPTURE_MODES[mode_index][0]}."
                    )

                if flip_image_vars[eye_id].get():
                    frame = cv2.flip(frame, 0)

                if eye_id == "eye":
                    process_frame(frame)
                else:
                    frame = cv2.resize(frame, (EXT_WIDTH, EXT_HEIGHT))
                    draw_front_calibration_overlay(frame)
                    cv2.imshow("External Camera (Gaze)", frame)

        if any_camera_active():
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                stop_all_cameras()
            elif key == ord(' '):
                camera_paused = not camera_paused
            elif key in (ord('f'), ord('F')):
                toggle_eye_sphere_adjustment()
                sync_fixed_sphere_checkbox()
            elif key == ord('e'):
                capture_stuck_ellipses = not capture_stuck_ellipses
                capture_frame_counter = 0
                print(f"Ellipse capture mode: {'ON' if capture_stuck_ellipses else 'OFF'}")
            elif key in (ord('g'), ord('G')):
                toggle_accuracy_grid()
            elif key in (ord('t'), ord('T')):
                if camera_paused or not all(active_captures[role] is not None for role in camera_roles):
                    print("Resume both camera streams before recording accuracy samples.")
                else:
                    advance_accuracy_grid_test()
            elif key in (ord('c'), ord('C')):
                calibrate()
            elif key in (ord('x'), ord('X')):
                stuck_ellipses.clear()
                print("Cleared stuck ellipses.")

        sync_fixed_sphere_checkbox()
        root.after(1, process_camera_frame)

    def calibrate():
        if camera_paused:
            print("Resume the camera streams before capturing a calibration sample.")
            return
        if all(active_captures[role] is not None for role in camera_roles):
            calibrate_gaze_to_external()
            sync_fixed_sphere_checkbox()
        else:
            print("Start both cameras before calibrating.")

    def close_app():
        stop_all_cameras()
        root.destroy()

    def browse_video():
        stop_all_cameras()
        try:
            process_video(parent=root)
        finally:
            sync_fixed_sphere_checkbox()
            root.lift()

    tk.Checkbutton(
        root,
        text="fixed eye sphere",
        variable=fixed_sphere_var,
        command=set_fixed_sphere_from_checkbox,
    ).pack(pady=5)
    tk.Button(root, text="Capture Calibration Point (C)", command=calibrate).pack(pady=5)
    tk.Button(root, text="Browse Video", command=browse_video).pack(pady=5)
    tk.Label(root, text="Q: stop | Space: pause | C: calibrate | F: fixed/auto sphere\nE: capture ellipses | X: clear ellipses\nG: grid/reset | T: start/record accuracy").pack(pady=5)

    if GL_SPHERE_AVAILABLE:
        # Start GL sphere window once
        app = gl_sphere.start_gl_window() 

    reset_tracking_state()
    sync_fixed_sphere_checkbox()
    root.protocol("WM_DELETE_WINDOW", close_app)
    root.after(1, process_camera_frame)
    root.mainloop()

# Run GUI to select camera or video
if __name__ == "__main__":
    selection_gui()
