import cv2
import numpy as np
import os
import mediapipe as mp
import time
import math
from scipy.spatial.transform import Rotation as Rscipy
from collections import deque
import pyautogui
import threading
import keyboard

# Screen and mouse control setup (from old script)
MONITOR_WIDTH, MONITOR_HEIGHT = pyautogui.size()
CENTER_X = MONITOR_WIDTH // 2
CENTER_Y = MONITOR_HEIGHT // 2
mouse_control_enabled = False
filter_length = 6
gaze_length = 350

# === Grid classification output parameters ===
# Change these to set how many cells the screen is divided into
OUTPUT_GRID_ROWS = 2   # rows (vertical divisions)
OUTPUT_GRID_COLS = 2   # columns (horizontal divisions)
NUM_CLASSES = OUTPUT_GRID_ROWS * OUTPUT_GRID_COLS

# Output smoothing (exponential averaging on class probabilities)
prob_smooth_alpha = 0.45   # 0 = full smoothing / 1 = no smoothing
smooth_probs = None        # running average of class probability vector

# --- Orbit camera state for the debug view ---
orbit_yaw   = -151.0          # radians, left/right
orbit_pitch = 00.0          # radians, up/down
orbit_radius = 1500.0       # distance from head center
orbit_fov_deg = 50.0       # horizontal FOV for projection

# --- Debug-view world freeze (pivot fixed after center calibration) ---
debug_world_frozen = False
orbit_pivot_frozen = None  # world-space point the debug camera orbits (monitor center at calib)

# Stored gaze markers on the monitor plane (as (a,b) in plane coords)
# a = 0..1 across width (p0->p1), b = 0..1 down height (p0->p3)
gaze_markers = []

# --- 3D monitor plane state (world space) ---
monitor_corners = None   # list of 4 world points (p0..p3)
monitor_center_w = None  # world center of the plane
monitor_normal_w = None  # world normal
units_per_cm = None      # world units per centimeter (computed at calibration)

# Shared mouse target position
mouse_target = [CENTER_X, CENTER_Y]
mouse_lock = threading.Lock()

# Calibration offsets for screen mapping
calibration_offset_yaw = 0
calibration_offset_pitch = 0

# --- Simple monitor-edge calibration state ---
# 0 = waiting for center, 1 = waiting for left edge, 2 = done
calib_step = 0

# Buffers to store recent gaze data for smoothing
combined_gaze_directions = deque(maxlen=filter_length)

# reference matrices to fix coordinate flipping issue
# These help keep the axes consistent from frame to frame by stabilizing eigenvector directions
R_ref_nose = [None]
R_ref_forehead = [None]
calibration_nose_scale = None

# Initialize MediaPipe FaceMesh - Updated for new API
BaseOptions = mp.tasks.BaseOptions
FaceLandmarker = mp.tasks.vision.FaceLandmarker
FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

# Download model if not present
model_path = "face_landmarker.task"
if not os.path.exists(model_path):
    print("Downloading face landmarker model...")
    import urllib.request
    model_url = "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task"
    try:
        urllib.request.urlretrieve(model_url, model_path)
        print("Model downloaded successfully!")
    except Exception as e:
        print(f"Error downloading model: {e}")
        print("Please download manually from: https://developers.google.com/mediapipe/solutions/vision/face_landmarker")
        exit(1)

# Create FaceLandmarker with options
options = FaceLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5,
    output_face_blendshapes=False,
    output_facial_transformation_matrixes=False
)

face_mesh = FaceLandmarker.create_from_options(options)
USE_NEW_API = True

# === Open webcam ===
cap = cv2.VideoCapture(0)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# === Nose-only landmark indices (for stable up/down eye sphere tracking) ===
# These landmarks are near the nose and are less affected by lateral head movement
nose_indices = [4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
                461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
                3, 248]

# ===== NEW: File writing for screen position =====
screen_position_file = "screen_position.txt"

def write_screen_position(grid_row, grid_col, confidence, cell_index):
    """Write predicted grid cell to file, overwriting the same line"""
    try:
        with open(screen_position_file, 'w') as f:
            f.write(f"{grid_row},{grid_col},{cell_index},{confidence:.3f}\n")
    except Exception as e:
        # Silently ignore file write errors to not interrupt tracking
        pass

# ============================================================
# Learned gaze-to-screen model (polynomial regression, numpy)
# ============================================================

CALIB_FILE = "calib_data.npz"

# Calibration grid: (norm_x, norm_y) fractions of screen size
# 5x5 = 25 points (rows top→bottom, cols left→right)
CALIB_GRID_COLS = 5
CALIB_GRID_ROWS = 5
_margin = 0.1
CALIB_POINTS = [
    (_margin + c * (1.0 - 2 * _margin) / (CALIB_GRID_COLS - 1),
     _margin + r * (1.0 - 2 * _margin) / (CALIB_GRID_ROWS - 1))
    for r in range(CALIB_GRID_ROWS)
    for c in range(CALIB_GRID_COLS)
]
CALIB_SAMPLES_PER_POINT = 30   # frames to average per target point
CALIB_SETTLE_FRAMES     = 15   # frames to skip (let eyes settle on dot)

# Head-pose sub-phases during calibration
# Each calibration point is collected at multiple head orientations for robustness
CALIB_HEAD_POSES = [
    ("Center",        0.0,  0.0),   # (label, target_yaw_deg, target_pitch_deg)
    ("Esquerda",  -8.0,  0.0),
    ("Direita",  8.0,  0.0),
]
CALIB_HEAD_YAW_TOL   = 4.0   # degrees tolerance for head yaw matching
CALIB_HEAD_PITCH_TOL = 4.0   # degrees tolerance for head pitch matching

# Calibration mode state
calib_mode          = False    # True while collecting calibration data
calib_point_index   = 0        # which target point we're on
calib_pose_index    = 0        # which head-pose sub-phase we're on
calib_settle_count  = 0        # settle-frame counter
calib_sample_count  = 0        # collected-sample counter for current point
calib_X             = []       # accumulated feature vectors
calib_Y             = []       # accumulated target grid cell indices
calib_feat_buf      = []       # raw feature samples for current point (to be averaged)

# Feature smoothing deque for model inference (separate from gaze direction deque)
MODEL_SMOOTH_LEN = 5
model_feat_history = deque(maxlen=MODEL_SMOOTH_LEN)


class GazeModel:
    """Softmax classification model mapping gaze features → screen grid cell.
    
    Uses degree-2 polynomial features with L2-regularised softmax (cross-entropy)
    trained via L-BFGS optimisation.  Pure numpy + scipy.optimize — no sklearn.
    """

    POLY_DEGREE = 2

    def __init__(self, num_classes=NUM_CLASSES):
        self._W   = None   # weight matrix [n_features_poly, num_classes]
        self._mu  = None   # feature mean for normalisation
        self._sig = None   # feature std  for normalisation
        self.num_classes = num_classes
        self.fitted = False

    # ------------------------------------------------------------------
    def _poly_features(self, X):
        """Degree-2 polynomial expansion (with bias column) of X [N, d]."""
        X = np.atleast_2d(X)
        cols = [np.ones((X.shape[0], 1))]
        # degree 1
        cols.append(X)
        # degree 2: xi*xj for i<=j
        n = X.shape[1]
        for i in range(n):
            for j in range(i, n):
                cols.append((X[:, i] * X[:, j]).reshape(-1, 1))
        return np.hstack(cols)

    # ------------------------------------------------------------------
    @staticmethod
    def _softmax(logits):
        """Numerically stable softmax over last axis."""
        e = np.exp(logits - logits.max(axis=1, keepdims=True))
        return e / e.sum(axis=1, keepdims=True)

    # ------------------------------------------------------------------
    def fit(self, X_raw, Y_labels):
        """Fit softmax classifier.
        X_raw:    [N, d] feature vectors.
        Y_labels: [N]    integer class labels in 0..num_classes-1.
        """
        from scipy.optimize import minimize as sp_minimize

        X_raw    = np.array(X_raw, dtype=float)
        Y_labels = np.array(Y_labels, dtype=int).ravel()
        N = len(Y_labels)
        K = self.num_classes

        # z-score normalise inputs
        self._mu  = X_raw.mean(axis=0)
        self._sig = X_raw.std(axis=0) + 1e-8
        Xn = (X_raw - self._mu) / self._sig
        Xp = self._poly_features(Xn)          # [N, p]
        p = Xp.shape[1]

        # One-hot encode labels
        Y_oh = np.zeros((N, K), dtype=float)
        Y_oh[np.arange(N), Y_labels] = 1.0

        lam = 0.1   # L2 regularisation strength

        def loss_and_grad(w_flat):
            W = w_flat.reshape(p, K)
            logits = Xp @ W                    # [N, K]
            probs  = self._softmax(logits)     # [N, K]
            # Cross-entropy + L2 penalty
            log_probs = np.log(probs + 1e-12)
            ce = -np.sum(Y_oh * log_probs) / N
            reg = 0.5 * lam * np.sum(W ** 2)
            loss = ce + reg
            # Gradient
            dW = Xp.T @ (probs - Y_oh) / N + lam * W   # [p, K]
            return loss, dW.ravel()

        # Initial weights (small random)
        w0 = np.random.randn(p * K) * 0.01
        result = sp_minimize(loss_and_grad, w0, jac=True, method='L-BFGS-B',
                             options={'maxiter': 500, 'ftol': 1e-8})
        self._W = result.x.reshape(p, K)
        self.fitted = True

        # Training accuracy report
        logits = Xp @ self._W
        preds = np.argmax(logits, axis=1)
        acc = np.mean(preds == Y_labels) * 100
        # Per-class count
        class_counts = np.bincount(Y_labels, minlength=K)
        print(f"[GazeModel] Fitted softmax on {N} samples, "
              f"{p} poly features, {K} classes.  "
              f"Train accuracy: {acc:.1f}%")
        print(f"[GazeModel] Samples per class: {class_counts.tolist()}")

    # ------------------------------------------------------------------
    def predict(self, feat):
        """Predict grid cell from a single feature vector.
        Returns (cell_index, confidence, probabilities) or None.
        """
        if not self.fitted:
            return None
        x = np.array(feat, dtype=float).reshape(1, -1)
        xn = (x - self._mu) / self._sig
        xp = self._poly_features(xn)           # [1, p]
        logits = xp @ self._W                  # [1, K]
        probs  = self._softmax(logits)[0]      # [K]
        cell_idx = int(np.argmax(probs))
        confidence = float(probs[cell_idx])
        return cell_idx, confidence, probs

    # ------------------------------------------------------------------
    def save(self, path=CALIB_FILE):
        np.savez(path, W=self._W, mu=self._mu, sig=self._sig,
                 num_classes=np.array([self.num_classes]))
        print(f"[GazeModel] Calibration saved to '{path}'.")

    # ------------------------------------------------------------------
    def load(self, path=CALIB_FILE):
        try:
            data = np.load(path)
            W   = data['W']
            mu  = data['mu']
            sig = data['sig']
            nc  = int(data['num_classes'][0]) if 'num_classes' in data else 0
            # Validate feature dimension (11 = current extract_gaze_features length)
            test_dim = 11
            if mu.shape[0] != test_dim:
                print(f"[GazeModel] Stale calibration (expected {test_dim} features, "
                      f"got {mu.shape[0]}). Ignoring '{path}'.")
                return False
            if nc != self.num_classes:
                print(f"[GazeModel] Grid size mismatch (file has {nc} classes, "
                      f"current config has {self.num_classes}). Ignoring '{path}'.")
                return False
            self._W  = W
            self._mu = mu
            self._sig = sig
            self.fitted = True
            print(f"[GazeModel] Calibration loaded from '{path}' "
                  f"({nc} classes, {mu.shape[0]} features).")
            return True
        except Exception as e:
            print(f"[GazeModel] Could not load '{path}': {e}")
            return False


# Singleton model instance
gaze_model = GazeModel()
gaze_model.load()   # load previous calibration if it exists


def extract_gaze_features(combined_dir, R_final, iris_3d_left, iris_3d_right,
                           sphere_world_l, sphere_world_r, head_center):
    """Return a compact feature vector for the gaze classification model.

    Features (11 total):
      [0:3]  left iris offset from sphere, in HEAD-LOCAL frame (unit vector)
      [3:6]  right iris offset from sphere, in HEAD-LOCAL frame (unit vector)
      [6:8]  combined gaze yaw/pitch proxy (head-local x/y of combined dir)
      [8:11] head Euler angles (yaw, pitch, roll) in radians — helps the
             model compensate for head-pose variation seen during calibration.
    """
    # --- iris offsets in head-local frame (most stable gaze signal) ---
    def iris_local(iris3d, sphere_w):
        raw = np.asarray(iris3d, dtype=float) - np.asarray(sphere_w, dtype=float)
        local = R_final.T @ raw
        n = np.linalg.norm(local)
        return local / (n + 1e-9)

    il = iris_local(iris_3d_left,  sphere_world_l)
    ir = iris_local(iris_3d_right, sphere_world_r)

    # --- combined gaze in head-local frame (yaw/pitch proxy) ---
    d = np.asarray(combined_dir, dtype=float)
    d = d / (np.linalg.norm(d) + 1e-9)
    d_local = R_final.T @ d
    gaze_xy = d_local[:2]   # 2 values

    # --- head Euler angles (yaw, pitch, roll) ---
    r_scipy = Rscipy.from_matrix(R_final)
    head_yaw, head_pitch, head_roll = r_scipy.as_euler('yxz', degrees=False)

    return np.array([il[0], il[1], il[2],
                     ir[0], ir[1], ir[2],
                     gaze_xy[0], gaze_xy[1],
                     head_yaw, head_pitch, head_roll], dtype=float)


def draw_calibration_overlay(frame, point_idx, pose_idx, settle_count, sample_count,
                              current_head_yaw_deg=0.0):
    """Draw calibration dot, head-pose guidance, ring countdown, and progress HUD."""
    total  = len(CALIB_POINTS)
    nx, ny = CALIB_POINTS[point_idx]
    tx = int(nx * MONITOR_WIDTH)
    ty = int(ny * MONITOR_HEIGHT)

    n_poses = len(CALIB_HEAD_POSES)
    pose_label, target_yaw, target_pitch = CALIB_HEAD_POSES[pose_idx]

    # --- Webcam frame annotation ---
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (frame.shape[1], frame.shape[0]), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.45, frame, 0.55, 0, frame)

    # Progress text
    prog_text = f"Calibration: point {point_idx + 1}/{total}, pose {pose_idx + 1}/{n_poses}"
    cv2.putText(frame, prog_text, (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 220, 0), 2, cv2.LINE_AA)

    # Head-pose guidance
    head_instr = f"Head: {pose_label}  (yaw target: {target_yaw:+.0f} deg, current: {current_head_yaw_deg:+.1f} deg)"
    cv2.putText(frame, head_instr, (10, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (180, 220, 255), 1, cv2.LINE_AA)

    # Collection status
    if settle_count < CALIB_SETTLE_FRAMES:
        instr = "Look at the dot, match head pose..."
        color_dot = (100, 100, 255)
    else:
        instr = f"Collecting... {sample_count}/{CALIB_SAMPLES_PER_POINT}"
        color_dot = (0, 255, 100)
    cv2.putText(frame, instr, (10, 75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    # --- Fullscreen calibration window ---
    cal_win = np.zeros((MONITOR_HEIGHT, MONITOR_WIDTH, 3), dtype=np.uint8)

    # Head-pose arrow indicator (top-center)
    arrow_cx = MONITOR_WIDTH // 2
    arrow_cy = 60
    arrow_len = 40
    # Arrow pointing in target yaw direction
    dx = int(arrow_len * math.sin(math.radians(target_yaw)))
    dy = 0
    cv2.arrowedLine(cal_win, (arrow_cx, arrow_cy), (arrow_cx + dx, arrow_cy + dy),
                    (180, 220, 255), 3, tipLength=0.35)
    cv2.putText(cal_win, f"Turn head: {pose_label}",
                (arrow_cx - 120, arrow_cy + 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (180, 220, 255), 2, cv2.LINE_AA)
    cv2.putText(cal_win, f"Your yaw: {current_head_yaw_deg:+.1f} deg",
                (arrow_cx - 100, arrow_cy + 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 180, 220), 1, cv2.LINE_AA)

    # Outer ring (progress arc)
    if settle_count >= CALIB_SETTLE_FRAMES and CALIB_SAMPLES_PER_POINT > 0:
        frac = sample_count / CALIB_SAMPLES_PER_POINT
        angle = int(360 * frac)
        cv2.ellipse(cal_win, (tx, ty), (28, 28), -90, 0, angle, (0, 200, 80), 4)
    # Dot
    cv2.circle(cal_win, (tx, ty), 14, color_dot, -1)
    cv2.circle(cal_win, (tx, ty),  4, (255, 255, 255), -1)
    # Point index + pose
    cv2.putText(cal_win, f"{point_idx + 1}/{total} [{pose_label}]",
                (tx + 18, ty - 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
    cv2.imshow("Calibration", cal_win)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)


def _rot_x(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0],
                     [0, ca, -sa],
                     [0, sa,  ca]], dtype=float)

def _rot_y(a):
    ca, sa = math.cos(a), math.sin(a)
    return np.array([[ ca, 0, sa],
                     [  0, 1,  0],
                     [-sa, 0, ca]], dtype=float)

def _normalize(v):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v

def _focal_px(width, fov_deg):
    # horizontal pinhole focal length
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


def create_monitor_plane(head_center, R_final, face_landmarks, w, h, 
                         forward_hint=None, gaze_origin=None, gaze_dir=None):
    """
    Build a 60cm x 40cm plane 50cm in front of the face, in world units.
    Monitor is oriented horizontally like a real monitor (top edge parallel to global X-axis).
    """
    # 1) Estimate scale from chin<->forehead distance
    try:
        lm_chin = face_landmarks[152]
        lm_fore = face_landmarks[10]
        chin_w = np.array([lm_chin.x * w,  lm_chin.y * h,  lm_chin.z * w], dtype=float)
        fore_w = np.array([lm_fore.x * w,  lm_fore.y * h,  lm_fore.z * w], dtype=float)
        face_h_units = np.linalg.norm(fore_w - chin_w)
        upc = face_h_units / 15.0  # units per cm
    except Exception:
        upc = 5.0
    
    # 2) Monitor geometry in world units
    dist_cm = 50.0

    mon_w_cm, mon_h_cm = 60.0, 40.0
    half_w = (mon_w_cm * 0.5) * upc
    half_h = (mon_h_cm * 0.5) * upc

    # Head forward vector
    head_forward = -R_final[:, 2]
    if forward_hint is not None:
        head_forward = forward_hint / np.linalg.norm(forward_hint)

    # --- NEW: use gaze ray intersection ---
    if gaze_origin is not None and gaze_dir is not None:
        gaze_dir = gaze_dir / np.linalg.norm(gaze_dir)

        # Place the monitor so its center is exactly at some point on the gaze ray
        # For simplicity: choose intersection at 50 cm from head_center along head_forward
        plane_point = head_center + head_forward * (50.0 * upc)
        plane_normal = head_forward

        denom = np.dot(plane_normal, gaze_dir)
        if abs(denom) > 1e-6:
            t = np.dot(plane_normal, plane_point - gaze_origin) / denom
            center_w = gaze_origin + t * gaze_dir
        else:
            # fallback: use fixed distance
            center_w = head_center + head_forward * (50.0 * upc)
    else:
        # fallback: original placement
        center_w = head_center + head_forward * (50.0 * upc)

    # Compute right/up using head orientation
    world_up = np.array([0, -1, 0], dtype=float)
    head_right = np.cross(world_up, head_forward)
    head_right /= np.linalg.norm(head_right)
    head_up = np.cross(head_forward, head_right)
    head_up /= np.linalg.norm(head_up)

    # Corners
    p0 = center_w - head_right * half_w - head_up * half_h
    p1 = center_w + head_right * half_w - head_up * half_h
    p2 = center_w + head_right * half_w + head_up * half_h
    p3 = center_w - head_right * half_w + head_up * half_h

    normal_w = head_forward / (np.linalg.norm(head_forward) + 1e-9)
    return [p0, p1, p2, p3], center_w, normal_w, upc




def update_orbit_from_keys():
    """Keyboard orbit controls that PRINT every frame while a key is held."""
    global orbit_yaw, orbit_pitch, orbit_radius
    yaw_step   = math.radians(1.5)
    pitch_step = math.radians(1.5)
    zoom_step  = 12.0

    changed = False

    # Rotate
    if keyboard.is_pressed('j'):  # yaw left
        orbit_yaw -= yaw_step; changed = True
    if keyboard.is_pressed('l'):  # yaw right
        orbit_yaw += yaw_step; changed = True
    if keyboard.is_pressed('i'):  # pitch up
        orbit_pitch += pitch_step; changed = True
    if keyboard.is_pressed('k'):  # pitch down
        orbit_pitch -= pitch_step; changed = True

    # Zoom
    if keyboard.is_pressed('['):  # zoom out
        orbit_radius += zoom_step; changed = True
    if keyboard.is_pressed(']'):  # zoom in
        orbit_radius = max(80.0, orbit_radius - zoom_step); changed = True

    # Reset (prints every frame while held)
    if keyboard.is_pressed('r'):
        orbit_yaw = 0.0
        orbit_pitch = 0.0
        orbit_radius = 600.0
        changed = True

    # Clamp pitch & radius
    orbit_pitch = max(math.radians(-89), min(math.radians(89), orbit_pitch))
    orbit_radius = max(80.0, orbit_radius)

    if changed:
        print(f"[Orbit Debug] yaw={math.degrees(orbit_yaw):.2f}°, "
              f"pitch={math.degrees(orbit_pitch):.2f}°, "
              f"radius={orbit_radius:.2f}, "
              f"fov={orbit_fov_deg:.1f}°")




def compute_scale(points_3d):
    # Use average pairwise distance for robustness
    n = len(points_3d)
    total = 0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0

def draw_gaze(frame, eye_center, iris_center, eye_radius, color, gaze_length):
    # Gaze vector
    gaze_direction = iris_center - eye_center
    gaze_direction /= np.linalg.norm(gaze_direction)
    gaze_endpoint = eye_center + gaze_direction * gaze_length

    cv2.line(frame, tuple(int(v) for v in eye_center[:2]), tuple(int(v) for v in gaze_endpoint[:2]), color, 2)

    # Segment points
    iris_offset = eye_center + gaze_direction * (1.2 * eye_radius)

    # ---- PART 1: back segment (behind iris) ----
    cv2.line(
        frame,
        (int(eye_center[0]), int(eye_center[1])),
        (int(iris_offset[0]), int(iris_offset[1])),
        color,
        1
    )

    # ---- IRIS (occludes part of the ray) ----
    up_dir = np.array([0, -1, 0])
    right_dir = np.cross(gaze_direction, up_dir)
    if np.linalg.norm(right_dir) < 1e-6:
        right_dir = np.array([1, 0, 0])
    up_dir = np.cross(right_dir, gaze_direction)
    up_dir /= np.linalg.norm(up_dir)
    right_dir /= np.linalg.norm(right_dir)
    ellipse_axes = (
        int((eye_radius / 3) * np.linalg.norm(right_dir[:2])),
        int((eye_radius / 3) * np.linalg.norm(up_dir[:2]))
    )
    angle = math.degrees(math.atan2(gaze_direction[1], gaze_direction[0]))

    # ---- PART 2: front segment (on top of iris) ----
    cv2.line(
        frame,
        (int(iris_offset[0]), int(iris_offset[1])),
        (int(gaze_endpoint[0]), int(gaze_endpoint[1])),
        color,
        1
    )

def draw_wireframe_cube(frame, center, R, size=80):
    # Given a center and rotation matrix, draw a cube aligned to that orientation
    right = R[:, 0]
    up = -R[:, 1]
    forward = -R[:, 2]

    hw, hh, hd = size * 1, size * 1, size * 1

    def corner(x_sign, y_sign, z_sign):
        return (center +
                x_sign * hw * right +
                y_sign * hh * up +
                z_sign * hd * forward)

    # 8 corners of the cube
    corners = [corner(x, y, z) for x in [-1, 1] for y in [1, -1] for z in [-1, 1]]
    projected = [(int(pt[0]), int(pt[1])) for pt in corners]

    # Edges connecting the corners
    edges = [
        (0, 1), (1, 3), (3, 2), (2, 0),
        (4, 5), (5, 7), (7, 6), (6, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    for i, j in edges:
        cv2.line(frame, projected[i], projected[j], (255, 128, 0), 2)

def compute_and_draw_coordinate_box(frame, face_landmarks, indices, ref_matrix_container, color=(0, 255, 0), size=80):
    # Extract 3D positions of selected landmarks
    points_3d = np.array([
        [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
        for i in indices
    ])

    # Compute the average position as the center of this substructure
    center = np.mean(points_3d, axis=0)

    # Draw the raw 2D landmark points
    for i in indices:
        x, y = int(face_landmarks[i].x * w), int(face_landmarks[i].y * h)
        cv2.circle(frame, (x, y), 3, color, -1)

    # PCA-based orientation: Compute eigenvectors of the covariance matrix
    centered = points_3d - center
    cov = np.cov(centered.T)
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvecs = eigvecs[:, np.argsort(-eigvals)]  # Sort by descending eigenvalue (major axes)

    # Ensure the orientation matrix is right-handed
    if np.linalg.det(eigvecs) < 0:
        eigvecs[:, 2] *= -1

    # Convert to Euler angles and re-construct rotation matrix (optional but clarifies the transform)
    r = Rscipy.from_matrix(eigvecs)
    roll, pitch, yaw = r.as_euler('zyx', degrees=False)
    yaw *= 1
    roll *= 1
    R_final = Rscipy.from_euler('zyx', [roll, pitch, yaw]).as_matrix()

    # === Stabilize rotation with reference matrix to avoid flipping during eigenvector sign change ===
    if ref_matrix_container[0] is None:
        ref_matrix_container[0] = R_final.copy()
    else:
        R_ref = ref_matrix_container[0]
        for i in range(3):
            if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                R_final[:, i] *= -1

    # Draw cube and orientation axes on the image
    draw_wireframe_cube(frame, center, R_final, size)

    # Draw X (green), Y (blue), Z (red) axes
    axis_length = size * 1.2
    axis_dirs = [R_final[:, 0], -R_final[:, 1], -R_final[:, 2]]
    axis_colors = [(0, 255, 0), (0, 0, 255), (255, 0, 0)]

    for i in range(3):
        end_pt = center + axis_dirs[i] * axis_length
        cv2.line(frame, (int(center[0]), int(center[1])), (int(end_pt[0]), int(end_pt[1])), axis_colors[i], 2)

    return center, R_final, points_3d

def convert_gaze_to_screen_coordinates(combined_gaze_direction, calibration_offset_yaw, calibration_offset_pitch):
    """
    Convert 3D gaze direction vector to 2D screen coordinates
    This function is adapted from the old script's vector-to-screen mapping logic
    """
    # Reference forward direction (camera looking straight ahead)
    reference_forward = np.array([0, 0, -1])  # Z-axis into the screen

    # Normalize the gaze direction
    avg_direction = combined_gaze_direction / np.linalg.norm(combined_gaze_direction)

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

    # Convert left rotations to 0-180 (from old script logic)
    if yaw_deg < 0:
        yaw_deg = -(yaw_deg)
    elif yaw_deg > 0:
        yaw_deg = - yaw_deg

    #yaw is now converted to -90 (looking directly left) to +90 (looking directly right), wrt camera
    #pitch is now converted to +90 (looking straight up) and -90 (looking straight down), wrt camera
    
    raw_yaw_deg = yaw_deg
    raw_pitch_deg = pitch_deg

    
    # Specify degrees at which screen border will be reached
    yawDegrees = 5 * 3  # x degrees left or right
    pitchDegrees = 2.0 * 2.5  # x degrees up or down

    # Apply calibration offsets
    yaw_deg += calibration_offset_yaw
    pitch_deg += calibration_offset_pitch

    # Map to full screen resolution
    screen_x = int(((yaw_deg + yawDegrees) / (2 * yawDegrees)) * MONITOR_WIDTH)
    screen_y = int(((pitchDegrees - pitch_deg) / (2 * pitchDegrees)) * MONITOR_HEIGHT)

    # Clamp screen position to monitor bounds
    screen_x = max(10, min(screen_x, MONITOR_WIDTH - 10))
    screen_y = max(10, min(screen_y, MONITOR_HEIGHT - 10))

    return screen_x, screen_y, raw_yaw_deg, raw_pitch_deg

def render_debug_view_orbit(
    h, w,
    head_center3d=None,
    sphere_world_l=None, scaled_radius_l=None,
    sphere_world_r=None, scaled_radius_r=None,
    iris3d_l=None, iris3d_r=None,
    left_locked=False, right_locked=False,
    landmarks3d=None,
    combined_dir=None,
    gaze_len=430,
    monitor_corners=None,
    monitor_center=None,
    monitor_normal=None,
    gaze_markers=None,
):
    if head_center3d is None:
        return

    debug = np.zeros((h, w, 3), dtype=np.uint8)

    # --- Choose orbit pivot ---
    head_w = np.asarray(head_center3d, dtype=float)

    # NEW: if we've frozen the world, orbit around the frozen pivot (monitor center at calib)
    global debug_world_frozen, orbit_pivot_frozen
    if debug_world_frozen and orbit_pivot_frozen is not None:
        pivot_w = np.asarray(orbit_pivot_frozen, dtype=float)
    else:
        if monitor_center is not None:
            pivot_w = (head_w + np.asarray(monitor_center, dtype=float)) * 0.5
        else:
            pivot_w = head_w

    # --- Camera pose (orbit around pivot_w) ---
    f_px = _focal_px(w, orbit_fov_deg)
    cam_offset = _rot_y(orbit_yaw) @ (_rot_x(orbit_pitch) @ np.array([0.0, 0.0, orbit_radius]))
    cam_pos = pivot_w + cam_offset

    up_world = np.array([0.0, -1.0, 0.0])   # image-space up is -Y
    fwd = _normalize(pivot_w - cam_pos)     # look at pivot
    right = _normalize(np.cross(fwd, up_world))
    up = _normalize(np.cross(right, fwd))
    V = np.stack([right, up, fwd], axis=0)

    def project_point(P):
        Pw = np.asarray(P, dtype=float)
        Pc = V @ (Pw - cam_pos)
        if Pc[2] <= 1e-3:
            return None
        x = f_px * (Pc[0] / Pc[2]) + w * 0.5
        y = -f_px * (Pc[1] / Pc[2]) + h * 0.5
        if not (np.isfinite(x) and np.isfinite(y)):
            return None
        return (int(x), int(y)), Pc[2]

    # --- helper draws ---
    def draw_poly_3d(pts, color=(0, 200, 255), thickness=2):
        projs = [project_point(p) for p in pts]
        if any(p is None for p in projs): return
        p2 = [p[0] for p in projs]
        for a, b in zip(p2, p2[1:] + [p2[0]]):
            cv2.line(debug, a, b, color, thickness)

    def draw_cross_3d(P, size=12, color=(255, 0, 255), thickness=2):
        res = project_point(P)
        if res is None: return
        (x, y), _ = res
        cv2.line(debug, (x - size, y), (x + size, y), color, thickness)
        cv2.line(debug, (x, y - size), (x, y + size), color, thickness)

    def draw_arrow_3d(P0, P1, color=(0, 200, 255), thickness=3):
        a = project_point(P0); b = project_point(P1)
        if a is None or b is None: return
        p0, p1 = a[0], b[0]
        cv2.line(debug, p0, p1, color, thickness)
        v = np.array([p1[0]-p0[0], p1[1]-p0[1]], dtype=float)
        n = np.linalg.norm(v)
        if n > 1e-3:
            v /= n
            l = np.array([-v[1], v[0]])
            ah = 10
            a1 = (int(p1[0] - v[0]*ah + l[0]*ah*0.6), int(p1[1] - v[1]*ah + l[1]*ah*0.6))
            a2 = (int(p1[0] - v[0]*ah - l[0]*ah*0.6), int(p1[1] - v[1]*ah - l[1]*ah*0.6))
            cv2.line(debug, p1, a1, color, thickness)
            cv2.line(debug, p1, a2, color, thickness)

    # --- Landmarks ---
    if landmarks3d is not None:
        for P in landmarks3d:
            res = project_point(P)
            if res is not None:
                cv2.circle(debug, res[0], 0, (200, 200, 200), -1)

    # --- Head center ---
    draw_cross_3d(head_w, size=12, color=(255, 0, 255), thickness=2)
    hc2d = project_point(head_w)
    if hc2d is not None:
        cv2.putText(debug, "Head Center", (hc2d[0][0] + 12, hc2d[0][1] - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1, cv2.LINE_AA)

    # --- Pivot visual (small cross + line to head and monitor) ---
    draw_cross_3d(pivot_w, size=8, color=(180, 120, 255), thickness=2)
    if monitor_center is not None:
        mc2d = project_point(monitor_center)
        pv2d = project_point(pivot_w)
        if mc2d is not None and pv2d is not None and hc2d is not None:
            cv2.line(debug, pv2d[0], hc2d[0], (160, 100, 255), 1)
            cv2.line(debug, pv2d[0], mc2d[0], (160, 100, 255), 1)

    # --- Eyes + per-eye gaze (unchanged from your version) ---
    left_dir = None
    right_dir = None

    if left_locked and sphere_world_l is not None:
        res = project_point(sphere_world_l)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_l if scaled_radius_l else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (255, 255, 25), 1)
            if iris3d_l is not None:
                left_dir = np.asarray(iris3d_l) - np.asarray(sphere_world_l)
                p1 = project_point(np.asarray(sphere_world_l) + _normalize(left_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (155, 155, 25), 1)
    elif iris3d_l is not None:
        res = project_point(iris3d_l)
        if res is not None:
            cv2.circle(debug, res[0], 2, (255, 255, 25), 1)

    if right_locked and sphere_world_r is not None:
        res = project_point(sphere_world_r)
        if res is not None:
            (cx, cy), z = res
            r_px = max(2, int((scaled_radius_r if scaled_radius_r else 6) * f_px / max(z, 1e-3)))
            cv2.circle(debug, (cx, cy), r_px, (25, 255, 255), 1)
            if iris3d_r is not None:
                right_dir = np.asarray(iris3d_r) - np.asarray(sphere_world_r)
                p1 = project_point(np.asarray(sphere_world_r) + _normalize(right_dir) * gaze_len)
                if p1 is not None:
                    cv2.line(debug, (cx, cy), p1[0], (25, 155, 155), 1)
    elif iris3d_r is not None:
        res = project_point(iris3d_r)
        if res is not None:
            cv2.circle(debug, res[0], 2, (25, 255, 255), 1)

    if left_locked and right_locked and sphere_world_l is not None and sphere_world_r is not None:
        origin_mid = (np.asarray(sphere_world_l) + np.asarray(sphere_world_r)) / 2.0
        if combined_dir is None and (left_dir is not None or right_dir is not None):
            parts = []
            if left_dir is not None:  parts.append(_normalize(left_dir))
            if right_dir is not None: parts.append(_normalize(right_dir))
            if parts:
                combined_dir = _normalize(np.mean(parts, axis=0))
        if combined_dir is not None:
            p0 = project_point(origin_mid)
            p1 = project_point(origin_mid + _normalize(combined_dir) * (gaze_len * 1.2))
            if p0 is not None and p1 is not None:
                cv2.line(debug, p0[0], p1[0], (155, 200, 10), 2)

    # --- Monitor plane ---
    if monitor_corners is not None:
        def draw_poly(points, color, thickness):
            projs = [project_point(p) for p in points]
            if any(p is None for p in projs): return
            p2 = [p[0] for p in projs]
            for a, b in zip(p2, p2[1:] + [p2[0]]):
                cv2.line(debug, a, b, color, thickness)
        draw_poly(monitor_corners, (0, 200, 255), 2)
        draw_poly([monitor_corners[0], monitor_corners[2]], (0, 150, 210), 1)
        draw_poly([monitor_corners[1], monitor_corners[3]], (0, 150, 210), 1)
        if monitor_center is not None:
            draw_cross_3d(monitor_center, size=8, color=(0, 200, 255), thickness=2)
            if monitor_normal is not None:
                tip = np.asarray(monitor_center) + np.asarray(monitor_normal) * (20.0 * (units_per_cm or 1.0))
                draw_arrow_3d(monitor_center, tip, color=(0, 220, 255), thickness=2)

    # --- Stored gaze markers on the monitor plane (green circles) ---
    if (gaze_markers and monitor_corners is not None):
        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
        u = p1 - p0  # width direction
        v = p3 - p0  # height direction
        width_world = float(np.linalg.norm(u))
        if width_world > 1e-9:
            u_hat = u / width_world
            r_world = 0.01 * width_world  # 2% of width
            for (a, b) in gaze_markers:
                Pm = p0 + a * u + b * v
                projP = project_point(Pm)
                projR = project_point(Pm + u_hat * r_world)
                if projP is not None and projR is not None:
                    center_px = projP[0]
                    r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                    cv2.circle(debug, center_px, r_px, (0, 255, 0), 1, lineType=cv2.LINE_AA)



    # --- Gaze hit on monitor plane (circle at intersection) ---
    if (monitor_corners is not None and monitor_center is not None and monitor_normal is not None
        and combined_dir is not None
        and sphere_world_l is not None and sphere_world_r is not None):

        # Ray: origin at midpoint between eyes; direction = combined gaze
        O = (np.asarray(sphere_world_l, dtype=float) + np.asarray(sphere_world_r, dtype=float)) * 0.5
        D = _normalize(np.asarray(combined_dir, dtype=float))

        # Plane: through monitor_center with normal = monitor_normal
        C = np.asarray(monitor_center, dtype=float)
        N = _normalize(np.asarray(monitor_normal, dtype=float))

        denom = float(np.dot(N, D))
        if abs(denom) > 1e-6:
            t = float(np.dot(N, (C - O)) / denom)
            if t > 0.0:
                P = O + t * D  # world-space intersection point

                # Inside-quad test using monitor's local axes (top-left p0, top-right p1, bottom-left p3)
                p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                u = p1 - p0             # horizontal (width) vector
                v = p3 - p0             # vertical (height) vector
                wv = P  - p0

                u_len2 = float(np.dot(u, u))
                v_len2 = float(np.dot(v, v))
                if u_len2 > 1e-9 and v_len2 > 1e-9:
                    a = float(np.dot(wv, u) / u_len2)  # 0..1 across width
                    b = float(np.dot(wv, v) / v_len2)  # 0..1 across height

                    if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                        # Project center to pixels
                        projP = project_point(P)
                        if projP is not None:
                            center_px = projP[0]

                            # Circle radius = 5% of monitor width (world), projected to pixels
                            width_world = math.sqrt(u_len2)
                            r_world = 0.05 * width_world
                            u_hat = u / max(width_world, 1e-9)

                            projR = project_point(P + u_hat * r_world)
                            if projR is not None:
                                r_px = int(max(1, np.linalg.norm(np.array(projR[0]) - np.array(center_px))))
                                cv2.circle(debug, center_px, r_px, (0, 255, 255), 2, lineType=cv2.LINE_AA)


    # --- Key command help text in lower-left ---
    help_text = [
        "C = lock eye spheres",
        "M = start model calibration",
        "S = re-centre (fallback)",
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

    font        = cv2.FONT_HERSHEY_SIMPLEX
    font_scale  = 0.5
    thickness   = 1
    line_height = 18  # pixels between lines

    # Start a bit above the bottom-left corner
    y0 = h - (len(help_text) * line_height) - 10
    x0 = 10

    for i, text in enumerate(help_text):
        y = y0 + i * line_height
        cv2.putText(debug, text, (x0, y), font, font_scale, (200, 200, 200), thickness, cv2.LINE_AA)


    cv2.imshow("Head/Eye Debug", debug)



def mouse_mover():
    """Mouse movement thread from old script"""
    while True:
        if mouse_control_enabled:
            with mouse_lock:
                x, y = mouse_target
            pyautogui.moveTo(x, y)
        time.sleep(0.01)  # adjust for responsiveness

# Start mouse movement thread
threading.Thread(target=mouse_mover, daemon=True).start()

# Eye sphere tracking variables (from new script)
left_sphere_locked = False
left_sphere_local_offset = None
left_calibration_nose_scale = None

right_sphere_locked = False
right_sphere_local_offset = None
right_calibration_nose_scale = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    combined_dir = None  # will be filled once you compute a smoothed direction

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Process frame with appropriate API
    if USE_NEW_API:
        # New API: convert frame to MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=frame_rgb)
        timestamp_ms = int(time.time() * 1000)
        results = face_mesh.detect_for_video(mp_image, timestamp_ms)
        face_landmarks = results.face_landmarks[0] if results.face_landmarks else None
    else:
        # Old API
        results = face_mesh.process(frame_rgb)
        face_landmarks = results.multi_face_landmarks[0].landmark if results.multi_face_landmarks else None

    if face_landmarks:

        # Index for left iris center point (from MediaPipe's iris model)
        left_iris_idx = 468
        right_iris_idx = 473
        left_iris = face_landmarks[left_iris_idx]
        right_iris = face_landmarks[right_iris_idx]

        # Compute and draw stabilized coordinate frame from nose region
        head_center, R_final, nose_points_3d = compute_and_draw_coordinate_box(
            frame,
            face_landmarks,
            nose_indices,
            R_ref_nose,
            color=(0, 255, 0),
            size=80
        )

        # TODO compute this radius using canthus during calibration
        base_radius = 20  # radius at calibration distance

        x_iris_l = int(left_iris.x * w)
        y_iris_l = int(left_iris.y * h)
        # === LEFT EYE visualization ===
        if not left_sphere_locked:
            cv2.circle(frame, (x_iris_l, y_iris_l), 10, (255, 25, 25), 2)
        else:
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scaled_offset = left_sphere_local_offset * scale_ratio
            sphere_world_l = head_center + R_final @ scaled_offset
            x_sphere_l, y_sphere_l = int(sphere_world_l[0]), int(sphere_world_l[1])
            scaled_radius_l = int(base_radius * scale_ratio)
            cv2.circle(frame, (x_sphere_l, y_sphere_l), scaled_radius_l, (255, 255, 25), 2)

        x_iris_r = int(right_iris.x * w)
        y_iris_r = int(right_iris.y * h)
        # === RIGHT EYE visualization ===
        if not right_sphere_locked:
            cv2.circle(frame, (x_iris_r, y_iris_r), 10, (25, 255, 25), 2)
        else:
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            scaled_offset_r = right_sphere_local_offset * scale_ratio_r
            sphere_world_r = head_center + R_final @ scaled_offset_r
            x_sphere_r, y_sphere_r = int(sphere_world_r[0]), int(sphere_world_r[1])
            scaled_radius_r = int(base_radius * scale_ratio_r)
            cv2.circle(frame, (x_sphere_r, y_sphere_r), scaled_radius_r, (25, 255, 255), 2)

        iris_3d_left = np.array([left_iris.x * w, left_iris.y * h, left_iris.z * w])
        iris_3d_right = np.array([right_iris.x * w, right_iris.y * h, right_iris.z * w])
        
        if left_sphere_locked and right_sphere_locked:
            # ==== DRAW LEFT AND RIGHT GAZE ====
            draw_gaze(frame, sphere_world_l, iris_3d_left, scaled_radius_l, (55, 255, 0), 130)   
            draw_gaze(frame, sphere_world_r, iris_3d_right, scaled_radius_r, (55, 255, 0), 130)  

            # ==== COMPUTE COMBINED GAZE DIRECTION FOR SCREEN MAPPING ====
            # Calculate individual gaze directions
            left_gaze_dir = iris_3d_left - sphere_world_l
            left_gaze_dir /= np.linalg.norm(left_gaze_dir)
            
            right_gaze_dir = iris_3d_right - sphere_world_r
            right_gaze_dir /= np.linalg.norm(right_gaze_dir)
            
            # Combine gaze directions (average)
            raw_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
            raw_combined_direction /= np.linalg.norm(raw_combined_direction)

            # Update direction buffer for smoothing
            combined_gaze_directions.append(raw_combined_direction)

            # Smoothed direction
            avg_combined_direction = np.mean(combined_gaze_directions, axis=0)
            avg_combined_direction /= np.linalg.norm(avg_combined_direction)

            combined_dir = avg_combined_direction

            # ==== EXTRACT FEATURES FOR MODEL ====
            feat = extract_gaze_features(
                avg_combined_direction,
                R_final,
                iris_3d_left, iris_3d_right,
                sphere_world_l, sphere_world_r,
                head_center
            )

            # ==== CALIBRATION DATA COLLECTION ====
            if calib_mode:
                # Extract current head yaw for guidance display
                r_head = Rscipy.from_matrix(R_final)
                current_head_yaw_deg = r_head.as_euler('yxz', degrees=True)[0]

                draw_calibration_overlay(frame, calib_point_index, calib_pose_index,
                                         calib_settle_count, calib_sample_count,
                                         current_head_yaw_deg)

                if calib_settle_count < CALIB_SETTLE_FRAMES:
                    calib_settle_count += 1
                else:
                    # Accumulate raw feature for this point+pose
                    calib_feat_buf.append(feat.copy())
                    calib_sample_count += 1

                    if calib_sample_count >= CALIB_SAMPLES_PER_POINT:
                        # Average features collected for this pose → one training row
                        avg_feat = np.mean(calib_feat_buf, axis=0)

                        # Compute grid cell index as target label
                        nx, ny = CALIB_POINTS[calib_point_index]
                        grid_col = min(int(nx * OUTPUT_GRID_COLS), OUTPUT_GRID_COLS - 1)
                        grid_row = min(int(ny * OUTPUT_GRID_ROWS), OUTPUT_GRID_ROWS - 1)
                        cell_idx = grid_row * OUTPUT_GRID_COLS + grid_col

                        calib_X.append(avg_feat)
                        calib_Y.append(cell_idx)

                        # Advance to next head-pose sub-phase
                        calib_pose_index += 1
                        calib_settle_count = 0
                        calib_sample_count = 0
                        calib_feat_buf     = []

                        if calib_pose_index >= len(CALIB_HEAD_POSES):
                            # All poses done for this point → next calibration point
                            calib_pose_index = 0
                            calib_point_index += 1

                            if calib_point_index >= len(CALIB_POINTS):
                                # All points collected — fit model
                                calib_mode = False
                                cv2.destroyWindow("Calibration")
                                gaze_model.fit(calib_X, calib_Y)
                                gaze_model.save()
                                # Reset output smoothing
                                smooth_probs = None
                                model_feat_history.clear()
                                print("[Calibration] Complete. Model fitted and saved.")

            # ==== CONVERT GAZE TO GRID CELL ====
            predicted_cell = None
            predicted_row = None
            predicted_col = None
            cell_confidence = 0.0

            if gaze_model.fitted and not calib_mode:
                # Smooth the feature vector over recent frames before predicting
                model_feat_history.append(feat.copy())
                smooth_feat = np.mean(model_feat_history, axis=0)

                pred = gaze_model.predict(smooth_feat)
                if pred is not None:
                    cell_idx, conf, probs = pred

                    # Exponential moving average on class probabilities
                    if smooth_probs is None:
                        smooth_probs = probs.copy()
                    else:
                        smooth_probs = smooth_probs + prob_smooth_alpha * (probs - smooth_probs)

                    predicted_cell = int(np.argmax(smooth_probs))
                    cell_confidence = float(smooth_probs[predicted_cell])
                    predicted_row = predicted_cell // OUTPUT_GRID_COLS
                    predicted_col = predicted_cell % OUTPUT_GRID_COLS

                    # Map to cell centre pixel for optional mouse control
                    cell_cx = int((predicted_col + 0.5) / OUTPUT_GRID_COLS * MONITOR_WIDTH)
                    cell_cy = int((predicted_row + 0.5) / OUTPUT_GRID_ROWS * MONITOR_HEIGHT)
                    screen_x, screen_y = cell_cx, cell_cy
                else:
                    screen_x, screen_y = CENTER_X, CENTER_Y

                mode_label = "GRID"
            else:
                screen_x, screen_y, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
                    avg_combined_direction,
                    calibration_offset_yaw,
                    calibration_offset_pitch
                )
                mode_label = "FALLBACK"

            # Update mouse target (snaps to cell centre when using grid model)
            if mouse_control_enabled:
                with mouse_lock:
                    mouse_target[0] = screen_x
                    mouse_target[1] = screen_y

            # ===== Write grid cell to file =====
            if predicted_cell is not None:
                write_screen_position(predicted_row, predicted_col,
                                      cell_confidence, predicted_cell)
            else:
                write_screen_position(-1, -1, 0.0, -1)

            # Draw combined gaze ray for visualization
            combined_origin = (sphere_world_l + sphere_world_r) / 2
            combined_target = combined_origin + avg_combined_direction * gaze_length
            cv2.line(
                frame,
                (int(combined_origin[0]), int(combined_origin[1])),
                (int(combined_target[0]), int(combined_target[1])),
                (255, 255, 10), 3
            )

            # ==== DRAW GRID OVERLAY ON WEBCAM FRAME ====
            if predicted_cell is not None:
                # Draw grid lines on the webcam frame (scaled to webcam resolution)
                for r in range(1, OUTPUT_GRID_ROWS):
                    y_line = int(r * h / OUTPUT_GRID_ROWS)
                    cv2.line(frame, (0, y_line), (w, y_line), (80, 80, 80), 1)
                for c in range(1, OUTPUT_GRID_COLS):
                    x_line = int(c * w / OUTPUT_GRID_COLS)
                    cv2.line(frame, (x_line, 0), (x_line, h), (80, 80, 80), 1)

                # Highlight the predicted cell with a semi-transparent overlay
                cell_x0 = int(predicted_col * w / OUTPUT_GRID_COLS)
                cell_y0 = int(predicted_row * h / OUTPUT_GRID_ROWS)
                cell_x1 = int((predicted_col + 1) * w / OUTPUT_GRID_COLS)
                cell_y1 = int((predicted_row + 1) * h / OUTPUT_GRID_ROWS)
                cell_overlay = frame.copy()
                cv2.rectangle(cell_overlay, (cell_x0, cell_y0), (cell_x1, cell_y1),
                              (0, 255, 0), -1)
                alpha = min(0.35, cell_confidence * 0.5)  # brighter when more confident
                cv2.addWeighted(cell_overlay, alpha, frame, 1 - alpha, 0, frame)
                cv2.rectangle(frame, (cell_x0, cell_y0), (cell_x1, cell_y1),
                              (0, 255, 0), 2)

                # Cell label inside the highlighted cell
                cell_label = f"R{predicted_row}C{predicted_col} ({cell_confidence:.0%})"
                cv2.putText(frame, cell_label,
                            (cell_x0 + 5, cell_y0 + 22),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

            # HUD text
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2

            if predicted_cell is not None:
                hud_lines = [
                    f"Grid: R{predicted_row} C{predicted_col}  ({cell_confidence:.0%})  [{mode_label}]",
                    f"Grid size: {OUTPUT_GRID_ROWS}x{OUTPUT_GRID_COLS}",
                ]
            else:
                hud_lines = [
                    f"Screen: ({screen_x}, {screen_y})  [{mode_label}]",
                ]
            for i, text in enumerate(hud_lines):
                (tw, _), _ = cv2.getTextSize(text, font, font_scale, thickness)
                cx = (w - tw) // 2
                cv2.putText(frame, text, (cx, 30 + i * 30), font, font_scale, (0, 255, 0), thickness)

        # Draw all landmark points in white
        for idx, lm in enumerate(face_landmarks):
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (x, y), 0, (255, 255, 255), -1)

        # Smooth orbit controls each frame
        update_orbit_from_keys()

        # Build 3D landmarks in your existing scale (x*w, y*h, z*w)
        landmarks3d = None
        if face_landmarks:
            landmarks3d = np.array([[p.x * w, p.y * h, p.z * w] for p in face_landmarks], dtype=float)

        render_debug_view_orbit(
            h, w,
            head_center3d=head_center if 'head_center' in locals() else None,
            sphere_world_l=sphere_world_l if left_sphere_locked and 'sphere_world_l' in locals() else None,
            scaled_radius_l=scaled_radius_l if left_sphere_locked and 'scaled_radius_l' in locals() else None,
            sphere_world_r=sphere_world_r if right_sphere_locked and 'sphere_world_r' in locals() else None,
            scaled_radius_r=scaled_radius_r if right_sphere_locked and 'scaled_radius_r' in locals() else None,
            iris3d_l=iris_3d_left if 'iris_3d_left' in locals() else None,
            iris3d_r=iris_3d_right if 'iris_3d_right' in locals() else None,
            left_locked=left_sphere_locked,
            right_locked=right_sphere_locked,
            landmarks3d=landmarks3d,
            combined_dir=avg_combined_direction if 'avg_combined_direction' in locals() else None,
            gaze_len=5230,
            monitor_corners=monitor_corners,
            monitor_center=monitor_center_w,
            monitor_normal=monitor_normal_w,
            gaze_markers=gaze_markers
        )


    cv2.imshow("Integrated Eye Tracking", frame)

    # Handle keyboard input
    if keyboard.is_pressed('f7'):
        mouse_control_enabled = not mouse_control_enabled
        print(f"[Mouse Control] {'Enabled' if mouse_control_enabled else 'Disabled'}")
        time.sleep(0.3)  # debounce to prevent rapid toggling

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c') and not (left_sphere_locked and right_sphere_locked):
        current_nose_scale = compute_scale(nose_points_3d)
        # Lock LEFT eye
        left_sphere_local_offset = R_final.T @ (iris_3d_left - head_center)
        camera_dir_world = np.array([0, 0, 1])
        camera_dir_local = R_final.T @ camera_dir_world
        left_sphere_local_offset += base_radius * camera_dir_local
        left_calibration_nose_scale = current_nose_scale
        left_sphere_locked = True

        # Lock RIGHT eye
        right_sphere_local_offset = R_final.T @ (iris_3d_right - head_center)
        right_sphere_local_offset += base_radius * camera_dir_local  # use same camera_dir_local
        right_calibration_nose_scale = current_nose_scale
        right_sphere_locked = True

        # === Create 3D monitor plane at calibration ===
        # Compute instantaneous sphere positions at calibration distance (scale=1)
        sphere_world_l_calib = head_center + R_final @ left_sphere_local_offset
        sphere_world_r_calib = head_center + R_final @ right_sphere_local_offset

        # Estimate a forward gaze direction from the two eyes
        left_dir  = iris_3d_left  - sphere_world_l_calib
        right_dir = iris_3d_right - sphere_world_r_calib
        # Normalize (guard zero)
        if np.linalg.norm(left_dir)  > 1e-9: left_dir  /= np.linalg.norm(left_dir)
        if np.linalg.norm(right_dir) > 1e-9: right_dir /= np.linalg.norm(right_dir)
        forward_hint = (left_dir + right_dir) * 0.5
        if np.linalg.norm(forward_hint) > 1e-9:
            forward_hint /= np.linalg.norm(forward_hint)
        else:
            forward_hint = None  # fallback to head frame

        gaze_origin = (sphere_world_l_calib + sphere_world_r_calib) / 2
        gaze_dir = forward_hint  # already normalized

        monitor_corners, monitor_center_w, monitor_normal_w, units_per_cm = create_monitor_plane(
            head_center, R_final, face_landmarks, w, h,
            forward_hint=forward_hint,
            gaze_origin=gaze_origin,
            gaze_dir=gaze_dir
        )

        # Freeze the debug world's orbit pivot at the calibrated monitor center
        #global debug_world_frozen, orbit_pivot_frozen
        debug_world_frozen = True
        orbit_pivot_frozen = monitor_center_w.copy()
        print("[Debug View] World pivot frozen at monitor center.")

        print(f"[Monitor] units_per_cm={units_per_cm:.3f}, center={monitor_center_w}, normal={monitor_normal_w}")


        print("[Both Spheres Locked] Eye sphere calibration complete.")
    elif key == ord('m') and left_sphere_locked and right_sphere_locked:
        # Start multi-point model calibration with head-pose variation
        calib_mode        = True
        calib_point_index = 0
        calib_pose_index  = 0
        calib_settle_count = 0
        calib_sample_count = 0
        calib_X           = []
        calib_Y           = []
        calib_feat_buf    = []
        smooth_probs      = None
        model_feat_history.clear()
        total = len(CALIB_POINTS)
        n_poses = len(CALIB_HEAD_POSES)
        print(f"[Calibration] Starting {total}-point x {n_poses}-pose calibration "
              f"({total * n_poses} total collections).  "
              f"Grid: {OUTPUT_GRID_ROWS}x{OUTPUT_GRID_COLS} = {NUM_CLASSES} cells.")
    elif key == ord('s') and left_sphere_locked and right_sphere_locked:
        # Fallback single-point centre calibration (kept for convenience)
        left_gaze_dir = iris_3d_left - sphere_world_l
        left_gaze_dir /= np.linalg.norm(left_gaze_dir)
        right_gaze_dir = iris_3d_right - sphere_world_r
        right_gaze_dir /= np.linalg.norm(right_gaze_dir)
        current_combined_direction = (left_gaze_dir + right_gaze_dir) / 2
        current_combined_direction /= np.linalg.norm(current_combined_direction)
        _, _, raw_yaw, raw_pitch = convert_gaze_to_screen_coordinates(
            current_combined_direction, 0, 0
        )
        calibration_offset_yaw   = 0 - raw_yaw
        calibration_offset_pitch = 0 - raw_pitch
        print(f"[Screen Calibrated] Offset Yaw: {calibration_offset_yaw:.2f}, "
              f"Offset Pitch: {calibration_offset_pitch:.2f}")
    elif key == ord('x'):
        # Drop a marker at the current gaze∩monitor point
        if (monitor_corners is not None and monitor_center_w is not None and monitor_normal_w is not None
            and left_sphere_locked and right_sphere_locked):
            # Recompute current eye-sphere positions (scale-aware)
            current_nose_scale = compute_scale(nose_points_3d)
            scale_ratio_l = current_nose_scale / left_calibration_nose_scale if left_calibration_nose_scale else 1.0
            scale_ratio_r = current_nose_scale / right_calibration_nose_scale if right_calibration_nose_scale else 1.0
            sphere_world_l_now = head_center + R_final @ (left_sphere_local_offset * scale_ratio_l)
            sphere_world_r_now = head_center + R_final @ (right_sphere_local_offset * scale_ratio_r)

            # Combined gaze direction (use smoothed if available; otherwise instantaneous)
            if 'avg_combined_direction' in locals() and avg_combined_direction is not None:
                D = _normalize(np.asarray(avg_combined_direction, dtype=float))
            else:
                lg = iris_3d_left  - sphere_world_l_now
                rg = iris_3d_right - sphere_world_r_now
                if np.linalg.norm(lg) < 1e-9 or np.linalg.norm(rg) < 1e-9:
                    print("[Marker] Gaze direction invalid; try again.")
                    D = None
                else:
                    lg /= np.linalg.norm(lg)
                    rg /= np.linalg.norm(rg)
                    D = _normalize(lg + rg)

            if D is not None:
                O = (sphere_world_l_now + sphere_world_r_now) * 0.5
                C = np.asarray(monitor_center_w, dtype=float)
                N = _normalize(np.asarray(monitor_normal_w, dtype=float))
                denom = float(np.dot(N, D))
                if abs(denom) < 1e-6:
                    print("[Marker] Gaze ray parallel to monitor; no marker.")
                else:
                    t = float(np.dot(N, (C - O)) / denom)
                    if t <= 0.0:
                        print("[Marker] Intersection behind/at eye; no marker.")
                    else:
                        P = O + t * D  # world-space intersection
                        # Map P to monitor local (a,b), then store if inside the quad
                        p0, p1, p2, p3 = [np.asarray(p, dtype=float) for p in monitor_corners]
                        u = p1 - p0
                        v = p3 - p0
                        u_len2 = float(np.dot(u, u))
                        v_len2 = float(np.dot(v, v))
                        if u_len2 > 1e-9 and v_len2 > 1e-9:
                            wv = P - p0
                            a = float(np.dot(wv, u) / u_len2)
                            b = float(np.dot(wv, v) / v_len2)
                            if 0.0 <= a <= 1.0 and 0.0 <= b <= 1.0:
                                gaze_markers.append((a, b))
                                print(f"[Marker] Added at a={a:.3f}, b={b:.3f}")
                            else:
                                print("[Marker] Gaze not on monitor; no marker.")
                        else:
                            print("[Marker] Monitor dimensions degenerate; no marker.")
        else:
            print("[Marker] Monitor/gaze not ready; complete center calibration first.")


cap.release()

cv2.destroyAllWindows()
