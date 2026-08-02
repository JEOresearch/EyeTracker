"""Core detection module for face landmarks and head pose estimation using MediaPipe."""

import cv2
import numpy as np
from typing import List, Optional, Tuple
import mediapipe as mp
from scipy.spatial.transform import Rotation as Rscipy
from ..config.settings import Settings
from ..utils.geometry import compute_scale


class Detection:
    """Handles face landmark detection and head pose computation."""

    def __init__(self, settings: Settings):
        self.settings = settings
        self.nose_indices = settings.nose_indices

        mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = mp_face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def process_frame(self, frame: np.ndarray, w: int, h: int) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[List]]:
        """Process frame to get face landmarks and compute head pose from nose region."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(frame_rgb)

        if not results.multi_face_landmarks:
            return None, None, None, None

        face_landmarks = results.multi_face_landmarks[0].landmark

        # Compute head pose from nose landmarks
        points_3d = np.array([
            [face_landmarks[i].x * w, face_landmarks[i].y * h, face_landmarks[i].z * w]
            for i in self.nose_indices
        ])

        center = np.mean(points_3d, axis=0)

        # PCA for orientation
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

        # Stabilize with reference (using settings.r_ref_nose)
        if self.settings.r_ref_nose[0] is None:
            self.settings.r_ref_nose[0] = R_final.copy()
        else:
            R_ref = self.settings.r_ref_nose[0]
            for i in range(3):
                if np.dot(R_final[:, i], R_ref[:, i]) < 0:
                    R_final[:, i] *= -1

        # Update calibration scale if needed
        current_scale = compute_scale(points_3d)
        if self.settings.calibration_nose_scale is None:
            self.settings.calibration_nose_scale = current_scale

        return center, R_final, points_3d, face_landmarks