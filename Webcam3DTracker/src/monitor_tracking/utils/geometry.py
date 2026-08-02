"""Geometric utility functions for rotations, normalization, and scaling in 3D eye tracking."""

import math
import numpy as np
from typing import Union


def rot_x(a: float) -> np.ndarray:
    """Rotation matrix around X-axis by angle a (radians)."""
    ca, sa = math.cos(a), math.sin(a)
    return np.array([
        [1, 0, 0],
        [0, ca, -sa],
        [0, sa, ca]
    ], dtype=float)


def rot_y(a: float) -> np.ndarray:
    """Rotation matrix around Y-axis by angle a (radians)."""
    ca, sa = math.cos(a), math.sin(a)
    return np.array([
        [ca, 0, sa],
        [0, 1, 0],
        [-sa, 0, ca]
    ], dtype=float)


def normalize(v: Union[np.ndarray, list]) -> np.ndarray:
    """Normalize a vector to unit length."""
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v


def focal_px(width: float, fov_deg: float) -> float:
    """Compute horizontal pinhole focal length in pixels."""
    return 0.5 * width / math.tan(math.radians(fov_deg) * 0.5)


def compute_scale(points_3d: np.ndarray) -> float:
    """Compute average pairwise distance for scale estimation."""
    n = len(points_3d)
    total = 0.0
    count = 0
    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(points_3d[i] - points_3d[j])
            total += dist
            count += 1
    return total / count if count > 0 else 1.0