"""Configuration and initial state for the monitor tracking system."""

import pyautogui
from dataclasses import dataclass, field
from typing import List, Optional, Deque
from collections import deque
import numpy as np

@dataclass
class Settings:
    """Central configuration class for eye tracking parameters and initial state."""
    
    # Screen and mouse setup
    monitor_width: int = field(init=False)
    monitor_height: int = field(init=False)
    center_x: int = field(init=False)
    center_y: int = field(init=False)
    mouse_control_enabled: bool = False
    filter_length: int = 10
    gaze_length: int = 350
    
    # Orbit camera state for debug view
    orbit_yaw: float = -151.0
    orbit_pitch: float = 0.0
    orbit_radius: float = 1500.0
    orbit_fov_deg: float = 50.0
    debug_world_frozen: bool = False
    orbit_pivot_frozen: Optional[np.ndarray] = None
    
    # 3D monitor plane state
    monitor_corners: Optional[List[np.ndarray]] = None
    monitor_center_w: Optional[np.ndarray] = None
    monitor_normal_w: Optional[np.ndarray] = None
    units_per_cm: Optional[float] = None
    
    # Calibration
    calib_step: int = 0
    calibration_offset_yaw: float = 0.0
    calibration_offset_pitch: float = 0.0
    
    # Buffers and references
    combined_gaze_directions: Deque[np.ndarray] = field(default_factory=lambda: deque(maxlen=10))
    r_ref_nose: List[Optional[np.ndarray]] = field(default_factory=lambda: [None])
    r_ref_forehead: List[Optional[np.ndarray]] = field(default_factory=lambda: [None])
    calibration_nose_scale: Optional[float] = None
    
    # Gaze markers and file
    gaze_markers: List[tuple] = field(default_factory=list)
    screen_position_file: str = "./screen_position.txt"
    
    # Nose landmark indices
    nose_indices: List[int] = field(default_factory=lambda: [
        4, 45, 275, 220, 440, 1, 5, 51, 281, 44, 274, 241, 
        461, 125, 354, 218, 438, 195, 167, 393, 165, 391,
        3, 248
    ])
    
    # Eye tracking
    base_radius: int = 20
    
    def __post_init__(self):
        """Initialize computed screen dimensions."""
        w, h = pyautogui.size()
        self.monitor_width = w
        self.monitor_height = h
        self.center_x = w // 2
        self.center_y = h // 2