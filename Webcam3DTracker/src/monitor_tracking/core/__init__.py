"""Core logic for detection, gaze tracking, and monitor modeling in eye tracking."""

from .detection import Detection
from .gaze import GazeTracker
from .monitor import MonitorPlane

__all__ = ["Detection", "GazeTracker", "MonitorPlane"]