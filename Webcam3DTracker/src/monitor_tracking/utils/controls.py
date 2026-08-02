"""Control utilities for keyboard input, mouse movement, and file I/O in eye tracking."""

import threading
import time
import math
import keyboard
import pyautogui
from typing import Optional
from ..config.settings import Settings


def write_screen_position(x: int, y: int, screen_position_file: str = "./screen_position.txt"):
    """Write screen position to file, overwriting the same line."""
    with open(screen_position_file, 'w') as f:
        f.write(f"{x},{y}\n")


def update_orbit_from_keys(settings: Settings):
    """Update orbit camera parameters based on keyboard input."""
    yaw_step = math.radians(1.5)
    pitch_step = math.radians(1.5)
    zoom_step = 12.0

    changed = False

    if keyboard.is_pressed('j'):  # yaw left
        settings.orbit_yaw -= yaw_step
        changed = True
    if keyboard.is_pressed('l'):  # yaw right
        settings.orbit_yaw += yaw_step
        changed = True
    if keyboard.is_pressed('i'):  # pitch up
        settings.orbit_pitch += pitch_step
        changed = True
    if keyboard.is_pressed('k'):  # pitch down
        settings.orbit_pitch -= pitch_step
        changed = True

    if keyboard.is_pressed('['):  # zoom out
        settings.orbit_radius += zoom_step
        changed = True
    if keyboard.is_pressed(']'):  # zoom in
        settings.orbit_radius = max(80.0, settings.orbit_radius - zoom_step)
        changed = True

    if keyboard.is_pressed('r'):
        settings.orbit_yaw = 0.0
        settings.orbit_pitch = 0.0
        settings.orbit_radius = 600.0
        changed = True

    # Clamp
    settings.orbit_pitch = max(math.radians(-89), min(math.radians(89), settings.orbit_pitch))
    settings.orbit_radius = max(80.0, settings.orbit_radius)

    if changed:
        print(f"[Orbit Debug] yaw={math.degrees(settings.orbit_yaw):.2f}°, "
              f"pitch={math.degrees(settings.orbit_pitch):.2f}°, "
              f"radius={settings.orbit_radius:.2f}, "
              f"fov={settings.orbit_fov_deg:.1f}°")


def mouse_mover(settings: Settings, mouse_target: list, mouse_lock: threading.Lock):
    """Thread function to move mouse to target position when enabled."""
    while True:
        if settings.mouse_control_enabled:
            with mouse_lock:
                x, y = mouse_target[:2]
            pyautogui.moveTo(x, y)
        time.sleep(0.01)  # Responsiveness adjustment


def start_mouse_thread(settings: Settings, mouse_target: list, mouse_lock: threading.Lock):
    """Start the mouse movement thread."""
    thread = threading.Thread(target=mouse_mover, args=(settings, mouse_target, mouse_lock), daemon=True)
    thread.start()
    return thread