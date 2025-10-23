"""
HeadTracker Test Script

Use this file as a refernce for different ways to use the HeadTracker class
for head tracking and mouse control functionality.

Author: findpiyush
Based on original work by JEOresearch
"""

from head_tracker import HeadTracker
import cv2
import time


print("Controls:")
print("  'c' - Calibrate (look at screen center first)")
print("  't' - Toggle mouse control")
print("  'q' - Quit test")
print()

# Create and start tracker
tracker = HeadTracker(camera_index=1)
tracker.start()


while True:
    # Process frame
    img, landmarks_frame, main_frame, success = tracker.process_frame()
    view = landmarks_frame
    
    # Get current mouse position
    x, y = tracker.get_current_position()
    
    # Add position display
    cv2.putText(view, f"Position: {x},{y}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.putText(view, f"Mouse Control: {'ON' if tracker.mouse_control_enabled else 'OFF'}", 
                (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display frames
    cv2.imshow("window", view)
    
    # Handle controls
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('c'):
        calibration_offset_yaw, calibration_offset_pitch = tracker.calibrate()
        print(f"[Calibrated] Yaw: {calibration_offset_yaw}, Pitch: {calibration_offset_pitch}")
    elif key == ord('t'):
        flag = tracker.toggle_mouse_control()
        print(f"[Mouse Control] {'Enabled' if flag else 'Disabled'}")
        time.sleep(0.3)  # debounce            

tracker.stop()
cv2.destroyAllWindows()

