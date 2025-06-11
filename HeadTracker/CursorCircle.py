import sys
import cv2
import numpy as np
import pyautogui
from PyQt5 import QtWidgets, QtGui, QtCore

class CursorOverlay(QtWidgets.QWidget):
    def __init__(self, radius=30):
        super().__init__()
        self.radius = radius
        self.diameter = 2 * self.radius + 4
        self.setWindowFlags(
            QtCore.Qt.FramelessWindowHint |
            QtCore.Qt.WindowStaysOnTopHint |
            QtCore.Qt.Tool |
            QtCore.Qt.X11BypassWindowManagerHint  # no taskbar on Linux
        )
        self.setAttribute(QtCore.Qt.WA_TranslucentBackground)
        self.setAttribute(QtCore.Qt.WA_NoSystemBackground)
        self.setFixedSize(self.diameter, self.diameter)

        self.label = QtWidgets.QLabel(self)
        self.label.setGeometry(0, 0, self.diameter, self.diameter)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_position)
        self.timer.start(10)

    def update_position(self):
        x, y = pyautogui.position()
        self.move(x - self.radius, y - self.radius)
        self.draw_circle()

    def draw_circle(self):
        # Create transparent OpenCV image
        img = np.zeros((self.diameter, self.diameter, 4), dtype=np.uint8)

        # Draw light green ring
        cv2.circle(img, (self.radius + 2, self.radius + 2), self.radius -5, (0, 255, 0, 255), 10)

        # Convert to Qt image
        qimg = QtGui.QImage(img.data, self.diameter, self.diameter, QtGui.QImage.Format_RGBA8888)
        pixmap = QtGui.QPixmap.fromImage(qimg)
        self.label.setPixmap(pixmap)

app = QtWidgets.QApplication(sys.argv)
overlay = CursorOverlay(radius=80)
overlay.show()
sys.exit(app.exec_())
