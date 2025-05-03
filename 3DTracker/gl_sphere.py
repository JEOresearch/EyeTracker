# gl_sphere.py

import sys
import numpy as np
from PyQt5.QtWidgets import QApplication, QMainWindow, QOpenGLWidget
from PyQt5.QtCore import Qt, QTimer
from OpenGL.GL import *
from OpenGL.GLU import *

# Singleton references
app = None
window = None
sphere_widget = None
CV_pupil_x = 0
CV_pupil_y = 0

class SphereWidget(QOpenGLWidget):
    def __init__(self):
        super().__init__()
        self.setFocusPolicy(Qt.StrongFocus)
        self.sphere_rot_x = 0
        self.sphere_rot_y = 0
        self.last_x, self.last_y = 0, 0
        self.rot_x, self.rot_y = 0, 0
        self.sphere_vertices, self.sphere_indices = self.generate_wireframe_sphere(30, 30)
        self.circle_vertices = self.generate_circle_on_sphere(r_sphere=1.0, r_circle=0.2, num_segments=100)
        self.camera_position = np.array([0.0, 0.0, -3.0])
        self.ray_origin = None
        self.ray_direction = None

        # New: Sphere center in screen coordinates (defaults to screen center)
        self.sphere_center_x = 320
        self.sphere_center_y = 240

    def draw_2d_circle(self, x, y, radius=10, segments=32):
        w = self.width()
        h = self.height()

        glMatrixMode(GL_PROJECTION)
        glPushMatrix()
        glLoadIdentity()
        glOrtho(0, w, 0, h, -1, 1)

        glMatrixMode(GL_MODELVIEW)
        glPushMatrix()
        glLoadIdentity()

        # Flip y because OpenGL's 0,0 is bottom-left, but PyQt uses top-left
        y_flipped = h - y

        glColor3f(1.0, 1.0, 0.0)  # Yellow circle
        glLineWidth(2.0)
        glBegin(GL_LINE_LOOP)
        for i in range(segments):
            angle = 2 * np.pi * i / segments
            cx = x + np.cos(angle) * radius
            cy = y_flipped + np.sin(angle) * radius
            glVertex2f(cx, cy)
        glEnd()

        glPopMatrix()
        glMatrixMode(GL_PROJECTION)
        glPopMatrix()
        glMatrixMode(GL_MODELVIEW)

    #create outer sphere
    def generate_wireframe_sphere(self, lat_div, lon_div):
        vertices = []
        indices = []
        for i in range(lat_div + 1):
            lat = np.pi * (-0.5 + float(i) / lat_div)
            z = np.sin(lat)
            zr = np.cos(lat)
            for j in range(lon_div + 1):
                lon = 2 * np.pi * float(j) / lon_div
                x = np.cos(lon) * zr
                y = np.sin(lon) * zr
                vertices.append((x, y, z))
        for i in range(lat_div):
            for j in range(lon_div):
                p1 = i * (lon_div + 1) + j
                p2 = p1 + lon_div + 1
                indices.append((p1, p2))
                indices.append((p1, p1 + 1))
        return np.array(vertices, dtype=np.float32), np.array(indices, dtype=np.int32)

    #create bisecting circle
    def generate_circle_on_sphere(self, r_sphere=1.0, r_circle=0.8, num_segments=100):
        circle_vertices = []

        # Correct plane_z using Pythagoras so circle is tangent to the sphere
        plane_z = np.sqrt(r_sphere**2 - r_circle**2)

        for i in range(num_segments):
            angle = 2.0 * np.pi * i / num_segments
            x = np.cos(angle) * r_circle
            y = np.sin(angle) * r_circle
            z = plane_z  # z is constant
            circle_vertices.append((x, y, z))

        return np.array(circle_vertices, dtype=np.float32)

    def initializeGL(self):
        glEnable(GL_DEPTH_TEST)
        glPolygonMode(GL_FRONT_AND_BACK, GL_LINE)

    def resizeGL(self, w, h):
        glViewport(0, 0, w, h)
        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(45, w / max(1, h), 0.1, 100)
        glMatrixMode(GL_MODELVIEW)

    def paintGL(self):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glLoadIdentity()

        glTranslatef(0.0, 0.0, -3)  # Move back so we can see it

        # Translate sphere to desired center position
        viewport_width = self.width()
        viewport_height = self.height()

        # Map center_x, center_y from screen to GL coordinates
        gl_x = (self.sphere_center_x / viewport_width) * 2.0 - 1.0
        gl_y = 1.0 - (self.sphere_center_y / viewport_height) * 2.0

        # Scale appropriately to match world space
        glTranslatef(gl_x * 1.5, gl_y * 1.5, 0.0)
        

        # Draw ray (NOT rotated by sphere rotation)
        if self.ray_origin is not None and self.ray_direction is not None:
            glBegin(GL_LINES)
            glColor3f(1.0, 1.0, 1.0)  
            glVertex3fv(self.ray_origin)
            glColor3f(0.0, 0.0, 0.0)  
            glVertex3fv(self.ray_origin - self.ray_direction * 5.5)
            glEnd()


        plane_z = 1/1.05
        inner_radius = abs(plane_z)

        # Draw ray-to-inner-sphere intersection marker
        if self.ray_origin is not None and self.ray_direction is not None:
            inner_radius = abs(plane_z)
            origin = self.ray_origin
            direction = -self.ray_direction

            # Ray-sphere intersection (centered at origin)
            # Solve: ||origin + t*direction||^2 = inner_radius^2
            a = np.dot(direction, direction)
            b = 2 * np.dot(origin, direction)
            c = np.dot(origin, origin) - inner_radius**2

            discriminant = b**2 - 4*a*c
            
            if discriminant >= 0:
                sqrt_disc = np.sqrt(discriminant)
                t1 = (-b - sqrt_disc) / (2 * a)
                t2 = (-b + sqrt_disc) / (2 * a)

                # Choose the smallest *positive* t
                t = None
                if t1 > 0 and t2 > 0:
                    t = min(t1, t2)
                elif t1 > 0:
                    t = t1
                elif t2 > 0:
                    t = t2
                print(t)
                if t is not None:
                    intersection = origin + t * direction
                    print(discriminant)
                    # Draw small red sphere at intersection
                    glPushMatrix()
                    glColor3f(1.0, 1.0, 1.0)  # white marker
                    glTranslatef(intersection[0], intersection[1], intersection[2])
                    quad = gluNewQuadric()
                    gluSphere(quad, 0.02, 10, 10)
                    glPopMatrix()

        glPushMatrix()
        #rotate sphere only
        glRotatef(self.sphere_rot_x, 1, 0, 0)
        glRotatef(self.sphere_rot_y, 0, 1, 0)

        glColor3f(0.3, 0.3, 0.8)
        glLineWidth(1.5)
        glBegin(GL_LINES)
        for i1, i2 in self.sphere_indices:
            glVertex3fv(self.sphere_vertices[i1])
            glVertex3fv(self.sphere_vertices[i2])
        glEnd()

        glColor3f(1.0, 0.0, 0.0)
        glLineWidth(2.0)
        glBegin(GL_LINES)
        glVertex3f(0.0, 0.0, 0.0)
        glVertex3f(0.0, 0.0, 1.2)
        glEnd()

        #Render an internal sphere for testing
        #glColor3f(0.3, 0.3, 0.3)  # Magenta for visibility
        #glLineWidth(1.0)
        #quadric = gluNewQuadric()
        #gluSphere(quadric, inner_radius, 20, 20)

        glColor3f(0.0, 1.0, 0.0)
        glLineWidth(3.0)
        glBegin(GL_LINE_LOOP)
        for vertex in self.circle_vertices:
            glVertex3fv(vertex)
        glEnd()

        self.draw_2d_circle(CV_pupil_x, CV_pupil_y)

        glPopMatrix()

def start_gl_window():
    global app, window, sphere_widget
    app = QApplication(sys.argv)
    window = QMainWindow()
    sphere_widget = SphereWidget()
    window.setCentralWidget(sphere_widget)
    window.setWindowTitle("Wireframe Sphere Eye Tracker Display")
    window.resize(640, 480)
    window.show()

    timer = QTimer()
    timer.timeout.connect(lambda: None)
    timer.start(16)  # ~60 FPS event loop tick

    return app

def update_sphere_rotation(x, y, center_x, center_y, screen_width=640, screen_height=480):
    """Call this from the eye tracker with the latest pupil center (x,y) and the center of the sphere (center_x, center_y)."""
    if sphere_widget is None:
        return

    # Save pupil center globally for overlay rendering
    global CV_pupil_x
    CV_pupil_x = x
    global CV_pupil_y
    CV_pupil_y = y

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

    inner_radius = 1.0 / 1.05

    # Compute sphere center offset in world space (same as in render)
    sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
    sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
    sphere_center = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

    # Ray origin and direction are defined in world space
    origin = ray_origin
    direction = -ray_direction  # Make sure ray points forward

    # Compute intersection with sphere centered at sphere_center
    L = origin - sphere_center

    a = np.dot(direction, direction)
    b = 2 * np.dot(direction, L)
    c = np.dot(L, L) - inner_radius**2

    discriminant = b**2 - 4 * a * c
    if discriminant < 0:
        return  # No intersection

    sqrt_disc = np.sqrt(discriminant)
    t1 = (-b - sqrt_disc) / (2 * a)
    t2 = (-b + sqrt_disc) / (2 * a)

    # Choose the nearest positive intersection point
    t = None
    if t1 > 0 and t2 > 0:
        t = min(t1, t2)
    elif t1 > 0:
        t = t1
    elif t2 > 0:
        t = t2
    if t is None:
        return

    # Final world-space intersection point
    intersection_point = origin + t * direction

    # Convert screen center of sphere to world-space offset
    sphere_offset_x = (center_x / screen_width) * 2.0 - 1.0
    sphere_offset_y = 1.0 - (center_y / screen_height) * 2.0
    sphere_center_offset = np.array([sphere_offset_x * 1.5, sphere_offset_y * 1.5, 0.0])

    # Shift intersection point into the local coordinate space of the sphere
    intersection_local = intersection_point - sphere_center_offset

    # Normalize to get direction from sphere center to intersection
    target_direction = intersection_local / np.linalg.norm(intersection_local)

    # Define local +Z axis of the unrotated sphere (toward green ring)
    circle_local_center = np.array([0.0, 0.0, inner_radius])
    circle_local_center /= np.linalg.norm(circle_local_center)

    # Compute axis and angle to rotate circle_local_center to target_direction
    rotation_axis = np.cross(circle_local_center, target_direction)
    rotation_axis_norm = np.linalg.norm(rotation_axis)
    if rotation_axis_norm < 1e-6:
        return

    rotation_axis /= rotation_axis_norm
    dot = np.dot(circle_local_center, target_direction)
    dot = np.clip(dot, -1.0, 1.0)
    angle_rad = np.arccos(dot)

    # Build rotation matrix from axis-angle
    c = np.cos(angle_rad)
    s = np.sin(angle_rad)
    t_ = 1 - c
    x_, y_, z_ = rotation_axis

    rotation_matrix = np.array([
        [t_*x_*x_ + c, t_*x_*y_ - s*z_, t_*x_*z_ + s*y_],
        [t_*x_*y_ + s*z_, t_*y_*y_ + c, t_*y_*z_ - s*x_],
        [t_*x_*z_ - s*y_, t_*y_*z_ + s*x_, t_*z_*z_ + c]
    ])

    # Extract Euler angles from rotation matrix
    sy = np.sqrt(rotation_matrix[0, 0] ** 2 + rotation_matrix[1, 0] ** 2)
    if sy < 1e-6:
        x_rot = np.arctan2(-rotation_matrix[1, 2], rotation_matrix[1, 1])
        y_rot = np.arctan2(-rotation_matrix[2, 0], sy)
    else:
        x_rot = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
        y_rot = np.arctan2(-rotation_matrix[2, 0], sy)


    # Apply computed rotation and center to sphere rendering
    sphere_widget.sphere_center_x = center_x
    sphere_widget.sphere_center_y = center_y
    sphere_widget.sphere_rot_x = np.degrees(x_rot)
    sphere_widget.sphere_rot_y = np.degrees(y_rot)

    # === Compute rotated gaze vector (from sphere center toward rotated green circle) ===

    # Local direction vector of green circle before rotation (points along +Z)
    gaze_local = np.array([0.0, 0.0, inner_radius])

    # Apply the rotation matrix to get the world-space gaze vector
    gaze_rotated = rotation_matrix @ gaze_local

    # Normalize the result to get a direction vector
    gaze_rotated /= np.linalg.norm(gaze_rotated)

    # Redraw the OpenGL scene
    sphere_widget.update()

    # Flush OpenGL pipeline and read back pixels
    glFinish()
    w = sphere_widget.width()
    h = sphere_widget.height()
    glReadBuffer(GL_FRONT)
    pixels = glReadPixels(0, 0, w, h, GL_RGB, GL_UNSIGNED_BYTE)
    image = np.frombuffer(pixels, dtype=np.uint8).reshape((h, w, 3))
    image = np.flipud(image)

    return image

