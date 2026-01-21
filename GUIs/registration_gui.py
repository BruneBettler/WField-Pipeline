from PyQt5.QtWidgets import QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QHBoxLayout, QGridLayout, QFileDialog, QDoubleSpinBox, QCheckBox, QSlider, QComboBox
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
from PyQt5.QtCore import Qt, QPoint
import numpy as np
import cv2
import os
import json


from GUIs.base_gui import BaseMaskApp
from GUIs.gui_utils import cv_to_qt



class RegistrationGUI(BaseMaskApp):
    def __init__(self, blue_image, violet_image):
        super().__init__(blue_image, violet_image)

        # Midline functionality
        self.midline_mode = True  # Start with midline mode enabled
        self.midline_x_intercept = 270.0  # Middle of width (540/2)
        self.midline_angle_degrees = 90.0  # 90 = vertical, 0 = horizontal
        self.midline_defined = False

        # Drawing functionality
        self.image_height, self.image_width = self.current_image.shape[:2]
        self.drawing_canvas = QPixmap(self.image_width, self.image_height)
        self.drawing_canvas.fill(Qt.transparent)
        # Initialize mask with 0s, drawing will set to 1 (drawn region)
        self.mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        
        # Store stroke paths for regenerating reflections when midline changes
        self.stroke_paths = []  # List of [(start_point, end_point), ...]
        
        # Drawing settings
        self.drawing = False
        self.last_point = QPoint()
        self.pen_color = QColor(255, 0, 0, 128)  # Semi-transparent red
        self.brush_size = 10

        # Brain orientation setting
        self.brain_orientation = None  # Will be set by dropdown

        # Create control panel layout
        control_layout = QGridLayout()

        # Midline controls
        self.create_midline_button = QPushButton("Reset to Default Midline")
        self.create_midline_button.clicked.connect(self.reset_midline)
        control_layout.addWidget(self.create_midline_button, 0, 0, 1, 2)

        # Slope and intercept controls
        self.x_intercept_label = QLabel("X-Intercept:")
        self.x_intercept_spinbox = QDoubleSpinBox()
        self.x_intercept_spinbox.setRange(0.0, 640.0)  # Image width range
        self.x_intercept_spinbox.setSingleStep(1.0)
        self.x_intercept_spinbox.setDecimals(1)
        self.x_intercept_spinbox.setValue(self.midline_x_intercept)
        self.x_intercept_spinbox.valueChanged.connect(self.update_midline)
        
        self.angle_label = QLabel("Angle (degrees):")
        self.angle_spinbox = QDoubleSpinBox()
        self.angle_spinbox.setRange(-90.0, 90.0)  # Reasonable range for brain midlines
        self.angle_spinbox.setSingleStep(0.5)
        self.angle_spinbox.setDecimals(1)
        self.angle_spinbox.setValue(self.midline_angle_degrees)
        self.angle_spinbox.valueChanged.connect(self.update_midline)

        control_layout.addWidget(self.x_intercept_label, 1, 0)
        control_layout.addWidget(self.x_intercept_spinbox, 1, 1)
        control_layout.addWidget(self.angle_label, 2, 0)
        control_layout.addWidget(self.angle_spinbox, 2, 1)

        self.save_midline_button = QPushButton("Save Midline")
        self.save_midline_button.clicked.connect(self.save_midline)
        control_layout.addWidget(self.save_midline_button, 3, 0, 1, 2)

        # Drawing controls
        self.pen_thickness_label = QLabel("Pen Thickness:")
        self.pen_thickness_slider = QSlider(Qt.Horizontal)
        self.pen_thickness_slider.setMinimum(1)
        self.pen_thickness_slider.setMaximum(50)
        self.pen_thickness_slider.setValue(self.brush_size)
        self.pen_thickness_slider.valueChanged.connect(self.update_brush_size)
        self.pen_thickness_slider.setFocusPolicy(Qt.NoFocus)
        
        control_layout.addWidget(self.pen_thickness_label, 4, 0)
        control_layout.addWidget(self.pen_thickness_slider, 4, 1)

        # Brain orientation dropdown
        self.orientation_label = QLabel("Brain Orientation:")
        self.orientation_dropdown = QComboBox()
        self.orientation_dropdown.addItems(["Select orientation...", "Up (nose up)", "Down (nose down)"])
        self.orientation_dropdown.currentTextChanged.connect(self.update_brain_orientation)
        
        control_layout.addWidget(self.orientation_label, 5, 0)
        control_layout.addWidget(self.orientation_dropdown, 5, 1)

        self.clear_drawing_button = QPushButton("Clear Drawing")
        self.clear_drawing_button.clicked.connect(self.clear_drawing)
        control_layout.addWidget(self.clear_drawing_button, 6, 0, 1, 2)

        self.save_mask_button = QPushButton("Save Mask")
        self.save_mask_button.clicked.connect(self.save_mask)
        control_layout.addWidget(self.save_mask_button, 7, 0, 1, 2)

        # Integrate into main layout
        main_layout = QHBoxLayout()
        image_panel = self.centralWidget().layout()
        wrapper = QWidget()
        wrapper.setLayout(image_panel)

        main_layout.addWidget(wrapper)

        control_panel = QWidget()
        control_panel.setLayout(control_layout)
        main_layout.addWidget(control_panel)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

    def reset_midline(self):
        """Reset midline to default (vertical line through center)"""
        self.midline_x_intercept = 270.0  # Middle of width (540/2)
        self.midline_angle_degrees = 90.0  # Vertical
        self.midline_defined = False
        
        # Update spinboxes
        self.x_intercept_spinbox.setValue(self.midline_x_intercept)
        self.angle_spinbox.setValue(self.midline_angle_degrees)
        
        self.update()
        print(f"Midline reset to default: X-intercept={self.midline_x_intercept:.1f}, Angle={self.midline_angle_degrees:.1f}°")

    def update_midline(self):
        """Update midline parameters from spinboxes"""
        self.midline_x_intercept = self.x_intercept_spinbox.value()
        self.midline_angle_degrees = self.angle_spinbox.value()
        
        # When midline changes, we need to regenerate the reflections
        # Store the original drawing points and regenerate the full canvas
        self.regenerate_drawing_with_new_midline()
        
        self.update()

    def regenerate_drawing_with_new_midline(self):
        """Regenerate the drawing canvas and mask with new midline reflections"""
        if not self.stroke_paths:
            return
            
        # Clear current drawing
        self.drawing_canvas.fill(Qt.transparent)
        self.mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        
        # Redraw all strokes with new reflections
        painter = QPainter(self.drawing_canvas)
        pen = QPen(self.pen_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
        painter.setPen(pen)
        
        for start_point, end_point in self.stroke_paths:
            # Draw original stroke
            painter.drawLine(start_point, end_point)
            cv2.line(self.mask, (start_point.x(), start_point.y()), 
                    (end_point.x(), end_point.y()), 1, self.brush_size)
            
            # Draw reflected stroke if reflection is valid
            reflected_start = self.reflect_point_across_midline(start_point)
            reflected_end = self.reflect_point_across_midline(end_point)
            if reflected_start is not None and reflected_end is not None:
                painter.drawLine(reflected_start, reflected_end)
                cv2.line(self.mask, (reflected_start.x(), reflected_start.y()), 
                        (reflected_end.x(), reflected_end.y()), 1, self.brush_size)
        
        painter.end()

    def create_hemisphere_masks(self):
        """Create separate masks for left and right hemispheres based on midline"""
        left_hemisphere_mask = np.zeros_like(self.mask)
        right_hemisphere_mask = np.zeros_like(self.mask)
        
        height, width = self.mask.shape
        
        # Handle vertical line case (90 degrees)
        if abs(self.midline_angle_degrees - 90.0) < 0.1:
            x_line = int(self.midline_x_intercept)
            # Left hemisphere (left side of vertical line)
            left_hemisphere_mask[:, :x_line] = self.mask[:, :x_line]
            # Right hemisphere (right side of vertical line)
            right_hemisphere_mask[:, x_line:] = self.mask[:, x_line:]
            return left_hemisphere_mask, right_hemisphere_mask
        
        # Get line parameters for non-vertical lines
        _, _, slope, intercept = self.get_line_endpoints(width, height)
        
        # Check which side of the line the left edge of the image is on
        left_edge_y = height / 2
        line_y_at_left = slope * 0 + intercept
        left_edge_is_above_line = left_edge_y < line_y_at_left
        
        # For each pixel, determine which hemisphere it belongs to
        for x in range(width):
            y_line = slope * x + intercept
            
            for y in range(height):
                pixel_is_above_line = y < y_line
                
                # Assign to hemisphere based on consistent logic
                if left_edge_is_above_line:
                    # Left edge is above line, so above = left hemisphere
                    if pixel_is_above_line:
                        left_hemisphere_mask[y, x] = self.mask[y, x]
                    else:
                        right_hemisphere_mask[y, x] = self.mask[y, x]
                else:
                    # Left edge is below line, so above = right hemisphere
                    if pixel_is_above_line:
                        right_hemisphere_mask[y, x] = self.mask[y, x]
                    else:
                        left_hemisphere_mask[y, x] = self.mask[y, x]
        
        return left_hemisphere_mask, right_hemisphere_mask

    def update_brush_size(self):
        """Update brush size from slider"""
        self.brush_size = self.pen_thickness_slider.value()

    def update_brain_orientation(self, orientation_text):
        """Update brain orientation from dropdown"""
        if orientation_text == "Up (nose up)":
            self.brain_orientation = "up"
        elif orientation_text == "Down (nose down)":
            self.brain_orientation = "down"
        else:
            self.brain_orientation = None

    def clear_drawing(self):
        """Clear the drawing canvas and reset mask"""
        self.drawing_canvas.fill(Qt.transparent)
        # Reset mask to all 0s
        self.mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
        # Clear stored stroke paths
        self.stroke_paths = []
        self.update()

    def reflect_point_across_midline(self, point):
        """Reflect a point across the current midline"""
        x, y = point.x(), point.y()
        
        # Handle vertical line case (90 degrees)
        if abs(self.midline_angle_degrees - 90.0) < 0.1:
            # Simple reflection across vertical line
            reflected_x = 2 * self.midline_x_intercept - x
            reflected_y = y
        else:
            # Get line parameters
            _, _, slope, intercept = self.get_line_endpoints(self.image_width, self.image_height)
            
            # For a line ax + by + c = 0, reflection formula is:
            # x' = x - 2a(ax + by + c)/(a² + b²)
            # y' = y - 2b(ax + by + c)/(a² + b²)
            
            # Convert y = mx + c to ax + by + c = 0 form
            # y = mx + c => mx - y + c = 0
            # So a = slope, b = -1, c = intercept
            a = slope
            b = -1
            c = intercept
            
            # Calculate reflection
            denominator = a*a + b*b
            numerator = a*x + b*y + c
            
            reflected_x = x - 2*a*numerator/denominator
            reflected_y = y - 2*b*numerator/denominator
        
        # Check if reflected point is within image bounds
        if 0 <= reflected_x < self.image_width and 0 <= reflected_y < self.image_height:
            return QPoint(int(reflected_x), int(reflected_y))
        else:
            return None

    def save_mask(self):
        """Save the current drawing as a binary mask and separate hemisphere masks"""
        # Check if there's any drawing (mask values that are 1, since we draw with 1s now)
        if np.all(self.mask == 0):
            print("No drawing to save. Please draw something first.")
            return

        # Check if brain orientation is selected
        if self.brain_orientation is None:
            print("Please select brain orientation (Up or Down) before saving the mask.")
            return

        save_path, _ = QFileDialog.getSaveFileName(self, "Save Mask", "", "NumPy Files (*.npy);;PNG Files (*.png)")
        if save_path:
            # Create hemisphere masks
            left_mask, right_mask = self.create_hemisphere_masks()
            
            # Determine naming based on brain orientation
            if self.brain_orientation == "down":
                # When nose is down, left side of image = right hemisphere, right side of image = left hemisphere
                anatomical_left_mask = right_mask  # Right side of image is anatomical left
                anatomical_right_mask = left_mask   # Left side of image is anatomical right
                left_label = "left_hemisphere"
                right_label = "right_hemisphere"
            else:  # orientation == "up"
                # When nose is up, left side of image = left hemisphere, right side of image = right hemisphere
                anatomical_left_mask = left_mask   # Left side of image is anatomical left
                anatomical_right_mask = right_mask  # Right side of image is anatomical right
                left_label = "left_hemisphere"
                right_label = "right_hemisphere"
            
            # Remove extension for base filename
            base_path = save_path.rsplit('.', 1)[0] if '.' in save_path else save_path
            extension = save_path.rsplit('.', 1)[1] if '.' in save_path else 'npy'
            
            if extension == 'npy':
                # Save full mask
                np.save(f"{base_path}_full_mask.npy", self.mask)
                print(f"Full mask saved as NumPy array to {base_path}_full_mask.npy")
                
                # Save hemisphere masks with anatomically correct naming
                np.save(f"{base_path}_{left_label}.npy", anatomical_left_mask)
                np.save(f"{base_path}_{right_label}.npy", anatomical_right_mask)
                print(f"Left hemisphere mask saved to {base_path}_{left_label}.npy")
                print(f"Right hemisphere mask saved to {base_path}_{right_label}.npy")
                print(f"Brain orientation: {self.brain_orientation} (nose pointing {self.brain_orientation})")
                print(f"Mask format: 1 = drawn region, 0 = background")
                
            elif extension == 'png':
                # Convert to binary format for PNG (255 for drawn, 0 for background)
                binary_mask = self.mask.astype(np.uint8) * 255
                binary_left = anatomical_left_mask.astype(np.uint8) * 255
                binary_right = anatomical_right_mask.astype(np.uint8) * 255
                
                cv2.imwrite(f"{base_path}_full_mask.png", binary_mask)
                cv2.imwrite(f"{base_path}_{left_label}.png", binary_left)
                cv2.imwrite(f"{base_path}_{right_label}.png", binary_right)
                print(f"Full mask saved as binary PNG to {base_path}_full_mask.png")
                print(f"Left hemisphere mask saved to {base_path}_{left_label}.png")
                print(f"Right hemisphere mask saved to {base_path}_{right_label}.png")
                print(f"Brain orientation: {self.brain_orientation} (nose pointing {self.brain_orientation})")
                print(f"PNG format: 255 = drawn region, 0 = background")
            else:
                # Default to .npy if no extension specified
                np.save(f"{base_path}_full_mask.npy", self.mask)
                np.save(f"{base_path}_{left_label}.npy", anatomical_left_mask)
                np.save(f"{base_path}_{right_label}.npy", anatomical_right_mask)
                print(f"Full mask saved as NumPy array to {base_path}_full_mask.npy")
                print(f"Left hemisphere mask saved to {base_path}_{left_label}.npy")
                print(f"Right hemisphere mask saved to {base_path}_{right_label}.npy")
                print(f"Brain orientation: {self.brain_orientation} (nose pointing {self.brain_orientation})")
                print(f"Mask format: 1 = drawn region, 0 = background")

    def mousePressEvent(self, event):
        """Start drawing when left mouse button is pressed"""
        if event.button() == Qt.LeftButton:
            # Convert global position to label-relative position
            pos = event.pos() - self.label.pos()
            if 0 <= pos.x() < self.image_width and 0 <= pos.y() < self.image_height:
                self.drawing = True
                self.last_point = pos

    def mouseMoveEvent(self, event):
        """Draw when mouse is moved while drawing"""
        if self.drawing:
            # Convert global position to label-relative position
            pos = event.pos() - self.label.pos()
            if 0 <= pos.x() < self.image_width and 0 <= pos.y() < self.image_height:
                # Store the stroke path for later regeneration
                self.stroke_paths.append((QPoint(self.last_point), QPoint(pos)))
                
                # Draw on the canvas
                painter = QPainter(self.drawing_canvas)
                pen = QPen(self.pen_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
                painter.setPen(pen)
                painter.drawLine(self.last_point, pos)
                
                # Draw reflected line if reflection point is valid
                reflected_last = self.reflect_point_across_midline(self.last_point)
                reflected_pos = self.reflect_point_across_midline(pos)
                if reflected_last is not None and reflected_pos is not None:
                    painter.drawLine(reflected_last, reflected_pos)
                
                painter.end()

                # Update mask array (drawn pixels = 1, representing drawn region)
                cv2.line(self.mask, (self.last_point.x(), self.last_point.y()), 
                        (pos.x(), pos.y()), 1, self.brush_size)
                
                # Update reflected mask if reflection is valid
                if reflected_last is not None and reflected_pos is not None:
                    cv2.line(self.mask, (reflected_last.x(), reflected_last.y()), 
                            (reflected_pos.x(), reflected_pos.y()), 1, self.brush_size)
                
                self.last_point = pos
                self.update()

    def mouseReleaseEvent(self, event):
        """Stop drawing when left mouse button is released"""
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def save_midline(self):
        """Save the current midline definition"""
        self.midline_defined = True
        save_path, _ = QFileDialog.getSaveFileName(self, "Save Midline", "", "Json Files (*.json)")
        if save_path:
            if not save_path.endswith(".json"):
                save_path += ".json"
            
            # Create midline data
            midline_data = {
                'midline_equation': {
                    'x_intercept': float(self.midline_x_intercept),
                    'angle_degrees': float(self.midline_angle_degrees)
                }
            }
            
            # Save to file
            with open(save_path, "w") as f:
                json.dump(midline_data, f, indent=4, separators=(", ", ": "))

            print(f"Midline saved to {save_path}")
            print(f"Midline equation: X-intercept={self.midline_x_intercept:.1f}, Angle={self.midline_angle_degrees:.1f}°")
            self.close()

    def get_line_endpoints(self, width, height):
        """Calculate line endpoints for drawing based on x-intercept and angle"""
        # Convert angle to radians
        angle_rad = np.radians(self.midline_angle_degrees)
        
        # Handle vertical line case (90 degrees)
        if abs(self.midline_angle_degrees - 90.0) < 0.1:
            # Vertical line - just draw from top to bottom at x_intercept
            x = self.midline_x_intercept
            return (x, 0), (x, height), float('inf'), 0
        
        # Calculate slope from angle
        if abs(np.cos(angle_rad)) > 1e-6:
            slope = np.tan(angle_rad)
        else:
            # Nearly vertical line
            slope = 1e6 if self.midline_angle_degrees > 0 else -1e6
        
        # Line passes through (x_intercept, height/2)
        reference_y = height / 2
        
        # Line equation: y - reference_y = slope * (x - x_intercept)
        # Rearranged: y = slope * x + (reference_y - slope * x_intercept)
        intercept = reference_y - slope * self.midline_x_intercept
        
        # Calculate endpoints at image boundaries
        endpoints = []
        
        # Check intersection with left edge (x = 0)
        y_at_x0 = intercept
        if 0 <= y_at_x0 <= height:
            endpoints.append((0, y_at_x0))
        
        # Check intersection with right edge (x = width)
        y_at_width = slope * width + intercept
        if 0 <= y_at_width <= height:
            endpoints.append((width, y_at_width))
        
        # Check intersection with top edge (y = 0)
        if abs(slope) > 1e-6:
            x_at_y0 = -intercept / slope
            if 0 <= x_at_y0 <= width:
                endpoints.append((x_at_y0, 0))
        
        # Check intersection with bottom edge (y = height)
        if abs(slope) > 1e-6:
            x_at_height = (height - intercept) / slope
            if 0 <= x_at_height <= width:
                endpoints.append((x_at_height, height))
        
        # Return the first two valid endpoints
        if len(endpoints) >= 2:
            return endpoints[0], endpoints[1], slope, intercept
        else:
            # Fallback for edge cases
            return (0, reference_y), (width, reference_y), 0, reference_y

    def paintEvent(self, event):
        pixmap = QPixmap.fromImage(cv_to_qt(self.current_image))
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.Antialiasing)

        # Draw hemisphere shading and midline
        self.draw_hemisphere_shading(painter, pixmap.width(), pixmap.height())
        self.draw_midline(painter, pixmap.width(), pixmap.height())
        
        # Draw the user's drawing on top
        painter.drawPixmap(0, 0, self.drawing_canvas)

        painter.end()
        self.label.setPixmap(pixmap)

    def draw_hemisphere_shading(self, painter, width, height):
        """Draw green and blue shading for left and right hemispheres"""
        # Create semi-transparent brushes
        left_brush = QBrush(QColor(0, 255, 0, 25))  # Green with alpha (left hemisphere)
        right_brush = QBrush(QColor(0, 0, 255, 25))  # Blue with alpha (right hemisphere)
        
        # Handle vertical line case (90 degrees) - simple left/right split
        if abs(self.midline_angle_degrees - 90.0) < 0.1:
            x_line = int(self.midline_x_intercept)
            # Left hemisphere (left side of vertical line) - green
            painter.fillRect(0, 0, x_line, height, left_brush)
            # Right hemisphere (right side of vertical line) - blue
            painter.fillRect(x_line, 0, width - x_line, height, right_brush)
            return
        
        # Get line parameters
        _, _, slope, intercept = self.get_line_endpoints(width, height)
        
        # Determine which side of the midline corresponds to left/right hemispheres
        # We'll use the midline's orientation to maintain consistent hemisphere assignment
        
        # Check which side of the line the left edge of the image is on
        # Point on left edge at middle height
        left_edge_y = height / 2
        line_y_at_left = slope * 0 + intercept
        left_edge_is_above_line = left_edge_y < line_y_at_left
        
        # For non-vertical lines, determine which side of the midline each pixel is on
        for x in range(width):
            y_line = slope * x + intercept
            
            # For pixels above the line
            if y_line > 0:
                y_end = int(min(y_line, height))
                # Assign color based on consistent hemisphere logic
                if left_edge_is_above_line:
                    # Left edge is above line, so above = left hemisphere
                    painter.fillRect(x, 0, 1, y_end, left_brush)
                else:
                    # Left edge is below line, so above = right hemisphere
                    painter.fillRect(x, 0, 1, y_end, right_brush)
            
            # For pixels below the line
            if y_line < height:
                y_start = int(max(y_line, 0))
                # Assign color based on consistent hemisphere logic
                if left_edge_is_above_line:
                    # Left edge is above line, so below = right hemisphere
                    painter.fillRect(x, y_start, 1, height - y_start, right_brush)
                else:
                    # Left edge is below line, so below = left hemisphere
                    painter.fillRect(x, y_start, 1, height - y_start, left_brush)

    def draw_midline(self, painter, width, height):
        """Draw the midline"""
        pen = QPen(QColor(255, 255, 255), 1)  # White line
        painter.setPen(pen)
        
        # Get line endpoints
        (x1, y1), (x2, y2), _, _ = self.get_line_endpoints(width, height)
        
        painter.drawLine(int(x1), int(y1), int(x2), int(y2))

if __name__ == "__main__":
    from PyQt5.QtWidgets import QApplication
    import sys
    import h5py
    #sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

    from ..pipeline_utils import get_exp_file_path, get_hdf5, normalize_arr

    app = QApplication(sys.argv)

    blue_frame = None
    violet_fram = None

    EXP_GROUP = 'WT'
    ANIMAL_NUM = '7202'

    # no need to edit further down 
    EXPERIMENT_PATH = rf"D:\wfield\NicoleData\{EXP_GROUP}\{ANIMAL_NUM}\LED_530_R_F0.5_ND0_FW1"
    FRAMETIME_MAT_PATH = get_exp_file_path(EXPERIMENT_PATH, 'T', dig=True)
    HDF5_EXPERIMENT_PATH = get_hdf5(EXPERIMENT_PATH[:-23], verbose=True) 
    HDF5_DATASET_PATH =  f"{EXP_GROUP}_{ANIMAL_NUM}_LED_530_R_F0.5_ND0_FW1/motion_corrected" 
    
    with h5py.File(HDF5_EXPERIMENT_PATH, 'r') as f:
        blue_img = (normalize_arr(f[HDF5_DATASET_PATH][0,0,...]) * 255).astype('uint8')
        violet_img = (normalize_arr(f[HDF5_DATASET_PATH][0,1,...]) * 255).astype('uint8')

    window = RegistrationGUI(blue_img, violet_img)
    window.show()
    sys.exit(app.exec_())
