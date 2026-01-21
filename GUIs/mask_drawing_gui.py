# Author: Brune Bettler
# Last Modified: 2025-03-17

import sys
import numpy as np
import os
import cv2
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QVBoxLayout, QWidget, QSlider
from PyQt5.QtGui import QPixmap, QPainter, QPen, QColor, QImage
from PyQt5.QtCore import Qt, QPoint
from gui_utils import cv_to_qt
from base_gui import BaseMaskApp


class MaskDrawingApp(BaseMaskApp):
    def __init__(self, blue_image, violet_image, output_path, template_mask_paths=[None, None]):
        '''
        template_mask_paths = [contour_mask_path, segmented_mask_path]
        '''
        super().__init__(blue_image, violet_image)

        self.output_path = output_path
        self.contour_mask_path, self.segmented_mask_path = template_mask_paths

        self.current_image = self.blue_image.copy()
        self.show_blue = True

        # Get image dimensions
        self.image_height, self.image_width, _ = self.current_image.shape
        self.canvas = QPixmap(self.image_width, self.image_height)
        self.canvas.fill(Qt.transparent)

        self.masks_loaded = False
        self.mask_visible = False
        self.mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)  
        self.seg_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)  
        self.contour_mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)  

        self.seg_mask_original = None
        self.contour_mask_original = None

        if self.contour_mask_path and self.segmented_mask_path:
            self.load_mask()

        # GUI Components
        self.label = QLabel(self)
        self.label.setPixmap(QPixmap.fromImage(cv_to_qt(self.current_image)))

        self.toggle_button = QPushButton("Toggle Image (Blue/Violet)", self)
        self.clear_button = QPushButton("Clear", self)
        self.submit_button = QPushButton("Submit Mask", self)

        self.mode_toggle_button = QPushButton("Switch to Mask Edit Mode", self)


        self.brush_size_slider = QSlider(Qt.Horizontal, self)

        self.brush_size_slider.setMinimum(1)
        self.brush_size_slider.setMaximum(100)
        self.brush_size_slider.setValue(10)
        self.brush_size = 10

        self.mask_scale_slider = QSlider(Qt.Horizontal, self)
        self.mask_scale_slider.setMinimum(10)  # 10% of original size
        self.mask_scale_slider.setMaximum(200)  # 200% of original size
        self.mask_scale_slider.setValue(100)  # Default: 100% (no scaling)
        self.mask_scale_slider.setVisible(False)  # Hidden until edit mode is enabled
        self.mask_scale_slider.valueChanged.connect(self.update_mask_scale)


        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.toggle_button)
        layout.addWidget(self.brush_size_slider)
        layout.addWidget(self.clear_button)
        layout.addWidget(self.submit_button)
        layout.addWidget(self.mode_toggle_button)
        layout.addWidget(self.mask_scale_slider)

        central_widget = QWidget()
        central_widget.setLayout(layout)
        self.setCentralWidget(central_widget)

        # Connect signals
        self.toggle_button.clicked.connect(self.toggle_image)
        self.clear_button.clicked.connect(self.clear_canvas)
        self.submit_button.clicked.connect(self.submit_mask)
        self.brush_size_slider.valueChanged.connect(self.update_brush_size)
        self.brush_size_slider.setFocusPolicy(Qt.NoFocus)
        self.mode_toggle_button.clicked.connect(self.toggle_mode)

        # Drawing settings
        self.is_drawing_mode = True
        self.drawing = False
        self.editing_mask = False
        self.last_point = QPoint()
        self.pen_color = QColor(255, 0, 0, 50)
        
        # Mask transformation variables
        self.mask_offset = QPoint(0, 0)
        self.mask_scale = 1.0

    def toggle_mode(self):
            """ Toggle between drawing mode and mask editing mode """
            self.is_drawing_mode = not self.is_drawing_mode

            if self.is_drawing_mode:
                self.mask = np.zeros((self.image_height, self.image_width), dtype=np.uint8)
                self.mask_visible = False
                self.editing_mask = False
                self.mask_scale_slider.setVisible(False)
                self.mode_toggle_button.setText("Switch to Mask Edit Mode")
            else:
                self.load_mask()
                self.mask_visible = True
                self.editing_mask = True
                self.mask_scale_slider.setVisible(True)
                self.mode_toggle_button.setText("Switch to Drawing Mode")
            self.update()

    def load_mask(self):
        """Load and process an existing mask, resizing if needed."""
        seg_mask = cv2.imread(self.segmented_mask_path, cv2.IMREAD_UNCHANGED)
        contour_mask = cv2.imread(self.contour_mask_path, cv2.IMREAD_UNCHANGED)

        if seg_mask is None or contour_mask is None:
            print("Error loading masks.")
            return

        # Ensure it has an alpha channel
        if seg_mask.shape[-1] == 3:
            seg_mask = cv2.cvtColor(seg_mask, cv2.COLOR_BGR2BGRA)
        if contour_mask.shape[-1] == 3:
            contour_mask = cv2.cvtColor(contour_mask, cv2.COLOR_BGR2BGRA)

        mask_height, mask_width = seg_mask.shape[:2]

        # Store original mask before transformations
        self.seg_mask_original = seg_mask.copy()
        self.contour_mask_original = contour_mask.copy()

        # Adjust mask size if needed
        if (mask_height, mask_width) != (self.image_height, self.image_width):
            print("Resizing mask to fit image dimensions.")
            self.seg_mask_original = cv2.resize(seg_mask, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)
            self.contour_mask_original = cv2.resize(contour_mask, (self.image_width, self.image_height), interpolation=cv2.INTER_LINEAR)


        self.seg_mask = self.seg_mask_original.copy()
        self.contour_mask = self.contour_mask_original.copy()
        self.masks_loaded = True


    def update_mask_scale(self):
        """Update the mask scale based on the slider value."""
        scale_factor = self.mask_scale_slider.value() / 100.0  # Convert from percentage
        self.mask_scale = scale_factor
        self.apply_mask_transformation()
        self.update()
    
    def apply_mask_transformation(self):
        """Scale and translate the mask."""
        if self.seg_mask_original is None or self.contour_mask_original is None:
            return
        
        # Scale the mask
        new_size = (int(self.image_width * self.mask_scale), int(self.image_height * self.mask_scale))
        scaled_seg_mask = cv2.resize(self.seg_mask_original, new_size, interpolation=cv2.INTER_LINEAR)
        scaled_contour_mask = cv2.resize(self.contour_mask_original, new_size, interpolation=cv2.INTER_LINEAR)

        # Create a blank transparent mask
        transformed_seg_mask = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)
        transformed_contour_mask = np.zeros((self.image_height, self.image_width, 4), dtype=np.uint8)

        # Get new mask position based on translation
        x_offset = self.mask_offset.x()
        y_offset = self.mask_offset.y()

        # Ensure mask doesn't go out of bounds
        x_start = max(0, x_offset)
        y_start = max(0, y_offset)
        x_end = min(self.image_width, x_offset + scaled_seg_mask.shape[1])
        y_end = min(self.image_height, y_offset + scaled_seg_mask.shape[0])

        x_src_start = max(0, -x_offset)
        y_src_start = max(0, -y_offset)

        # Copy scaled mask into transformed mask
        transformed_seg_mask[y_start:y_end, x_start:x_end] = scaled_seg_mask[y_src_start:y_src_start + (y_end - y_start), x_src_start:x_src_start + (x_end - x_start)]
        transformed_contour_mask[y_start:y_end, x_start:x_end] = scaled_contour_mask[y_src_start:y_src_start + (y_end - y_start), x_src_start:x_src_start + (x_end - x_start)]

        self.seg_mask = transformed_seg_mask
        self.contour_mask = transformed_contour_mask

    def paintEvent(self, event):
        """ Redraw the canvas with the background, user's drawing, and optional mask overlay """

        # Start with the base image (blue or violet)
        pixmap = QPixmap.fromImage(cv_to_qt(self.current_image))
        canvas_painter = QPainter(pixmap)
        canvas_painter.drawPixmap(0, 0, self.canvas)  

        # If the mask is visible, draw it on top of everything
        if self.masks_loaded and self.mask_visible:
            self.apply_mask_transformation()
            mask_pixmap = QPixmap.fromImage(QImage(self.seg_mask.data, self.seg_mask.shape[1], self.seg_mask.shape[0], QImage.Format_RGBA8888))
            canvas_painter.drawPixmap(0, 0, mask_pixmap)  # Draw transformed mask
            self.update()
        canvas_painter.end()
        self.label.setPixmap(pixmap)
        

    def clear_canvas(self):
        """ Clear the canvas and reset mask """
        self.canvas.fill(Qt.transparent)  # Properly clears the overlay
        self.mask[:] = 0  # Reset mask
        self.seg_mask[:] = 0
        self.contour_mask[:] = 0
        self.paintEvent(None)

    def update_brush_size(self):
        """ Update brush size from slider """
        self.brush_size = self.brush_size_slider.value()

    def mousePressEvent(self, event):
        """ Start drawing or moving mask """
        if event.button() == Qt.LeftButton:
            if self.editing_mask:
                self.last_point = event.pos()
            else:
                self.drawing = True
                self.last_point = event.pos()

    def mouseMoveEvent(self, event):
        """ Draw or move the mask """
        if self.drawing:
            painter = QPainter(self.canvas)
            pen = QPen(self.pen_color, self.brush_size, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
            painter.setPen(pen)
            painter.drawLine(self.last_point, event.pos())

            # Update mask array (brain pixels = 1)
            cv2.line(self.mask, (self.last_point.x(), self.last_point.y()), (event.pos().x(), event.pos().y()), 1, self.brush_size)
            self.last_point = event.pos()
            self.paintEvent(None)
        elif self.editing_mask:
            delta = event.pos() - self.last_point
            self.mask_offset += delta
            self.last_point = event.pos()
            self.paintEvent(None)

    def mouseReleaseEvent(self, event):
        """ Stop drawing or moving mask """
        if event.button() == Qt.LeftButton:
            self.drawing = False

    def keyPressEvent(self, event):
        """ Handle key events for scaling mask """
        if self.editing_mask:
            if event.key() == Qt.Key_Up:
                self.mask_scale += 0.1
            elif event.key() == Qt.Key_Down:
                self.mask_scale = max(0.1, self.mask_scale - 0.1)
            self.paintEvent(None)
            self.apply_mask_transformation()  # Apply changes
            self.update()  # Refresh UI

    def toggle_image(self):
        """ Toggle between blue and violet images """
        self.show_blue = not self.show_blue
        self.current_image = self.blue_image if self.show_blue else self.violet_image
        self.paintEvent(None)

    def toggle_mask_edit_mode(self):
        """ Enable or disable mask editing mode """
        self.editing_mask = not self.editing_mask
        self.mask_scale_slider.setVisible(self.editing_mask)  # Show/hide the scale slider
        self.mask_edit_button.setText("Exit Mask Edit" if self.editing_mask else "Edit Mask")

    def submit_mask(self):
        """ Save the mask as a binary image or NumPy file """

        # Ensure the mask exists
        if self.seg_mask is None or self.contour_mask is None:
            print("Error: No masks to save.")
            return

        # Determine if we're in drawing mode (mask is a 2D NumPy array) or mask edit mode (mask is RGBA)
        if self.is_drawing_mode:
            # Convert drawn mask to binary format (0 for background, 255 for mask)
            binary_mask = (self.mask > 0).astype(np.uint8) * 255  # Convert 1s to 255

            if self.output_path != None:
                # Save as PNG (binary mask)
                cv2.imwrite(os.path.join(self.output_path, "brain_mask.png"), binary_mask)
                print(f"Drawn mask saved as binary PNG at {os.path.join(self.output_path, 'brain_mask.png')}")
                # Save as .npy for numerical processing 
                np.save(os.path.join(self.output_path, "brain_mask.npy"), self.mask)  
                print(f"Drawn mask saved as binary NPY at {os.path.join(self.output_path, 'brain_mask.npy')}")
            #else: # return the npy mask only, does not save as png
                #return self.mask
        else:
            # Ensure the mask has an alpha channel before extracting transparency
            if self.contour_mask.shape[-1] != 4:
                print("Error: Mask does not have an alpha channel.")
                return

            # Extract the alpha channel (invert so transparent areas are 0)
            alpha_channel_contour = self.contour_mask[:, :, 3]
            alpha_channel_seg = self.seg_mask[:, :, 3]

            # Convert to binary format (0 for background, 255 for mask)
            binary_alpha_contour = (alpha_channel_contour > 0).astype(np.uint8) * 255  
            binary_alpha_seg = (alpha_channel_seg > 0).astype(np.uint8) * 255

            # Save as PNG
            cv2.imwrite(os.path.join(self.output_path, "brain_mask.png"), binary_alpha_contour)
            cv2.imwrite(os.path.join(self.output_path, "segmented_mask.png"), binary_alpha_seg)

            print(f"Edited PNG masks saved as binary PNGs at {os.path.join(self.output_path, 'brain_mask.png')} and {os.path.join(self.output_path, 'segmented_mask.png')}")

            # Save as .npy for numerical processing 
            np.save(os.path.join(self.output_path, "brain_mask.npy"), alpha_channel_contour)  
            np.save(os.path.join(self.output_path, "segmented_mask.npy"), alpha_channel_seg) 

            print(f"Masks also saved as NumPy array at {os.path.join(self.output_path, 'brain_mask.npy')} and {os.path.join(self.output_path, 'segmented_mask.npy')}")

        self.close()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MaskDrawingApp(np.random.randint(0, 255, (512, 512), dtype=np.uint8),
                             np.random.randint(0, 255, (512, 512), dtype=np.uint8),
                             output_path=r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\PIPELINE\notebook_images",template_mask_paths=[r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\PIPELINE\notebook_images\default_brain_mask_down.png", r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\PIPELINE\notebook_images\default_segmentation_down.png"])
    window.show()
    sys.exit(app.exec_())