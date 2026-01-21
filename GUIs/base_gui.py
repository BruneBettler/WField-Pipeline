# GUIs/base_gui.py

from PyQt5.QtWidgets import QMainWindow, QLabel, QPushButton, QVBoxLayout, QWidget
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import numpy as np
import cv2
from gui_utils import cv_to_qt

class BaseMaskApp(QMainWindow):
    def __init__(self, blue_image: np.ndarray, violet_image: np.ndarray):
        super().__init__()

        # Convert grayscale images to RGB if needed
        if blue_image.ndim == 2:
            blue_image = cv2.cvtColor(blue_image, cv2.COLOR_GRAY2RGB)
        if violet_image.ndim == 2:
            violet_image = cv2.cvtColor(violet_image, cv2.COLOR_GRAY2RGB)

        self.blue_image = blue_image
        self.violet_image = violet_image
        self.current_image = self.blue_image.copy()
        self.show_blue = True

        # GUI Elements
        self.label = QLabel(self)
        self.label.setPixmap(QPixmap.fromImage(cv_to_qt(self.current_image)))

        self.toggle_button = QPushButton("Toggle Image (Blue/Violet)", self)
        self.toggle_button.clicked.connect(self.toggle_image)

        # Layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)
        layout.addWidget(self.toggle_button)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def toggle_image(self):
        """ Toggle between blue and violet images """
        self.show_blue = not self.show_blue
        self.current_image = self.blue_image if self.show_blue else self.violet_image
        self.label.setPixmap(QPixmap.fromImage(cv_to_qt(self.current_image)))