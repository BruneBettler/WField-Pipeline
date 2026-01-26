# GUIs/utils.py
from PyQt5.QtGui import QImage

def cv_to_qt(cv_img):
    height, width, channel = cv_img.shape
    bytes_per_line = 3 * width
    return QImage(cv_img.data, width, height, bytes_per_line, QImage.Format_RGB888)
