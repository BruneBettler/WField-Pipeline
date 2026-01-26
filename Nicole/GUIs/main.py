# main.py
from mask_drawing_gui import MaskDrawingApp
from PyQt5.QtWidgets import QApplication
import sys, numpy as np

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MaskDrawingApp(
        np.random.randint(0, 255, (512, 512), dtype=np.uint8),
        np.random.randint(0, 255, (512, 512), dtype=np.uint8),
        output_path=r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\PIPELINE\OUTPUT_MASKS_TEMP",
        template_mask_paths=[r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\PIPELINE\notebook_data\notebook_images\default_brain_mask_down.png", r"C:\Users\bbettl\PycharmProjects\wfield_pipeline\PIPELINE\notebook_data\notebook_images\default_segmentation_down.png"]
    )
    window.show()
    sys.exit(app.exec_())
