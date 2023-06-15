from PyQt5 import QtWidgets
from detector import Detector
from utils import config
from qt_material import apply_stylesheet

import utils.help_functions as hf
import sys


class DetectorGeoTIFF(Detector):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Detector GeoTIFF")


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             'primaryTextColor': '#ffffff'}

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = DetectorGeoTIFF()
    w.show()
    sys.exit(app.exec_())
