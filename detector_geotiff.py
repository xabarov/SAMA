import os
import sys

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QCursor
from qt_material import apply_stylesheet

import utils.help_functions as hf
from detector import Detector
from utils import config
from utils.gdal_translate import convert_geotiff


class DetectorGeoTIFF(Detector):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Detector GeoTIFF")

        # For GeoTIFF support
        self.image_types = ['jpg', 'png', 'tiff', 'jpeg', 'tif']
        self.map_geotiff_names = {}
        self.view.mouse_move_conn.on_mouse_move.connect(self.on_view_mouse_move)
        self.block_geo_coords_message = False

    def on_view_mouse_move(self, x, y):
        if self.image_set and not self.block_geo_coords_message:
            if self.lrm:
                geo_x, geo_y = hf.convert_point_coords_to_geo(x, y, self.tek_image_path)
                self.statusBar().showMessage(
                    f"{geo_x:.6f}, {geo_y:.6f}")

    def detect(self):
        # на вход воркера - исходное изображение

        jpg_path = self.get_jpg_name(self.tek_image_path)

        img_name = os.path.basename(jpg_path)
        img_path = os.path.dirname(jpg_path)

        self.run_detection(img_name=img_name, img_path=img_path)

    def start_grounddino(self):
        prompt = self.prompt_input_dialog.getPrompt()
        self.prompt_input_dialog.close()

        jpg_path = self.get_jpg_name(self.tek_image_path)

        if prompt:
            self.run_gd(jpg_path, prompt)

    def on_image_set(self):

        if len(self.queue_to_image_setter) != 0:
            image_name = self.queue_to_image_setter[-1]  # geo_tif_names
            jpg_name = self.get_jpg_name(image_name)
            self.set_image(jpg_name)

        else:
            self.statusBar().showMessage(
                "Нейросеть SAM готова к сегментации" if self.settings.read_lang() == 'RU' else "SAM ready to work",
                3000)
            self.image_set = True

    def get_jpg_name(self, image_name):

        suffix = image_name.split('.')[-1]
        if suffix in ['tif', 'tiff']:
            if image_name in self.map_geotiff_names:
                jpg_path = self.map_geotiff_names[image_name]
            else:
                temp_folder = self.handle_temp_folder()  # if not exist
                jpg_path = os.path.join(temp_folder,
                                        os.path.basename(image_name).split('.')[0] + '.jpg')

                convert_geotiff(image_name, save_path=jpg_path)

                self.map_geotiff_names[image_name] = jpg_path
        else:
            jpg_path = image_name

        return jpg_path

    def open_image(self, image_name):
        message = f"Загружаю {os.path.basename(image_name)}..." if self.settings.read_lang() == 'RU' else f"Loading {os.path.basename(image_name)}..."
        self.statusBar().showMessage(
            message,
            3000)

        self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))

        jpg_path = self.get_jpg_name(image_name)
        super(DetectorGeoTIFF, self).open_image(jpg_path)

        self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             'primaryTextColor': '#ffffff'}

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = DetectorGeoTIFF()
    w.show()
    sys.exit(app.exec_())
