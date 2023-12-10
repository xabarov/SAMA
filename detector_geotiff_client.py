from PyQt5 import QtCore, QtWidgets, QtGui
from detector_client import DetectorClient
from PyQt5.QtGui import QCursor

from utils import config
from utils.gdal_translate import convert_geotiff

from qt_material import apply_stylesheet

import utils.help_functions as hf
import sys
import os
import cv2


class DetectorGeoTIFFClient(DetectorClient):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Detector GeoTIFF")

        # For GeoTIFF support
        self.image_types = ['jpg', 'png', 'tiff', 'jpeg', 'tif']
        self.map_geotiff_names = {}
        self.view.mouse_move_conn.on_mouse_move.connect(self.on_view_mouse_move)

    def on_view_mouse_move(self, x, y):
        if self.image_set:
            if self.lrm:
                geo_x, geo_y = hf.convert_point_coords_to_geo(x, y, self.tek_image_path)
                self.statusBar().showMessage(
                    f"{geo_x:.6f}, {geo_y:.6f}")

    def detect(self):
        # на вход воркера - исходное изображение

        if self.tek_image_name.split('.')[-1] == 'tif':
            img_name = os.path.basename(self.map_geotiff_names[self.tek_image_path])
            img_path = hf.handle_temp_folder(os.getcwd())
        else:
            img_name = os.path.basename(self.tek_image_path)
            img_path = self.dataset_dir

        self.run_detection(img_name=img_name, img_path=img_path)

    def on_image_set(self):

        if len(self.queue_to_image_setter) != 0:

            self.image_set = False

            image_name = self.queue_to_image_setter[-1]  # geo_tif_names

            if image_name.split('.')[-1] == 'tif':
                if image_name in self.map_geotiff_names:
                    jpg_path = self.map_geotiff_names[image_name]
                else:

                    temp_folder = hf.handle_temp_folder(os.getcwd())  # if not exist
                    jpg_path = os.path.join(temp_folder,
                                            os.path.basename(image_name).split('.')[0] + '.jpg')

                    convert_geotiff(image_name, jpg_path)

                    self.map_geotiff_names[image_name] = jpg_path

                self.image_setter.set_image(jpg_path)  # !!!

            else:
                self.image_setter.set_image(image_name)

            self.queue_to_image_setter = []
            self.statusBar().showMessage(
                "Нейросеть SAM еще не готова. Подождите секунду..." if self.settings.read_lang() == 'RU' else "SAM is loading. Please wait...",
                3000)
            self.image_setter.start()

        else:
            self.statusBar().showMessage(
                "Нейросеть SAM готова к сегментации" if self.settings.read_lang() == 'RU' else "SAM ready to work",
                3000)
            self.image_set = True

    def open_image(self, image_name):

        if image_name.split('.')[-1] == 'tif':
            # GeoTIFF
            message = f"Загружаю {os.path.basename(image_name)}..." if self.settings.read_lang() == 'RU' else f"Loading {os.path.basename(image_name)}..."
            self.statusBar().showMessage(
                message,
                3000)

            self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))

            if image_name in self.map_geotiff_names:
                jpg_path = self.map_geotiff_names[image_name]
                image = cv2.imread(jpg_path)
            else:

                temp_folder = hf.handle_temp_folder(os.getcwd())  # if not exist
                jpg_path = os.path.join(temp_folder,
                                        os.path.basename(image_name).split('.')[0] + '.jpg')

                image = convert_geotiff(image_name, jpg_path)

                self.map_geotiff_names[image_name] = jpg_path

            self.view.setPixmap(QtGui.QPixmap(jpg_path))
            self.view.fitInView(self.view.pixmap_item, QtCore.Qt.KeepAspectRatio)

            self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

            self.cv2_image = image


            self.toggle_act(True)

            self.image_set = False
            self.queue_image_to_sam(image_name)
            lrm = hf.try_read_lrm(image_name)
            if lrm:
                self.lrm = lrm

        else:
            super(DetectorGeoTIFFClient, self).open_image(image_name)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             'primaryTextColor': '#ffffff'}

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = DetectorGeoTIFFClient()
    w.show()
    sys.exit(app.exec_())
