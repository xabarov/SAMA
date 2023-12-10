from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QAction, QMessageBox, QToolBar, QToolButton
from PyQt5.QtGui import QIcon, QCursor

from annotator import Annotator
from utils import config

from ui.edit_with_button import EditWithButton

from utils import cls_settings
from utils.segmenter_worker import SegmenterWorker

from shapely import Polygon

import utils.help_functions as hf
import cv2
import os

import numpy as np
from utils.gdal_translate import convert_geotiff


class Detector(Annotator):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Detector")

        # For GeoTIFF support
        self.map_geotiff_names = {}
        self.view.mouse_move_conn.on_mouse_move.connect(self.on_view_mouse_move)
        self.block_geo_coords_message = False

    def on_view_mouse_move(self, x, y):
        if self.image_set and not self.block_geo_coords_message:
            if self.lrm:
                geo_x, geo_y = hf.convert_point_coords_to_geo(x, y, self.tek_image_path)
                if geo_x == 0 and geo_y == 0:
                    return
                self.statusBar().showMessage(
                    f"{geo_x:.6f}, {geo_y:.6f}")

    def createActions(self):

        self.detectScanAct = QAction(
            "Обнаружить объекты сканированием" if self.settings.read_lang() == 'RU' else "Detect objects with scanning",
            self, enabled=False,
            triggered=self.detect_scan)

        self.segmentImage = QAction(
            "Сегментация" if self.settings.read_lang() == 'RU' else "Segment image",
            self, enabled=False,
            triggered=self.segment_image)

        self.exportToESRIAct = QAction(
            "Экспорт в ESRI shapefile" if self.settings.read_lang() == 'RU' else "Export to ESRI shapefile", self,
            enabled=False,
            triggered=self.export_to_esri)

        super(Detector, self).createActions()

    def createMenus(self):
        super(Detector, self).createMenus()

        self.classifierMenu.addAction(self.detectScanAct)

        self.classifierMenu.addAction(self.segmentImage)

        self.annotatorExportMenu.addAction(self.exportToESRIAct)

    def createToolbar(self):

        self.create_top_toolbar()

        toolBar = QToolBar("Панель инструментов" if self.settings.read_lang() == 'RU' else "ToolBar", self)
        toolBar.addAction(self.createNewProjectAct)
        toolBar.addSeparator()
        toolBar.addAction(self.zoomInAct)
        toolBar.addAction(self.zoomOutAct)
        toolBar.addAction(self.fitToWindowAct)
        toolBar.addSeparator()

        self.annotatorToolButton = QToolButton(self)
        self.annotatorToolButton.setDefaultAction(self.polygonAct)
        self.annotatorToolButton.setPopupMode(QToolButton.MenuButtonPopup)
        self.annotatorToolButton.triggered.connect(self.ann_triggered)

        self.annotatorToolButton.setMenu(self.AnnotatorMethodMenu)

        toolBar.addWidget(self.annotatorToolButton)

        toolBar.addSeparator()
        toolBar.addAction(self.detectAct)
        toolBar.addAction(self.detectScanAct)

        toolBar.addSeparator()
        toolBar.addAction(self.settingsAct)
        toolBar.addSeparator()

        self.toolBarLeft = toolBar
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBarLeft)

    def set_icons(self):
        super(Detector, self).set_icons()
        # AI
        self.detectScanAct.setIcon(QIcon(self.icon_folder + "/slide.png"))
        self.segmentImage.setIcon(QIcon(self.icon_folder + "/seg.png"))
        self.exportToESRIAct.setIcon(QIcon(self.icon_folder + "/esri_shp.png"))

    def get_jpg_path(self, image_name):

        suffix = image_name.split('.')[-1]
        if suffix in ['tif', 'tiff']:
            if image_name in self.map_geotiff_names:
                jpg_path = self.map_geotiff_names[image_name]
            else:
                temp_folder = hf.handle_temp_folder(os.getcwd())  # if not exist
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

        jpg_path = self.get_jpg_path(image_name)
        super(Detector, self).open_image(jpg_path)

        self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

    def on_segment_start(self):
        print('Started')
        self.view.start_circle_progress()

    def on_segment_finished(self):
        print('Finished')
        predictions = self.seg_worker.results['predictions']
        shape = predictions.shape

        for cls_num, cls_name in enumerate(cls_settings.CLASSES_SEG):
            if cls_name == 'background':
                continue
            mask = predictions == cls_num

            segment_np = np.zeros((shape[0], shape[1], 3))
            segment_np[:, :] = [0, 0, 0]
            segment_np[mask, :] = cls_settings.PALETTE_SEG[cls_num]
            temp_folder = hf.handle_temp_folder(os.getcwd())
            segment_name = os.path.join(temp_folder, f'segment_{cls_name}.jpg')
            cv2.imwrite(segment_name, segment_np)
            self.view.add_segment_pixmap(QtGui.QPixmap(segment_name), opacity=0.5, z_value=100 + cls_num)

        self.view.stop_circle_progress()
        # for points in points_mass:
        #     cls_num = self.project_data.get_label_num(cls_name)
        #     if cls_num == -1:
        #         cls_num = 0
        #     self.add_segment_polygon_to_scene(points, cls_num)

    def add_segment_polygon_to_scene(self, points, cls_num):

        if len(points) > 0:
            shapely_pol = Polygon(points)
            area = shapely_pol.area
            if area > config.POLYGON_AREA_THRESHOLD:

                cls_name = self.cls_combo.itemText(cls_num)
                alpha_tek = self.settings.read_alpha()
                color = self.project_data.get_label_color(cls_name)

                self.view.add_polygon_to_scene(cls_num, points, color, alpha_tek)
                self.save_view_to_project()

            else:
                if self.settings.read_lang() == 'RU':
                    self.statusBar().showMessage(
                        f"Метку сделать не удалось. Площадь маски слишком мала {area:0.3f}. Попробуйте еще раз", 3000)
                else:
                    self.statusBar().showMessage(
                        f"Can't create label. Area of label is too small {area:0.3f}. Try again", 3000)

        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def segment_image(self):

        config = os.path.join(os.getcwd(), cls_settings.SEG_DICT['PSPNet']['config'])
        checkpoint = os.path.join(os.getcwd(), cls_settings.SEG_DICT['PSPNet']['weights'])

        im = self.cv2_image

        palette = cls_settings.PALETTE_SEG
        classes = cls_settings.CLASSES_SEG
        self.seg_worker = SegmenterWorker(image_path=im, config_path=config, checkpoint_path=checkpoint,
                                          palette=palette,
                                          classes=classes, device='cuda')

        self.seg_worker.started.connect(self.on_segment_start)
        self.seg_worker.finished.connect(self.on_segment_finished)

        if not self.seg_worker.isRunning():
            self.seg_worker.start()

    def export_to_esri(self):

        self.esri_path_window = EditWithButton(None, in_separate_window=True,
                                               theme=self.settings.read_theme(),
                                               on_button_clicked_callback=self.on_input_esri_path,
                                               is_dir=False, dialog_text='ESRI shapefile name',
                                               title=f"Choose ESRI shapefile name", file_type='shp',
                                               placeholder='ESRI shapefile name', is_existing_file_only=False)
        self.esri_path_window.show()

    def on_input_esri_path(self):

        esri_filename = self.esri_path_window.getEditText()

        self.esri_path_window.hide()

        if hf.get_extension(esri_filename) != 'shp':

            if self.settings.read_lang() == 'RU':
                message = f"Имя ESRI shapefile файла должно иметь расширение .shp"
            else:
                message = f"ESRI shapefile has to have '.shp' extension."

            self.statusBar().showMessage(
                message, 3000)

            return

        if self.lrm:
            esri_shapes = []
            view_shapes = self.view.get_all_shapes()
            for shape in view_shapes:
                shape_id = shape['id']
                is_found = False
                for det_shape in self.detected_shapes:
                    if det_shape['id'] == shape_id:
                        esri_shapes.append(det_shape)
                        is_found = True
                        break
                if not is_found:
                    esri_shapes.append(shape)

            hf.convert_shapes_to_esri(esri_shapes, self.tek_image_path, out_shapefile=esri_filename)

            if self.settings.read_lang() == 'RU':
                message = f"ESRI shapefile файл создан. Добавлено {len(esri_shapes)} объектов"
            else:
                message = f"ESRI shapefile has been created with {len(esri_shapes)} objects."

            self.statusBar().showMessage(
                message, 3000)

        else:
            if self.settings.read_lang() == 'RU':
                message = f"ESRI shapefile файл не создан. Не найден файл геоданных"
            else:
                message = f"Can't create ESRI shapefile. No geo data"

            self.statusBar().showMessage(
                message, 3000)

    def toggle_act(self, is_active):
        super(Detector, self).toggle_act(is_active)

        self.detectScanAct.setEnabled(is_active)

        self.segmentImage.setEnabled(is_active)

        self.exportToESRIAct.setEnabled(is_active)

    def about(self):
        """
        Окно о приложении
        """
        QMessageBox.about(self, "AI Detector",
                          "<p><b>AI Detector</b></p>"
                          "<p>Программа для обнаружения объектов</p>" if
                          self.settings.read_lang() == 'RU' else "<p>Object detection and instance segmentation program</p>")

    def detect_scan(self):
        self.scanning_mode = True

        self.detect()


if __name__ == '__main__':
    import sys
    from qt_material import apply_stylesheet

    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             # 'primaryTextColor': '#ffffff',
             }

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = Detector()
    w.show()
    sys.exit(app.exec_())
