import json
import os
import sys

import matplotlib.pyplot as plt
import requests
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import QAction, QMessageBox, QMenu
from PyQt5.QtWidgets import QToolBar, QToolButton
from qt_material import apply_stylesheet
from shapely import Polygon

import utils.help_functions as hf
from ui.base_window import MainWindow
from ui.edit_with_button import EditWithButton
from ui.settings_window_client import SettingsWindowClient
from ui.show_image_widget import ShowImgWindow
from utils import cls_settings
from utils import config
from utils.cnn_worker_client import CNNWorkerClient
from utils.sam_predictor_client import SAMImageSetterClient, SAMPredictByPointsClient, SAMPredictByMaskClient


class DetectorClient(MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Detector")

        # Detector
        self.started_cnn = None

        # SAM
        self.image_set = False
        self.image_setter = SAMImageSetterClient(self.settings.read_server_name())

        self.sam_by_points_client = SAMPredictByPointsClient(self.settings.read_server_name())
        self.sam_by_points_client.finished.connect(self.on_sam_by_points_finished)

        self.sam_by_mask_client = SAMPredictByMaskClient(self.settings.read_server_name())
        self.sam_by_mask_client.finished.connect(self.on_sam_by_mask_finished)

        self.image_setter.finished.connect(self.on_image_set)
        self.queue_to_image_setter = []

        self.view.mask_end_drawing.on_mask_end_drawing.connect(self.ai_mask_end_drawing)

        self.scanning_mode = False
        self.lrm = None

        self.detected_shapes = []

    def createActions(self):
        super(DetectorClient, self).createActions()
        self.balanceAct = QAction("Информация о датасете" if self.settings.read_lang() == 'RU' else "Dataset info",
                                  self,
                                  enabled=False, triggered=self.on_dataset_balance_clicked)

        self.detectAct = QAction(
            "Обнаружить объекты за один проход" if self.settings.read_lang() == 'RU' else "Detect objects", self,
            shortcut="Ctrl+Y", enabled=False,
            triggered=self.detect)

        self.detectScanAct = QAction(
            "Обнаружить объекты сканированием" if self.settings.read_lang() == 'RU' else "Detect objects with scanning",
            self, enabled=False,
            triggered=self.detect_scan)

        self.syncLabelsAct = QAction(
            "Синхронизировать имена меток" if self.settings.read_lang() == 'RU' else "Fill label names from AI model",
            self, enabled=False,
            triggered=self.sync_labels)

        self.aiAnnotatorPointsAct = QAction(
            "Сегментация по точкам" if self.settings.read_lang() == 'RU' else "SAM by points",
            self, enabled=False, shortcut="Ctrl+A",
            triggered=self.ai_points_pressed,
            checkable=True)

        self.aiAnnotatorMaskAct = QAction(
            "Сегментация внутри бокса" if self.settings.read_lang() == 'RU' else "SAM by box", self,
            enabled=False, shortcut="Ctrl+M",
            triggered=self.ai_mask_pressed,
            checkable=True)

        self.exportToESRIAct = QAction(
            "Экспорт в ESRI shapefile" if self.settings.read_lang() == 'RU' else "Export to ESRI shapefile", self,
            enabled=False,
            triggered=self.export_to_esri)

    def sync_labels(self):

        url = f'{self.settings.read_server_name()}/sync_names'
        response = requests.get(url)
        names = json.loads(response.text)

        self.cls_combo.clear()
        labels = []
        for key in names:
            label = names[key]
            self.cls_combo.addItem(label)
            labels.append(label)

        self.project_data.set_labels(labels)
        self.project_data.set_labels_colors(labels, rewrite=True)

    def detect(self):
        # на вход воркера - исходное изображение

        img_path = self.dataset_dir
        img_name = os.path.basename(self.tek_image_name)

        self.run_detection(img_name=img_name, img_path=img_path)

    def detect_scan(self):
        self.scanning_mode = True

        self.detect()

    def run_detection(self, img_name, img_path):
        """
        Запуск классификации
        img_name - имя изображения
        img_path - путь к изображению
        """

        self.started_cnn = self.settings.read_cnn_model()

        conf_thres = self.settings.read_conf_thres()
        iou_thres = self.settings.read_iou_thres()

        lrm = 0  # 0 === not scanning mode

        if self.scanning_mode:
            str_text = "Начинаю классифкацию СНС {0:s} сканирующим окном".format(self.started_cnn)
            if self.lrm:
                lrm = self.lrm
        else:
            str_text = "Начинаю классифкацию СНС {0:s}".format(self.started_cnn)

        self.statusBar().showMessage(str_text, 3000)

        self.CNN_worker = CNNWorkerClient(server=self.settings.read_server_name(),
                                          image_path=os.path.join(img_path, img_name), conf_thres=conf_thres,
                                          iou_thres=iou_thres, lrm=lrm)

        self.CNN_worker.started.connect(self.on_cnn_started)
        self.CNN_worker.finished.connect(self.on_cnn_finished)

        if not self.CNN_worker.isRunning():
            self.CNN_worker.start()

    def on_cnn_started(self):
        """
        При начале классификации
        """
        self.statusBar().showMessage(
            f"Начинаю поиск объектов на изображении..." if self.settings.read_lang() == 'RU' else f"Start searching object on image...",
            3000)
        self.view.start_circle_progress()

    def on_cnn_finished(self):
        """
        При завершении классификации
        """

        if self.scanning_mode:
            self.scanning_mode = False

        self.detected_shapes = []
        for res in self.CNN_worker.mask_results:

            shape_id = self.view.get_unique_label_id()

            cls_num = res['cls_num']
            points = res['points']

            color = None
            label = self.project_data.get_label_name(cls_num)
            if label:
                color = self.project_data.get_label_color(label)
            if not color:
                color = cls_settings.PALETTE[cls_num]

            self.view.add_polygon_to_scene(cls_num, points, color=color, id=shape_id)

            shape = {'id': shape_id, 'cls_num': cls_num, 'points': points, 'conf': res['conf']}
            self.detected_shapes.append(shape)

        self.write_scene_to_project_data()
        self.fill_labels_on_tek_image_list_widget()
        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

        self.statusBar().showMessage(
            f"Найдено {len(self.CNN_worker.mask_results)} объектов" if self.settings.read_lang() == 'RU' else f"{len(self.CNN_worker.mask_results)} objects has been detected",
            3000)

        self.view.stop_circle_progress()

    def toggle_act(self, is_active):
        super(DetectorClient, self).toggle_act(is_active)
        self.balanceAct.setEnabled(is_active)
        self.detectAct.setEnabled(is_active)
        self.detectScanAct.setEnabled(is_active)
        self.exportToESRIAct.setEnabled(is_active)

        self.aiAnnotatorPointsAct.setEnabled(is_active)
        self.aiAnnotatorMaskAct.setEnabled(is_active)

        self.syncLabelsAct.setEnabled(is_active)

    def createMenus(self):
        super(DetectorClient, self).createMenus()

        self.classifierMenu = QMenu("Классификатор" if self.settings.read_lang() == 'RU' else "Classifier", self)
        self.classifierMenu.addAction(self.detectAct)
        self.classifierMenu.addAction(self.detectScanAct)
        self.classifierMenu.addAction(self.syncLabelsAct)

        self.annotatorMenu.addAction(self.balanceAct)

        self.aiAnnotatorMethodMenu = QMenu("С помощью ИИ" if self.settings.read_lang() == 'RU' else "AI", self)

        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorPointsAct)
        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorMaskAct)

        self.AnnotatorMethodMenu.addMenu(self.aiAnnotatorMethodMenu)

        self.menuBar().clear()
        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.classifierMenu)
        self.menuBar().addMenu(self.settingsMenu)
        self.menuBar().addMenu(self.helpMenu)

    def createToolbar(self):
        self.create_right_toolbar()
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
        super(DetectorClient, self).set_icons()
        self.balanceAct.setIcon(QIcon(self.icon_folder + "/bar-chart.png"))

        self.aiAnnotatorMethodMenu.setIcon(QIcon(self.icon_folder + "/ai.png"))
        self.aiAnnotatorPointsAct.setIcon(QIcon(self.icon_folder + "/mouse.png"))
        self.aiAnnotatorMaskAct.setIcon(QIcon(self.icon_folder + "/ai_select.png"))

        self.syncLabelsAct.setIcon(QIcon(self.icon_folder + "/sync.png"))
        # classifier
        self.detectAct.setIcon(QIcon(self.icon_folder + "/detect.png"))
        self.detectScanAct.setIcon(QIcon(self.icon_folder + "/slide.png"))

    def open_image(self, image_name):
        super(DetectorClient, self).open_image(image_name)

        self.image_set = False
        self.queue_image_to_sam(image_name)
        lrm = hf.try_read_lrm(image_name)
        if lrm:
            self.lrm = lrm

    def reload_image(self):
        super(DetectorClient, self).reload_image()
        self.view.clear_ai_points()

    def queue_image_to_sam(self, image_name):

        if not self.image_setter.isRunning():
            self.image_setter.set_image(self.tek_image_path)
            self.statusBar().showMessage(
                "Начинаю загружать изображение в нейросеть SAM..." if self.settings.read_lang() == 'RU' else "Start loading image to SAM...",
                3000)
            self.image_setter.start()
        else:
            self.queue_to_image_setter.append(image_name)

            self.statusBar().showMessage(
                f"Изображение {os.path.split(image_name)[-1]} добавлено в очередь на обработку." if self.settings.read_lang() == 'RU' else f"Image {os.path.split(image_name)[-1]} is added to queue...",
                3000)

    def on_image_set(self):

        if len(self.queue_to_image_setter) != 0:
            image_name = self.queue_to_image_setter[-1]
            self.image_set = False
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

    def on_dataset_balance_clicked(self):
        balance_data = self.project_data.calc_dataset_balance()

        label_names = self.project_data.get_data()['labels']
        labels = list(balance_data.keys())
        labels = [label_names[int(i)] for i in labels]
        values = list(balance_data.values())

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.bar(labels, values,
               # color=config.THEMES_COLORS[self.theme_str],
               width=0.8)

        ax.set_xlabel("Label names")
        ax.set_ylabel("No. of labels")
        ax.tick_params(axis='x', rotation=70)
        plt.title('Баланс меток')

        temp_folder = self.handle_temp_folder()
        fileName = os.path.join(temp_folder, 'balance.jpg')
        plt.savefig(fileName)

        ShowImgWindow(self, title='Баланс меток', img_file=fileName, icon_folder=self.icon_folder)

    def showSettings(self):
        """
        Показать окно с настройками приложения
        """

        self.settings_window = SettingsWindowClient(self)

        self.settings_window.okBtn.clicked.connect(self.on_settings_closed)
        self.settings_window.cancelBtn.clicked.connect(self.on_settings_closed)

        self.settings_window.show()

    def about(self):
        """
        Окно о приложении
        """
        QMessageBox.about(self, "AI Detector Client",
                          "<p><b>AI Detector Client</b></p>"
                          "<p>Программа для обнаружения объектов на изображении</p>" if
                          self.settings.read_lang() == 'RU' else "<p>Object Detection and Instance Segmentation program.</p>")

    def ai_points_pressed(self):

        self.ann_type = "AiPoints"
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings.read_alpha()

        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def ai_mask_pressed(self):

        self.ann_type = "AiMask"
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings.read_alpha()
        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def start_drawing(self):
        super(DetectorClient, self).start_drawing()
        self.view.clear_ai_points()

    def break_drawing(self):
        super(DetectorClient, self).break_drawing()
        if self.ann_type == "AiPoints":
            self.view.clear_ai_points()
            self.view.remove_active()

    def end_drawing(self):
        super(DetectorClient, self).end_drawing()

        if self.ann_type == "AiPoints":

            self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))

            input_point, input_label = self.view.get_sam_input_points_and_labels()

            # print(input_point, input_label)

            if len(input_label):
                if self.image_set and not self.image_setter.isRunning():

                    self.sam_by_points_client.set_inputs(input_point, input_label)

                    if not self.sam_by_points_client.isRunning():
                        self.sam_by_points_client.start()

            else:
                self.view.remove_active()

            self.view.clear_ai_points()
            self.view.end_drawing()

            self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()

            self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def on_sam_by_points_finished(self):
        for points_mass in self.sam_by_points_client.points_mass:
            self.add_sam_polygon_to_scene(points_mass)

    def add_sam_polygon_to_scene(self, points_mass):

        if len(points_mass) > 0:
            filtered_points_mass = []
            for points in points_mass:
                shapely_pol = Polygon(points)
                area = shapely_pol.area

                if area > config.POLYGON_AREA_THRESHOLD:

                    filtered_points_mass.append(points)

                else:
                    if self.settings.read_lang() == 'RU':
                        self.statusBar().showMessage(
                            f"Метку сделать не удалось. Площадь маски слишком мала {area:0.3f}. Попробуйте еще раз",
                            3000)
                    else:
                        self.statusBar().showMessage(
                            f"Can't create label. Area of label is too small {area:0.3f}. Try again", 3000)

            cls_num = self.cls_combo.currentIndex()
            cls_name = self.cls_combo.itemText(cls_num)
            alpha_tek = self.settings.read_alpha()
            color = self.project_data.get_label_color(cls_name)

            self.view.add_polygons_group_to_scene(cls_num, filtered_points_mass, color, alpha_tek)
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()
        else:
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()

        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def ai_mask_end_drawing(self):

        self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))
        input_box = self.view.get_sam_mask_input()

        self.view.remove_active()

        if len(input_box):
            if self.image_set and not self.image_setter.isRunning():
                self.sam_by_mask_client.set_input_box(input_box)
                if not self.sam_by_mask_client.isRunning():
                    self.sam_by_mask_client.start()

        self.view.end_drawing()
        self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

    def on_sam_by_mask_finished(self):
        points_mass = self.sam_by_mask_client.points_mass
        self.add_sam_polygon_to_scene(points_mass=points_mass)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             'primaryTextColor': '#ffffff'}

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = DetectorClient()
    w.show()
    sys.exit(app.exec_())
