import json
import os
import sys

import matplotlib.pyplot as plt
import requests
from PyQt5 import QtCore
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QAction, QMessageBox, QMenu
from PyQt5.QtWidgets import QToolBar, QToolButton
from qt_material import apply_stylesheet

import utils.help_functions as hf
from ui.base_window import MainWindow
from ui.settings_window_client import SettingsWindowClient
from ui.show_image_widget import ShowImgWindow
from ui.edit_with_button import EditWithButton

from utils import cls_settings
from utils import config
from utils.cnn_worker_client import CNNWorkerClient


class DetectorClient(MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Detector")

        # Detector
        self.started_cnn = None

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

    def toggle_act(self, is_active):
        super(DetectorClient, self).toggle_act(is_active)
        self.balanceAct.setEnabled(is_active)
        self.detectAct.setEnabled(is_active)
        self.detectScanAct.setEnabled(is_active)
        self.exportToESRIAct.setEnabled(is_active)

        self.syncLabelsAct.setEnabled(is_active)

    def createMenus(self):
        super(DetectorClient, self).createMenus()

        self.classifierMenu = QMenu("Классификатор" if self.settings.read_lang() == 'RU' else "Classifier", self)
        self.classifierMenu.addAction(self.detectAct)
        self.classifierMenu.addAction(self.detectScanAct)
        self.classifierMenu.addAction(self.syncLabelsAct)

        self.annotatorMenu.addAction(self.balanceAct)

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

        # classifier
        self.detectAct.setIcon(QIcon(self.icon_folder + "/detect_all.png"))
        self.detectScanAct.setIcon(QIcon(self.icon_folder + "/slide.png"))

    def open_image(self, image_name):
        super(DetectorClient, self).open_image(image_name)
        lrm = hf.try_read_lrm(image_name)
        if lrm:
            self.lrm = lrm
            print(self.lrm)

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


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             'primaryTextColor': '#ffffff'}

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = DetectorClient()
    w.show()
    sys.exit(app.exec_())
