from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QMovie, QPainter, QIcon, QColor, QCursor
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QAction, QFileDialog, QMessageBox, QMenu, QToolBar, QToolButton, QComboBox, QLabel, \
    QColorDialog, QListWidget

from PyQt5.QtWidgets import QApplication

from torch import cuda

from utils import help_functions as hf
from utils.sam_predictor import load_model, mask_to_seg, predict_by_points, predict_by_box
from utils import config
from utils.predictor import SAMImageSetter
from utils.project import ProjectHandler
from utils.importer import Importer
from utils.edges_from_mask import yolo8masks2points
from gd.gd_worker import GroundingSAMWorker

from ui.settings_window import SettingsWindow
from ui.ask_del_polygon import AskDelWindow
from ui.splash_screen import MovieSplashScreen
from ui.view import GraphicsView
from ui.input_dialog import CustomInputDialog, PromptInputDialog, CustomComboDialog
from ui.show_image_widget import ShowImgWindow
from ui.panels import ImagesPanel, LabelsPanel
from ui.signals_and_slots import ImagesPanelCountConnection, LabelsPanelCountConnection, ThemeChangeConnection
from ui.import_dialogs import ImportFromYOLODialog, ImportFromCOCODialog

from shapely import Polygon

import cv2
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.view = GraphicsView()
        self.setCentralWidget(self.view)

        # Signals
        self.view.polygon_clicked.id_pressed.connect(self.polygon_pressed)
        self.view.polygon_delete.id_delete.connect(self.on_polygon_delete)
        self.view.polygon_end_drawing.on_end_drawing.connect(self.on_polygon_end_draw)
        self.view.mask_end_drawing.on_mask_end_drawing.connect(self.ai_mask_end_drawing)
        self.view.polygon_cls_num_change.pol_cls_num_and_id.connect(self.change_polygon_cls_num)

        self.im_panel_count_conn = ImagesPanelCountConnection()
        self.labels_count_conn = LabelsPanelCountConnection()

        self.on_theme_change_connection = ThemeChangeConnection()

        self.start_gif(is_prog_load=True)

        screen = app.primaryScreen()
        rect = screen.availableGeometry()

        self.resize(int(0.8 * rect.width()), int(0.8 * rect.height()))

        # Установка темы оформления
        self.theme_str = 'dark_blue.xml'
        self.is_dark_theme = True

        self.setWindowTitle("AI Annotator")
        self.settings_set = False
        self.init_settings()

        # создаем меню и тулбар
        self.createActions()
        self.createMenus()
        self.createToolbar()

        # Принтер
        self.printer = QPrinter()
        self.scaleFactor = 1.0

        self.project_data = ProjectHandler()

        self.ann_type = "Polygon"
        self.loaded_proj_name = None
        self.labels_on_tek_image_ids = None
        self.image_types = ['jpg', 'png', 'tiff', 'jpeg']
        self.tek_image_name = None
        self.dataset_dir = None
        self.dataset_images = []

        self.sam_model_path = os.path.join(os.getcwd(), "sam_models\sam_vit_h_4b8939.pth")

        self.sam = load_model(self.sam_model_path, model_type="vit_h", device=self.settings_['platform'])
        self.image_setted = False
        self.image_setter = SAMImageSetter()
        self.image_setter.set_predictor(self.sam)
        self.image_setter.finished.connect(self.on_image_setted)
        self.queue_to_image_setter = []

        # GroundingDINO saved promts
        self.prompts = []

        # import settings
        self.is_seg_import = False

        self.splash.finish(self)
        self.statusBar().showMessage(
            "Загрузите проект или набор изображений" if self.settings_['lang'] == 'RU' else "Load dataset or project")

    def set_movie_gif(self):
        """
        Установка гифки на заставку
        """
        self.movie_gif = "ui/icons/15.gif"
        self.ai_gif = "ui/icons/15.gif"

    def start_gif(self, is_prog_load=False, mode="Loading"):
        self.set_movie_gif()
        if mode == "Loading":
            self.movie = QMovie(self.movie_gif)
        elif mode == "AI":
            self.movie = QMovie(self.ai_gif)
        if is_prog_load:
            self.splash = MovieSplashScreen(self.movie)
        else:
            self.splash = MovieSplashScreen(self.movie, parent_geo=self.geometry())

        self.splash.setWindowFlags(
            QtCore.Qt.WindowStaysOnTopHint | QtCore.Qt.FramelessWindowHint
        )

        self.splash.showMessage(
            "<h1><font color='red'></font></h1>",
            QtCore.Qt.AlignTop | QtCore.Qt.AlignCenter,
            QtCore.Qt.white,
        )

        self.splash.show()

    def createActions(self):
        """
        Задать действия
        """
        self.openAct = QAction("Загрузить набор изображений" if self.settings_['lang'] == 'RU' else "Load dataset",
                               self,
                               shortcut="Ctrl+O", triggered=self.open)
        self.openProjAct = QAction("Загрузить проект" if self.settings_['lang'] == 'RU' else "Load Project", self,
                                   triggered=self.open_project)
        self.saveProjAsAct = QAction(
            "Сохранить проект как..." if self.settings_['lang'] == 'RU' else "Save project as...",
            self, triggered=self.save_project_as)
        self.saveProjAct = QAction("Сохранить проект" if self.settings_['lang'] == 'RU' else "Save project", self,
                                   shortcut="Ctrl+S", triggered=self.save_project)

        self.printAct = QAction("Печать" if self.settings_['lang'] == 'RU' else "Print", self, shortcut="Ctrl+P",
                                enabled=False, triggered=self.print_)
        self.exitAct = QAction("Выход" if self.settings_['lang'] == 'RU' else "Exit", self, shortcut="Ctrl+Q",
                               triggered=self.close)
        self.zoomInAct = QAction("Увеличить" if self.settings_['lang'] == 'RU' else "Zoom In", self, shortcut="Ctrl++",
                                 enabled=False,
                                 triggered=self.zoomIn)
        self.zoomOutAct = QAction("Уменьшить" if self.settings_['lang'] == 'RU' else "Zoom Out", self,
                                  shortcut="Ctrl+-",
                                  enabled=False,
                                  triggered=self.zoomOut)

        self.fitToWindowAct = QAction(
            "Подогнать под размер окна" if self.settings_['lang'] == 'RU' else "Fit to window size",
            self, enabled=False,
            shortcut="Ctrl+F",
            triggered=self.fitToWindow)
        self.aboutAct = QAction("О модуле" if self.settings_['lang'] == 'RU' else "About", self, triggered=self.about)
        self.tutorialAct = QAction("Горячие клавиши" if self.settings_['lang'] == 'RU' else "Shortcuts", self,
                                   triggered=self.show_tutorial)

        self.settingsAct = QAction("Настройки приложения" if self.settings_['lang'] == 'RU' else "Settings", self,
                                   enabled=True, triggered=self.showSettings)

        self.balanceAct = QAction("Информация о датасете" if self.settings_['lang'] == 'RU' else "Dataset info", self,
                                  enabled=False, triggered=self.on_dataset_balance_clicked)

        # Annotators
        self.polygonAct = QAction("Полигон" if self.settings_['lang'] == 'RU' else "Polygon", self, enabled=False,
                                  triggered=self.polygon_tool_pressed, checkable=True)
        self.circleAct = QAction("Эллипс" if self.settings_['lang'] == 'RU' else "Ellips", self, enabled=False,
                                 triggered=self.circle_pressed, checkable=True)
        self.squareAct = QAction("Прямоугольник" if self.settings_['lang'] == 'RU' else "Box", self, enabled=False,
                                 triggered=self.square_pressed,
                                 checkable=True)
        self.aiAnnotatorPointsAct = QAction(
            "Сегментация по точкам" if self.settings_['lang'] == 'RU' else "SAM by points",
            self, enabled=False, shortcut="Ctrl+A",
            triggered=self.ai_points_pressed,
            checkable=True)
        self.aiAnnotatorMaskAct = QAction(
            "Сегментация внутри бокса" if self.settings_['lang'] == 'RU' else "SAM by box", self,
            enabled=False, shortcut="Ctrl+M",
            triggered=self.ai_mask_pressed,
            checkable=True)

        self.GroundingDINOSamAct = QAction(
            "GroundingDINO + SAM" if self.settings_['lang'] == 'RU' else "GroundingDINO + SAM", self,
            enabled=False, shortcut="Ctrl+G",
            triggered=self.grounding_sam_pressed,
            checkable=True)

        # Export
        self.exportAnnToYoloBoxAct = QAction(
            "YOLO (Box)" if self.settings_['lang'] == 'RU' else "YOLO (Boxes)", self,
            enabled=False,
            triggered=self.exportToYOLOBox)
        self.exportAnnToYoloSegAct = QAction(
            "YOLO (Seg)" if self.settings_['lang'] == 'RU' else "YOLO (Seg)", self,
            enabled=False,
            triggered=self.exportToYOLOSeg)
        self.exportAnnToCOCOAct = QAction(
            "COCO" if self.settings_['lang'] == 'RU' else "COCO", self, enabled=False,
            triggered=self.exportToCOCO)

        # Import
        self.importAnnFromYoloBoxAct = QAction(
            "YOLO (Box)" if self.settings_['lang'] == 'RU' else "YOLO (Boxes)", self,
            enabled=True,
            triggered=self.importFromYOLOBox)
        self.importAnnFromYoloSegAct = QAction(
            "YOLO (Seg)" if self.settings_['lang'] == 'RU' else "YOLO (Seg)", self,
            enabled=True,
            triggered=self.importFromYOLOSeg)
        self.importAnnFromCOCOAct = QAction(
            "COCO" if self.settings_['lang'] == 'RU' else "COCO", self, enabled=True,
            triggered=self.importFromCOCO)

        # Labels
        self.add_label = QAction("Добавить новый класс" if self.settings_['lang'] == 'RU' else "Add new label", self,
                                 enabled=True, triggered=self.add_label_button_clicked)
        self.del_label = QAction("Удалить текущий класс" if self.settings_['lang'] == 'RU' else "Delete current label",
                                 self,
                                 enabled=True, triggered=self.del_label_button_clicked)
        self.change_label_color = QAction(
            "Изменить цвет разметки для текущего класса" if self.settings_['lang'] == 'RU' else "Change label color",
            self,
            enabled=True,
            triggered=self.change_label_color_button_clicked)
        self.rename_label = QAction("Изменить имя класса" if self.settings_['lang'] == 'RU' else "Rename", self,
                                    enabled=True,
                                    triggered=self.rename_label_button_clicked)

        self.set_icons()

    def createMenus(self):

        """
        Создание меню
        """

        self.fileMenu = QMenu("&Файл" if self.settings_['lang'] == 'RU' else "&File", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.openProjAct)
        self.fileMenu.addAction(self.saveProjAct)
        self.fileMenu.addAction(self.saveProjAsAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
        #
        self.viewMenu = QMenu("&Изображение" if self.settings_['lang'] == 'RU' else "&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)
        #
        self.annotatorMenu = QMenu("&Аннотация" if self.settings_['lang'] == 'RU' else "&Labeling", self)

        self.AnnotatorMethodMenu = QMenu("Способ выделения" if self.settings_['lang'] == 'RU' else "Method", self)
        self.AnnotatorMethodMenu.setIcon(QIcon(self.icon_folder + "/label.png"))

        self.aiAnnotatorMethodMenu = QMenu("С помощью ИИ" if self.settings_['lang'] == 'RU' else "AI", self)
        self.aiAnnotatorMethodMenu.setIcon(QIcon(self.icon_folder + "/ai.png"))
        self.aiAnnotatorMethodMenu.setEnabled(False)

        self.AnnotatorMethodMenu.addAction(self.polygonAct)
        self.AnnotatorMethodMenu.addAction(self.squareAct)
        self.AnnotatorMethodMenu.addAction(self.circleAct)

        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorPointsAct)
        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorMaskAct)
        self.aiAnnotatorMethodMenu.addAction(self.GroundingDINOSamAct)

        self.AnnotatorMethodMenu.addMenu(self.aiAnnotatorMethodMenu)
        self.annotatorMenu.addMenu(self.AnnotatorMethodMenu)

        self.annotatorExportMenu = QMenu("Экспорт" if self.settings_['lang'] == 'RU' else "Export", self)
        self.annotatorExportMenu.addAction(self.exportAnnToYoloBoxAct)
        self.annotatorExportMenu.addAction(self.exportAnnToYoloSegAct)
        self.annotatorExportMenu.addAction(self.exportAnnToCOCOAct)
        self.annotatorExportMenu.setIcon(QIcon(self.icon_folder + "/export.png"))
        self.annotatorMenu.addMenu(self.annotatorExportMenu)

        self.annotatorImportMenu = QMenu("Импорт" if self.settings_['lang'] == 'RU' else "Import", self)
        self.annotatorImportMenu.addAction(self.importAnnFromYoloBoxAct)
        self.annotatorImportMenu.addAction(self.importAnnFromYoloSegAct)
        self.annotatorImportMenu.addAction(self.importAnnFromCOCOAct)
        self.annotatorImportMenu.setIcon(QIcon(self.icon_folder + "/import.png"))
        self.annotatorMenu.addMenu(self.annotatorImportMenu)
        self.annotatorMenu.addAction(self.balanceAct)

        #
        self.settingsMenu = QMenu("Настройки" if self.settings_['lang'] == 'RU' else "Settings", self)
        self.settingsMenu.addAction(self.settingsAct)
        #
        self.helpMenu = QMenu("&Помощь" if self.settings_['lang'] == 'RU' else "Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.tutorialAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.annotatorMenu)
        self.menuBar().addMenu(self.settingsMenu)
        self.menuBar().addMenu(self.helpMenu)

    def createToolbar(self):

        """
        Создание тулбаров
        """

        # Слева

        toolBar = QToolBar("Панель инструментов" if self.settings_['lang'] == 'RU' else "ToolBar", self)
        toolBar.addAction(self.openAct)
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
        toolBar.addAction(self.settingsAct)
        toolBar.addSeparator()
        # toolBar.addAction(self.printAct)
        # toolBar.addAction(self.exitAct)

        labelSettingsToolBar = QToolBar("Настройки разметки" if self.settings_['lang'] == 'RU' else "Current Label Bar",
                                        self)
        self.cls_combo = QComboBox()

        label = QLabel("Текущий класс:   " if self.settings_['lang'] == 'RU' else "Current label:   ")
        cls_names = np.array(['no name'])
        self.cls_combo.addItems(cls_names)
        self.cls_combo.setMinimumWidth(150)
        self.cls_combo.setEnabled(True)

        labelSettingsToolBar.addWidget(label)
        labelSettingsToolBar.addWidget(self.cls_combo)

        labelSettingsToolBar.addAction(self.change_label_color)
        labelSettingsToolBar.addAction(self.rename_label)
        labelSettingsToolBar.addAction(self.del_label)
        labelSettingsToolBar.addSeparator()
        labelSettingsToolBar.addAction(self.add_label)

        self.toolBarLeft = toolBar

        # Правая панель
        self.toolBarRight = QToolBar("Менеджер разметок" if self.settings_['lang'] == 'RU' else "Labeling Bar", self)

        # # Панель меток - заголовок
        self.toolBarRight.addWidget(LabelsPanel(self, self.break_drawing, self.icon_folder,
                                                on_color_change_signal=self.on_theme_change_connection.on_theme_change,
                                                on_labels_count_change=self.labels_count_conn.on_labels_count_change))

        # Панель меток - список
        self.labels_on_tek_image = QListWidget()
        self.labels_on_tek_image.itemClicked.connect(self.labels_on_tek_image_clicked)
        self.toolBarRight.addWidget(self.labels_on_tek_image)

        # Панель изображений - заголовок
        self.toolBarRight.addWidget(
            ImagesPanel(self, self.add_im_to_proj_clicked, self.del_im_from_proj_clicked, self.icon_folder,
                        on_color_change_signal=self.on_theme_change_connection.on_theme_change,
                        on_images_list_change=self.im_panel_count_conn.on_image_count_change))

        # Панель изображений - список
        self.images_list_widget = QListWidget()
        self.images_list_widget.itemClicked.connect(self.images_list_widget_clicked)
        self.toolBarRight.addWidget(self.images_list_widget)

        # Устанавливаем layout в правую панель

        self.addToolBar(QtCore.Qt.TopToolBarArea, labelSettingsToolBar)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBarLeft)
        self.addToolBar(QtCore.Qt.RightToolBarArea, self.toolBarRight)

    def change_polygon_cls_num(self, cls_num, cls_id):
        labels = self.project_data.get_labels()
        self.combo_dialog = CustomComboDialog(self,
                                              title_name="Изменение имени метки" if self.settings_[
                                                                                        'lang'] == 'RU' else "Label name change",
                                              question_name="Введите имя класса:" if self.settings_[
                                                                                         'lang'] == 'RU' else "Enter label name:",
                                              variants=[labels[i] for i in range(len(labels)) if i != cls_num])

        self.combo_dialog.okBtn.clicked.connect(self.change_cls_num_ok_clicked)
        self.combo_dialog.show()
        self.changed_cls_num = cls_num
        self.changed_cls_id = cls_id

    def change_cls_num_ok_clicked(self):
        new_cls_name = self.combo_dialog.getText()
        labels = self.project_data.get_labels()
        new_cls_num = 0
        for i, lbl in enumerate(labels):
            new_cls_num = i
            if lbl == new_cls_name:
                break

        # print(f'cls_num: {self.changed_cls_num} -> {new_cls_num}, id {self.changed_cls_id}')
        self.project_data.change_cls_num_by_id(self.changed_cls_id, new_cls_num)
        self.reload_image()
        self.combo_dialog.close()

    def grounding_sam_pressed(self):
        self.prompt_input_dialog = PromptInputDialog(self,
                                                     class_names=self.project_data.get_labels(),
                                                     on_ok_clicked=self.start_grounddino, prompts_variants=self.prompts)
        self.prompt_input_dialog.show()

    def start_grounddino(self):
        prompt = self.prompt_input_dialog.getPrompt()

        if prompt:

            self.prompt_cls_name = self.prompt_input_dialog.getClsName()
            self.prompt_cls_num = self.prompt_input_dialog.getClsNumber()

            if prompt not in self.prompts:
                self.prompts.append(prompt)

            config_file = os.path.join(os.getcwd(),
                                       config.PATH_TO_GROUNDING_DINO_CONFIG)  # 'D:\python\\aia_git\\ai_annotator\gd\GroundingDINO\groundingdino\config\GroundingDINO_SwinT_OGC.py'
            grounded_checkpoint = os.path.join(os.getcwd(),
                                               config.PATH_TO_GROUNDING_DINO_CHECKPOINT)  # 'D:\python\\aia_git\\ai_annotator\gd\groundingdino_swint_ogc.pth'
            # sam_checkpoint = os.path.join(os.getcwd(), config.PATH_TO_SAM_CHECKPOINT)

            self.gd_worker = GroundingSAMWorker(config_file=config_file, grounded_checkpoint=grounded_checkpoint,
                                                sam_predictor=self.sam, tek_image_path=self.tek_image_path,
                                                prompt=prompt)

            self.prompt_input_dialog.set_progress(10)

            self.gd_worker.finished.connect(self.on_gd_worker_finished)

            if not self.gd_worker.isRunning():
                self.statusBar().showMessage(
                    f"Начинаю поиск {self.prompt_cls_name} на изображении..." if self.settings_[
                                                                                     'lang'] == 'RU' else f"Start searching {self.prompt_cls_name} on image...",
                    3000)
                self.gd_worker.start()

    def on_gd_worker_finished(self):
        masks = self.gd_worker.getMasks()

        shape = self.cv2_image.shape
        for i, mask in enumerate(masks):
            self.prompt_input_dialog.set_progress(10 + int(90.0 * i / len(masks)))
            points = yolo8masks2points(mask[0] * 255, simplify_factor=3, width=shape[1], height=shape[0])
            if points:
                self.view.add_polygon_to_scene(self.prompt_cls_num, points,
                                               color=self.project_data.get_label_color(self.prompt_cls_name))

        self.write_scene_to_project_data()
        self.fill_labels_on_tek_image_list_widget()
        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())
        self.prompt_input_dialog.close()

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

        plt.savefig('temp.jpg')
        fileName = 'temp.jpg'
        ShowImgWindow(self, title='Баланс меток', img_file=fileName, icon_folder=self.icon_folder)

    def add_im_to_proj_clicked(self):

        if self.dataset_dir:
            images, _ = QFileDialog.getOpenFileNames(self,
                                                     'Загрузка изображений в проект' if self.settings_[
                                                                                            'lang'] == 'RU' else "Loading dataset",
                                                     'images',
                                                     'Images Files (*.jpeg *.png *.jpg *.tiff)')
            if images:
                for im in images:
                    im_base_name = os.path.basename(im)
                    im_new_name = os.path.join(self.dataset_dir, im_base_name)
                    if os.path.basename(im_new_name) != os.path.basename(im):
                        shutil.copy(im, im_new_name)

                    if im_base_name not in self.dataset_images:
                        self.dataset_images.append(im_base_name)

                self.fill_images_label(self.dataset_images)

        else:
            self.open()

        self.im_panel_count_conn.on_image_count_change.emit(len(self.dataset_images))

    def del_im_from_proj_clicked(self):

        if self.tek_image_name:

            current_idx = self.images_list_widget.currentRow()

            tek_im_name = self.tek_image_name
            os.remove(os.path.join(self.dataset_dir, tek_im_name))

            # del labels from project
            self.del_image_labels_from_project(tek_im_name)

            if len(self.dataset_images) == 1:
                # последнее изображение
                self.view.clearScene()
                self.dataset_dir = None
                self.dataset_images = []
                self.tek_image_name = None

            else:
                self.get_next_image_name()
                self.reload_image()
                self.dataset_images = [image for image in self.dataset_images if image != tek_im_name]

            self.fill_images_label(self.dataset_images)

            self.im_panel_count_conn.on_image_count_change.emit(len(self.dataset_images))

            self.images_list_widget.setCurrentRow(current_idx)

            self.fill_labels_on_tek_image_list_widget()

    def fill_labels_on_tek_image_list_widget(self):

        self.labels_on_tek_image.clear()

        im = self.project_data.get_image_data(self.tek_image_name)

        if im:
            shapes = im["shapes"]
            for shape in shapes:
                cls_counter = {}
                cls_num = shape["cls_num"]
                shape_id = shape["id"]
                if cls_num in cls_counter:
                    cls_counter[cls_num] += 1
                else:
                    cls_counter[cls_num] = 1

                cls_name = self.cls_combo.itemText(cls_num)

                self.labels_on_tek_image.addItem(f"{cls_name} id {shape_id}")

    def images_list_widget_clicked(self, item):
        self.write_scene_to_project_data()

        self.tek_image_name = item.text()
        self.tek_image_path = os.path.join(self.dataset_dir, self.tek_image_name)
        self.open_image(self.tek_image_path)
        self.load_image_data(self.tek_image_name)
        self.fill_labels_on_tek_image_list_widget()
        self.view.setFocus()

    def labels_on_tek_image_clicked(self, item):
        item_id = item.text().split(" ")[-1]
        self.view.activate_item_by_id(int(item_id))

    def exportToYOLOBox(self):
        self.save_project()
        export_dir = QFileDialog.getExistingDirectory(self,
                                                      'Выберите папку для сохранения разметки' if self.settings_[
                                                                                                      'lang'] == 'RU' else "Set folder",
                                                      'images')
        if export_dir:
            self.project_data.exportToYOLOBox(export_dir)
            self.on_project_export(export_format="YOLO Box")

    def exportToYOLOSeg(self):
        self.save_project()
        export_dir = QFileDialog.getExistingDirectory(self,
                                                      'Выберите папку для сохранения разметки' if self.settings_[
                                                                                                      'lang'] == 'RU' else "Set folder",
                                                      'images')
        if export_dir:
            self.project_data.exportToYOLOSeg(export_dir)
            self.on_project_export(export_format="YOLO Seg")

    def exportToCOCO(self):
        self.save_project()
        export_сoco_file, _ = QFileDialog.getSaveFileName(self,
                                                          'Выберите имя сохраняемого файла' if self.settings_[
                                                                                                   'lang'] == 'RU' else "Set export file name",
                                                          'images',
                                                          'JSON File (*.json)')

        if export_сoco_file:
            self.project_data.exportToCOCO(export_сoco_file)

            self.on_project_export(export_format="COCO")

    def importFromYOLOBox(self):

        self.import_dialog = ImportFromYOLODialog(self, on_ok_clicked=self.on_import_yolo_clicked)
        self.is_seg_import = False
        self.import_dialog.show()

    def on_import_yolo_clicked(self):
        yaml_data = self.import_dialog.getData()

        copy_images_path = None
        if yaml_data['is_copy_images']:
            copy_images_path = yaml_data['save_images_dir']

        self.importer = Importer(yaml_data=yaml_data, alpha=self.settings_['alpha'], is_seg=self.is_seg_import,
                                 copy_images_path=copy_images_path, dataset=yaml_data["selected_dataset"],
                                 is_coco=False)

        self.importer.load_percent_conn.percent.connect(self.on_import_percent_change)
        self.importer.info_conn.info_message.connect(self.on_importer_message)
        self.importer.err_conn.error_message.connect(self.on_importer_message)

        self.importer.finished.connect(self.on_import_finished)

        if not self.importer.isRunning():
            self.importer.start()

    def on_importer_message(self, message):
        self.statusBar().showMessage(message, 3000)

    def on_import_percent_change(self, persent):
        self.import_dialog.set_progress(persent)

    def importFromYOLOSeg(self):
        self.import_dialog = ImportFromYOLODialog(self, on_ok_clicked=self.on_import_yolo_clicked)
        self.is_seg_import = True
        self.import_dialog.show()

    def importFromCOCO(self):
        self.import_dialog = ImportFromCOCODialog(self, on_ok_clicked=self.on_import_coco_clicked)
        self.import_dialog.show()

    def on_import_coco_clicked(self):
        proj_data = self.import_dialog.getData()
        if proj_data:
            label_names = self.import_dialog.get_label_names()
            self.importer = Importer(coco_data=proj_data, alpha=self.settings_['alpha'], label_names=label_names,
                                     copy_images_path=self.import_dialog.get_copy_images_path(),
                                     coco_name=self.import_dialog.get_coco_name(), is_coco=True)

            self.importer.finished.connect(self.on_import_finished)
            self.importer.load_percent_conn.percent.connect(self.on_import_percent_change)

            if not self.importer.isRunning():
                self.importer.start()

    def on_import_finished(self):
        self.project_data.set_data(self.importer.get_project())

        proj_path = os.path.join(os.getcwd(), 'projects', 'saved.json')
        self.project_data.save(proj_path)
        self.load_project(proj_path)

        self.import_dialog.hide()

    def set_color_to_cls(self, cls_name):

        self.project_data.set_label_color(cls_name, alpha=self.settings_["alpha"])

    def on_project_export(self, export_format="YOLO Seg"):
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Information)
        msgbox.setText(
            f"Экспорт в формат {export_format} завершен успешно" if self.settings_[
                                                                        'lang'] == 'RU' else f"Export to {export_format} was successful")
        msgbox.setWindowTitle("Экспорт завершен")
        msgbox.exec()

    def add_new_name_to_combobox(self, new_name):
        # Обновлем список комбо-бокса
        cls_names = [self.cls_combo.itemText(i) for i in range(self.cls_combo.count())]
        new_cls = []
        for cls in cls_names:
            new_cls.append(cls)

        new_cls.append(new_name)
        self.cls_combo.clear()
        self.cls_combo.addItems(new_cls)
        self.cls_combo.setCurrentIndex(self.cls_combo.count() - 1)

    def add_label_button_clicked(self):

        self.input_dialog = CustomInputDialog(self,
                                              title_name="Добавление нового класса" if self.settings_[
                                                                                           'lang'] == 'RU' else "New label",
                                              question_name="Введите имя класса:" if self.settings_[
                                                                                         'lang'] == 'RU' else "Enter label name:")
        self.input_dialog.okBtn.clicked.connect(self.add_label_ok_clicked)
        self.input_dialog.show()
        self.input_dialog.edit.setFocus()

    def add_label_ok_clicked(self):
        new_name = self.input_dialog.getText()
        self.input_dialog.close()
        if new_name:

            cls_names = [self.cls_combo.itemText(i) for i in range(self.cls_combo.count())]

            if new_name in cls_names:
                msgbox = QMessageBox()
                msgbox.setIcon(QMessageBox.Information)
                msgbox.setText(
                    f"Ошибка добавления класса" if self.settings_['lang'] == 'RU' else "Error in setting new label")
                msgbox.setWindowTitle("Ошибка добавления класса" if self.settings_['lang'] == 'RU' else "Error")
                msgbox.setInformativeText(
                    f"Класс с именем {new_name} уже существует" if self.settings_[
                                                                       'lang'] == 'RU' else f"Label with name {new_name} is already exist. Try again")
                msgbox.exec()
                return

            self.set_labels_color()  # сохранение информации о цветах масок
            self.set_color_to_cls(new_name)  # Добавляем новый цвет в proj_data

            self.add_new_name_to_combobox(new_name)

            # Сохраняем проект

            self.fill_project_labels()
            self.write_scene_to_project_data()

            self.open_image(self.tek_image_path)
            self.load_image_data(self.tek_image_name)
            self.view.setFocus()

    def del_label_button_clicked(self):

        cls_names = [self.cls_combo.itemText(i) for i in range(self.cls_combo.count())]

        if len(cls_names) <= 1:
            msgbox = QMessageBox()
            msgbox.setIcon(QMessageBox.Information)
            msgbox.setText(f"Ошибка удаления класса" if self.settings_['lang'] == 'RU' else "Error in deleting label")
            msgbox.setWindowTitle(f"Ошибка удаления класса" if self.settings_['lang'] == 'RU' else "Error")
            msgbox.setInformativeText(
                "Количество классов должно быть хотя бы 2 для удаления текущего" if self.settings_[
                                                                                        'lang'] == 'RU' else "The last label left. If you don't like the name - just rename it")
            msgbox.exec()
            return

        self.del_index = self.cls_combo.currentIndex()  # номер удаляемого класса

        self.ask_del_label = AskDelWindow(self, cls_names, self.cls_combo.currentText())

        self.ask_del_label.okBtn.clicked.connect(self.remove_label_with_change)
        self.ask_del_label.cancelBtn.clicked.connect(self.ask_del_label.close)
        self.ask_del_label.del_all_btn.clicked.connect(self.on_ask_del_all)

        self.ask_del_label.show()

    def write_scene_to_project_data(self):
        # Добавление данных текущей сцены в proj_data

        if self.tek_image_name:
            shapes = self.view.get_all_shapes()
            image_updated = {"filename": self.tek_image_name, "shapes": shapes}
            self.project_data.set_image_data(image_updated)

    def remove_label_with_change(self):
        # 1. Сохраняем последние изменения на сцене в проект
        self.write_scene_to_project_data()

        new_name = self.ask_del_label.cls_combo.currentText()  # на что меняем
        old_name = self.cls_combo.itemText(self.del_index)

        self.ask_del_label.close()  # закрываем окно

        # 2. Убираем цвет из данных
        self.project_data.delete_label_color(old_name)

        # Берем индекс класса, на который меняем.
        change_to_idx = self.get_label_index_by_name(new_name)

        # 3. Убираем имя из комбобокса
        self.del_label_from_combobox(old_name)  # теперь в комбобоксе нет имени
        # 4. Перезаписываем данные о именах классов в проекте
        self.fill_project_labels()

        # 5. Обновляем все полигоны
        self.change_project_data_labels_from_to(self.del_index, change_to_idx)

        # 6. Обновляем панель справа
        self.fill_labels_on_tek_image_list_widget()

        # 7. Переоткрываем изображение и рисуем полигоны из проекта
        self.open_image(self.tek_image_path)
        self.load_image_data(self.tek_image_name)
        self.view.setFocus()

    def on_ask_del_all(self):
        # 1. Сохраняем данные сцены в проект
        self.write_scene_to_project_data()

        del_name = self.ask_del_label.cls_name
        del_idx = self.get_label_index_by_name(del_name)

        self.ask_del_label.close()

        # 2. Удаляем данные о цвете из проекта
        self.project_data.delete_label_color(del_name)

        # 3. Убираем имя класса из комбобокса
        self.del_label_from_combobox(del_name)
        # 4. Перезаписываем данные о именах классов в проекте
        self.fill_project_labels()

        # 5. Обновляем все полигоны
        self.del_labels_from_project(del_idx)

        # 6. Обновляем панель справа
        self.fill_labels_on_tek_image_list_widget()

        # 7. Переоткрываем изображение и рисуем полигоны из проекта
        self.open_image(self.tek_image_path)
        self.load_image_data(self.tek_image_name)

        self.view.setFocus()

    def del_label_from_combobox(self, label):
        cls_names = [self.cls_combo.itemText(i) for i in range(self.cls_combo.count())]

        labels = []
        for name in cls_names:
            if name != label:
                labels.append(name)

        self.cls_combo.clear()
        self.cls_combo.addItems(labels)

    def get_label_index_by_name(self, label_name):
        cls_names = [self.cls_combo.itemText(i) for i in range(self.cls_combo.count())]
        for i, name in enumerate(cls_names):
            if name == label_name:
                return i

        return -1

    def del_labels_from_project(self, del_index):

        self.project_data.set_labels([self.cls_combo.itemText(i) for i in range(self.cls_combo.count())])
        self.project_data.delete_data_by_class_number(del_index)

    def del_image_labels_from_project(self, image_name):

        self.project_data.set_labels([self.cls_combo.itemText(i) for i in range(self.cls_combo.count())])

        self.project_data.del_image(image_name)

    def change_project_data_labels_from_to(self, from_index, to_index):

        self.write_scene_to_project_data()

        self.project_data.set_labels([self.cls_combo.itemText(i) for i in range(self.cls_combo.count())])
        self.project_data.change_images_class_from_to(from_index, to_index)

    def change_label_color_button_clicked(self):

        self.write_scene_to_project_data()

        color_dialog = QColorDialog()
        cls_txt = self.cls_combo.currentText()
        color_dialog.setWindowTitle(
            f"Выберите цвет для класса {cls_txt}" if self.settings_[
                                                         'lang'] == 'RU' else f"Enter color to label {cls_txt}")
        current_color = self.project_data.get_label_color(cls_txt)
        if not current_color:
            current_color = config.COLORS[self.cls_combo.currentIndex()]

        color_dialog.setCurrentColor(QColor(*current_color))
        color_dialog.setWindowIcon(QIcon(self.icon_folder + "/color.png"))
        color_dialog.exec()
        rgb = color_dialog.selectedColor().getRgb()
        rgba = (rgb[0], rgb[1], rgb[2], self.settings_["alpha"])

        self.project_data.set_label_color(cls_txt, color=rgba)

        self.open_image(self.tek_image_path)
        self.load_image_data(self.tek_image_name)

        self.view.setFocus()

    def rename_label_button_clicked(self):

        self.write_scene_to_project_data()

        cls_name = self.cls_combo.currentText()
        self.input_dialog = CustomInputDialog(self,
                                              title_name=f"Редактирование имени класса {cls_name}" if self.settings_[
                                                                                                          'lang'] == 'RU' else f"Rename label {cls_name}",
                                              question_name="Введите новое имя класса:" if self.settings_[
                                                                                               'lang'] == 'RU' else "Enter new label name:")

        self.input_dialog.okBtn.clicked.connect(self.rename_label_ok_clicked)
        self.input_dialog.show()
        self.input_dialog.edit.setFocus()

    def rename_label_ok_clicked(self):

        new_name = self.input_dialog.getText()
        self.input_dialog.close()

        if new_name:

            tek_idx = self.cls_combo.currentIndex()
            cls_name = self.cls_combo.currentText()

            cls_names = [self.cls_combo.itemText(i) for i in range(self.cls_combo.count())]

            new_cls = []
            for i, cls in enumerate(cls_names):
                if i == tek_idx:
                    new_cls.append(new_name)
                else:
                    new_cls.append(cls_names[i])
            self.cls_combo.clear()
            self.cls_combo.addItems(new_cls)
            self.project_data.rename_color(cls_name, new_name)

            self.fill_labels_on_tek_image_list_widget()

        self.view.setFocus()

    def set_icons(self):
        """
        Задать иконки
        """

        theme_type = self.theme_str.split('.')[0]

        self.icon_folder = "ui/icons/" + theme_type

        self.setWindowIcon(QIcon(self.icon_folder + "/neural.png"))
        self.openAct.setIcon(QIcon(self.icon_folder + "/folder.png"))
        self.printAct.setIcon(QIcon(self.icon_folder + "/printer.png"))
        self.exitAct.setIcon(QIcon(self.icon_folder + "/logout.png"))
        self.zoomInAct.setIcon(QIcon(self.icon_folder + "/zoom-in.png"))
        self.zoomOutAct.setIcon(QIcon(self.icon_folder + "/zoom-out.png"))
        self.fitToWindowAct.setIcon(QIcon(self.icon_folder + "/fit.png"))
        self.aboutAct.setIcon(QIcon(self.icon_folder + "/info.png"))
        self.settingsAct.setIcon(QIcon(self.icon_folder + "/settings.png"))

        # save load
        self.openProjAct.setIcon(QIcon(self.icon_folder + "/load.png"))
        self.saveProjAsAct.setIcon(QIcon(self.icon_folder + "/save_project.png"))
        self.saveProjAct.setIcon(QIcon(self.icon_folder + "/save.png"))

        # export
        self.exportAnnToYoloBoxAct.setIcon(QIcon(self.icon_folder + "/yolo.png"))
        self.exportAnnToYoloSegAct.setIcon(QIcon(self.icon_folder + "/yolo_white.png"))
        self.exportAnnToCOCOAct.setIcon(QIcon(self.icon_folder + "/coco.png"))

        # import
        self.importAnnFromYoloBoxAct.setIcon(QIcon(self.icon_folder + "/yolo.png"))
        self.importAnnFromYoloSegAct.setIcon(QIcon(self.icon_folder + "/yolo_white.png"))
        self.importAnnFromCOCOAct.setIcon(QIcon(self.icon_folder + "/coco.png"))

        # tutorial
        self.tutorialAct.setIcon(QIcon(self.icon_folder + "/keyboard.png"))

        self.polygonAct.setIcon(QIcon(self.icon_folder + "/polygon.png"))
        self.circleAct.setIcon(QIcon(self.icon_folder + "/circle.png"))
        self.squareAct.setIcon(QIcon(self.icon_folder + "/square.png"))
        self.aiAnnotatorPointsAct.setIcon(QIcon(self.icon_folder + "/mouse.png"))
        self.aiAnnotatorMaskAct.setIcon(QIcon(self.icon_folder + "/ai_select.png"))
        self.GroundingDINOSamAct.setIcon(QIcon(self.icon_folder + "/dino.png"))

        self.balanceAct.setIcon(QIcon(self.icon_folder + "/bar-chart.png"))

        # labeling
        self.add_label.setIcon((QIcon(self.icon_folder + "/add.png")))
        self.del_label.setIcon((QIcon(self.icon_folder + "/del.png")))
        self.change_label_color.setIcon((QIcon(self.icon_folder + "/color.png")))
        self.rename_label.setIcon((QIcon(self.icon_folder + "/rename.png")))

    def open_image(self, image_name):

        self.view.setPixmap(QtGui.QPixmap(image_name))
        self.view.fitInView(self.view.pixmap_item, QtCore.Qt.KeepAspectRatio)
        self.printAct.setEnabled(True)

        self.polygonAct.setEnabled(True)
        self.circleAct.setEnabled(True)
        self.squareAct.setEnabled(True)

        self.aiAnnotatorPointsAct.setEnabled(True)
        self.aiAnnotatorMaskAct.setEnabled(True)
        self.aiAnnotatorMethodMenu.setEnabled(True)
        self.GroundingDINOSamAct.setEnabled(True)

        self.exportAnnToYoloBoxAct.setEnabled(True)
        self.exportAnnToYoloSegAct.setEnabled(True)
        self.exportAnnToCOCOAct.setEnabled(True)

        image = cv2.imread(image_name)
        self.cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.image_setted = False
        self.image_setter.set_image(self.cv2_image)

        if not self.image_setter.isRunning():
            self.statusBar().showMessage(
                "Начинаю загружать изображение в нейросеть SAM..." if self.settings_[
                                                                          'lang'] == 'RU' else "Start loading image to SAM...",
                3000)
            self.image_setter.start()
        else:
            image_copy = cv2.imread(image_name)
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
            self.queue_to_image_setter.append(image_copy)

            self.statusBar().showMessage(
                f"Изображение {os.path.split(image_name)[-1]} добавлено в очередь на обработку." if self.settings_[
                                                                                                        'lang'] == 'RU' else f"Image {os.path.split(image_name)[-1]} is added to queue...",
                3000)

    def on_image_setted(self):
        if len(self.queue_to_image_setter) != 0:
            self.cv2_image = self.queue_to_image_setter[-1]
            self.image_setted = False
            self.image_setter.set_image(self.cv2_image)
            self.queue_to_image_setter = []
            self.statusBar().showMessage(
                "Нейросеть SAM еще не готова. Подождите секунду..." if self.settings_[
                                                                           'lang'] == 'RU' else "SAM is loading. Please wait...",
                3000)
            self.image_setter.start()

        else:
            self.statusBar().showMessage(
                "Нейросеть SAM готова к сегментации" if self.settings_['lang'] == 'RU' else "SAM ready to work", 3000)
            self.image_setted = True

    def print_(self):
        """
        Вывод изображения на печать
        """
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.view.pixmap_item.pixmap().size()
            size.scale(rect.size(), QtCore.Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.view.pixmap_item.pixmap().rect())
            painter.drawPixmap(0, 0, self.view.pixmap_item.pixmap())

    def about(self):
        """
        Окно о приложении
        """
        QMessageBox.about(self, "AI Annotator",
                          "<p><b>AI Annotator</b></p>"
                          "<p>Программа для разметки изображений с поддержкой автоматической сегментации</p>" if
                          self.settings_[
                              'lang'] == 'RU' else "<p>Labeling Data for Object Detection and Instance Segmentation with The Segment Anything Model (SAM).</p>")

    def filter_images_names(self, file_names):
        im_names_valid = []
        for f in file_names:
            is_valid = False
            for t in self.image_types:
                if is_valid:
                    break
                if f.endswith('.' + t):
                    is_valid = True
                    im_names_valid.append(f)
        return im_names_valid

    def open(self):

        dataset_dir = QFileDialog.getExistingDirectory(self,
                                                       'Выберите папку с изображениями для разметки' if self.settings_[
                                                                                                            'lang'] == 'RU' else "Set dataset folder",
                                                       'images')

        if dataset_dir:
            self.dataset_images = self.filter_images_names(os.listdir(dataset_dir))

            if self.dataset_images:
                self.start_gif(mode="Loading")

                self.dataset_dir = dataset_dir
                self.project_data.set_path_to_images(dataset_dir)

                self.tek_image_name = self.dataset_images[0]
                self.tek_image_path = os.path.join(self.dataset_dir, self.tek_image_name)

                self.open_image(self.tek_image_path)

                self.fitToWindowAct.setEnabled(True)
                self.zoomInAct.setEnabled(True)
                self.zoomOutAct.setEnabled(True)
                self.balanceAct.setEnabled(True)

                main_geom = self.geometry().getCoords()
                self.scaleFactor = (main_geom[2] - main_geom[0]) / self.cv2_image.shape[1]

                self.fill_labels_on_tek_image_list_widget()

                self.fill_images_label(self.dataset_images)

                self.splash.finish(self)

                self.im_panel_count_conn.on_image_count_change.emit(len(self.dataset_images))
                self.images_list_widget.setCurrentRow(0)

                self.statusBar().showMessage(
                    f"Число загруженны в проект изображений: {len(self.dataset_images)}" if self.settings_[
                                                                                                'lang'] == 'RU' else f"Loaded images count: {len(self.dataset_images)}",
                    3000)
            else:
                self.statusBar().showMessage(
                    f"В указанной папке изображений не обнаружено" if self.settings_[
                                                                          'lang'] == 'RU' else "Folder is empty",
                    3000)

    def fill_images_label(self, image_names):

        self.images_list_widget.clear()
        for name in image_names:
            self.images_list_widget.addItem(name)

    def load_project(self, project_name):
        self.loaded_proj_name = project_name
        is_success = self.project_data.load(self.loaded_proj_name)
        self.view.set_ids_from_project(self.project_data.get_data())
        self.balanceAct.setEnabled(True)

        if not is_success:
            msgbox = QMessageBox()
            msgbox.setIcon(QMessageBox.Information)
            msgbox.setText(
                f"Ошибка открытия файла {self.loaded_proj_name}" if self.settings_[
                                                                        'lang'] == 'RU' else f"Error in opening file {self.loaded_proj_name}")
            if self.settings_['lang'] == 'RU':
                msgbox.setInformativeText(
                    f"Файл {self.loaded_proj_name} должен быть в формате .json и содержать поля:\n\t"
                    f"path_to_images\n\timages\n\t\tfilename\n\t\tshapes\n\t")
            else:
                msgbox.setInformativeText(
                    f"File {self.loaded_proj_name} not in project format")
            msgbox.setWindowTitle(
                f"Ошибка открытия файла {self.loaded_proj_name}" if self.settings_['lang'] == 'RU' else "Error")
            msgbox.exec()

            return

        dataset_dir = self.project_data.get_image_path()

        if not os.path.exists(dataset_dir):
            msgbox = QMessageBox()
            msgbox.setIcon(QMessageBox.Information)
            msgbox.setText(
                f"Ошибка загрузки проекта. " if self.settings_['lang'] == 'RU' else f"Error in loading project")
            if self.settings_['lang'] == 'RU':
                msgbox.setInformativeText(
                    f"Директория {dataset_dir} не существует"
                    )
            else:
                msgbox.setInformativeText(
                    f"Directory {dataset_dir} doesn't exist")
            msgbox.setWindowTitle(
                f"Ошибка загрузки проекта {self.loaded_proj_name}" if self.settings_['lang'] == 'RU' else "Error")
            msgbox.exec()

            return

        self.dataset_dir = dataset_dir
        self.dataset_images = self.filter_images_names(os.listdir(self.dataset_dir))

        self.fill_labels_combo_from_project()

        if self.dataset_images:
            self.tek_image_name = self.dataset_images[0]
            self.tek_image_path = os.path.join(self.dataset_dir, self.tek_image_name)

            # Открытие изображения
            self.open_image(self.tek_image_path)

            self.fitToWindowAct.setEnabled(True)
            self.zoomInAct.setEnabled(True)
            self.zoomOutAct.setEnabled(True)

            main_geom = self.geometry().getCoords()
            self.scaleFactor = (main_geom[2] - main_geom[0]) / self.cv2_image.shape[1]

            self.load_image_data(self.tek_image_name)
            self.view.setMouseTracking(True)  # Starange
            self.fill_labels_on_tek_image_list_widget()
            self.fill_images_label(self.dataset_images)

            self.im_panel_count_conn.on_image_count_change.emit(len(self.dataset_images))
            self.images_list_widget.setCurrentRow(0)

        self.statusBar().showMessage(
            f"Число загруженных в проект изображений: {len(self.dataset_images)}" if self.settings_[
                                                                                         'lang'] == 'RU' else f"Loaded images count: {len(self.dataset_images)}",
            3000)

    def open_project(self):
        """
        Загрузка проекта
        """
        loaded_proj_name, _ = QFileDialog.getOpenFileName(self,
                                                          'Загрузка проекта' if self.settings_[
                                                                                    'lang'] == 'RU' else "Loading project",
                                                          'projects',
                                                          'JSON Proj File (*.json)')

        if loaded_proj_name:
            self.load_project(loaded_proj_name)

    def on_polygon_end_draw(self, is_end_draw):
        if is_end_draw:
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()

    def fill_project_labels(self):
        self.project_data.set_labels([self.cls_combo.itemText(i) for i in range(self.cls_combo.count())])

    def fill_labels_combo_from_project(self):
        self.cls_combo.clear()
        self.cls_combo.addItems(np.array(self.project_data.get_labels()))

    def save_project(self):
        """
        Сохранение проекта
        """
        if self.loaded_proj_name:

            self.set_labels_color()  # сохранение информации о цветах масок
            self.write_scene_to_project_data()

            self.project_data.save(self.loaded_proj_name)

            self.fill_project_labels()

            self.statusBar().showMessage(
                f"Проект успешно сохранен" if self.settings_['lang'] == 'RU' else "Project is saved", 3000)

        else:
            self.save_project_as()

    def save_project_as(self):
        """
        Сохранение проекта как...
        """
        proj_name, _ = QFileDialog.getSaveFileName(self, 'QFileDialog.getOpenFileName()', 'projects',
                                                   'JSON Proj File (*.json)')

        if proj_name:
            self.loaded_proj_name = proj_name
            self.set_labels_color()  # сохранение информации о цветах масок
            self.write_scene_to_project_data()
            self.fill_project_labels()

            self.project_data.save(proj_name)

            self.loaded_proj_name = proj_name

            self.statusBar().showMessage(
                f"Проект успешно сохранен" if self.settings_['lang'] == 'RU' else "Project is saved", 3000)

    def zoomIn(self):
        """
        Увеличить на 25%
        """
        self.scaleImage(factor=1.1)

    def zoomOut(self):
        """
        Уменьшить
        """
        self.scaleImage(factor=0.9)

    def scaleImage(self, factor=1.0):
        """
        Масштабировать картинку
        """
        self.scaleFactor *= factor
        self.view.scale(factor, factor)
        self.view.setVerticalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.view.setHorizontalScrollBarPolicy(QtCore.Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        self.zoomInAct.setEnabled(self.scaleFactor < 3.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.1)

    def fitToWindow(self):
        """
        Подогнать под экран
        """
        self.view.fitInView(self.view.pixmap_item, QtCore.Qt.KeepAspectRatio)

    def showSettings(self):
        """
        Показать осно с настройками приложения
        """

        if self.settings_set:
            self.settings_window = SettingsWindow(self, self.settings_)
        else:
            self.settings_window = SettingsWindow(self)

        self.settings_window.okBtn.clicked.connect(self.on_settings_closed)
        self.settings_window.cancelBtn.clicked.connect(self.on_settings_closed)

        self.settings_window.show()

    def init_settings(self):
        """
        Инициализация настроек приложения
        """
        self.settings_ = {}
        self.settings_["theme"] = self.theme_str

        self.settings_["alpha"] = 50
        self.settings_["fat_width"] = 50
        self.settings_['lang'] = config.LANGUAGE

        if cuda.is_available():
            print("CUDA is available")
            if self.settings_['lang'] == 'RU':
                self.statusBar().showMessage(
                    "Найдено устройство NVIDIA CUDA. Нейросеть будет использовать ее для ускорения", 3000)
            else:
                self.statusBar().showMessage(
                    "NVIDIA CUDA is found. SAM will use it for acceleration", 3000)
            self.settings_['platform'] = "cuda"
        else:
            if self.settings_['lang'] == 'RU':
                self.statusBar().showMessage(
                    "Не найдено устройство NVIDIA CUDA. Нейросеть будет использовать ресурсы процессора", 3000)
            else:
                self.statusBar().showMessage(
                    "Cant't find NVIDIA CUDA. SAM will use CPU", 3000)

            self.settings_['platform'] = "cpu"

        self.settings_["labels_color"] = {}

        self.settings_['density'] = config.DENSITY_SCALE

        self.settings_set = True

    def change_theme(self, theme_str):
        """
        Изменение темы приложения
        """
        app = QApplication.instance()

        primary_color = "#ffffff"
        if 'light' in theme_str:
            primary_color = "#000000"

        density = hf.density_slider_to_value(self.settings_["density"])
        # print(f'Density: {self.settings_["density"]} -> {density}')

        extra = {'density_scale': density,
                 # 'font_size': '14px',
                 'primaryTextColor': primary_color}

        apply_stylesheet(app, theme=theme_str, extra=extra)

        theme_str = self.settings_['theme']
        theme_type = theme_str.split('.')[0]

        icon_folder = "ui/icons/" + theme_type

        self.on_theme_change_connection.on_theme_change.emit(icon_folder)

    def on_settings_closed(self):
        """
        При закрытии окна настроек приложения
        Осуществляет сохранение настроек
        """

        if len(self.settings_window.settings) != 0:
            density_old = self.settings_['density']
            lang_old = self.settings_['lang']

            self.settings_ = self.settings_window.settings
            self.settings_set = True

            str_settings = "Настройки сохранены.\n" if self.settings_['lang'] == 'RU' else "Setting is saved\n"

            if self.settings_['platform'] == 'Auto':
                if cuda.is_available():
                    if self.settings_['lang'] == 'RU':
                        self.statusBar().showMessage(
                            "Найдено устройство NVIDIA CUDA. Нейросеть будет использовать ее для ускорения", 3000)
                    else:
                        self.statusBar().showMessage(
                            "NVIDIA CUDA is found. SAM will use it for acceleration", 3000)

                    if self.settings_['platform'] == 'cpu':
                        self.settings_['platform'] = "cuda"
                        self.sam = load_model(self.sam_model_path, model_type="vit_h",
                                              device=self.settings_['platform'])
                else:
                    if self.settings_['lang'] == 'RU':
                        self.statusBar().showMessage(
                            "Не найдено устройство NVIDIA CUDA. Нейросеть будет использовать ресурсы процессора", 3000)
                    else:
                        self.statusBar().showMessage(
                            "Cant't find NVIDIA CUDA. SAM will use CPU", 3000)

                    if self.settings_['platform'] == 'cuda':
                        self.settings_['platform'] = "cpu"
                        self.sam = load_model(self.sam_model_path, model_type="vit_h",
                                              device=self.settings_['platform'])

            if self.settings_['theme'] != self.theme_str or self.settings_['density'] != density_old or lang_old != \
                    self.settings_['lang']:
                self.statusBar().showMessage(
                    "Тема приложения изменена" if self.settings_['lang'] == 'RU' else "Theme is changed", 3000)
                self.theme_str = self.settings_['theme']
                self.change_theme(self.settings_['theme'])
                self.set_icons()

            self.settings_["alpha"] = self.settings_window.settings["alpha"]
            self.settings_["fat_width"] = self.settings_window.settings["fat_width"]
            self.view.set_fat_width(self.settings_["fat_width"])

            QMessageBox.about(self, "Сохранение настроек приложения" if self.settings_['lang'] == 'RU' else "Settings",
                              str_settings)

            self.statusBar().showMessage(
                f"Настройки проекта изменены" if self.settings_['lang'] == 'RU' else "Settings is saved", 3000)

    def ai_points_pressed(self):
        self.ann_type = "AiPoints"
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings_["alpha"]

        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def ai_mask_pressed(self):
        self.ann_type = "AiMask"
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings_["alpha"]
        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def polygon_tool_pressed(self):

        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings_["alpha"]
        self.ann_type = "Polygon"
        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def polygon_pressed(self, pressed_id):
        for i in range(self.labels_on_tek_image.count()):
            item = self.labels_on_tek_image.item(i)
            item_id = item.text().split(" ")[-1]
            if int(item_id) == pressed_id:
                self.labels_on_tek_image.setCurrentItem(item)
                break

    def on_polygon_delete(self, delete_id):
        self.write_scene_to_project_data()
        self.fill_labels_on_tek_image_list_widget()

    def square_pressed(self):
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings_["alpha"]
        self.ann_type = "Box"
        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def circle_pressed(self):
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data["labels_color"][cls_txt]

        alpha_tek = self.settings_["alpha"]
        self.ann_type = "Ellips"
        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def ann_triggered(self, ann):
        self.annotatorToolButton.setDefaultAction(ann)

    def set_labels_color(self):

        labels = [self.cls_combo.itemText(i) for i in range(self.cls_combo.count())]
        self.project_data.set_labels_colors(labels)

    def load_image_data(self, image_name):
        # Проверка наличия записи о цветах полигонов

        im = self.project_data.get_image_data(image_name)
        if im:
            for shape in im["shapes"]:
                cls_num = shape["cls_num"]
                cls_name = self.cls_combo.itemText(cls_num)
                points = shape["points"]
                alpha_tek = self.settings_["alpha"]

                color = self.project_data.get_label_color(cls_name)

                self.view.add_polygon_to_scene(cls_num, points, color=color, alpha=alpha_tek, id=shape["id"])

    def get_next_image_name(self):

        if self.tek_image_name:

            if len(self.dataset_images) == 1:
                return self.tek_image_name

            id_tek = 0
            for i, name in enumerate(self.dataset_images):
                if name == self.tek_image_name:
                    id_tek = i
                    break

            if id_tek == len(self.dataset_images) - 1:
                # last image
                id_tek = 0
            else:
                id_tek += 1

            self.tek_image_name = self.dataset_images[id_tek]
            self.tek_image_path = os.path.join(self.dataset_dir, self.tek_image_name)

            return self.dataset_images[id_tek]

    def get_before_image_name(self):

        if self.tek_image_name:

            if len(self.dataset_images) == 1:
                return self.tek_image_name

            id_tek = 0
            for i, name in enumerate(self.dataset_images):
                if name == self.tek_image_name:
                    id_tek = i
                    break
            if id_tek == 0:
                # first image
                id_tek = len(self.dataset_images) - 1
            else:
                id_tek -= 1

            self.tek_image_name = self.dataset_images[id_tek]
            self.tek_image_path = os.path.join(self.dataset_dir, self.tek_image_name)
            return self.dataset_images[id_tek]

    def add_sam_polygon_to_scene(self, sam_mask, id):
        points_mass = mask_to_seg(sam_mask)

        if points_mass:
            shapely_pol = Polygon(points_mass)
            area = shapely_pol.area
            if area > config.POLYGON_AREA_THRESHOLD:

                cls_num = self.cls_combo.currentIndex()
                cls_name = self.cls_combo.itemText(cls_num)
                alpha_tek = self.settings_["alpha"]
                color = self.project_data.get_label_color(cls_name)

                self.view.add_polygon_to_scene(cls_num, points_mass, color, alpha_tek, id=id)
                self.write_scene_to_project_data()
                self.fill_labels_on_tek_image_list_widget()
            else:
                if self.settings_['lang'] == 'RU':
                    self.statusBar().showMessage(
                        f"Метку сделать не удалось. Площадь маски слишком мала {area:0.3f}. Попробуйте еще раз", 3000)
                else:
                    self.statusBar().showMessage(
                        f"Can't create label. Area of label is too small {area:0.3f}. Try again", 3000)

                self.view.remove_label_id(id)
                self.write_scene_to_project_data()
                self.fill_labels_on_tek_image_list_widget()
        else:
            self.view.remove_label_id(id)
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()

        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def ai_mask_end_drawing(self):

        self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))
        input_box = self.view.get_sam_mask_input()

        self.view.remove_active()

        if len(input_box):
            if self.image_setted and not self.image_setter.isRunning():
                mask = predict_by_box(self.sam, input_box)
                self.add_sam_polygon_to_scene(mask, self.view.get_unique_label_id())

        self.view.end_drawing()
        self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

    def end_drawing(self):

        if not self.image_setted:
            return

        if self.ann_type in ["Polygon", "Box", "Ellips"]:
            self.view.end_drawing()  # save it to

        elif self.ann_type == "AiPoints":

            self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))

            input_point, input_label = self.view.get_sam_input_points_and_labels()

            if len(input_label):
                if self.image_setted and not self.image_setter.isRunning():
                    mask = predict_by_points(self.sam, input_point, input_label)
                    self.add_sam_polygon_to_scene(mask, id=self.view.get_unique_label_id())

            else:
                self.view.remove_active()

            self.view.clear_ai_points()
            self.view.end_drawing()

            self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

        self.write_scene_to_project_data()
        self.fill_labels_on_tek_image_list_widget()

        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def start_drawing(self):
        if not self.image_setted:
            return

        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings_["alpha"]

        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)
        self.view.clear_ai_points()

    def break_drawing(self):
        if not self.image_setted:
            return

        if self.ann_type == "AiPoints":
            self.view.clear_ai_points()
            self.view.remove_active()

        if not "Continue" in self.view.drag_mode:
            self.view.remove_active()

        if self.tek_image_name:
            self.view.end_drawing()
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()

        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def reload_image(self):
        self.open_image(self.tek_image_path)
        self.load_image_data(self.tek_image_name)
        self.fill_labels_on_tek_image_list_widget()
        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())
        self.view.clear_ai_points()

    def show_tutorial(self):
        path_to_png = os.path.join(os.getcwd(), 'ui', 'tutorial', 'shortcuts.png')
        print(path_to_png)
        self.tutorial = ShowImgWindow(self, title='Горячие клавиши', img_file=path_to_png, icon_folder=self.icon_folder,
                                      is_fit_button=False)
        self.tutorial.scaleImage(0.4)
        self.tutorial.show()

    def keyPressEvent(self, e):
        # e.accept()
        # print(e.key())
        modifierPressed = QApplication.keyboardModifiers()
        modifierName = ''
        # if (modifierPressed & QtCore.Qt.AltModifier) == QtCore.Qt.AltModifier:
        #     modifierName += 'Alt'

        if (modifierPressed & QtCore.Qt.ControlModifier) == QtCore.Qt.ControlModifier:
            modifierName += 'Ctrl'

        if e.key() == 83 or e.key() == 1067:  # start poly
            # S
            self.start_drawing()

        elif e.key() == 32:  # end poly
            # Space
            self.end_drawing()

        elif e.key() == 44 or e.key() == 1041:

            if not self.image_setted:
                return

            # <<< Before image
            self.write_scene_to_project_data()

            current_idx = self.images_list_widget.currentRow()
            next_idx = current_idx - 1 if current_idx > 0 else self.images_list_widget.count() - 1
            if self.get_before_image_name():
                self.reload_image()
                self.images_list_widget.setCurrentRow(next_idx)

        elif e.key() == 46 or e.key() == 1070:

            if not self.image_setted:
                return
            # >>> Next image
            self.write_scene_to_project_data()

            current_idx = self.images_list_widget.currentRow()
            next_idx = current_idx + 1 if current_idx < self.images_list_widget.count() - 1 else 0
            if self.get_next_image_name():
                self.reload_image()
                self.images_list_widget.setCurrentRow(next_idx)

        elif e.key() == 68 or e.key() == 1042:
            # D

            self.break_drawing()

        elif (e.key() == 67 or e.key() == 1057) and 'Ctrl' in modifierName:
            # Ctrl + C

            self.view.copy_active_item_to_buffer()
            # self.write_scene_to_project_data()

        elif (e.key() == 86 or e.key() == 1052) and 'Ctrl' in modifierName:
            # Ctrl + V
            if not self.image_setted:
                return

            self.view.paste_buffer()
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()
            self.view.clear_ai_points()


if __name__ == '__main__':
    import sys
    from qt_material import apply_stylesheet

    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             'primaryTextColor': '#ffffff'}

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
