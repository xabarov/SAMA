import os
import cv2

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QMovie, QPainter, QIcon, QColor, QCursor
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QAction, QFileDialog, QMessageBox, QMenu, QToolBar, QToolButton, QComboBox, QLabel, \
    QColorDialog, QListWidget

from PyQt5.QtWidgets import QApplication

from torch import cuda

import numpy as np
import json
import datetime

from utils import help_functions as hf
from utils.sam_predictor import load_model, mask_to_seg, predict_by_points, predict_by_box, predictor_set_image
from utils import config
from utils.predictor import SAMImageSetter

from ui.settings_window import SettingsWindow
from ui.ask_del_polygon import AskDelWindow
from ui.splash_screen import MovieSplashScreen
from ui.view import GraphicsView
from ui.input_dialog import CustomInputDialog
from ui.tutorial_window import Tutorial
from ui.panels import ImagesPanel, LabelsPanel
from utils.project import ProjectHandler
from ui.signals_and_slots import ThemeChangeConnection

from shapely import Polygon
import shutil


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

        self.on_theme_change_connection = ThemeChangeConnection()

        self.start_gif(is_prog_load=True)

        screen = app.primaryScreen()
        rect = screen.availableGeometry()

        self.resize(int(0.8 * rect.width()), int(0.8 * rect.height()))

        # Установка темы оформления
        self.theme_str = 'dark_blue.xml'
        self.is_dark_theme = True

        self.setWindowTitle("AI Annotator")

        # создаем меню и тулбар
        self.createActions()
        self.createMenus()
        self.createToolbar()

        # Принтер
        self.printer = QPrinter()
        self.scaleFactor = 1.0
        self.settings_set = False
        self.init_settings()
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

        self.splash.finish(self)
        self.statusBar().showMessage("Загрузите проект или набор изображений")

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
        self.openAct = QAction("Загрузить набор изображений", self, shortcut="Ctrl+O", triggered=self.open)
        self.openProjAct = QAction("Загрузить проект", self, triggered=self.open_project)
        self.saveProjAsAct = QAction("Сохранить проект как...", self, triggered=self.save_project_as)
        self.saveProjAct = QAction("Сохранить проект", self, shortcut="Ctrl+S", triggered=self.save_project)

        self.printAct = QAction("Печать...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
        self.exitAct = QAction("Выход", self, shortcut="Ctrl+Q", triggered=self.close)
        self.zoomInAct = QAction("Увеличить на (25%)", self, shortcut="Ctrl++", enabled=False,
                                 triggered=self.zoomIn)
        self.zoomOutAct = QAction("Уменьшить на (25%)", self, shortcut="Ctrl+-", enabled=False,
                                  triggered=self.zoomOut)

        self.fitToWindowAct = QAction("Подогнать под размер окна", self, enabled=False,
                                      shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)
        self.aboutAct = QAction("О модуле", self, triggered=self.about)
        self.tutorialAct = QAction("Горячие клавиши", self, triggered=self.show_tutorial)

        self.settingsAct = QAction("Настройки приложения", self, enabled=True, triggered=self.showSettings)

        # Annotators
        self.polygonAct = QAction("Полигон", self, enabled=False, triggered=self.polygon_tool_pressed, checkable=True)
        self.circleAct = QAction("Эллипс", self, enabled=False, triggered=self.circle_pressed, checkable=True)
        self.squareAct = QAction("Прямоугольник (Box)", self, enabled=False, triggered=self.square_pressed,
                                 checkable=True)
        self.aiAnnotatorPointsAct = QAction("Сегментация по точкам", self, enabled=False,
                                            triggered=self.ai_points_pressed,
                                            checkable=True)
        self.aiAnnotatorMaskAct = QAction("Сегментация внутри бокса", self, enabled=False,
                                          triggered=self.ai_mask_pressed,
                                          checkable=True)

        # Export
        self.exportAnnToYoloBoxAct = QAction("Экспорт в формат YOLO (Box)", self, enabled=False,
                                             triggered=self.exportToYOLOBox)
        self.exportAnnToYoloSegAct = QAction("Экспорт в формат YOLO (Seg)", self, enabled=False,
                                             triggered=self.exportToYOLOSeg)
        self.exportAnnToCOCOAct = QAction("Экспорт в формат COCO", self, enabled=False,
                                          triggered=self.exportToCOCO)

        # Labels
        self.add_label = QAction("Добавить новый класс", self, enabled=True, triggered=self.add_label_button_clicked)
        self.del_label = QAction("Удалить текущий класс", self, enabled=True, triggered=self.del_label_button_clicked)
        self.change_label_color = QAction("Изменить цвет разметки для текущего класса", self, enabled=True,
                                          triggered=self.change_label_color_button_clicked)
        self.rename_label = QAction("Изменить имя класса", self, enabled=True,
                                    triggered=self.rename_label_button_clicked)

        self.set_icons()

    def createMenus(self):

        """
        Создание меню
        """

        self.fileMenu = QMenu("&Файл", self)
        self.fileMenu.addAction(self.openAct)
        self.fileMenu.addAction(self.openProjAct)
        self.fileMenu.addAction(self.saveProjAct)
        self.fileMenu.addAction(self.saveProjAsAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
        #
        self.viewMenu = QMenu("&Изображение", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)
        #
        self.annotatorMenu = QMenu("&Аннотация", self)
        self.AnnotatorMethodMenu = QMenu("Способ выделения", self)
        self.aiAnnotatorMethodMenu = QMenu("Сегментация", self)
        self.aiAnnotatorMethodMenu.setIcon(QIcon(self.icon_folder + "/ai.png"))

        self.AnnotatorMethodMenu.addAction(self.polygonAct)
        self.AnnotatorMethodMenu.addAction(self.squareAct)
        self.AnnotatorMethodMenu.addAction(self.circleAct)

        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorPointsAct)
        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorMaskAct)

        self.AnnotatorMethodMenu.addMenu(self.aiAnnotatorMethodMenu)
        self.annotatorMenu.addMenu(self.AnnotatorMethodMenu)

        self.annotatorExportMenu = QMenu("Экспорт", self)
        self.annotatorExportMenu.addAction(self.exportAnnToYoloBoxAct)
        self.annotatorExportMenu.addAction(self.exportAnnToYoloSegAct)
        self.annotatorExportMenu.addAction(self.exportAnnToCOCOAct)
        self.annotatorMenu.addMenu(self.annotatorExportMenu)
        #
        self.settingsMenu = QMenu("Настройки", self)
        self.settingsMenu.addAction(self.settingsAct)
        #
        self.helpMenu = QMenu("&Помощь", self)
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

        toolBar = QToolBar("Панель инструментов", self)
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

        labelSettingsToolBar = QToolBar("Настройки разметки", self)
        self.cls_combo = QComboBox()

        label = QLabel("Текущий класс:   ")
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
        self.toolBarRight = QToolBar("Менеджер разметок", self)

        # # Панель меток - заголовок
        self.toolBarRight.addWidget(LabelsPanel(self, self.break_drawing, self.icon_folder,
                                                on_color_change_signal=self.on_theme_change_connection.on_theme_change))

        # Панель меток - список
        self.labels_on_tek_image = QListWidget()
        self.labels_on_tek_image.itemClicked.connect(self.labels_on_tek_image_clicked)
        self.toolBarRight.addWidget(self.labels_on_tek_image)

        # Панель изображений - заголовок
        self.toolBarRight.addWidget(
            ImagesPanel(self, self.add_im_to_proj_clicked, self.del_im_from_proj_clicked, self.icon_folder,
                        on_color_change_signal=self.on_theme_change_connection.on_theme_change))

        # Панель изображений - список
        self.images_list_widget = QListWidget()
        self.images_list_widget.itemClicked.connect(self.images_list_widget_clicked)
        self.toolBarRight.addWidget(self.images_list_widget)

        # Устанавливаем layout в правую панель

        self.addToolBar(QtCore.Qt.TopToolBarArea, labelSettingsToolBar)
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBarLeft)
        self.addToolBar(QtCore.Qt.RightToolBarArea, self.toolBarRight)

    def add_im_to_proj_clicked(self):

        if self.dataset_dir:
            images, _ = QFileDialog.getOpenFileNames(self, 'Загрузка изображений в проект', 'images',
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

    def del_im_from_proj_clicked(self):

        if self.tek_image_name:
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
        self.view.setFocus()

    def labels_on_tek_image_clicked(self, item):
        item_id = item.text().split(" ")[-1]
        self.view.activate_item_by_id(int(item_id))

    def exportToYOLOBox(self):
        self.save_project()
        export_dir = QFileDialog.getExistingDirectory(self, 'Выберите папку для сохранения разметки',
                                                      'images')
        if export_dir:
            self.project_data.exportToYOLOBox(export_dir)
            self.on_project_export(export_format="YOLO Box")

    def exportToYOLOSeg(self):
        self.save_project()
        export_dir = QFileDialog.getExistingDirectory(self, 'Выберите папку для сохранения разметки',
                                                      'images')
        if export_dir:
            self.project_data.exportToYOLOSeg(export_dir)
            self.on_project_export(export_format="YOLO Seg")

    def exportToCOCO(self):
        self.save_project()
        export_сoco_file, _ = QFileDialog.getSaveFileName(self, 'Выберите имя сохраняемого файла', 'images',
                                                          'JSON File (*.json)')

        if export_сoco_file:
            self.project_data.exportToCOCO(export_сoco_file)

            self.on_project_export(export_format="COCO")

    def set_color_to_cls(self, cls_name):

        self.project_data.set_label_color(cls_name, alpha=self.settings_["alpha"])

    def on_project_export(self, export_format="YOLO Seg"):
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Information)
        msgbox.setText(f"Экспорт в формат {export_format} завершен успешно")
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

        self.input_dialog = CustomInputDialog(self, title_name="Добавление нового класса",
                                              question_name="Введите имя класса:")
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
                msgbox.setText(f"Ошибка добавления класса")
                msgbox.setWindowTitle("Ошибка добавления класса")
                msgbox.setInformativeText(f"Класс с именем {new_name} уже существует")
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
            msgbox.setText(f"Ошибка удаления класса")
            msgbox.setWindowTitle(f"Ошибка удаления класса")
            msgbox.setInformativeText("Количество классов должно быть хотя бы 2 для удаления текущего")
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
        color_dialog.setWindowTitle(f"Выберите цвет для класса {cls_txt}")
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
        self.input_dialog = CustomInputDialog(self, title_name=f"Редактирование имени класса {cls_name}",
                                              question_name="Введите новое имя класса:")

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

        self.polygonAct.setIcon(QIcon(self.icon_folder + "/polygon.png"))
        self.circleAct.setIcon(QIcon(self.icon_folder + "/circle.png"))
        self.squareAct.setIcon(QIcon(self.icon_folder + "/square.png"))
        self.aiAnnotatorPointsAct.setIcon(QIcon(self.icon_folder + "/mouse.png"))
        self.aiAnnotatorMaskAct.setIcon(QIcon(self.icon_folder + "/ai_select.png"))

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

        self.exportAnnToYoloBoxAct.setEnabled(True)
        self.exportAnnToYoloSegAct.setEnabled(True)
        self.exportAnnToCOCOAct.setEnabled(True)

        image = cv2.imread(image_name)
        self.cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        self.image_setted = False
        self.image_setter.set_image(self.cv2_image)

        if not self.image_setter.isRunning():
            self.statusBar().showMessage("Начинаю загружать изображение в нейросеть SAM...", 3000)
            self.image_setter.start()
        else:
            image_copy = cv2.imread(image_name)
            image_copy = cv2.cvtColor(image_copy, cv2.COLOR_BGR2RGB)
            self.queue_to_image_setter.append(image_copy)
            self.statusBar().showMessage(
                f"Изображение {os.path.split(image_name)[-1]} добавлено в очередь на обработку.", 3000)

    def on_image_setted(self):
        if len(self.queue_to_image_setter) != 0:
            self.cv2_image = self.queue_to_image_setter[-1]
            self.image_setted = False
            self.image_setter.set_image(self.cv2_image)
            self.queue_to_image_setter = []
            self.statusBar().showMessage("Нейросеть SAM еще не готова. Подождите секунду...", 3000)
            self.image_setter.start()

        else:
            self.statusBar().showMessage("Нейросеть SAM готова к сегментации", 3000)
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
                          "<p>Программа для разметки изображений с поддержкой автоматической сегментации</p>")

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

        dataset_dir = QFileDialog.getExistingDirectory(self, 'Выберите папку с изображениями для разметки',
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

                main_geom = self.geometry().getCoords()
                self.scaleFactor = (main_geom[2] - main_geom[0]) / self.cv2_image.shape[1]

                self.fill_labels_on_tek_image_list_widget()

                self.fill_images_label(self.dataset_images)

                self.splash.finish(self)

                self.statusBar().showMessage(
                    f"Число загруженны в проект изображений: {len(self.dataset_images)}", 3000)
            else:
                self.statusBar().showMessage(
                    f"В указанной папке изображений не обнаружено", 3000)

    def fill_images_label(self, image_names):

        self.images_list_widget.clear()
        for name in image_names:
            self.images_list_widget.addItem(name)

    def open_project(self):
        """
        Загрузка проекта
        """
        loaded_proj_name, _ = QFileDialog.getOpenFileName(self, 'Загрузка проекта', 'projects',
                                                          'JSON Proj File (*.json)')

        if loaded_proj_name:
            self.loaded_proj_name = loaded_proj_name
            is_success = self.project_data.load(self.loaded_proj_name)

            if not is_success:
                msgbox = QMessageBox()
                msgbox.setIcon(QMessageBox.Information)
                msgbox.setText(f"Ошибка открытия файла {self.loaded_proj_name}")
                msgbox.setInformativeText(
                    f"Файл {self.loaded_proj_name} должен быть в формате .json и содержать поля:\n\t"
                    f"path_to_images\n\timages\n\t\tfilename\n\t\tshapes\n\t")
                msgbox.setWindowTitle(f"Ошибка открытия файла {self.loaded_proj_name}")
                msgbox.exec()

            else:

                self.dataset_dir = self.project_data.get_image_path()
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

                self.statusBar().showMessage(
                    f"Число загруженны в проект изображений: {len(self.dataset_images)}", 3000)

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
                f"Проект успешно сохранен", 3000)

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
                f"Проект успешно сохранен", 3000)

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
        self.settings_['is_fit_window'] = True

        self.settings_["alpha"] = 50
        self.settings_["fat_width"] = 50

        if cuda.is_available():
            print("CUDA is available")
            self.statusBar().showMessage(
                "Найдено устройство NVIDIA CUDA. Нейросеть будет использовать ее для ускорения", 3000)
            self.settings_['platform'] = "cuda"
        else:
            self.statusBar().showMessage(
                "Не найдено устройство NVIDIA CUDA. Нейросеть будет использовать ресурсы процессора", 3000)
            self.settings_['platform'] = "cpu"

        self.settings_["labels_color"] = {}

        self.settings_set = True

    def change_theme(self, theme_str):
        """
        Изменение темы приложения
        """
        app = QApplication.instance()

        primary_color = "#ffffff"
        if 'light' in theme_str:
            primary_color = "#000000"

        extra = {'density_scale': config.DENSITY_SCALE,
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
            self.settings_ = self.settings_window.settings
            self.settings_set = True

            str_settings = "Настройки сохранены.\n"

            if self.settings_['platform'] == 'Auto':
                if cuda.is_available():
                    self.statusBar().showMessage(
                        "Найдено устройство NVIDIA CUDA. Нейросеть будет использовать ее для ускорения", 3000)
                    if self.settings_['platform'] == 'cpu':
                        self.settings_['platform'] = "cuda"
                        self.sam = load_model(self.sam_model_path, model_type="vit_h",
                                              device=self.settings_['platform'])
                else:
                    self.statusBar().showMessage(
                        "Не найдено устройство NVIDIA CUDA. Нейросеть будет использовать ресурсы процессора", 3000)
                    if self.settings_['platform'] == 'cuda':
                        self.settings_['platform'] = "cpu"
                        self.sam = load_model(self.sam_model_path, model_type="vit_h",
                                              device=self.settings_['platform'])

            if self.settings_['theme'] != self.theme_str:
                self.statusBar().showMessage(
                    "Тема приложения изменена", 3000)
                self.theme_str = self.settings_['theme']
                self.change_theme(self.settings_['theme'])
                self.set_icons()

            self.settings_["alpha"] = self.settings_window.settings["alpha"]
            self.settings_["fat_width"] = self.settings_window.settings["fat_width"]
            self.view.set_fat_width(self.settings_["fat_width"])

            QMessageBox.about(self, "Сохранение настроек приложения",
                              str_settings)

            self.statusBar().showMessage(
                f"Настройки проекта изменены", 3000)

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
                self.statusBar().showMessage(
                    f"Метку сделать не удалось. Площадь маски слишком мала {area}. Попробуйте еще раз", 3000)
                self.view.remove_label_id(id)
                self.write_scene_to_project_data()
                self.fill_labels_on_tek_image_list_widget()
        else:
            self.view.remove_label_id(id)
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()

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

    def start_drawing(self):
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings_["alpha"]

        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)
        self.view.clear_ai_points()

    def break_drawing(self):
        if self.ann_type == "AiPoints":
            self.view.clear_ai_points()
            self.view.remove_active()

        if not "Continue" in self.view.drag_mode:
            self.view.remove_active()

        if self.tek_image_name:
            self.view.end_drawing()
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()

    def reload_image(self):
        self.open_image(self.tek_image_path)
        self.load_image_data(self.tek_image_name)
        self.fill_labels_on_tek_image_list_widget()
        self.view.clear_ai_points()

    def show_tutorial(self):
        self.tutorial = Tutorial(self)
        self.tutorial.show()

    def keyPressEvent(self, e):
        # print(e.key())
        modifierPressed = QApplication.keyboardModifiers()
        modifierName = ''
        # if (modifierPressed & QtCore.Qt.AltModifier) == QtCore.Qt.AltModifier:
        #     modifierName += 'Alt'

        if (modifierPressed & QtCore.Qt.ControlModifier) == QtCore.Qt.ControlModifier:
            modifierName += 'Ctrl'

        if e.key() == 83 or e.key() == 1067:  # start poly
            self.start_drawing()

        elif e.key() == 32:  # end poly
            self.end_drawing()

        elif e.key() == 44 or e.key() == 1041:
            self.write_scene_to_project_data()

            if self.get_before_image_name():
                self.reload_image()

        elif e.key() == 46 or e.key() == 1070:

            self.write_scene_to_project_data()
            if self.get_next_image_name():
                self.reload_image()

        elif e.key() == 68 or e.key() == 1042:

            self.break_drawing()

        elif (e.key() == 67 or e.key() == 1057) and 'Ctrl' in modifierName:
            self.view.copy_active_item_to_buffer()
            # self.write_scene_to_project_data()

        elif (e.key() == 86 or e.key() == 1052) and 'Ctrl' in modifierName:
            self.view.paste_buffer()
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()
            self.view.clear_ai_points()


if __name__ == '__main__':
    import sys
    from qt_material import apply_stylesheet

    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': config.DENSITY_SCALE,
             # 'font_size': '14px',
             'primaryTextColor': '#ffffff'}

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
