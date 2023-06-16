from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QMovie, QPainter, QIcon, QColor
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QAction, QFileDialog, QMessageBox, QMenu, QToolBar, QToolButton, QComboBox, QLabel, \
    QColorDialog, QListWidget, QProgressBar
from PyQt5.QtWidgets import QApplication

from utils import help_functions as hf

from utils import config

from utils.project import ProjectHandler
from utils.importer import Importer
from utils.settings_handler import AppSettings

from ui.settings_window_base import SettingsWindowBase
from ui.ask_del_polygon import AskDelWindow
from ui.splash_screen import MovieSplashScreen
from ui.view import GraphicsView
from ui.input_dialog import CustomInputDialog, CustomComboDialog
from ui.show_image_widget import ShowImgWindow
from ui.panels import ImagesPanel, LabelsPanel
from ui.signals_and_slots import ImagesPanelCountConnection, LabelsPanelCountConnection, ThemeChangeConnection, \
    RubberBandModeConnection
from ui.import_dialogs import ImportFromYOLODialog, ImportFromCOCODialog
from ui.edit_with_button import EditWithButton
from ui.ok_cancel_dialog import OkCancelDialog
from ui.progress import ProgressWindow
from ui.create_project_dialog import CreateProjectDialog

import cv2
import numpy as np
import os
import shutil

import sys
from qt_material import apply_stylesheet


class MainWindow(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Annotator")

        # Start on Loading Animation
        self.start_gif(is_prog_load=True)

        # Rubber band
        self.is_rubber_band_mode = False
        self.rubber_band_change_conn = RubberBandModeConnection()

        # GraphicsView
        self.view = GraphicsView(on_rubber_band_mode=self.rubber_band_change_conn.on_rubber_mode_change)
        self.setCentralWidget(self.view)

        # Signals with View
        self.view.polygon_clicked.id_pressed.connect(self.polygon_pressed)
        self.view.polygon_delete.id_delete.connect(self.on_polygon_delete)
        self.view.polygon_end_drawing.on_end_drawing.connect(self.on_polygon_end_draw)

        self.view.polygon_cls_num_change.pol_cls_num_and_id.connect(self.change_polygon_cls_num)

        # Signals with Right Panels
        self.im_panel_count_conn = ImagesPanelCountConnection()
        self.labels_count_conn = LabelsPanelCountConnection()
        self.on_theme_change_connection = ThemeChangeConnection()

        self.resize(1200, 800)

        # Settings
        self.settings = AppSettings()
        self.icon_folder = self.settings.get_icon_folder()
        # last_ for not recreate if not change
        self.last_theme = self.settings.read_theme()

        # Menu and toolbars
        self.createActions()
        self.createMenus()
        self.createToolbar()

        # Icons
        self.change_theme()
        self.set_icons()

        # Printer
        self.printer = QPrinter()

        # Work with project
        self.project_data = ProjectHandler()

        self.init_global_values()

        self.splash.finish(self)
        self.statusBar().showMessage(
            "Загрузите проект или набор изображений" if self.settings.read_lang() == 'RU' else "Load dataset or project")

    def init_global_values(self):
        """
        Set some app global values
        """

        self.scaleFactor = 1.0
        self.ann_type = "Polygon"

        self.loaded_proj_name = None
        self.labels_on_tek_image_ids = None
        self.tek_image_name = None
        self.tek_image_path = None
        self.dataset_dir = None

        self.last_alpha = None
        self.last_fat_width = None

        self.image_types = ['jpg', 'png', 'tiff', 'jpeg']

        self.dataset_images = []

        # import settings
        self.is_seg_import = False

        # close window flag
        self.is_asked_before_close = False

        # Set window size and pos from last state
        self.read_size_pos()

    def write_size_pos(self):
        """
        Save window pos and size
        """

        self.settings.write_size_pos_settings(self.size(), self.pos())

    def read_size_pos(self):
        """
        Read saved window pos and size
        """

        size, pos = self.settings.read_size_pos_settings()

        if size and pos:
            self.resize(size)
            self.move(pos)

    def set_movie_gif(self):
        self.movie_gif = "ui/icons/15.gif"
        self.ai_gif = "ui/icons/15.gif"

    def start_gif(self, is_prog_load=False, mode="Loading"):
        """
        Show animation while do something
        """
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

        self.splash.show()

    def createActions(self):

        self.createNewProjectAct = QAction(
            "Создать новый проект" if self.settings.read_lang() == 'RU' else "Create new project",
            self,
            shortcut="Ctrl+N", triggered=self.createNewProject)
        self.openProjAct = QAction("Загрузить проект" if self.settings.read_lang() == 'RU' else "Load Project", self,
                                   shortcut='Ctrl+O',
                                   triggered=self.open_project)
        self.saveProjAsAct = QAction(
            "Сохранить проект как..." if self.settings.read_lang() == 'RU' else "Save project as...",
            self, triggered=self.save_project_as, enabled=False)
        self.saveProjAct = QAction("Сохранить проект" if self.settings.read_lang() == 'RU' else "Save project", self,
                                   shortcut="Ctrl+S", triggered=self.save_project, enabled=False)

        self.printAct = QAction("Печать" if self.settings.read_lang() == 'RU' else "Print", self, shortcut="Ctrl+P",
                                enabled=False, triggered=self.print_)
        self.exitAct = QAction("Выход" if self.settings.read_lang() == 'RU' else "Exit", self, shortcut="Ctrl+Q",
                               triggered=self.close)
        self.zoomInAct = QAction("Увеличить" if self.settings.read_lang() == 'RU' else "Zoom In", self,
                                 shortcut="Ctrl++",
                                 enabled=False,
                                 triggered=self.zoomIn)
        self.zoomOutAct = QAction("Уменьшить" if self.settings.read_lang() == 'RU' else "Zoom Out", self,
                                  shortcut="Ctrl+-",
                                  enabled=False,
                                  triggered=self.zoomOut)

        self.fitToWindowAct = QAction(
            "Подогнать под размер окна" if self.settings.read_lang() == 'RU' else "Fit to window size",
            self, enabled=False,
            shortcut="Ctrl+F",
            triggered=self.fitToWindow)
        self.aboutAct = QAction("О модуле" if self.settings.read_lang() == 'RU' else "About", self,
                                triggered=self.about)
        self.tutorialAct = QAction("Горячие клавиши" if self.settings.read_lang() == 'RU' else "Shortcuts", self,
                                   triggered=self.show_tutorial)

        self.settingsAct = QAction("Настройки приложения" if self.settings.read_lang() == 'RU' else "Settings", self,
                                   enabled=True, triggered=self.showSettings)

        # Annotators
        self.polygonAct = QAction("Полигон" if self.settings.read_lang() == 'RU' else "Polygon", self, enabled=False,
                                  triggered=self.polygon_tool_pressed, checkable=True)
        self.circleAct = QAction("Эллипс" if self.settings.read_lang() == 'RU' else "Ellips", self, enabled=False,
                                 triggered=self.circle_pressed, checkable=True)
        self.squareAct = QAction("Прямоугольник" if self.settings.read_lang() == 'RU' else "Box", self, enabled=False,
                                 triggered=self.square_pressed,
                                 checkable=True)

        # Export
        self.exportAnnToYoloBoxAct = QAction(
            "YOLO (Box)" if self.settings.read_lang() == 'RU' else "YOLO (Boxes)", self,
            enabled=False,
            triggered=self.exportToYOLOBox)
        self.exportAnnToYoloSegAct = QAction(
            "YOLO (Seg)" if self.settings.read_lang() == 'RU' else "YOLO (Seg)", self,
            enabled=False,
            triggered=self.exportToYOLOSeg)
        self.exportAnnToCOCOAct = QAction(
            "COCO" if self.settings.read_lang() == 'RU' else "COCO", self, enabled=False,
            triggered=self.exportToCOCO)

        # Import
        self.importAnnFromYoloBoxAct = QAction(
            "YOLO (Box)" if self.settings.read_lang() == 'RU' else "YOLO (Boxes)", self,
            enabled=True,
            triggered=self.importFromYOLOBox)
        self.importAnnFromYoloSegAct = QAction(
            "YOLO (Seg)" if self.settings.read_lang() == 'RU' else "YOLO (Seg)", self,
            enabled=True,
            triggered=self.importFromYOLOSeg)
        self.importAnnFromCOCOAct = QAction(
            "COCO" if self.settings.read_lang() == 'RU' else "COCO", self, enabled=True,
            triggered=self.importFromCOCO)

        # Labels
        self.add_label = QAction("Добавить новый класс" if self.settings.read_lang() == 'RU' else "Add new label", self,
                                 enabled=True, triggered=self.add_label_button_clicked)
        self.del_label = QAction(
            "Удалить текущий класс" if self.settings.read_lang() == 'RU' else "Delete current label",
            self,
            enabled=True, triggered=self.del_label_button_clicked)
        self.change_label_color = QAction(
            "Изменить цвет разметки для текущего класса" if self.settings.read_lang() == 'RU' else "Change label color",
            self,
            enabled=True,
            triggered=self.change_label_color_button_clicked)
        self.rename_label = QAction("Изменить имя класса" if self.settings.read_lang() == 'RU' else "Rename", self,
                                    enabled=True,
                                    triggered=self.rename_label_button_clicked)

        # Image actions

        self.selectAreaAct = QAction("Выделить область" if self.settings.read_lang() == 'RU' else "Select an area",
                                     self,
                                     shortcut="Ctrl+I", enabled=False, triggered=self.getArea)

    def createMenus(self):

        self.fileMenu = QMenu("&Файл" if self.settings.read_lang() == 'RU' else "&File", self)
        self.fileMenu.addAction(self.createNewProjectAct)
        self.fileMenu.addAction(self.openProjAct)
        self.fileMenu.addAction(self.saveProjAct)
        self.fileMenu.addAction(self.saveProjAsAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
        #
        self.viewMenu = QMenu("&Изображение" if self.settings.read_lang() == 'RU' else "&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.selectAreaAct)

        #
        self.annotatorMenu = QMenu("&Аннотация" if self.settings.read_lang() == 'RU' else "&Labeling", self)

        self.AnnotatorMethodMenu = QMenu("Способ выделения" if self.settings.read_lang() == 'RU' else "Method", self)

        self.AnnotatorMethodMenu.addAction(self.polygonAct)
        self.AnnotatorMethodMenu.addAction(self.squareAct)
        self.AnnotatorMethodMenu.addAction(self.circleAct)

        self.annotatorMenu.addMenu(self.AnnotatorMethodMenu)

        self.annotatorExportMenu = QMenu("Экспорт" if self.settings.read_lang() == 'RU' else "Export", self)
        self.annotatorExportMenu.addAction(self.exportAnnToYoloBoxAct)
        self.annotatorExportMenu.addAction(self.exportAnnToYoloSegAct)
        self.annotatorExportMenu.addAction(self.exportAnnToCOCOAct)

        self.annotatorMenu.addMenu(self.annotatorExportMenu)

        self.annotatorImportMenu = QMenu("Импорт" if self.settings.read_lang() == 'RU' else "Import", self)
        self.annotatorImportMenu.addAction(self.importAnnFromYoloBoxAct)
        self.annotatorImportMenu.addAction(self.importAnnFromYoloSegAct)
        self.annotatorImportMenu.addAction(self.importAnnFromCOCOAct)

        self.annotatorMenu.addMenu(self.annotatorImportMenu)

        #
        self.settingsMenu = QMenu("Настройки" if self.settings.read_lang() == 'RU' else "Settings", self)
        self.settingsMenu.addAction(self.settingsAct)
        #
        self.helpMenu = QMenu("&Помощь" if self.settings.read_lang() == 'RU' else "Help", self)
        self.helpMenu.addAction(self.aboutAct)
        self.helpMenu.addAction(self.tutorialAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)

        self.menuBar().addMenu(self.annotatorMenu)
        self.menuBar().addMenu(self.settingsMenu)
        self.menuBar().addMenu(self.helpMenu)

    def create_left_toolbar(self):
        # Left

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
        toolBar.addAction(self.settingsAct)
        toolBar.addSeparator()

        self.toolBarLeft = toolBar
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBarLeft)

    def create_right_toolbar(self):
        # Right toolbar
        self.toolBarRight = QToolBar("Менеджер разметок" if self.settings.read_lang() == 'RU' else "Labeling Bar", self)

        # Labels
        self.toolBarRight.addWidget(LabelsPanel(self, self.break_drawing, self.icon_folder,
                                                on_color_change_signal=self.on_theme_change_connection.on_theme_change,
                                                on_labels_count_change=self.labels_count_conn.on_labels_count_change))

        self.labels_on_tek_image = QListWidget()
        self.labels_on_tek_image.itemClicked.connect(self.labels_on_tek_image_clicked)
        self.toolBarRight.addWidget(self.labels_on_tek_image)

        # Images
        self.toolBarRight.addWidget(
            ImagesPanel(self, self.add_im_to_proj_clicked, self.del_im_from_proj_clicked, self.icon_folder,
                        on_color_change_signal=self.on_theme_change_connection.on_theme_change,
                        on_images_list_change=self.im_panel_count_conn.on_image_count_change))

        self.images_list_widget = QListWidget()
        self.images_list_widget.itemClicked.connect(self.images_list_widget_clicked)
        self.toolBarRight.addWidget(self.images_list_widget)

        # Add panels to toolbars

        self.addToolBar(QtCore.Qt.RightToolBarArea, self.toolBarRight)

    def create_top_toolbar(self):

        self.labelSettingsToolBar = QToolBar(
            "Настройки разметки" if self.settings.read_lang() == 'RU' else "Current Label Bar",
            self)
        self.cls_combo = QComboBox()

        label = QLabel("Текущий класс:   " if self.settings.read_lang() == 'RU' else "Current label:   ")
        cls_names = np.array(['no name'])
        self.cls_combo.addItems(cls_names)
        self.cls_combo.setMinimumWidth(150)
        self.cls_combo.setEnabled(True)

        self.labelSettingsToolBar.addWidget(label)
        self.labelSettingsToolBar.addWidget(self.cls_combo)

        self.labelSettingsToolBar.addAction(self.change_label_color)
        self.labelSettingsToolBar.addAction(self.rename_label)
        self.labelSettingsToolBar.addAction(self.del_label)
        self.labelSettingsToolBar.addSeparator()
        self.labelSettingsToolBar.addAction(self.add_label)
        self.labelSettingsToolBar.addSeparator()

        self.addToolBar(QtCore.Qt.TopToolBarArea, self.labelSettingsToolBar)

    def createToolbar(self):

        self.create_right_toolbar()
        self.create_left_toolbar()
        self.create_top_toolbar()

    def getArea(self):
        """
        Действие - выбрать область
        """
        self.break_drawing()  # Если до этого что-то рисовали - сбросить
        self.is_rubber_band_mode = True
        self.rubber_band_change_conn.on_rubber_mode_change.emit(True)

    def change_polygon_cls_num(self, cls_num, cls_id):
        labels = self.project_data.get_labels()
        self.combo_dialog = CustomComboDialog(self,
                                              title_name="Изменение имени метки" if self.settings.read_lang() == 'RU' else "Label name change",
                                              question_name="Введите имя класса:" if self.settings.read_lang() == 'RU' else "Enter label name:",
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

        self.project_data.change_cls_num_by_id(self.changed_cls_id, new_cls_num)
        self.reload_image()
        self.combo_dialog.close()

    def add_im_to_proj_clicked(self):

        if self.dataset_dir:
            images, _ = QFileDialog.getOpenFileNames(self,
                                                     'Загрузка изображений в проект' if self.settings.read_lang() == 'RU' else "Loading dataset",
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
            self.createNewProject()

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
                                                      'Выберите папку для сохранения разметки' if self.settings.read_lang() == 'RU' else "Set folder",
                                                      'images')
        if export_dir:
            self.project_data.exportToYOLOBox(export_dir)
            self.on_project_export(export_format="YOLO Box")

    def exportToYOLOSeg(self):
        self.save_project()
        export_dir = QFileDialog.getExistingDirectory(self,
                                                      'Выберите папку для сохранения разметки' if self.settings.read_lang() == 'RU' else "Set folder",
                                                      'images')
        if export_dir:
            self.project_data.exportToYOLOSeg(export_dir)
            self.on_project_export(export_format="YOLO Seg")

    def exportToCOCO(self):
        self.save_project()
        export_сoco_file, _ = QFileDialog.getSaveFileName(self,
                                                          'Выберите имя сохраняемого файла' if self.settings.read_lang() == 'RU' else "Set export file name",
                                                          'images',
                                                          'JSON File (*.json)')

        if export_сoco_file:
            self.project_data.exportToCOCO(export_сoco_file)

            self.on_project_export(export_format="COCO")

    def importFromYOLOBox(self):

        self.close_project()

        self.import_dialog = ImportFromYOLODialog(self, on_ok_clicked=self.on_import_yolo_clicked)
        self.is_seg_import = False
        self.import_dialog.show()

    def on_import_yolo_clicked(self):
        yaml_data = self.import_dialog.getData()

        copy_images_path = None
        if yaml_data['is_copy_images']:
            copy_images_path = yaml_data['save_images_dir']

        self.importer = Importer(yaml_data=yaml_data, alpha=self.settings.read_alpha(), is_seg=self.is_seg_import,
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
        self.close_project()

        self.import_dialog = ImportFromYOLODialog(self, on_ok_clicked=self.on_import_yolo_clicked)
        self.is_seg_import = True
        self.import_dialog.show()

    def importFromCOCO(self):
        self.close_project()

        self.import_dialog = ImportFromCOCODialog(self, on_ok_clicked=self.on_import_coco_clicked)
        self.import_dialog.show()

    def on_import_coco_clicked(self):
        proj_data = self.import_dialog.getData()
        if proj_data:
            label_names = self.import_dialog.get_label_names()
            self.importer = Importer(coco_data=proj_data, alpha=self.settings.read_alpha(), label_names=label_names,
                                     copy_images_path=self.import_dialog.get_copy_images_path(),
                                     coco_name=self.import_dialog.get_coco_name(), is_coco=True)

            self.importer.finished.connect(self.on_import_finished)
            self.importer.load_percent_conn.percent.connect(self.on_import_percent_change)

            if not self.importer.isRunning():
                self.importer.start()

    def on_import_finished(self):
        self.project_data.set_data(self.importer.get_project())

        if self.loaded_proj_name:
            self.project_data.save(self.loaded_proj_name)
            self.load_project(self.loaded_proj_name)

        self.import_dialog.hide()

    def set_color_to_cls(self, cls_name):

        self.project_data.set_label_color(cls_name, alpha=self.settings.read_alpha())

    def on_project_export(self, export_format="YOLO Seg"):
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Information)
        msgbox.setText(
            f"Экспорт в формат {export_format} завершен успешно" if self.settings.read_lang() == 'RU' else f"Export to {export_format} was successful")
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
                                              title_name="Добавление нового класса" if self.settings.read_lang() == 'RU' else "New label",
                                              question_name="Введите имя класса:" if self.settings.read_lang() == 'RU' else "Enter label name:")
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
                    f"Ошибка добавления класса" if self.settings.read_lang() == 'RU' else "Error in setting new label")
                msgbox.setWindowTitle("Ошибка добавления класса" if self.settings.read_lang() == 'RU' else "Error")
                msgbox.setInformativeText(
                    f"Класс с именем {new_name} уже существует" if self.settings.read_lang() == 'RU' else f"Label with name {new_name} is already exist. Try again")
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
            msgbox.setText(
                f"Ошибка удаления класса" if self.settings.read_lang() == 'RU' else "Error in deleting label")
            msgbox.setWindowTitle(f"Ошибка удаления класса" if self.settings.read_lang() == 'RU' else "Error")
            msgbox.setInformativeText(
                "Количество классов должно быть хотя бы 2 для удаления текущего" if self.settings.read_lang() == 'RU' else "The last label left. If you don't like the name - just rename it")
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

        new_name = self.ask_del_label.cls_combo.currentText()  # на что меням
        old_name = self.cls_combo.itemText(self.del_index)

        self.ask_del_label.close()  # закрываем окно

        # 2. Убираем имя из комбобокса
        self.del_label_from_combobox(old_name)  # теперь в комбобоксе нет имени

        # 3. Обновляем все полигоны
        self.project_data.change_data_class_from_to(old_name, new_name)

        # 4. Обновляем панель справа
        self.fill_labels_on_tek_image_list_widget()

        # 5. Переоткрываем изображение и рисуем полигоны из проекта
        self.open_image(self.tek_image_path)
        self.load_image_data(self.tek_image_name)
        self.view.setFocus()

    def on_ask_del_all(self):
        # 1. Сохраняем данные сцены в проект
        self.write_scene_to_project_data()

        del_name = self.ask_del_label.cls_name

        self.ask_del_label.close()

        # 2. Удаляем данные о цвете из проекта
        self.project_data.delete_data_by_class_name(del_name)

        # 3. Убираем имя класса из комбобокса
        self.del_label_from_combobox(del_name)

        # 4. Обновляем панель справа
        self.fill_labels_on_tek_image_list_widget()

        # 5. Переоткрываем изображение и рисуем полигоны из проекта
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

    def del_image_labels_from_project(self, image_name):

        self.project_data.del_image(image_name)

    def change_label_color_button_clicked(self):

        self.write_scene_to_project_data()

        color_dialog = QColorDialog()
        cls_txt = self.cls_combo.currentText()
        color_dialog.setWindowTitle(
            f"Выберите цвет для класса {cls_txt}" if self.settings.read_lang() == 'RU' else f"Enter color to label {cls_txt}")
        current_color = self.project_data.get_label_color(cls_txt)
        if not current_color:
            current_color = config.COLORS[self.cls_combo.currentIndex()]

        color_dialog.setCurrentColor(QColor(*current_color))
        color_dialog.setWindowIcon(QIcon(self.icon_folder + "/color.png"))
        color_dialog.exec()
        rgb = color_dialog.selectedColor().getRgb()
        rgba = (rgb[0], rgb[1], rgb[2], self.settings.read_alpha())
        print(rgba)

        self.project_data.set_label_color(cls_txt, color=rgba)

        self.open_image(self.tek_image_path)
        self.load_image_data(self.tek_image_name)

        self.view.setFocus()

    def rename_label_button_clicked(self):

        self.write_scene_to_project_data()

        cls_name = self.cls_combo.currentText()
        self.input_dialog = CustomInputDialog(self,
                                              title_name=f"Редактирование имени класса {cls_name}" if self.settings.read_lang() == 'RU' else f"Rename label {cls_name}",
                                              question_name="Введите новое имя класса:" if self.settings.read_lang() == 'RU' else "Enter new label name:")

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
            self.project_data.change_name(cls_name, new_name)

            self.fill_labels_on_tek_image_list_widget()

        self.view.setFocus()

    def set_icons(self):
        """
        Задать иконки
        """

        self.icon_folder = self.settings.get_icon_folder()

        self.setWindowIcon(QIcon(self.icon_folder + "/neural.png"))
        self.createNewProjectAct.setIcon(QIcon(self.icon_folder + "/folder.png"))
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

        # rubber band
        self.selectAreaAct.setIcon(QIcon(self.icon_folder + "/select.png"))

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

        # labeling
        self.add_label.setIcon((QIcon(self.icon_folder + "/add.png")))
        self.del_label.setIcon((QIcon(self.icon_folder + "/del.png")))
        self.change_label_color.setIcon((QIcon(self.icon_folder + "/color.png")))
        self.rename_label.setIcon((QIcon(self.icon_folder + "/rename.png")))

        # menus

        self.AnnotatorMethodMenu.setIcon(QIcon(self.icon_folder + "/label.png"))
        self.annotatorExportMenu.setIcon(QIcon(self.icon_folder + "/export.png"))
        self.annotatorImportMenu.setIcon(QIcon(self.icon_folder + "/import.png"))

    def open_image(self, image_name):

        self.view.setPixmap(QtGui.QPixmap(image_name))
        self.view.fitInView(self.view.pixmap_item, QtCore.Qt.KeepAspectRatio)
        self.printAct.setEnabled(True)

        self.polygonAct.setEnabled(True)
        self.circleAct.setEnabled(True)
        self.squareAct.setEnabled(True)

        self.exportAnnToYoloBoxAct.setEnabled(True)
        self.exportAnnToYoloSegAct.setEnabled(True)
        self.exportAnnToCOCOAct.setEnabled(True)


        image = cv2.imread(image_name)
        self.cv2_image = image

        self.image_set = True

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
        QMessageBox.about(self, "Annotator Light",
                          "<p><b>Annotator Light</b></p>"
                          "<p>Программа для разметки изображений</p>" if
                          self.settings.read_lang() == 'RU' else "<p>Labeling Data for Object Detection and Instance Segmentation</p>")

    def filter_images_names(self, images_dir):
        im_names_valid = []
        file_names = os.listdir(images_dir)
        for f in file_names:
            if not os.path.isfile(os.path.join(images_dir, f)):
                continue
            is_valid = False
            for t in self.image_types:
                if is_valid:
                    break
                if f.endswith('.' + t):
                    is_valid = True
                    im_names_valid.append(f)
        return im_names_valid

    def createNewProject(self):

        if self.loaded_proj_name:
            self.close_project()

        self.create_new_proj_dialog = CreateProjectDialog(self, on_ok_clicked=self.on_create_proj_ok)
        self.create_new_proj_dialog.show()

    def on_create_proj_ok(self):

        dataset_dir = self.create_new_proj_dialog.get_image_folder()

        if not dataset_dir:
            return

        self.dataset_dir = dataset_dir
        self.project_data.set_path_to_images(dataset_dir)
        dataset_images = self.filter_images_names(dataset_dir)

        if not dataset_images:
            self.statusBar().showMessage(
                f"В указанной папке изображений не обнаружено" if self.settings.read_lang() == 'RU' else "Folder is empty",
                3000)
            return

        self.dataset_images = dataset_images

        self.statusBar().showMessage(
            f"Число загруженны в проект изображений: {len(self.dataset_images)}" if self.settings.read_lang() == 'RU' else f"Loaded images count: {len(self.dataset_images)}",
            3000)
        self.loaded_proj_name = self.create_new_proj_dialog.get_project_name()
        self.create_new_proj_dialog.hide()
        self.save_project_as(proj_name=self.loaded_proj_name, on_save_callback=self.on_created_project_saved)

    def on_created_project_saved(self):
        self.load_project(self.loaded_proj_name)

    def fill_images_label(self, image_names):

        self.images_list_widget.clear()
        for name in image_names:
            self.images_list_widget.addItem(name)

    def progress_bar_changed(self, percent):
        self.progress_bar.set_progress(percent)
        if percent == 100:
            self.progress_bar.hide()

    def on_change_project_name(self):
        dataset_dir = self.change_path_window.getEditText()

        if not os.path.exists(dataset_dir):

            msgbox = QMessageBox()
            msgbox.setIcon(QMessageBox.Information)
            msgbox.setText(
                f"Ошибка загрузки проекта. " if self.settings.read_lang() == 'RU' else f"Error in loading project")
            if self.settings.read_lang() == 'RU':
                msgbox.setInformativeText(
                    f"Директория {dataset_dir} не существует"
                )
            else:
                msgbox.setInformativeText(
                    f"Directory {dataset_dir} doesn't exist")
            msgbox.setWindowTitle(
                f"Ошибка загрузки проекта {self.loaded_proj_name}" if self.settings.read_lang() == 'RU' else "Error")
            msgbox.exec()

            return

        self.project_data.set_path_to_images(dataset_dir)
        self.on_checking_project_success(dataset_dir)

    def on_checking_project_success(self, dataset_dir):

        self.progress_bar = ProgressWindow(self,
                                           title='Загрузка проекта...' if self.settings.read_lang() == 'RU' else 'Loading project...')

        self.view.load_ids_conn.percent.connect(self.progress_bar_changed)

        self.view.set_ids_from_project(self.project_data.get_data())

        self.dataset_dir = dataset_dir
        self.dataset_images = self.filter_images_names(dataset_dir)

        self.fill_labels_combo_from_project()

        if self.dataset_images:
            self.tek_image_name = self.dataset_images[0]
            self.tek_image_path = os.path.join(self.dataset_dir, self.tek_image_name)

            # Открытие изображения
            self.open_image(self.tek_image_path)

            self.fitToWindowAct.setEnabled(True)
            self.zoomInAct.setEnabled(True)
            self.zoomOutAct.setEnabled(True)

            self.selectAreaAct.setEnabled(True)

            self.saveProjAct.setEnabled(True)
            self.saveProjAsAct.setEnabled(True)

            main_geom = self.geometry().getCoords()
            self.scaleFactor = (main_geom[2] - main_geom[0]) / self.cv2_image.shape[1]

            self.load_image_data(self.tek_image_name)

            self.view.setMouseTracking(True)
            self.fill_labels_on_tek_image_list_widget()
            self.fill_images_label(self.dataset_images)

            self.im_panel_count_conn.on_image_count_change.emit(len(self.dataset_images))
            self.images_list_widget.setCurrentRow(0)

            self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

        self.statusBar().showMessage(
            f"Число загруженных в проект изображений: {len(self.dataset_images)}" if self.settings.read_lang() == 'RU' else f"Loaded images count: {len(self.dataset_images)}",
            3000)

    def load_project(self, project_name):

        if not project_name:
            return

        if self.loaded_proj_name:
            self.close_project()

        self.loaded_proj_name = project_name
        is_success = self.project_data.load(project_name)

        if not is_success:
            msgbox = QMessageBox()
            msgbox.setIcon(QMessageBox.Information)
            msgbox.setText(
                f"Ошибка открытия файла {self.loaded_proj_name}" if self.settings.read_lang() == 'RU' else f"Error in opening file {self.loaded_proj_name}")
            if self.settings.read_lang() == 'RU':
                msgbox.setInformativeText(
                    f"Файл {self.loaded_proj_name} должен быть в формате .json и содержать поля:\n\t"
                    f"path_to_images\n\timages\n\t\tfilename\n\t\tshapes\n\t")
            else:
                msgbox.setInformativeText(
                    f"File {self.loaded_proj_name} not in project format")
            msgbox.setWindowTitle(
                f"Ошибка открытия файла {self.loaded_proj_name}" if self.settings.read_lang() == 'RU' else "Error")
            msgbox.exec()

            return

        dataset_dir = self.project_data.get_image_path()

        if not os.path.exists(dataset_dir):
            # Ask for change path

            self.change_path_window = EditWithButton(None, in_separate_window=True,
                                                     theme=self.settings.read_theme(),
                                                     on_button_clicked_callback=self.on_change_project_name,
                                                     is_dir=True, dialog_text='Input path to images',
                                                     title=f"Директория {dataset_dir} не существует",
                                                     placeholder='Input path to images')
            self.change_path_window.show()

            return

        self.on_checking_project_success(dataset_dir)

    def open_project(self):
        """
        Загрузка проекта
        """
        loaded_proj_name, _ = QFileDialog.getOpenFileName(self,
                                                          'Загрузка проекта' if self.settings.read_lang() == 'RU' else "Loading project",
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

    def close_project(self):
        if self.project_data.is_loaded:
            self.project_data.clear()
            self.cls_combo.clear()
            self.labels_on_tek_image.clear()
            self.images_list_widget.clear()
            self.del_im_from_proj_clicked()

    def save_project(self):
        """
        Сохранение проекта
        """
        self.start_gif(is_prog_load=True)
        if self.loaded_proj_name:

            self.set_labels_color()  # сохранение информации о цветах масок
            self.write_scene_to_project_data()

            self.project_data.save(self.loaded_proj_name)

            self.fill_project_labels()

            self.statusBar().showMessage(
                f"Проект успешно сохранен" if self.settings.read_lang() == 'RU' else "Project is saved", 3000)

        else:
            self.save_project_as()

        self.splash.finish(self)

    def save_project_as(self, proj_name=None, on_save_callback=None):
        """
        Сохранение проекта как...
        """
        if not proj_name:
            proj_name, _ = QFileDialog.getSaveFileName(self,
                                                       'Выберите имя нового проекта' if self.settings.read_lang == 'RU' else 'Type new project name',
                                                       'projects',
                                                       'JSON Proj File (*.json)')

        if proj_name:
            self.loaded_proj_name = proj_name
            self.set_labels_color()  # сохранение информации о цветах масок
            self.write_scene_to_project_data()
            self.fill_project_labels()
            self.loaded_proj_name = proj_name

            if not on_save_callback:
                on_save_callback = self.on_project_saved
            self.project_data.save(proj_name, on_save_callback=on_save_callback)

    def on_project_saved(self):
        self.statusBar().showMessage(
            f"Проект успешно сохранен" if self.settings.read_lang() == 'RU' else "Project is saved", 3000)

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
        Показать окно с настройками приложения
        """

        self.settings_window = SettingsWindowBase(self)

        self.settings_window.okBtn.clicked.connect(self.on_settings_closed)
        self.settings_window.cancelBtn.clicked.connect(self.on_settings_closed)

        self.settings_window.show()

    def change_theme(self):
        """
        Изменение темы приложения
        """
        app = QApplication.instance()

        # primary_color = "#ffffff"

        theme = self.settings.read_theme()
        icon_folder = self.settings.get_icon_folder()

        # if 'light' in theme:
        #     primary_color = "#000000"

        density = hf.density_slider_to_value(self.settings.read_density())

        extra = {'density_scale': density,
                 # 'font_size': '14px',
                 # 'primaryTextColor': primary_color
                 }

        apply_stylesheet(app, theme=theme, extra=extra)

        self.on_theme_change_connection.on_theme_change.emit(icon_folder)

    def on_settings_closed(self):
        """
        При закрытии окна настроек приложения

        """
        lang = self.settings.read_lang()

        theme = self.settings.read_theme()
        if theme != self.last_theme:
            self.last_theme = theme
            self.change_theme()

        self.set_icons()

        fat_width = self.settings.read_fat_width()
        alpha = self.settings.read_alpha()

        if alpha != self.last_alpha:
            self.last_alpha = alpha
            self.reload_image()

        if fat_width != self.last_fat_width:
            self.last_fat_width = fat_width
            self.view.set_fat_width(self.settings.read_fat_width())
            self.reload_image()

        self.statusBar().showMessage(
            f"Настройки проекта изменены" if lang == 'RU' else "Settings is saved", 3000)

    def polygon_tool_pressed(self):

        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings.read_alpha()
        self.ann_type = "Polygon"
        self.view.start_drawing(self.ann_type, cls_num, color=label_color, alpha=alpha_tek)

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

        alpha_tek = self.settings.read_alpha()
        self.ann_type = "Box"
        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def circle_pressed(self):
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings.read_alpha()
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
                alpha_tek = self.settings.read_alpha()

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

    def end_drawing(self):

        if not self.image_set:
            return

        if self.is_rubber_band_mode:
            # Режим селекции области на изображении

            # Получаем полигон селекта
            pol = self.view.get_rubber_band_polygon()
            left_top_x, left_top_y = int(pol[0][0]), int(pol[0][1])
            right_bottom_x, right_bottom_y = int(pol[2][0]), int(pol[2][1])

            # Обрезаем изображение и сохраняем
            cropped_image = self.cv2_image[left_top_y:right_bottom_y, left_top_x:right_bottom_x]
            cropped_image = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2RGB)

            image_name = hf.create_unique_image_name(self.tek_image_name)
            im_full_name = os.path.join(self.dataset_dir, image_name)

            cv2.imwrite(im_full_name, cropped_image)

            # Добавляем в набор картинок и в панель
            if image_name not in self.dataset_images:
                self.dataset_images.append(image_name)

            self.fill_images_label(self.dataset_images)

            # Отключаем режим выделения области
            self.is_rubber_band_mode = False
            self.rubber_band_change_conn.on_rubber_mode_change.emit(False)

            return

        if self.ann_type in ["Polygon", "Box", "Ellips"]:
            self.view.end_drawing()  # save it to

            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()

            self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def start_drawing(self):
        if not self.image_set:
            return

        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings.read_alpha()

        if self.is_rubber_band_mode:
            self.rubber_band_change_conn.on_rubber_mode_change.emit(False)
            self.is_rubber_band_mode = False

        self.view.start_drawing(self.ann_type, cls_num=cls_num, color=label_color, alpha=alpha_tek)

    def break_drawing(self):
        if not self.image_set:
            return

        if not "Continue" in self.view.drag_mode:
            self.view.remove_active()

        if self.tek_image_name:
            self.view.end_drawing()
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()

        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def reload_image(self):
        if not self.tek_image_path:
            return

        self.open_image(self.tek_image_path)
        self.load_image_data(self.tek_image_name)
        self.fill_labels_on_tek_image_list_widget()
        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def show_tutorial(self):
        path_to_png = os.path.join(os.getcwd(), 'ui', 'tutorial', 'shortcuts.png')
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

            if not self.image_set:
                return

            # <<< Before image
            self.write_scene_to_project_data()

            current_idx = self.images_list_widget.currentRow()
            next_idx = current_idx - 1 if current_idx > 0 else self.images_list_widget.count() - 1
            if self.get_before_image_name():
                self.reload_image()
                self.images_list_widget.setCurrentRow(next_idx)

        elif e.key() == 46 or e.key() == 1070:

            if not self.image_set:
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
            if not self.image_set:
                return

            self.view.paste_buffer()
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()
            self.view.clear_ai_points()

    def on_quit(self):
        self.exit_box.hide()

        self.write_size_pos()

        self.hide()  # Скрываем окно

        self.is_asked_before_close = True
        self.close()

    def closeEvent(self, event):
        if self.is_asked_before_close:
            event.accept()
        else:
            self.exit_box = OkCancelDialog(self, title='Quit', text='Are you really want to quit?', on_ok=self.on_quit)
            event.ignore()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             # 'primaryTextColor': '#ffffff'
             }

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = MainWindow()
    w.show()
    sys.exit(app.exec_())
