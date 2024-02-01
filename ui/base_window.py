import os
import shutil
import sys

import cv2
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QMovie, QIcon, QKeySequence, QColor, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QAction, QFileDialog, QMessageBox, QMenu, QToolBar, QToolButton, QLabel, \
    QColorDialog, QProgressBar, QApplication
from qt_material import apply_stylesheet

from ui.ask_del_polygon import AskDelWindow
from ui.custom_widgets.edit_with_button import EditWithButton
from ui.custom_widgets.styled_widgets import StyledComboBox
from ui.custom_widgets.toolbars import ProgressBarToolbar
from ui.dialogs.create_project_dialog import CreateProjectDialog
from ui.dialogs.export_dialog import ExportDialog
from ui.dialogs.import_dialogs import ImportFromYOLODialog, ImportFromCOCODialog, ImportLRMSDialog
from ui.dialogs.input_dialog import CustomInputDialog, CustomComboDialog
from ui.dialogs.ok_cancel_dialog import OkCancelDialog
from ui.list_widgets.images_widget import ImagesWidget
from ui.list_widgets.labels_widget import LabelsWidget
from ui.panels import ImagesPanel, LabelsPanel
from ui.settings_window_base import SettingsWindowBase
from ui.shortcuts_editor import ShortCutsEditor
from ui.signals_and_slots import ImagesPanelCountConnection, LabelsPanelCountConnection, ThemeChangeConnection, \
    RubberBandModeConnection
from ui.splash_screen import MovieSplashScreen
from ui.view import GraphicsView
from utils import config
from utils import help_functions as hf
from utils.importer import Importer
from utils.project import ProjectHandler
from utils.settings_handler import AppSettings
from utils.settings_handler import shortcuts as shortcuts_init
from utils.states import DrawState, WindowState, ViewState

basedir = os.path.dirname(__file__)


class MainWindow(QtWidgets.QMainWindow):
    """
    Базовое окно для работы с разметкой без поддержки моделей ИИ
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Annotator")

        # Settings
        self.settings = AppSettings()
        self.icon_folder = self.settings.get_icon_folder()
        # Work with project
        self.project_data = ProjectHandler()
        # Some settings
        self.init_global_values()

        # Start on Loading Animation
        # self.start_gif(is_prog_load=True)

        # Rubber band
        self.window_state = WindowState.normal
        self.rubber_band_change_conn = RubberBandModeConnection()

        # GraphicsView
        self.view = GraphicsView(on_rubber_band_mode=self.rubber_band_change_conn.on_rubber_mode_change)

        # Signals with View
        self.view.polygon_clicked.id_pressed.connect(self.polygon_pressed)
        self.view.polygon_delete.id_delete.connect(self.on_polygon_delete)
        self.view.polygon_end_drawing.on_end_drawing.connect(self.on_polygon_end_draw)
        self.view.list_of_polygons_conn.list_of_polygons.connect(self.on_rubber_band_end)

        self.view.polygon_cls_num_change.pol_cls_num_and_id.connect(self.change_polygon_cls_num)
        self.view.info_conn.info_message.connect(self.info_message)

        # Signals with Right Panels
        self.im_panel_count_conn = ImagesPanelCountConnection()
        self.labels_count_conn = LabelsPanelCountConnection()
        self.selected_count_conn = LabelsPanelCountConnection()
        self.on_theme_change_connection = ThemeChangeConnection()

        lay = QtWidgets.QSplitter(QtCore.Qt.Horizontal)  # QtWidgets.QWidget()

        self.create_right_toolbar()
        lay.addWidget(self.view)
        lay.addWidget(self.toolBarRight)

        lay.setStretchFactor(0, 4)
        self.setCentralWidget(lay)

        # Menu and toolbars
        self.createActions()
        self.createMenus()
        self.createToolbar()

        # Icons
        self.change_theme()
        self.set_icons()

        # Printer
        self.printer = QPrinter()

        # self.splash.finish(self)
        self.statusBar().showMessage(
            "Загрузите проект или набор изображений" if self.lang == 'RU' else "Load dataset or project")

    def init_global_values(self):
        """
        Set some app global values
        """
        self.lang = self.settings.read_lang()

        self.scaleFactor = 1.0
        self.draw_state = DrawState.polygon

        self.loaded_proj_name = None
        self.labels_on_tek_image_ids = None
        self.tek_image_name = None
        self.tek_image_path = None
        self.dataset_dir = None

        self.last_fat_width = None
        self.last_theme = self.settings.read_theme()
        self.last_density = self.settings.read_density()

        self.lrm = None  # ЛРМ снимка

        self.image_types = ['jpg', 'png', 'tiff', 'jpeg', 'tif']

        self.dataset_images = []

        # import settings
        self.is_seg_import = False

        # close window flag
        self.is_asked_before_close = False

        # Set window size and pos from last state
        self.read_size_pos()

        # Image hide labels
        self.saved_image_data = None

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

    def info_message(self, message):
        time = 3000
        self.statusBar().showMessage(message, time)

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

    def reset_shortcuts(self):
        shortcuts = self.settings.read_shortcuts()
        # print(shortcuts)

        for sc, act in zip(
                ['copy', 'crop', 'del', 'end_drawing', 'fit', 'hand_move', 'image_after', 'image_before',
                 'open_project', 'paste', 'polygon', 'print', 'quit',
                 'save_project', 'settings', 'start_drawing', 'undo', 'zoom_in', 'zoom_out', 'change_polygon_label',
                 'hide_labels', 'toggle_image_status'],
                [self.copyAct, self.selectAreaAct, self.deleteLabelAct, self.stopDrawAct, self.fitToWindowAct,
                 self.handMoveWindowAct,
                 self.goNextAct, self.goBeforeAct, self.openProjAct, self.pasteAct, self.polygonAct, self.printAct,
                 self.exitAct,
                 self.saveProjAct, self.settingsAct, self.startDrawAct, self.undoAct, self.zoomInAct, self.zoomOutAct,
                 self.changePolygonLabelAct, self.hideLabelsAct, self.toggleImageStatusAct]):
            shortcut = shortcuts.get(sc, shortcuts_init[sc])
            appearance = shortcut['appearance']
            if appearance == 'Ctrl+Plus':
                seq = QKeySequence.ZoomIn
            else:
                seq = QKeySequence(appearance)
            act.setShortcut(seq)

    def createActions(self):

        self.copyAct = QAction(
            "Копировать метку" if self.lang == 'RU' else "Copy label",
            self, triggered=self.copy_label)

        self.pasteAct = QAction(
            "Вставить метку" if self.lang == 'RU' else "Paste label",
            self, triggered=self.paste_label)

        self.undoAct = QAction(
            "Отменить" if self.lang == 'RU' else "Undo",
            self, triggered=self.undo)

        self.deleteLabelAct = QAction(
            "Удалить метку" if self.lang == 'RU' else "Delete label",
            self, triggered=self.break_drawing)

        self.startDrawAct = QAction(
            "Начать рисовать метку" if self.lang == 'RU' else "Start drawing label",
            self, triggered=self.start_drawing)

        self.stopDrawAct = QAction(
            "Закончить рисовать метку" if self.lang == 'RU' else "Stop drawing label",
            self, triggered=self.end_drawing)

        self.goNextAct = QAction(
            "Следующее изображение" if self.lang == 'RU' else "Next Image",
            self, triggered=self.go_next)

        self.goBeforeAct = QAction(
            "Предыдущее изображение" if self.lang == 'RU' else "Before Image",
            self, triggered=self.go_before)

        self.createNewProjectAct = QAction(
            "Создать новый проект" if self.lang == 'RU' else "Create new project",
            self, triggered=self.createNewProject)
        self.openProjAct = QAction("Загрузить проект" if self.lang == 'RU' else "Load Project", self,
                                   triggered=self.open_project)
        self.saveProjAsAct = QAction(
            "Сохранить проект как..." if self.lang == 'RU' else "Save project as...",
            self, triggered=self.save_project_as, enabled=False)
        self.saveProjAct = QAction("Сохранить проект" if self.lang == 'RU' else "Save project", self,
                                   triggered=self.save_project, enabled=False)

        self.printAct = QAction("Печать" if self.lang == 'RU' else "Print", self,
                                enabled=False, triggered=self.print_)
        self.exitAct = QAction("Выход" if self.lang == 'RU' else "Exit", self,
                               triggered=self.close)

        self.zoomInAct = QAction("Увеличить" if self.lang == 'RU' else "Zoom In", self,
                                 enabled=False,
                                 triggered=self.zoomIn)
        self.zoomOutAct = QAction("Уменьшить" if self.lang == 'RU' else "Zoom Out", self,
                                  enabled=False,
                                  triggered=self.zoomOut)
        self.fitToWindowAct = QAction(
            "Подогнать под размер окна" if self.lang == 'RU' else "Fit to window size",
            self, enabled=False,
            triggered=self.fitToWindow)
        self.handMoveWindowAct = QAction(
            "Перемещение рукой" if self.lang == 'RU' else "Hand navigation",
            self, enabled=False, checkable=True,
            triggered=self.on_hand_move_clicked)

        self.aboutAct = QAction("О модуле" if self.lang == 'RU' else "About", self,
                                triggered=self.about)
        self.shortCutsEditAct = QAction("Горячие клавиши" if self.lang == 'RU' else "Shortcuts", self,
                                        triggered=self.show_shortcuts)

        self.settingsAct = QAction("Настройки приложения" if self.lang == 'RU' else "Settings", self,
                                   enabled=True, triggered=self.showSettings)

        # Annotators
        self.polygonAct = QAction("Полигон" if self.lang == 'RU' else "Polygon", self, enabled=False,
                                  triggered=self.polygon_tool_pressed, checkable=True)
        self.circleAct = QAction("Эллипс" if self.lang == 'RU' else "Ellipse", self, enabled=False,
                                 triggered=self.circle_pressed, checkable=True)
        self.squareAct = QAction("Прямоугольник" if self.lang == 'RU' else "Box", self, enabled=False,
                                 triggered=self.square_pressed,
                                 checkable=True)
        self.hideLabelsAct = QAction("Скрыть разметку" if self.lang == 'RU' else "Hide labels", self, enabled=False,
                                     triggered=self.toggle_hide_labels, checkable=True)

        # Данные о текущем разметчике
        self.addUserAct = QAction(
            "Добавить нового пользователя" if self.lang == 'RU' else "Add new user", self,
            enabled=True, triggered=self.add_user_clicked)
        self.renameUserAct = QAction(
            "Изменить имя пользователя" if self.lang == 'RU' else "Change user name", self,
            triggered=self.rename_user)
        self.deleteUserAct = QAction(
            "Удалить имя пользователя" if self.lang == 'RU' else "Delete user name", self,
            triggered=self.delete_user)

        # Export
        self.exportAnnAct = QAction(
            "Экспортировать" if self.lang == 'RU' else "Export", self,
            enabled=False,
            triggered=self.exportDataset)

        # Import
        self.importAnnFromYoloBoxAct = QAction(
            "YOLO (Box)" if self.lang == 'RU' else "YOLO (Boxes)", self,
            enabled=True,
            triggered=self.importFromYOLOBox)
        self.importAnnFromYoloSegAct = QAction(
            "YOLO (Seg)" if self.lang == 'RU' else "YOLO (Seg)", self,
            enabled=True,
            triggered=self.importFromYOLOSeg)
        self.importAnnFromCOCOAct = QAction(
            "COCO" if self.lang == 'RU' else "COCO", self, enabled=True,
            triggered=self.importFromCOCO)

        # Labels
        self.add_label = QAction("Добавить новый класс" if self.lang == 'RU' else "Add new label", self,
                                 enabled=True, triggered=self.add_label_button_clicked)
        self.del_label = QAction(
            "Удалить текущий класс" if self.lang == 'RU' else "Delete current label",
            self,
            enabled=True, triggered=self.del_label_button_clicked)
        self.change_label_color = QAction(
            "Изменить цвет разметки для текущего класса" if self.lang == 'RU' else "Change label color",
            self,
            enabled=True,
            triggered=self.change_label_color_button_clicked)
        self.rename_label = QAction("Изменить имя класса" if self.lang == 'RU' else "Rename", self,
                                    enabled=True,
                                    triggered=self.rename_label_button_clicked)
        self.changePolygonLabelAct = QAction(
            "Изменить имя метки" if self.lang == 'RU' else "Rename label", self,
            enabled=True,
            triggered=self.change_polygon_label_clicked)

        # Image actions

        self.selectAreaAct = QAction(
            "Выделить область как новое изображение" if self.lang == 'RU' else "Save image crop",
            self,
            enabled=False, triggered=self.getArea)

        self.saveSelectedPolygonAsImage = QAction(
            "Сохранить активную область как новое изображение" if self.lang == 'RU' else "Save active as image",
            self,
            enabled=False, triggered=self.save_active_item_as_image)

        self.load_lrm_data_act = QAction(
            "Загрузить данные о ЛРМ" if self.lang == 'RU' else "Load linear ground res. data", self,
            enabled=False, triggered=self.load_lrm_data_pressed)

        self.ruler_act = QAction(
            "Линейка" if self.lang == 'RU' else "Ruler", self,
            enabled=False, triggered=self.ruler_pressed, checkable=True)

        self.toggleImageStatusAct = QAction(
            "Переключить статус изображения" if self.lang == 'RU' else "Change image status", self,
            enabled=False, triggered=self.on_toggle_image_status)

        self.reset_shortcuts()

    def createMenus(self):

        self.fileMenu = QMenu("&Файл" if self.lang == 'RU' else "&File", self)
        self.fileMenu.addAction(self.createNewProjectAct)
        self.fileMenu.addAction(self.openProjAct)
        self.fileMenu.addAction(self.saveProjAct)
        self.fileMenu.addAction(self.saveProjAsAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        #
        self.viewMenu = QMenu("&Изображение" if self.lang == 'RU' else "&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)
        self.viewMenu.addAction(self.handMoveWindowAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.selectAreaAct)
        self.viewMenu.addAction(self.saveSelectedPolygonAsImage)

        #
        self.annotatorMenu = QMenu("&Аннотация" if self.lang == 'RU' else "&Labeling", self)

        self.AnnotatorMethodMenu = QMenu("Способ выделения" if self.lang == 'RU' else "Method", self)

        self.AnnotatorMethodMenu.addAction(self.polygonAct)
        self.AnnotatorMethodMenu.addAction(self.squareAct)
        self.AnnotatorMethodMenu.addAction(self.circleAct)

        self.annotatorMenu.addMenu(self.AnnotatorMethodMenu)

        self.annotatorImportMenu = QMenu("Импорт" if self.lang == 'RU' else "Import", self)
        self.annotatorImportMenu.addAction(self.importAnnFromYoloBoxAct)
        self.annotatorImportMenu.addAction(self.importAnnFromYoloSegAct)
        self.annotatorImportMenu.addAction(self.importAnnFromCOCOAct)

        self.annotatorMenu.addAction(self.changePolygonLabelAct)
        self.annotatorMenu.addAction(self.hideLabelsAct)
        self.annotatorMenu.addAction(self.toggleImageStatusAct)

        # Dataset menu
        self.datasetMenu = QMenu("Датасет" if self.lang == 'RU' else "Dataset", self)
        self.datasetMenu.addMenu(self.annotatorImportMenu)
        self.datasetMenu.addAction(self.exportAnnAct)
        self.datasetMenu.addAction(self.load_lrm_data_act)
        #
        self.settingsMenu = QMenu("Настройки" if self.lang == 'RU' else "Settings", self)
        self.settingsMenu.addAction(self.settingsAct)
        self.settingsMenu.addAction(self.shortCutsEditAct)
        #
        self.helpMenu = QMenu("&Помощь" if self.lang == 'RU' else "Help", self)
        self.helpMenu.addAction(self.aboutAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)

        self.menuBar().addMenu(self.annotatorMenu)
        self.menuBar().addMenu(self.datasetMenu)
        self.menuBar().addMenu(self.settingsMenu)
        self.menuBar().addMenu(self.helpMenu)

    def create_left_toolbar(self):
        # Left

        toolBar = QToolBar("Панель инструментов" if self.lang == 'RU' else "ToolBar", self)
        toolBar.addAction(self.openProjAct)
        toolBar.addSeparator()
        toolBar.addAction(self.zoomInAct)
        toolBar.addAction(self.zoomOutAct)
        toolBar.addAction(self.fitToWindowAct)
        toolBar.addAction(self.handMoveWindowAct)
        toolBar.addSeparator()
        toolBar.setMovable(True)

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

    def compose_label_panel(self):
        # Labels
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout()
        label_panel = LabelsPanel(self, self.break_drawing, self.clean_all_labels, self.icon_folder,
                                  on_color_change_signal=self.on_theme_change_connection.on_theme_change,
                                  on_labels_count_change=self.labels_count_conn.on_labels_count_change,
                                  on_selected_num_changed=self.selected_count_conn.on_labels_count_change)

        label_panel.setMaximumHeight(300)
        lay.addWidget(label_panel)

        self.labels_on_tek_image = LabelsWidget(None)
        self.labels_on_tek_image.itemClicked.connect(self.labels_on_tek_image_clicked)
        lay.addWidget(self.labels_on_tek_image)

        panel.setLayout(lay)

        return panel

    def compose_image_panel(self):
        # Images
        panel = QtWidgets.QWidget()
        lay = QtWidgets.QVBoxLayout()
        lay.addWidget(
            ImagesPanel(self, self.add_im_to_proj_clicked, self.del_im_from_proj_clicked, self.change_image_status,
                        self.icon_folder,
                        on_color_change_signal=self.on_theme_change_connection.on_theme_change,
                        on_images_list_change=self.im_panel_count_conn.on_image_count_change))

        self.image_panel_progress_bar = QProgressBar()
        self.image_panel_progress_bar.setMinimum(0)
        self.image_panel_progress_bar.setMaximum(100)
        self.image_panel_progress_bar.setMaximumHeight(5)
        lay.addWidget(self.image_panel_progress_bar)

        self.images_list_widget = ImagesWidget(self, self.settings.get_icon_folder())  # QListWidget()
        self.images_list_widget.itemClicked.connect(self.images_list_widget_clicked)
        lay.addWidget(self.images_list_widget)
        panel.setLayout(lay)

        return panel

    def create_right_toolbar(self):
        # Right toolbar
        self.toolBarRight = QtWidgets.QSplitter(
            QtCore.Qt.Vertical)  # "Менеджер разметок" if self.lang == 'RU' else "Labeling Bar",

        # Labels
        label_panel = self.compose_label_panel()
        self.toolBarRight.addWidget(label_panel)
        image_panel = self.compose_image_panel()
        self.toolBarRight.addWidget(image_panel)

    def save_view_to_project(self):
        """
        Записывает текущую картину со сцены в проект
        """
        self.write_scene_to_project_data()
        self.fill_labels_on_tek_image_list_widget()
        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def clean_all_labels(self):
        self.view.remove_all_polygons()

        if self.tek_image_name:
            cls_txt = self.cls_combo.currentText()
            cls_num = self.cls_combo.currentIndex()
            label_color = self.project_data.get_label_color(cls_txt)
            self.view.end_drawing(cls_num=cls_num, text=cls_txt, color=label_color)  #
            self.save_view_to_project()

    def create_top_toolbar(self):

        self.labelSettingsToolBar = QToolBar(
            "Настройки разметки" if self.lang == 'RU' else "Current Label Bar",
            self)

        # имена классов
        self.cls_combo = StyledComboBox()

        label = QLabel("  Текущий класс:   " if self.lang == 'RU' else "  Current label:   ")
        cls_names = np.array(['no name'])
        self.cls_combo.addItems(cls_names)
        self.cls_combo.setMinimumWidth(150)

        self.cls_combo.setEnabled(False)

        self.labelSettingsToolBar.addWidget(label)
        self.labelSettingsToolBar.addWidget(self.cls_combo)

        # кнопки справа от имен классов
        self.labelSettingsToolBar.addAction(self.change_label_color)
        self.labelSettingsToolBar.addAction(self.rename_label)
        self.labelSettingsToolBar.addAction(self.del_label)
        self.labelSettingsToolBar.addSeparator()
        self.labelSettingsToolBar.addAction(self.add_label)
        self.labelSettingsToolBar.addSeparator()

        # рулетка
        self.labelSettingsToolBar.addAction(self.ruler_act)
        self.labelSettingsToolBar.addSeparator()

        # имя пользователя
        self.user_names_combo = StyledComboBox()
        user_label = QLabel("  Текущий пользователь:   " if self.lang == 'RU' else "  Current user:   ")
        user_name_variants = np.array(self.settings.read_username_variants())
        cur_user = self.settings.read_username()
        self.user_names_combo.addItems(user_name_variants)
        self.user_names_combo.setCurrentText(cur_user)
        self.user_names_combo.currentTextChanged.connect(self.change_user_name)

        self.user_names_combo.setMinimumWidth(150)

        self.labelSettingsToolBar.addWidget(user_label)
        self.labelSettingsToolBar.addWidget(self.user_names_combo)
        self.labelSettingsToolBar.addSeparator()
        self.labelSettingsToolBar.addAction(self.addUserAct)
        self.labelSettingsToolBar.addAction(self.renameUserAct)
        self.labelSettingsToolBar.addAction(self.deleteUserAct)
        self.labelSettingsToolBar.addSeparator()

        # прогресс-бар
        self.progress_toolbar = ProgressBarToolbar(self,
                                                   right_padding=self.images_list_widget.width())

        self.labelSettingsToolBar.addWidget(self.progress_toolbar)

        self.addToolBar(QtCore.Qt.TopToolBarArea, self.labelSettingsToolBar)

    def createToolbar(self):

        self.create_left_toolbar()
        self.create_top_toolbar()

    def getArea(self):
        """
        Действие - выбрать область
        """
        self.break_drawing()  # Если до этого что-то рисовали - сбросить
        self.window_state = WindowState.rubber_band
        self.rubber_band_change_conn.on_rubber_mode_change.emit(True)

    def save_selected_polygon_as_image(self, pol, im_full_name):
        pts = np.array(pol)

        # (1) Crop the bounding rect
        rect = cv2.boundingRect(pts)
        x, y, w, h = rect
        cropped = self.cv2_image[y:y + h, x:x + w].copy()
        # (2) make mask
        pts = pts - pts.min(axis=0)

        mask = np.zeros(cropped.shape[:2], np.uint8)
        cv2.drawContours(mask, [pts], -1, (255, 255, 255), -1, cv2.LINE_AA)

        # (3) do bit-op
        dst = cv2.bitwise_and(cropped, cropped, mask=mask)

        cv2.imwrite(im_full_name, dst)

    def save_polygon_as_image(self, pol):
        if pol:
            image_name = hf.create_unique_image_name(self.tek_image_name)
            im_full_name = os.path.join(self.dataset_dir, image_name)

            self.save_selected_polygon_as_image(pol, im_full_name)

            # Добавляем в набор картинок и в панель
            if image_name not in self.dataset_images:
                self.dataset_images.append(image_name)

            self.fill_images_label(self.dataset_images)

            self.save_view_to_project()

            self.tek_image_name = image_name
            self.tek_image_path = im_full_name
            self.reload_image(is_tek_image_changed=True)
            self.images_list_widget.move_to_image_name(image_name)

        else:
            self.info_message(
                "Пожалуйста, выделите полигон на изображении" if self.lang == 'RU' else "Please select some polygon "
                                                                                        "on image as active")

        self.window_state = WindowState.normal

    def save_active_item_as_image(self):
        if not self.image_set:
            if self.lang == 'RU':
                message = "Подождите окончания загрузки изображения"
            else:
                message = "Wait for the image to finish loading"
            self.info_message(message)

            return

        # Получаем полигон выделенной области
        pol = self.view.get_active_item_polygon()
        self.save_polygon_as_image(pol)

    def change_polygon_label_clicked(self):
        """
        Изменение имени нажатого полигона горячей клавишей
        """
        self.view.on_change_cls_num()

    def change_polygon_cls_num(self, cls_num, cls_ids):
        """
        Изменение имени нажатого полигона по правой кнопке мыши. Сигнал от view
        """
        labels = self.project_data.get_labels()
        theme = self.settings.read_theme()
        variants = [labels[i] for i in range(len(labels)) if i != cls_num]

        if len(variants) < 2:
            if self.lang == 'RU':
                message = f"Добавьте еще один класс. Пока в списке только один класс {labels[cls_num]}"
            else:
                message = f"Add a different label name. At the moment you only have {labels[cls_num]}."
            self.info_message(message)
            return

        self.combo_dialog = CustomComboDialog(self, theme=theme,
                                              title_name="Изменение имени метки" if self.lang == 'RU' else "Label "
                                                                                                           "name "
                                                                                                           "change",
                                              question_name="Введите имя класса:" if self.lang == 'RU' else "Enter "
                                                                                                            "label "
                                                                                                            "name:",
                                              variants=[labels[i] for i in range(len(labels)) if i != cls_num])

        self.combo_dialog.okBtn.clicked.connect(self.change_cls_num_ok_clicked)
        self.combo_dialog.cancelBtn.clicked.connect(self.combo_dialog.hide)
        self.combo_dialog.show()
        self.changed_cls_num = cls_num
        self.changed_cls_ids = cls_ids

    def change_cls_num_cancel_clicked(self):
        self.combo_dialog.close()
        self.view.active_group.clear()

    def change_cls_num_ok_clicked(self):
        """
        При нажатии OK в окне изменения имени полигона
        """
        # Сохранить текущую сцену в проект.

        self.save_view_to_project()

        new_cls_name = self.combo_dialog.getText()
        self.combo_dialog.hide()

        labels = self.project_data.get_labels()
        new_cls_num = 0
        for i, lbl in enumerate(labels):
            new_cls_num = i
            if lbl == new_cls_name:
                break
        self.project_data.change_cls_num_by_ids(self.tek_image_name, self.changed_cls_ids, new_cls_num)

        self.reload_image(is_tek_image_changed=False)

    def add_im_to_proj_clicked(self):

        if self.dataset_dir:
            last_opened_path = self.settings.read_last_opened_path()
            txt = 'Загрузка изображений в проект' if self.lang == 'RU' else "Loading dataset"
            images, _ = QFileDialog.getOpenFileNames(self,
                                                     txt,
                                                     last_opened_path,
                                                     'Images Files (*.jpeg *.png *.jpg *.tiff)')
            if images and len(images) > 0:  # full names
                self.settings.write_last_opened_path(os.path.dirname(images[0]))

                for im in images:
                    im_base_name = os.path.basename(im)
                    im_folder = os.path.dirname(im)
                    dataset_folder = os.path.dirname(self.tek_image_name)

                    if im_folder != dataset_folder:

                        # Есть изображение с таким именем. Создаем новое уникальное имя
                        if im_base_name in self.dataset_images:
                            im_unique = hf.create_unique_image_name(im_base_name)

                            if self.lang == 'RU':
                                message = f"В датасете уже есть изображение с именем {im_base_name}. " \
                                          f"Будет добавлено как {im_unique}"
                            else:
                                message = f"The dataset already has an image with the name {im_base_name}." \
                                          f" Will be added as {im_unique}"
                            self.info_message(message)

                            im_base_name = im_unique

                        im_new_name = os.path.join(self.dataset_dir, im_base_name)
                        shutil.copy(im, im_new_name)

                    self.dataset_images.append(im_base_name)

                self.fill_images_label(self.dataset_images)

        else:
            self.createNewProject()

        self.im_panel_count_conn.on_image_count_change.emit(len(self.dataset_images))

    def del_im_from_proj_clicked(self):
        if self.lang:
            text = f'Удалить изображение {self.tek_image_name} из проекта?'
        else:
            text = f'Delete image {self.tek_image_name} from project?'

        title = 'Удаление изображения' if self.lang == 'RU' else 'Image delete'

        cancel_text = 'Отмена' if self.lang == 'RU' else 'Cancel'
        ok_text = 'Удалить' if self.lang == 'RU' else 'Delete'

        self.del_im_box = OkCancelDialog(self, title=title, text=text,
                                         on_ok=self.on_del_im_from_project_ok, ok_text=ok_text, cancel_text=cancel_text)
        self.del_im_box.setMinimumWidth(300)

    def on_del_im_from_project_ok(self):

        self.del_im_box.hide()

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
                self.tek_image_name = self.images_list_widget.get_next_name()
                self.tek_image_path = os.path.join(self.dataset_dir, self.tek_image_name)
                self.reload_image(is_tek_image_changed=True)
                self.dataset_images = [image for image in self.dataset_images if image != tek_im_name]

            self.fill_images_label(self.dataset_images)

            self.im_panel_count_conn.on_image_count_change.emit(len(self.dataset_images))

            self.images_list_widget.setCurrentRow(current_idx)

            self.save_view_to_project()

    def change_image_status(self):
        if self.tek_image_name:
            tek_im_name = self.tek_image_name

            tek_status = self.project_data.get_image_status(self.tek_image_name)
            variants = [var for var in ['empty', 'in_work', 'approve'] if var != tek_status]

            last_user = self.project_data.get_image_last_user(tek_im_name)
            if not last_user:
                last_user = ""
            else:
                if self.lang == 'RU':
                    last_user = f"Последние правки сделаны: {last_user}. Текущий статус: {tek_status}"
                else:
                    last_user = f"Last edit user: {last_user}. Current status: {tek_status}"

            if self.lang == 'RU':
                title = f'Изменение статуса изображения {tek_im_name}'
            else:
                title = f'Change image {tek_im_name} status '

            text = f'Выберите новый\nстатус изображения: ' if self.lang == 'RU' else f'Choose new image status: '

            theme = self.settings.read_theme()

            self.change_image_status_dialog = CustomComboDialog(self, theme=theme, variants=variants, editable=True,
                                                                title_name=title,
                                                                question_name=text, post_label=last_user)
            self.change_image_status_dialog.setMinimumWidth(500)

            self.change_image_status_dialog.okBtn.clicked.connect(self.on_change_image_status_ok)
            self.change_image_status_dialog.cancelBtn.clicked.connect(self.change_image_status_dialog.close)
            self.change_image_status_dialog.show()

    def on_change_image_status_ok(self):
        tek_im_name = self.tek_image_name
        new_status = self.change_image_status_dialog.getText()
        self.change_image_status_dialog.close()

        if new_status:
            print(f"Set status {new_status} for image {tek_im_name}")
            self.project_data.set_image_status(tek_im_name, new_status)
            self.project_data.set_image_last_user(tek_im_name, self.settings.read_username())
            self.images_list_widget.set_status(status=new_status)
            self.reset_image_panel_progress_bar()

    def fill_labels_on_tek_image_list_widget(self):

        self.labels_on_tek_image.clear()

        im = self.project_data.get_image_data(self.tek_image_name)  # im = {shapes:[], lrm:float, status:str}

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
            self.labels_on_tek_image.sortItems()

    def images_list_widget_clicked(self, item):

        if self.window_state == WindowState.drawing:
            self.window_state = WindowState.normal
            self.view.break_drawing()

        self.write_scene_to_project_data()

        self.tek_image_name = item.text()
        self.tek_image_path = os.path.join(self.dataset_dir, self.tek_image_name)

        self.open_image(self.tek_image_path)

        self.load_image_data(self.tek_image_name)

        self.fill_labels_on_tek_image_list_widget()
        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())
        items = self.labels_on_tek_image.selectedItems()
        self.selected_count_conn.on_labels_count_change.emit(len(items))

        self.view.setFocus()

    def labels_on_tek_image_clicked(self, item):
        item_id = item.text().split(" ")[-1]
        self.view.activate_item_by_id(int(item_id))
        items = self.labels_on_tek_image.selectedItems()
        self.selected_count_conn.on_labels_count_change.emit(len(items))

    def exportDataset(self):
        self.save_project()

        self.show_export_dialog()

    def show_export_dialog(self):
        test_image_path = os.path.join(self.dataset_dir, self.dataset_images[0])
        self.export_dialog = ExportDialog(test_image_path=test_image_path, on_ok_clicked=self.on_export_dialog_ok,
                                          label_names=self.project_data.get_labels())

        self.export_dialog.show()

    def on_export_dialog_ok(self):
        export_dir = self.export_dialog.get_export_path()
        export_map = self.export_dialog.get_labels_map()
        is_filter_null = self.export_dialog.is_filter_null()

        if export_dir:
            splits = self.export_dialog.get_splits()
            idx, variant = self.export_dialog.get_idx_text_variant()
            sim_idx, sim_text = self.export_dialog.get_idx_text_sim()
            export_format = self.export_dialog.get_export_format()
            new_image_size = self.export_dialog.get_new_image_size()  # None - if no changes

            self.project_data.export_percent_conn.percent.connect(self.export_dialog.set_progress)
            self.project_data.export_finished.on_finished.connect(self.on_project_export)
            self.project_data.export(export_dir, export_map=export_map, format=export_format, variant_idx=idx,
                                     splits=splits, sim=sim_idx, is_filter_null=is_filter_null,
                                     new_image_size=new_image_size)

    def importFromYOLOBox(self):

        self.close_project()

        self.import_dialog = ImportFromYOLODialog(self, on_ok_clicked=self.on_import_yolo_clicked)
        self.is_seg_import = False
        self.import_dialog.show()

    def importFromYOLOSeg(self):
        self.close_project()

        self.import_dialog = ImportFromYOLODialog(self, on_ok_clicked=self.on_import_yolo_clicked)
        self.is_seg_import = True
        self.import_dialog.show()

    def importFromCOCO(self):
        self.close_project()

        self.import_dialog = ImportFromCOCODialog(self, on_ok_clicked=self.on_import_coco_clicked)
        self.import_dialog.show()

    def on_import_yolo_clicked(self):
        yaml_data = self.import_dialog.getData()
        self.new_import_project_name = yaml_data['import_project_path']

        copy_images_path = None
        if yaml_data['is_copy_images']:
            copy_images_path = yaml_data['save_images_dir']

        self.importer = Importer(yaml_data=yaml_data, alpha=self.settings.read_alpha(), is_seg=self.is_seg_import,
                                 copy_images_path=copy_images_path, dataset=yaml_data["selected_dataset"],
                                 is_coco=False)

        self.start_importer()

    def start_importer(self):
        self.importer.load_percent_conn.percent.connect(self.on_import_percent_change)
        self.importer.info_conn.info_message.connect(self.on_importer_message)
        self.importer.err_conn.error_message.connect(self.on_importer_message)

        self.importer.finished.connect(self.on_import_finished)

        if not self.importer.isRunning():
            self.importer.start()

    def on_importer_message(self, message):
        self.info_message(message)

    def on_import_percent_change(self, percent):
        self.import_dialog.set_progress(percent)

    def on_import_coco_clicked(self):

        # dict with keys
        #       'coco_json' - json data,
        #       'coco_name' - path to coco file,
        #       'import_project_path' - path to new project file
        proj_data = self.import_dialog.getData()
        self.new_import_project_name = proj_data['import_project_path']

        if proj_data:
            self.importer = Importer(coco_data=proj_data['coco_json'], alpha=self.settings.read_alpha(),
                                     copy_images_path=self.import_dialog.get_copy_images_path(),
                                     coco_name=proj_data['coco_name'], is_coco=True)

            self.importer.finished.connect(self.on_import_finished)
            self.importer.load_percent_conn.percent.connect(self.on_import_percent_change)

            if not self.importer.isRunning():
                self.importer.start()

    def on_import_finished(self):
        self.project_data.set_data(self.importer.get_project())

        self.fill_labels_combo_from_project()
        self.save_project_as(self.new_import_project_name)

        self.reload_project()
        self.import_dialog.hide()

    def set_color_to_cls(self, cls_name):

        self.project_data.set_label_color(cls_name, alpha=self.settings.read_alpha())

    def on_project_export(self):

        export_format = self.export_dialog.get_export_format()
        self.export_dialog.hide()
        msgbox = QMessageBox()
        msgbox.setIcon(QMessageBox.Information)

        if self.lang == 'RU':
            text = f"Экспорт в формат {export_format} завершен успешно"
        else:
            text = f"Export to {export_format} was successful"

        msgbox.setText(text)
        msgbox.setWindowTitle("Экспорт завершен" if self.lang == 'RU' else "Export completed")
        msgbox.exec()

    def add_new_name_to_combobox(self, new_name):
        # Обновляем список комбо-бокса
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
                                              title_name="Добавление нового класса" if self.lang == 'RU' else "New "
                                                                                                              "label",
                                              question_name="Введите имя класса:" if self.lang == 'RU' else "Enter "
                                                                                                            "label "
                                                                                                            "name:")
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
                    f"Ошибка добавления класса" if self.lang == 'RU' else "Error in setting new label")
                msgbox.setWindowTitle("Ошибка добавления класса" if self.lang == 'RU' else "Error")
                if self.lang == 'RU':
                    info_text = f"Класс с именем {new_name} уже существует"
                else:
                    info_text = f"Label with name {new_name} is already exist. Try again"

                msgbox.setInformativeText(info_text)
                msgbox.exec()
                return

            self.set_labels_color()  # сохранение информации о цветах масок
            self.set_color_to_cls(new_name)  # Добавляем новый цвет в proj_data

            self.add_new_name_to_combobox(new_name)

            # Сохраняем проект

            self.fill_project_labels()
            self.save_view_to_project()

            self.open_image(self.tek_image_path)
            self.load_image_data(self.tek_image_name)
            self.view.setFocus()

    def del_label_button_clicked(self):

        cls_names = [self.cls_combo.itemText(i) for i in range(self.cls_combo.count())]

        if len(cls_names) <= 1:
            msgbox = QMessageBox()
            msgbox.setIcon(QMessageBox.Information)
            msgbox.setText(
                f"Ошибка удаления класса" if self.lang == 'RU' else "Error in deleting label")
            msgbox.setWindowTitle(f"Ошибка удаления класса" if self.lang == 'RU' else "Error")
            msgbox.setInformativeText(
                "Количество классов должно быть хотя бы 2 для удаления текущего" if self.lang == 'RU' else "The last "
                                                                                                           "label "
                                                                                                           "left. If "
                                                                                                           "you don't "
                                                                                                           "like the "
                                                                                                           "name - "
                                                                                                           "just "
                                                                                                           "rename it")
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
            status = self.project_data.get_image_status(self.tek_image_name)
            last_user = self.project_data.get_image_last_user(self.tek_image_name)
            image_updated = {"shapes": shapes, "lrm": self.lrm, "status": status,
                             "last_user": last_user}
            self.project_data.set_image_data(self.tek_image_name, image_updated)

    def remove_label_with_change(self):

        new_name = self.ask_del_label.cls_combo.currentText()  # на что меням
        old_name = self.cls_combo.itemText(self.del_index)

        self.ask_del_label.close()  # закрываем окно

        # 1. Убираем имя из комбо-бокса
        self.del_label_from_combobox(old_name)  # теперь в комбо-боксе нет имени

        # 2. Обновляем все полигоны
        self.project_data.change_data_class_from_to(old_name, new_name)

        # 3. Переоткрываем изображение и рисуем полигоны из проекта
        self.reload_image(is_tek_image_changed=False)

    def on_ask_del_all(self):

        del_name = self.ask_del_label.cls_name

        self.ask_del_label.close()

        # 1. Удаляем данные о цвете из проекта
        self.project_data.delete_data_by_class_name(del_name)

        # 2. Убираем имя класса из комбобокса
        self.del_label_from_combobox(del_name)

        # 3. Переоткрываем изображение и рисуем полигоны из проекта
        self.reload_image(is_tek_image_changed=False)

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

        self.project_data.delete_image(image_name)

    def change_label_color_button_clicked(self):

        color_dialog = QColorDialog()
        cls_txt = self.cls_combo.currentText()
        color_dialog.setWindowTitle(
            f"Выберите цвет для класса {cls_txt}" if self.lang == 'RU' else f"Enter color to label {cls_txt}")
        current_color = self.project_data.get_label_color(cls_txt)
        if not current_color:
            current_color = config.COLORS[self.cls_combo.currentIndex()]

        color_dialog.setCurrentColor(QColor(*current_color))
        color_dialog.setWindowIcon(QIcon(self.icon_folder + "/color.png"))
        color_dialog.exec()
        rgb = color_dialog.selectedColor().getRgb()
        if rgb[0] == 0 and rgb[1] == 0 and rgb[2] == 0:  # black
            return

        rgba = (rgb[0], rgb[1], rgb[2], self.settings.read_alpha())

        self.project_data.set_label_color(cls_txt, color=rgba)

        self.reload_image(is_tek_image_changed=False)

    def rename_label_button_clicked(self):

        cls_name = self.cls_combo.currentText()
        title = f"Редактирование имени класса {cls_name}" if self.lang == 'RU' else f"Rename label {cls_name}"
        question = "Введите новое имя класса:" if self.lang == 'RU' else "Enter new label name:"
        self.input_dialog = CustomInputDialog(self,
                                              title_name=title,
                                              question_name=question)

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

            self.reload_image(is_tek_image_changed=False)

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
        self.handMoveWindowAct.setIcon(QIcon(self.icon_folder + "/open-hand.png"))
        self.toggleImageStatusAct.setIcon(QIcon(self.icon_folder + "/clipboard.png"))

        self.aboutAct.setIcon(QIcon(self.icon_folder + "/info.png"))
        self.settingsAct.setIcon(QIcon(self.icon_folder + "/settings.png"))

        # save load
        self.openProjAct.setIcon(QIcon(self.icon_folder + "/load.png"))
        self.saveProjAsAct.setIcon(QIcon(self.icon_folder + "/save_project.png"))
        self.saveProjAct.setIcon(QIcon(self.icon_folder + "/save.png"))

        # rubber band
        self.selectAreaAct.setIcon(QIcon(self.icon_folder + "/select.png"))
        self.saveSelectedPolygonAsImage.setIcon(QIcon(self.icon_folder + "/polygon_image.png"))

        # export
        self.exportAnnAct.setIcon(QIcon(self.icon_folder + "/export.png"))

        # import
        self.importAnnFromYoloBoxAct.setIcon(QIcon(self.icon_folder + "/yolo.png"))
        self.importAnnFromYoloSegAct.setIcon(QIcon(self.icon_folder + "/yolo_white.png"))
        self.importAnnFromCOCOAct.setIcon(QIcon(self.icon_folder + "/coco.png"))

        # shortcuts
        self.shortCutsEditAct.setIcon(QIcon(self.icon_folder + "/keyboard.png"))

        self.polygonAct.setIcon(QIcon(self.icon_folder + "/polygon.png"))
        self.circleAct.setIcon(QIcon(self.icon_folder + "/circle.png"))
        self.squareAct.setIcon(QIcon(self.icon_folder + "/square.png"))
        self.hideLabelsAct.setIcon(QIcon(self.icon_folder + "/segment_area.png"))

        # labeling
        self.add_label.setIcon((QIcon(self.icon_folder + "/add.png")))
        self.del_label.setIcon((QIcon(self.icon_folder + "/del.png")))
        self.change_label_color.setIcon((QIcon(self.icon_folder + "/color.png")))
        self.rename_label.setIcon((QIcon(self.icon_folder + "/rename.png")))
        self.changePolygonLabelAct.setIcon(QIcon(self.icon_folder + "/label_rename.png"))

        # user
        self.addUserAct.setIcon((QIcon(self.icon_folder + "/add.png")))
        self.renameUserAct.setIcon((QIcon(self.icon_folder + "/rename.png")))
        self.deleteUserAct.setIcon((QIcon(self.icon_folder + "/del.png")))

        # menus

        self.AnnotatorMethodMenu.setIcon(QIcon(self.icon_folder + "/label.png"))
        self.annotatorImportMenu.setIcon(QIcon(self.icon_folder + "/import.png"))

        self.load_lrm_data_act.setIcon(QIcon(self.icon_folder + "/json.png"))
        self.ruler_act.setIcon(QIcon(self.icon_folder + "/ruler3.png"))

    def toggle_act(self, is_active):

        self.fitToWindowAct.setEnabled(is_active)
        self.handMoveWindowAct.setEnabled(is_active)
        self.toggleImageStatusAct.setEnabled(is_active)
        self.handMoveWindowAct.setChecked(not is_active)

        self.zoomInAct.setEnabled(is_active)
        self.zoomOutAct.setEnabled(is_active)

        self.selectAreaAct.setEnabled(is_active)
        self.saveSelectedPolygonAsImage.setEnabled(is_active)

        self.saveProjAct.setEnabled(is_active)
        self.saveProjAsAct.setEnabled(is_active)

        self.printAct.setEnabled(is_active)

        self.polygonAct.setEnabled(is_active)
        self.circleAct.setEnabled(is_active)
        self.squareAct.setEnabled(is_active)
        self.hideLabelsAct.setEnabled(is_active)
        self.hideLabelsAct.setChecked(not is_active)

        self.exportAnnAct.setEnabled(is_active)
        self.cls_combo.setEnabled(is_active)
        self.load_lrm_data_act.setEnabled(is_active)

    def open_image(self, image_name):

        self.view.setPixmap(QtGui.QPixmap(image_name))
        self.view.fitInView(self.view.pixmap_item, QtCore.Qt.KeepAspectRatio)

        image = cv2.imread(image_name)
        self.cv2_image = image

        self.image_set = True
        self.toggle_act(self.image_set)

        lrm = self.read_tek_image_lrm()

        if lrm:
            self.project_data.set_image_lrm(self.tek_image_name, lrm)
            if self.lang == 'RU':
                self.info_message(
                    f"Установлено ЛРМ для снимка {lrm:0.3f} м")
            else:
                self.info_message(
                    f"Linear ground resoxlutions is set to {lrm:0.3f} m")
            self.ruler_act.setEnabled(True)

        else:
            self.ruler_act.setEnabled(False)

        self.lrm = lrm  # Присвоить в любом случае, даже None. Иначе может использоваться lrm от предыдущего снимка

        self.ruler_act.setChecked(False)  # Выключить рулетку. Могла остаться включенной

    def read_tek_image_lrm(self):

        lrm = self.project_data.get_image_lrm(self.tek_image_name)

        if not lrm:
            # if not presented in project_data. Can be load from project by base_window
            # Try read if .tif or find geo file in the same dir.
            lrm = hf.try_read_lrm(self.tek_image_path)

        return lrm

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
                          self.lang == 'RU' else "<p>Labeling Data for Object Detection and Instance Segmentation</p>")

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
            self.info_message(
                f"В указанной папке изображений не обнаружено" if self.lang == 'RU' else "Folder is empty")
            return

        self.dataset_images = dataset_images

        text = "Число загруженных в проект изображений" if self.lang == 'RU' else f"Loaded images count"
        message = f"{text}: {len(self.dataset_images)}"
        self.info_message(message)

        self.loaded_proj_name = self.create_new_proj_dialog.get_project_name()
        self.create_new_proj_dialog.hide()
        self.save_project_as(proj_name=self.loaded_proj_name)
        self.reload_project()

    def fill_images_label(self, image_names):

        self.images_list_widget.clear()
        self.progress_toolbar.show_progressbar()
        self.info_message(
            f"Загружаю информацию о разметке..." if self.lang == 'RU' else f"Loading labels...")
        images_info = self.project_data.get_all_images_info()
        for i, name in enumerate(image_names):
            if name in images_info:
                status = images_info[name].get('status', None)
            else:
                status = None
            percent = int(100 * (i + 1) / len(image_names))
            self.progress_toolbar.set_percent(percent)
            self.images_list_widget.addItem(name, status)
        self.images_list_widget.sortItems()
        self.progress_toolbar.hide_progressbar()

    def progress_bar_changed(self, percent):
        if self.progress_bar.isVisible():
            self.progress_bar.set_progress(percent)
            if percent == 100:
                self.progress_bar.hide()

    def on_change_project_name(self):
        dataset_dir = self.change_path_window.getEditText()

        self.change_path_window.hide()

        if not os.path.exists(dataset_dir):

            msgbox = QMessageBox()
            msgbox.setIcon(QMessageBox.Information)
            msgbox.setText(
                f"Ошибка загрузки проекта. " if self.lang == 'RU' else f"Error in loading project")
            if self.lang == 'RU':
                msgbox.setInformativeText(
                    f"Директория {dataset_dir} не существует"
                )
            else:
                msgbox.setInformativeText(
                    f"Directory {dataset_dir} doesn't exist")
            msgbox.setWindowTitle(
                f"Ошибка загрузки проекта {self.loaded_proj_name}" if self.lang == 'RU' else "Error")
            msgbox.exec()

            return

        self.project_data.set_path_to_images(dataset_dir)
        self.on_checking_project_success(dataset_dir)

    def on_checking_project_success(self, dataset_dir):

        self.dataset_dir = dataset_dir
        # ProgressBar
        self.progress_toolbar.set_signal(self.view.load_ids_conn.percent)
        self.progress_toolbar.show_progressbar()

        # Задать id меткам на изображениях
        self.view.set_ids_from_project(self.project_data.get_data(), on_set_callback=self.on_view_ids_set)

    def on_view_ids_set(self):
        # Заданы id меткам на изображениях

        self.progress_toolbar.hide_progressbar()
        self.dataset_images = self.filter_images_names(self.dataset_dir)

        self.fill_labels_combo_from_project()

        if self.dataset_images:
            self.fill_images_label(self.dataset_images)

            self.tek_image_name = self.images_list_widget.item(0).text()

            self.tek_image_path = os.path.join(self.dataset_dir, self.tek_image_name)

            # Открытие изображения
            self.open_image(self.tek_image_path)

            main_geom = self.geometry().getCoords()
            self.scaleFactor = (main_geom[2] - main_geom[0]) / self.cv2_image.shape[1]

            self.load_image_data(self.tek_image_name)

            self.save_view_to_project()

            self.im_panel_count_conn.on_image_count_change.emit(len(self.dataset_images))
            self.images_list_widget.setCurrentRow(0)

            self.selected_count_conn.on_labels_count_change.emit(0)
            self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

            self.reset_image_panel_progress_bar()

        self.info_message(
            f"Число загруженных в проект изображений: {len(self.dataset_images)}" if self.lang == 'RU' else f"Loaded images count: {len(self.dataset_images)}")

    def reload_project(self):
        self.load_project(self.loaded_proj_name)

    def load_project(self, project_name):

        if not project_name:
            return

        if self.loaded_proj_name:
            self.close_project()

        self.loaded_proj_name = project_name
        self.project_data.load(project_name)
        self.on_load_project()

    def on_load_project(self):

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
        last_opened_path = self.settings.read_last_opened_path()
        title = 'Загрузка проекта' if self.lang == 'RU' else "Loading project"
        loaded_proj_name, _ = QFileDialog.getOpenFileName(self,
                                                          title,
                                                          last_opened_path,
                                                          'JSON Proj File (*.json)')

        if loaded_proj_name:
            self.settings.write_last_opened_path(os.path.dirname(loaded_proj_name))
            self.load_project(loaded_proj_name)

    def on_polygon_end_draw(self, is_end_draw):
        if is_end_draw:
            self.save_view_to_project()

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
            self.view.clearScene()
            self.dataset_images.clear()
            self.im_panel_count_conn.on_image_count_change.emit(0)  # zero images

        self.toggle_act(False)

    def save_project(self):
        """
        Сохранение проекта
        """

        if self.loaded_proj_name:
            self.view.start_circle_progress()

            self.set_labels_color()  # сохранение информации о цветах масок
            self.save_view_to_project()

            self.project_data.save(self.loaded_proj_name)

            self.fill_project_labels()

            self.view.stop_circle_progress()

            self.info_message(
                f"Проект успешно сохранен" if self.lang == 'RU' else "Project is saved")

        else:
            self.save_project_as()

    def save_project_as(self, proj_name=None):
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
            self.save_view_to_project()
            self.fill_project_labels()
            self.loaded_proj_name = proj_name

            self.view.start_circle_progress()
            self.project_data.save(proj_name)
            self.view.stop_circle_progress()

    def on_project_saved(self):

        self.info_message(
            f"Проект успешно сохранен" if self.lang == 'RU' else "Project is saved")

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

        self.zoomInAct.setEnabled(self.scaleFactor < 30.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.0001)

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

        self.settings_window.show()

    def change_theme(self):
        """
        Изменение темы приложения
        """
        app_new = QApplication.instance()

        # primary_color = "#ffffff"

        theme = self.settings.read_theme()
        icon_folder = self.settings.get_icon_folder()

        # if 'light' in theme:
        #     primary_color = "#000000"

        density = hf.density_slider_to_value(self.settings.read_density())

        extra = {'density_scale': density,
                 # 'font_size': '14px',
                 # 'primaryTextColor': primary_color,
                 # 'secondaryTextColor': '#ffffff'
                 }

        invert_secondary = False if 'dark' in theme else True

        apply_stylesheet(app_new, theme=theme, extra=extra, invert_secondary=invert_secondary)

        self.cls_combo.change_theme(theme)

        self.on_theme_change_connection.on_theme_change.emit(icon_folder)

    def on_settings_closed(self):
        """
        При закрытии окна настроек приложения

        """
        lang = self.lang

        theme = self.settings.read_theme()
        density = self.settings.read_density()
        if theme != self.last_theme or density != self.last_density:
            self.last_theme = theme
            self.last_density = density
            self.change_theme()

        self.set_icons()

        fat_width = self.settings.read_fat_width()

        if fat_width != self.last_fat_width:
            self.last_fat_width = fat_width
            self.view.set_fat_width(self.settings.read_fat_width())

        self.save_view_to_project()
        self.reload_image(is_tek_image_changed=False)
        self.info_message(
            f"Настройки проекта изменены" if lang == 'RU' else "Settings is saved")

    def start_view_drawing(self, draw_state):

        self.open_polygons_and_off_hand()

        self.window_state = WindowState.drawing

        if self.window_state == WindowState.rubber_band:
            self.rubber_band_change_conn.on_rubber_mode_change.emit(False)
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings.read_alpha()
        alpha_edge = self.settings.read_edges_alpha()

        label_text_params = self.settings.read_label_text_params()
        if label_text_params['hide']:
            text = None
        else:
            text = cls_txt
        self.view.start_drawing(draw_state, cls_num, color=label_color, alpha=alpha_tek, text=text,
                                alpha_edge=alpha_edge)

    def polygon_pressed(self, pressed_id):
        self.window_state = WindowState.normal

        if pressed_id == -1:  # нажали не по полигону
            items_selected = self.labels_on_tek_image.selectedItems()
            for item in items_selected:
                item.setSelected(False)

            self.selected_count_conn.on_labels_count_change.emit(0)
            return

        for i in range(self.labels_on_tek_image.count()):
            item = self.labels_on_tek_image.item(i)
            item_id = item.text().split(" ")[-1]
            if item_id:
                if int(item_id) == pressed_id:
                    self.labels_on_tek_image.setCurrentItem(item)
                    self.selected_count_conn.on_labels_count_change.emit(1)

                    break

    def on_polygon_delete(self, delete_ids):
        self.window_state = WindowState.normal
        self.save_view_to_project()

    def polygon_tool_pressed(self):

        self.draw_state = DrawState.polygon
        self.start_view_drawing(self.draw_state)

    def square_pressed(self):
        self.draw_state = DrawState.box

        self.start_view_drawing(self.draw_state)

    def circle_pressed(self):

        self.draw_state = DrawState.ellipse
        self.start_view_drawing(self.draw_state)

    def ann_triggered(self, ann):
        self.annotatorToolButton.setDefaultAction(ann)

    def set_labels_color(self):

        labels = [self.cls_combo.itemText(i) for i in range(self.cls_combo.count())]
        self.project_data.set_labels_colors(labels)

    def add_image_shapes_to_view(self, im_project_data):
        if im_project_data:

            for shape in im_project_data["shapes"]:
                cls_num = shape["cls_num"]
                cls_name = self.cls_combo.itemText(cls_num)
                points = shape["points"]
                alpha_tek = self.settings.read_alpha()
                alpha_edge = self.settings.read_edges_alpha()

                color = self.project_data.get_label_color(cls_name)

                label_text_params = self.settings.read_label_text_params()
                if label_text_params['hide']:
                    text = None
                else:
                    text = cls_name
                self.view.add_polygon_to_scene(cls_num, points, color=color, alpha=alpha_tek, id=shape["id"], text=text,
                                               alpha_edge=alpha_edge)

    def load_image_data(self, image_name):
        # Проверка наличия записи о цветах полигонов

        im = self.project_data.get_image_data(image_name)
        self.add_image_shapes_to_view(im)

    def on_rubber_band_end(self, points):

        if not self.image_set:
            if self.lang == 'RU':
                message = "Подождите окончания загрузки изображения"
            else:
                message = "Wait for the image to finish loading"
            self.info_message(message)

            return

        # Получаем полигон селекта
        self.save_polygon_as_image(points)

        self.rubber_band_change_conn.on_rubber_mode_change.emit(False)

    def end_drawing(self):

        if self.draw_state in [DrawState.polygon, DrawState.ellipse, DrawState.box]:
            if self.window_state == WindowState.drawing:
                cls_txt = self.cls_combo.currentText()
                cls_num = self.cls_combo.currentIndex()
                label_color = self.project_data.get_label_color(cls_txt)
                alpha_tek = self.settings.read_alpha()

                self.view.end_drawing(cls_num=cls_num, text=cls_txt, color=label_color,
                                      alpha_percent=alpha_tek)  # save it to
                self.window_state = WindowState.normal
                self.save_view_to_project()

        if self.window_state != WindowState.normal:  # pressed Space

            self.window_state = WindowState.normal
            self.save_view_to_project()

    def start_drawing(self):
        """
        Старт рисования метки
        """
        if not self.image_set:
            if self.lang == 'RU':
                message = "Подождите окончания загрузки изображения"
            else:
                message = "Wait for the image to finish loading"
            self.info_message(message)
            return

        self.start_view_drawing(self.draw_state)

    def break_drawing(self):

        """Delete polygon pressed"""

        self.view.break_drawing()
        self.window_state = WindowState.normal

        self.save_view_to_project()

    def reload_image(self, is_tek_image_changed=True):
        """
        Заново загружает текущее изображение с разметкой
        """
        if not self.tek_image_path:
            return

        if is_tek_image_changed:
            self.open_image(self.tek_image_path)
        else:
            self.view.setPixmap(QtGui.QPixmap(self.tek_image_path))

        self.load_image_data(self.tek_image_name)
        self.save_view_to_project()
        self.selected_count_conn.on_labels_count_change.emit(0)
        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def show_shortcuts(self):
        self.shortcuts_window = ShortCutsEditor(on_ok_act=self.reset_shortcuts)
        self.shortcuts_window.show()

    def undo(self):
        self.view.remove_last_changes()
        self.save_view_to_project()
        self.window_state = WindowState.normal

    def keyPressEvent(self, e):
        modifierPressed = QApplication.keyboardModifiers()
        modifierName = ''

        if (modifierPressed & QtCore.Qt.ControlModifier) == QtCore.Qt.ControlModifier:
            modifierName += 'Ctrl'
        if (modifierPressed & QtCore.Qt.AltModifier) == QtCore.Qt.AltModifier:
            modifierName += 'Alt'
        if (modifierPressed & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier:
            modifierName += 'Shift'

        shortcuts = self.settings.read_shortcuts()
        # print(shortcuts)

        for sc_key, act in zip(
                ['copy', 'del', 'end_drawing', 'image_after', 'image_before', 'paste', 'start_drawing', 'undo'],
                [self.copy_label, self.break_drawing, self.end_drawing, self.go_next, self.go_before, self.paste_label,
                 self.start_drawing, self.undo]):
            shortcut = shortcuts[sc_key]
            shortcut_modifiers = shortcut['modifier']
            if e.key() == shortcut['shortcut_key_eng'] or e.key() == shortcut['shortcut_key_ru'] and hf.match_modifiers(
                    shortcut_modifiers, modifierName):
                act()

    def go_next(self):

        if self.window_state == WindowState.drawing:
            self.window_state = WindowState.normal
            self.view.break_drawing()

        self.save_view_to_project()

        next_im_name = self.images_list_widget.get_next_name()
        if next_im_name:
            self.tek_image_name = next_im_name
            self.tek_image_path = os.path.join(self.dataset_dir, next_im_name)
            self.reload_image(is_tek_image_changed=True)
            self.images_list_widget.move_next()

    def go_before(self):

        if self.window_state == WindowState.drawing:
            self.window_state = WindowState.normal
            self.view.break_drawing()

        self.save_view_to_project()

        before_im_name = self.images_list_widget.get_before_name()
        if before_im_name:
            self.tek_image_name = before_im_name
            self.tek_image_path = os.path.join(self.dataset_dir, before_im_name)
            self.reload_image(is_tek_image_changed=True)
            self.images_list_widget.move_before()

    def copy_label(self):
        self.view.copy_active_item_to_buffer()

    def paste_label(self):

        self.view.paste_buffer()
        self.save_view_to_project()
        self.reload_image(is_tek_image_changed=False)

    def on_cancel_quit(self):
        self.exit_box.hide()
        self.hide()  # Скрываем окно

        self.write_size_pos()
        self.close_project()
        self.is_asked_before_close = True

        self.close()

    def on_ok_quit(self):
        self.save_project()
        self.on_cancel_quit()

    def closeEvent(self, event):
        if self.is_asked_before_close:
            hf.clear_temp_folder(os.getcwd())
            event.accept()
        else:
            if self.loaded_proj_name:
                event.ignore()

                title = 'Выйти' if self.lang == 'RU' else 'Quit'

                text = 'Сохранить проект перед выходом?' if self.lang == 'RU' else 'Save project before exit?'

                cancel_text = 'Нет' if self.lang == 'RU' else 'No'
                ok_text = 'Да' if self.lang == 'RU' else 'Yes'

                self.exit_box = OkCancelDialog(self, title=title, text=text, on_cancel=self.on_cancel_quit,
                                               on_ok=self.on_ok_quit, ok_text=ok_text, cancel_text=cancel_text)
                self.exit_box.setMinimumWidth(300)
            else:
                self.hide()  # Скрываем окно

                self.write_size_pos()

                self.close()

    def load_lrm_data_pressed(self):
        theme = self.settings.read_theme()
        self.import_lrms_dialog = ImportLRMSDialog(self, theme=theme, on_ok_clicked=self.on_load_lrm_ok)
        self.import_lrms_dialog.show()

    def on_load_lrm_ok(self):
        self.import_lrms_dialog.hide()
        lrms_data = self.import_lrms_dialog.lrms_data

        set_images_names, unset_images_names = self.project_data.set_lrm_for_all_images(lrms_data)

        set_num = len(set_images_names)
        img_num = self.project_data.get_images_num()

        # set lrm from tek_image

        if self.tek_image_name in lrms_data:
            self.lrm = lrms_data[self.tek_image_name]
            self.ruler_act.setEnabled(True)
        else:
            self.ruler_act.setEnabled(False)

        if self.lang == 'RU':
            message = f"Задан ЛРМ для {set_num} изображений из {img_num}"
        else:
            message = f"Linear ground resolution is set for {set_num} images from {img_num}"

        self.info_message(message)

    def open_polygons_and_off_hand(self):
        self.toggle_act(True)

        if self.view.view_state == ViewState.hide_polygons:
            # Открыть полигоны
            self.view.set_view_state(ViewState.normal)
            self.labels_on_tek_image.setEnabled(True)
            self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

        if self.view.view_state == ViewState.hand_move:
            self.view.set_view_state(ViewState.normal)
            self.labels_on_tek_image.setEnabled(True)
            self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def ruler_pressed(self):

        self.open_polygons_and_off_hand()

        if self.ruler_act.isChecked():
            self.view.on_ruler_mode_on(self.lrm)
        else:
            self.view.on_ruler_mode_off()

    def add_user_clicked(self):
        user_name = self.user_names_combo.currentText()
        title = f"Добавление нового пользователя {user_name}" if self.lang == 'RU' else f"Add new user {user_name}"
        question = "Введите имя нового пользователя:" if self.lang == 'RU' else "Enter new user name:"
        self.user_dialog = CustomInputDialog(self,
                                             title_name=title,
                                             question_name=question)
        self.user_dialog.setMinimumWidth(500)

        self.user_dialog.okBtn.clicked.connect(self.add_user_ok_clicked)
        self.user_dialog.show()

    def add_user_ok_clicked(self):
        new_name = self.user_dialog.getText()
        self.user_dialog.close()

        if new_name:
            if new_name not in self.user_names_combo.getAll():
                self.user_names_combo.addItem(new_name)
                idx = self.user_names_combo.getPos(new_name)
                self.user_names_combo.setCurrentIndex(idx)
            else:
                if self.lang == 'RU':
                    message = f"Не могу добавить пользователя {new_name}. Такой пользователь уже существует"
                else:
                    message = f"Can't add {new_name}. User already exists"
                self.info_message(message)

    def rename_user(self):
        user_name = self.user_names_combo.currentText()

        title = f"Редактирование имени пользователя {user_name}" if self.lang == 'RU' else f"Rename user {user_name}"
        question = "Введите новое имя пользователя:" if self.lang == 'RU' else "Enter new user name:"
        self.user_dialog = CustomInputDialog(self,
                                             title_name=title,
                                             question_name=question)
        self.user_dialog.setMinimumWidth(500)

        self.user_dialog.okBtn.clicked.connect(self.rename_user_ok_clicked)
        self.user_dialog.show()

    def rename_user_ok_clicked(self):
        new_name = self.user_dialog.getText()
        self.user_dialog.close()

        if new_name:
            items = [it for it in self.user_names_combo.getAll() if it != self.user_names_combo.currentText()]
            items.append(new_name)
            self.user_names_combo.clear()
            self.user_names_combo.addItems(items)
            idx = self.user_names_combo.getPos(new_name)
            self.user_names_combo.setCurrentIndex(idx)
            self.change_user_name(new_name)

    def delete_user(self):
        self.del_user_name = self.user_names_combo.currentText()

        title = f'Удаление пользователя' if self.lang == 'RU' else 'User delete'
        if self.lang == 'RU':
            text = f'Вы точно хотите удалить пользователя {self.del_user_name}?'
        else:
            text = f'Are you really want to delete {self.del_user_name}?'

        self.delete_box = OkCancelDialog(self, title=title, text=text, on_ok=self.on_delete_user_ok)
        self.delete_box.setMinimumWidth(300)

    def on_delete_user_ok(self):
        self.delete_box.hide()
        idx = self.user_names_combo.getPos(self.del_user_name)
        self.user_names_combo.removeItem(idx)

    def change_user_name(self, new_name):
        if new_name:
            self.settings.write_username(new_name)
            self.settings.write_username_variants(self.user_names_combo.getAll())

    def reset_image_panel_progress_bar(self):
        if len(self.dataset_images) == 0:
            return
        images_info = self.project_data.get_all_images_info()
        approved_count = 0
        for name in self.dataset_images:
            if name in images_info:
                status = images_info[name].get('status', None)
                if status == 'approve':
                    approved_count += 1
        percent = int(approved_count * 100.0 / len(self.dataset_images))
        self.image_panel_progress_bar.setValue(percent)

        message = f"Число изображений со статусом approved: " if self.lang == 'RU' else f"Approved images count: "
        message += f"{approved_count} из {len(self.dataset_images)}"
        self.info_message(message)

    def on_hand_move_clicked(self):
        if self.view.view_state == ViewState.hand_move:
            self.view.set_view_state(ViewState.normal)
            self.labels_on_tek_image.setEnabled(True)
            self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())
        else:
            self.view.set_view_state(ViewState.hand_move)
            self.labels_on_tek_image.setEnabled(False)
            self.labels_count_conn.on_labels_count_change.emit(0)

    def toggle_hide_labels(self):

        if self.view.view_state == ViewState.hide_polygons:
            # Открыть полигоны
            self.view.set_view_state(ViewState.normal)
            self.labels_on_tek_image.setEnabled(True)
            self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())
        else:
            self.view.set_view_state(ViewState.hide_polygons)
            self.labels_on_tek_image.setEnabled(False)
            self.labels_count_conn.on_labels_count_change.emit(0)

    def on_toggle_image_status(self):
        variants = ['empty', 'in_work', 'approve']
        status = self.project_data.get_image_status(self.tek_image_name)
        index = variants.index(status)
        next_index = 0 if index == len(variants) - 1 else index + 1
        new_status = variants[next_index]
        self.project_data.set_image_status(self.tek_image_name, new_status)
        self.project_data.set_image_last_user(self.tek_image_name, self.settings.read_username())
        self.images_list_widget.set_status(status=new_status)
        self.reset_image_panel_progress_bar()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             # 'primaryTextColor': '#ffffff'
             }

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra, invert_secondary=False)

    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
