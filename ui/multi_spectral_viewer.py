import os
import shutil
import sys

import cv2
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QAction, QFileDialog, QMessageBox, QMenu, QToolBar, QToolButton, QLabel, \
    QColorDialog, QListWidget
from PyQt5.QtGui import QMovie, QIcon
from PyQt5.QtPrintSupport import QPrinter
from PyQt5.QtWidgets import QAction, QMenu, QToolBar
from PyQt5.QtWidgets import QApplication
from qt_material import apply_stylesheet

from ui.ok_cancel_dialog import OkCancelDialog
from ui.signals_and_slots import ThemeChangeConnection
from ui.simple_view import SimpleView
from ui.splash_screen import MovieSplashScreen
from ui.toolbars import ProgressBarToolbar
from utils import config
from utils import help_functions as hf
from utils.gdal_translate_worker import GdalWorker
from utils.settings_handler import AppSettings

basedir = os.path.dirname(__file__)


class MultiSpectralViewer(QtWidgets.QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("Multi-spectral viewer")

        # GraphicsView

        self.view = SimpleView()
        self.setCentralWidget(self.view)

        self.on_theme_change_connection = ThemeChangeConnection()

        # Settings
        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        self.icon_folder = os.path.join(basedir, self.settings.get_icon_folder())
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

        self.init_global_values()

        # self.show()
        self.lrm = None
        self.block_geo_coords_message = False
        self.image_set = False
        self.view.mouse_move_conn.on_mouse_move.connect(self.on_view_mouse_move)
        self.tek_band_full_name = None
        self.geotiff_image_fullname = None

        # self.splash.finish(self)
        self.statusBar().showMessage(
            "Загрузите проект или набор изображений" if self.lang == 'RU' else "Load dataset or project")

    def init_global_values(self):
        """
        Set some app global values
        """

        self.scaleFactor = 1.0

        self.image_types = ['tif', 'tiff']

        self.bands_full_names = []

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

        self.openImageAct = QAction("Загрузить изображение" if self.lang == 'RU' else "Load Image",
                                    self,
                                    shortcut='Ctrl+O',
                                    triggered=self.open_image)

        self.exitAct = QAction("Выход" if self.lang == 'RU' else "Exit", self, shortcut="Ctrl+Q",
                               triggered=self.close)
        self.zoomInAct = QAction("Увеличить" if self.lang == 'RU' else "Zoom In", self,
                                 shortcut="Ctrl++",
                                 enabled=False,
                                 triggered=self.zoomIn)
        self.zoomOutAct = QAction("Уменьшить" if self.lang == 'RU' else "Zoom Out", self,
                                  shortcut="Ctrl+-",
                                  enabled=False,
                                  triggered=self.zoomOut)

        self.fitToWindowAct = QAction(
            "Подогнать под размер окна" if self.lang == 'RU' else "Fit to window size",
            self, enabled=False,
            shortcut="Ctrl+F",
            triggered=self.fitToWindow)

    def createMenus(self):

        self.fileMenu = QMenu("&Файл" if self.lang == 'RU' else "&File", self)
        self.fileMenu.addAction(self.openImageAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)
        #
        self.viewMenu = QMenu("&Изображение" if self.lang == 'RU' else "&View", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addSeparator()
        self.viewMenu.addAction(self.fitToWindowAct)

        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)

    def create_left_toolbar(self):
        # Left

        toolBar = QToolBar("Панель инструментов" if self.lang == 'RU' else "ToolBar", self)
        toolBar.addAction(self.openImageAct)
        toolBar.addSeparator()
        toolBar.addAction(self.zoomInAct)
        toolBar.addAction(self.zoomOutAct)
        toolBar.addAction(self.fitToWindowAct)
        toolBar.addSeparator()

        self.toolBarLeft = toolBar
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBarLeft)

    def create_right_toolbar(self):
        # Right toolbar
        self.toolBarRight = QToolBar("Менеджер разметок" if self.lang == 'RU' else "Labeling Bar", self)

        # Labels
        self.bands_list_widget = QListWidget()
        self.bands_list_widget.itemClicked.connect(self.bands_list_widget_clicked)
        self.toolBarRight.addWidget(self.bands_list_widget)

        # Add panels to toolbars

        self.addToolBar(QtCore.Qt.RightToolBarArea, self.toolBarRight)

    def create_top_toolbar(self):

        self.labelSettingsToolBar = QToolBar(
            "Настройки разметки" if self.lang == 'RU' else "Current Label Bar",
            self)

        self.progress_toolbar = ProgressBarToolbar(self,
                                                   right_padding=self.toolBarRight.width())
        self.labelSettingsToolBar.addWidget(self.progress_toolbar)

        self.addToolBar(QtCore.Qt.TopToolBarArea, self.labelSettingsToolBar)

    def createToolbar(self):

        self.create_left_toolbar()
        self.create_right_toolbar()
        self.create_top_toolbar()

    def set_icons(self):
        """
        Задать иконки
        """

        self.icon_folder = self.settings.get_icon_folder()

        self.setWindowIcon(QIcon(self.icon_folder + "/neural.png"))
        self.exitAct.setIcon(QIcon(self.icon_folder + "/logout.png"))
        self.zoomInAct.setIcon(QIcon(self.icon_folder + "/zoom-in.png"))
        self.zoomOutAct.setIcon(QIcon(self.icon_folder + "/zoom-out.png"))
        self.fitToWindowAct.setIcon(QIcon(self.icon_folder + "/fit.png"))

        # save load
        self.openImageAct.setIcon(QIcon(self.icon_folder + "/load.png"))

    def toggle_act(self, is_active):

        self.fitToWindowAct.setEnabled(is_active)
        self.zoomInAct.setEnabled(is_active)
        self.zoomOutAct.setEnabled(is_active)

    def open_image(self):

        image_name, _ = QFileDialog.getOpenFileName(self,
                                                    'Загрузка изображения' if self.lang == 'RU' else "Load image",
                                                    'projects',
                                                    'Tiff File (*.tiff, *.tif)')

        if not image_name:
            return

        self.geotiff_image_fullname = image_name

        self.start_gif(is_prog_load=True)

        self.gdal_worker = GdalWorker(image_name, save_folder=self.handle_temp_folder())

        self.gdal_worker.started.connect(self.on_gdal_worker_started)

        self.progress_toolbar.set_signal(self.gdal_worker.translate_conn.percent)

        self.gdal_worker.finished.connect(self.on_gdal_finished)

        if not self.gdal_worker.isRunning():
            self.gdal_worker.start()

    def on_gdal_worker_started(self):
        """
        При начале открытия изображения
        """
        self.progress_toolbar.show_progressbar()
        self.statusBar().showMessage(
            f"Начинаю преобрзование каналов изображения..." if self.lang == 'RU' else f"Start opening bands...",
            3000)

    def on_gdal_finished(self):
        self.bands_full_names = self.gdal_worker.get_bands_full_names()
        bands_names = self.gdal_worker.get_bands_names()
        self.fill_bands_list_widget(bands_names)

        self.open_band(self.bands_full_names[0])
        self.lrm = self.gdal_worker.get_lrm(from_crs='epsg:32636', to_crs='epsg:4326')
        print(f"LRM: {self.lrm}")

        self.splash.finish(self)

        self.progress_toolbar.hide_progressbar()

        self.statusBar().showMessage(
            f"Найдено {len(self.gdal_worker.bands_names)} каналов" if self.lang == 'RU' else f"{len(self.gdal_worker.bands_names)} bands has been found",
            3000)

    def open_band(self, band_name):

        self.view.setPixmap(QtGui.QPixmap(band_name))
        self.view.fitInView(self.view.pixmap_item, QtCore.Qt.KeepAspectRatio)

        image = cv2.imread(band_name)
        self.cv2_image = image

        self.image_set = True
        self.tek_band_full_name = band_name
        self.toggle_act(self.image_set)

    def bands_list_widget_clicked(self, item):
        band_name = os.path.join(self.handle_temp_folder(), item.text())
        self.open_band(band_name)

    def fill_bands_list_widget(self, bands_names):

        self.bands_list_widget.clear()
        for name in bands_names:
            self.bands_list_widget.addItem(name)

    def handle_temp_folder(self):
        temp_folder = os.path.join(os.getcwd(), 'temp')
        if not os.path.exists(temp_folder):
            os.makedirs(temp_folder)

        return temp_folder

    def clear_temp_folder(self):
        temp_folder = os.path.join(os.getcwd(), 'temp')
        # print(temp_folder)
        if os.path.exists(temp_folder):
            shutil.rmtree(temp_folder)

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

    def on_view_mouse_move(self, x, y):
        if self.image_set and not self.block_geo_coords_message:
            if self.lrm:
                geo_x, geo_y = hf.convert_point_coords_to_geo(x, y, self.geotiff_image_fullname, from_crs='epsg:32636',
                                                              to_crs='epsg:4326')
                self.statusBar().showMessage(
                    f"{geo_x:.6f}, {geo_y:.6f}")

    def fitToWindow(self):
        """
        Подогнать под экран
        """
        self.view.fitInView(self.view.pixmap_item, QtCore.Qt.KeepAspectRatio)

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
                 # 'primaryTextColor': primary_color,
                 # 'secondaryTextColor': '#ffffff'
                 }

        invert_secondary = False if 'dark' in theme else True

        apply_stylesheet(app, theme=theme, extra=extra, invert_secondary=invert_secondary)

        self.on_theme_change_connection.on_theme_change.emit(icon_folder)

    def keyPressEvent(self, e):

        pass

    def on_quit(self):
        self.exit_box.hide()
        self.hide()  # Скрываем окно

        self.write_size_pos()
        self.is_asked_before_close = True

        self.close()

    def closeEvent(self, event):

        if self.is_asked_before_close:
            self.clear_temp_folder()
            event.accept()
        else:
            event.ignore()
            title = 'Выйти' if self.lang == 'RU' else 'Quit'
            text = 'Вы точно хотите выйти?' if self.lang == 'RU' else 'Are you really want to quit?'
            self.exit_box = OkCancelDialog(self, title=title, text=text, on_ok=self.on_quit)
            self.exit_box.setMinimumWidth(300)


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             # 'primaryTextColor': '#ffffff'
             }

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra, invert_secondary=False)

    w = MultiSpectralViewer()
    w.show()
    sys.exit(app.exec_())
