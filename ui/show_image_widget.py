from PyQt5.QtCore import Qt
from PyQt5.QtGui import QIcon, QPalette
from PyQt5.QtGui import QImage
from PyQt5.QtGui import QPixmap, QPainter
from PyQt5.QtPrintSupport import QPrintDialog, QPrinter
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel, QSizePolicy, QScrollArea, QMessageBox, QMenu, QAction, QMenuBar, QToolBar, QWidget, \
    QVBoxLayout


class ShowImgWindow(QWidget):
    def __init__(self, parent, title="Изображение", img_file="", icon_folder="", is_fit_button=True):
        super().__init__(parent)
        self.setWindowTitle(title)

        self.setWindowFlag(Qt.Window)
        self.filename = img_file
        self.is_fit_button = is_fit_button

        self.printer = QPrinter()
        self.scaleFactor = 0.0

        if icon_folder != "":
            self.icon_folder = icon_folder
        else:
            self.icon_folder = "./icons"

        self.imageLabel = QLabel()
        self.imageLabel.setBackgroundRole(QPalette.Base)
        self.imageLabel.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Ignored)
        self.imageLabel.setScaledContents(True)

        self.scrollArea = QScrollArea()
        self.scrollArea.setBackgroundRole(QPalette.Dark)
        self.scrollArea.setWidget(self.imageLabel)
        self.scrollArea.setVisible(False)

        self.vert_layout = QVBoxLayout(self)

        self.createActions()
        self.createToolbar()
        self.vert_layout.addWidget(self.scrollArea)
        self.open()

        scale = self.get_scale_factor()
        if scale != 0:
            self.resize(int(self.image.width() * scale), int(self.image.height() * scale))
        else:
            self.resize(self.image.width(), self.image.height())

        self.show()

    def get_scale_factor(self, scale=0.6):

        app = QApplication.instance()
        screen = app.primaryScreen().size()

        if self.image.width() > screen.width() and self.image.height() > screen.height():
            factor = screen.width() / self.image.width()
            return factor * scale
        elif self.image.width() > screen.width():
            factor = screen.width() / self.image.width()
            return factor * scale
        elif self.image.height() > screen.height():
            factor = screen.height() / self.image.height()
            return factor * scale
        return 0

    def open(self):
        # fileName = QFileDialog.getOpenFileName(self, "Open File", QDir.currentPath())
        fileName = self.filename

        if fileName:
            image = QImage(fileName)
            self.image = image
            if image.isNull():
                QMessageBox.information(self, "Не могу загрузить изображение",
                                        "Не могу загрузить изображение %s." % fileName)
                return

            self.imageLabel.setPixmap(QPixmap.fromImage(image))
            self.scaleFactor = 1.0

            self.scrollArea.setVisible(True)
            self.printAct.setEnabled(True)

            self.updateActions()

            if self.is_fit_button:
                self.fitToWindowAct.setEnabled(True)
                if not self.fitToWindowAct.isChecked():
                    self.imageLabel.adjustSize()

                self.fitToWindowAct.setChecked(False)
                self.fitToWindow()


    def print_(self):
        dialog = QPrintDialog(self.printer, self)
        if dialog.exec_():
            painter = QPainter(self.printer)
            rect = painter.viewport()
            size = self.imageLabel.pixmap().size()
            size.scale(rect.size(), Qt.KeepAspectRatio)
            painter.setViewport(rect.x(), rect.y(), size.width(), size.height())
            painter.setWindow(self.imageLabel.pixmap().rect())
            painter.drawPixmap(0, 0, self.imageLabel.pixmap())

    def zoomIn(self):
        self.scaleImage(1.25)

    def zoomOut(self):
        self.scaleImage(0.8)

    def normalSize(self):
        self.imageLabel.adjustSize()
        self.scaleFactor = 1.0

    def fitToWindow(self):
        fitToWindow = self.fitToWindowAct.isChecked()
        self.scrollArea.setWidgetResizable(fitToWindow)
        if not fitToWindow:
            self.normalSize()

        self.updateActions()

    def createActions(self):

        self.printAct = QAction("Печать...", self, shortcut="Ctrl+P", enabled=False, triggered=self.print_)
        self.printAct.setIcon(QIcon(self.icon_folder + "/printer.png"))

        self.exitAct = QAction("Выход", self, shortcut="Ctrl+Q", triggered=self.close)
        self.exitAct.setIcon(QIcon(self.icon_folder + "/logout.png"))

        self.zoomInAct = QAction("Увеличить на (25%)", self, shortcut="Ctrl++", enabled=False, triggered=self.zoomIn)
        self.zoomInAct.setIcon(QIcon(self.icon_folder + "/zoom-in.png"))

        self.zoomOutAct = QAction("Уменьшить на (25%)", self, shortcut="Ctrl+-", enabled=False, triggered=self.zoomOut)
        self.zoomOutAct.setIcon(QIcon(self.icon_folder + "/zoom-out.png"))

        self.normalSizeAct = QAction("Исходный размер", self, shortcut="Ctrl+S", enabled=False,
                                     triggered=self.normalSize)
        self.normalSizeAct.setIcon(QIcon(self.icon_folder + "/reset.png"))

        if self.is_fit_button:
            self.fitToWindowAct = QAction("Подогнать под размер окна", self, enabled=False, checkable=True,
                                      shortcut="Ctrl+F",
                                      triggered=self.fitToWindow)
            self.fitToWindowAct.setIcon(QIcon(self.icon_folder + "/fit.png"))

    def createMenus(self):

        self.fileMenu = QMenu("&Файл", self)
        self.fileMenu.addAction(self.printAct)
        self.fileMenu.addSeparator()
        self.fileMenu.addAction(self.exitAct)

        self.viewMenu = QMenu("&Изображение", self)
        self.viewMenu.addAction(self.zoomInAct)
        self.viewMenu.addAction(self.zoomOutAct)
        self.viewMenu.addAction(self.normalSizeAct)

        if self.is_fit_button:
            self.viewMenu.addSeparator()
            self.viewMenu.addAction(self.fitToWindowAct)

        self.menubar = QMenuBar()
        self.menubar.addMenu(self.fileMenu)
        self.menubar.addMenu(self.viewMenu)
        self.vert_layout.addWidget(self.menubar)

    def createToolbar(self):

        toolBar = QToolBar("", self)

        toolBar.addAction(self.zoomInAct)
        toolBar.addAction(self.zoomOutAct)

        if self.is_fit_button:
            toolBar.addAction(self.fitToWindowAct)

        toolBar.addAction(self.normalSizeAct)

        toolBar.addSeparator()
        toolBar.addAction(self.printAct)
        toolBar.addAction(self.exitAct)

        self.vert_layout.addWidget(toolBar)

    def updateActions(self):
        if self.is_fit_button:
            self.zoomInAct.setEnabled(not self.fitToWindowAct.isChecked())
            self.zoomOutAct.setEnabled(not self.fitToWindowAct.isChecked())
            self.normalSizeAct.setEnabled(not self.fitToWindowAct.isChecked())
        else:
            self.zoomInAct.setEnabled(True)
            self.zoomOutAct.setEnabled(True)
            self.normalSizeAct.setEnabled(True)

    def scaleImage(self, factor):
        self.scaleFactor *= factor
        self.imageLabel.resize(self.scaleFactor * self.imageLabel.pixmap().size())

        self.adjustScrollBar(self.scrollArea.horizontalScrollBar(), factor)
        self.adjustScrollBar(self.scrollArea.verticalScrollBar(), factor)

        self.zoomInAct.setEnabled(self.scaleFactor < 10.0)
        self.zoomOutAct.setEnabled(self.scaleFactor > 0.001)

    def adjustScrollBar(self, scrollBar, factor):
        scrollBar.setValue(int(factor * scrollBar.value()
                               + ((factor - 1) * scrollBar.pageStep() / 2)))
