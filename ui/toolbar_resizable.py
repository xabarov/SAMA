from PyQt5.QtWidgets import QAction, QFileDialog, QMessageBox, QMenu, QToolBar, QToolButton, QLabel, \
    QColorDialog, QListWidget, QSizePolicy
import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QIcon, QCursor
from utils.help_functions import distance


class ToolBarResizable(QToolBar):

    def __init__(self, title, parent, type='top', tol=25):
        """
        Toolbar с изменяемыми размерами
        type = {top, left, right, bottom} - тип тулбара
        tol - зазор в пикселях для появления значка resize
        """
        super().__init__(title, parent)
        self.type = type
        self.tol = tol  # px
        self.state = 'static'  # 'movable'
        self.min_w_h = self.width(), self.height()
        self.pressed_x_y = None
        self.setMouseTracking(True)

    def mouseMoveEvent(self, e: QtGui.QMouseEvent) -> None:

        if self.type == "left":
            if self.state == 'static':
                if np.abs(e.x() - self.width()) < self.tol:
                    self.setCursor(QCursor(QtCore.Qt.SizeHorCursor))
                    print("Change state to movable")
                    self.state = 'movable'
            elif self.state == 'movable':
                if np.abs(e.x() - self.width()) > self.tol:
                    self.setCursor(QCursor(QtCore.Qt.ArrowCursor))
                    print("Change state to static")
                    self.state = 'static'
            elif self.state == 'start_move':
                print("In start move state")
                self.resize(e.x(), self.height())
        elif self.type == "right":
            if self.state == 'static':
                if e.x() < self.tol:
                    self.setCursor(QCursor(QtCore.Qt.SizeHorCursor))
                    print("Change state to movable")
                    self.state = 'movable'
            elif self.state == 'movable':
                if e.x() > self.tol:
                    self.setCursor(QCursor(QtCore.Qt.ArrowCursor))
                    print("Change state to static")
                    self.state = 'static'
            elif self.state == 'start_move':
                print("In start move state")
                self.resize(self.width() - e.x(), self.height())
                print(f"x {e.x()} width {self.width()}")

    def mousePressEvent(self, e: QtGui.QMouseEvent) -> None:
        if self.state == 'movable':
            if self.type == "left" or self.type == 'right':
                self.state = 'start_move'
                self.setMouseTracking(False)
                self.setCursor(QCursor(QtCore.Qt.SizeHorCursor))
                self.pressed_x_y = e.pos()
                print(f"x {e.x()} width {self.width()}")

    def mouseReleaseEvent(self, e: QtGui.QMouseEvent) -> None:
        if self.state == 'start_move':

            if self.pressed_x_y:
                if self.type == "left":
                    if e.x() > self.pressed_x_y.x() or e.x() >= self.min_w_h[0] - self.tol:
                        self.resize(e.x(), self.height())
                        print(f"resize to x {e.x()}")
                    else:
                        self.resize(self.min_w_h[0], self.height())
                        print(f"resize to min {self.min_w_h[0]}")
                elif self.type == "right":
                    print(f"x {e.x()} width {self.width()}")
                    self.resize(self.width() - e.x(), self.height())


            self.state = 'static'
            self.setMouseTracking(True)
