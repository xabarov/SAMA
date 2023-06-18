from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPolygonF, QColor, QPen

from utils import config
from utils import help_functions as hf

import math


class GrGroup(QtWidgets.QGraphicsItemGroup):

    def __init__(self, cls_name=None, color=None, alpha_percent=50, group_id=0):
        super(GrGroup, self).__init__()
        self.points = []
        self.cls_name = cls_name
        self.color = color
        self.alpha_percent = alpha_percent
        self.group_id = group_id
        self.brush = QtGui.QBrush(QColor(*self.color), QtCore.Qt.SolidPattern)
        self.pen = QPen(QColor(*hf.set_alpha_to_max(self.color)), 1, QtCore.Qt.SolidLine)

    def add_points(self, points):
        for point in points:
            ellipse = QtWidgets.QGraphicsEllipseItem(point[0], point[1], 1, 1)
            ellipse.setBrush(self.brush)
            ellipse.setPen(self.pen)
            self.points.append(point)
            self.addToGroup(ellipse)

    def contains(self, point):
        for p in self.points:
            if point.x() == p[0] and point.y() == p[1]:
                return True
        return False
