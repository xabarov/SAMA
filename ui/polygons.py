import math

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QFont, QColor
from PyQt5.QtGui import QPolygonF, QPen

from utils import config
from utils import help_functions as hf
from utils.settings_handler import AppSettings

settings = AppSettings()


def make_label(text, text_pos, color):
    label_text_params = settings.read_label_text_params()
    label = QtWidgets.QGraphicsTextItem()
    label.setHtml(
        f"<div style='background-color: rgba(0,0,0, 0.7)'>"
        f"<div style='margin: 10px'>{text}</div></div>")
    font = label_text_params['font']
    label.setFont(font)

    # set the color
    if not label_text_params['auto_color']:
        color = label_text_params['default_color']

    # Reset alpha
    color = [color[0], color[1], color[2], 255]
    label.setDefaultTextColor(QColor(*color))

    # set the alignment
    x = text_pos[0]
    y = text_pos[1]
    label.setPos(x, y)

    return label


def get_color(color, cls_num, alpha, alpha_min=15, alpha_max=200):
    """
    Определение и преобразование цвета
    Нормализация и ограницение прозрачности
    """
    if not color:
        if cls_num > len(config.COLORS) - 1:
            color_num = len(config.COLORS) % (cls_num + 1) - 1
            color = config.COLORS[color_num]
        else:
            color = config.COLORS[cls_num]

    alpha = hf.convert_percent_to_alpha(alpha, alpha_min=alpha_min, alpha_max=alpha_max)

    return (color[0], color[1], color[2], alpha)


class GrEllipsLabel(QtWidgets.QGraphicsEllipseItem):
    """
    Эллипс с поддержкой номера класса, цветов и т.д.
    """

    def __init__(self, parent=None, cls_num=0, color=None, alpha_percent=50, id=0, text=None, text_pos=None):

        super().__init__(parent)
        self.cls_num = cls_num
        self.id = id
        self.alpha_percent = alpha_percent
        self.color = get_color(color, cls_num, alpha_percent)
        self.label = None
        self.text = text
        self.text_pos = text_pos

        if text and text_pos:
            self.label = make_label(text, text_pos, self.color)

    def set_cls_num(self, cls_num):
        self.cls_num = cls_num

    def get_label(self):
        return self.label

    def set_label(self, text, text_pos, color=None):
        if not color:
            color = self.color
        self.label = make_label(text, text_pos, color)

    def reset_label_pos(self, x, y):
        if self.label:
            self.label.setPos(x, y)

    def check_ellips(self, min_height=10, min_width=10):
        # Проверка эллипса на мин кол-во пикселей. Можно посмотреть width, height. Если меньше порога - удалить
        rect = self.boundingRect()
        tl = rect.topLeft()
        br = rect.bottomRight()
        width = abs(br.x() - tl.x())
        height = abs(br.y() - tl.y())
        if width > min_width and height > min_height:
            return True

        return False

    def convert_to_polygon(self, points_count=30):
        rect = self.boundingRect()

        # взять кол-во точек. Определить делта-угол. Определить центр.
        # Посчитать точки и накопить полигон.
        polygon = QPolygonF()
        parent = self.parentItem()
        delta_angle_rad = 2 * math.pi / points_count
        tek_angle = 0
        for i in range(points_count):
            polygon.append(hf.calc_ellips_point_coords(rect, tek_angle))
            tek_angle += delta_angle_rad

        grpol = GrPolygonLabel(parent, color=self.color, cls_num=self.cls_num, alpha_percent=self.alpha_percent,
                               id=self.id)
        grpol.setBrush(QtGui.QBrush(QColor(*self.color), QtCore.Qt.SolidPattern))
        grpol.setPen(QPen(QColor(*hf.set_alpha_to_max(self.color)), 5, QtCore.Qt.SolidLine))
        grpol.setPolygon(polygon)

        return grpol

    def __str__(self):
        rect = self.rect()
        txt = f"Ellipse class = {self.cls_num} color = {self.color}points: "
        txt += f"\n\ttop left({rect.topLeft().x()}, {rect.topLeft().y()}) "
        txt += f"\n\tbottom right({rect.bottomRight().x()}, {rect.bottomRight().y()}) "
        return txt


class GrPolygonLabel(QtWidgets.QGraphicsPolygonItem):
    """
    Полигон с поддержкой номера класса, цветов и т.д.
    """

    def __init__(self, parent=None, cls_num=0, color=None, alpha_percent=50, id=0, text=None, text_pos=None):

        super().__init__(parent)
        self.cls_num = cls_num
        self.id = id
        self.alpha_percent = alpha_percent
        self.color = get_color(color, cls_num, alpha_percent)
        self.label = None
        self.text = text
        self.text_pos = text_pos

        if text and text_pos:
            self.label = make_label(text, text_pos, self.color)

    def check_polygon(self, min_height=10, min_width=10):
        # Проверка на мин кол-во пикселей. Можно посмотреть width, height. Если меньше порога - удалить
        rect = self.boundingRect()
        tl = rect.topLeft()
        br = rect.bottomRight()
        width = abs(br.x() - tl.x())
        height = abs(br.y() - tl.y())
        if width > min_width and height > min_height:
            return True

        return False

    def get_label(self):
        return self.label

    def set_cls_num(self, cls_num):
        self.cls_num = cls_num

    def set_label(self, text, text_pos, color=None):
        if not color:
            color = self.color
        self.label = make_label(text, text_pos, color)

    def __str__(self):
        pol = self.polygon()
        txt = f"Polygon class = {self.cls_num} color = {self.color}\n\tpoints: "
        for p in pol:
            txt += f"({p.x()}, {p.y()}) "
        return txt
