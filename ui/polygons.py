import math

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QPolygonF, QPen

from utils import config
from utils import help_functions as hf
from utils.settings_handler import AppSettings

settings = AppSettings()

from shapely import Polygon


def set_item_label(item, text=None, color=None):
    if not item.label:
        point_mass = hf.convert_item_polygon_to_point_mass(item.polygon())

        if len(point_mass) > 2:
            text_pos = hf.calc_label_pos(point_mass)
            if not text:
                text = item.text
            if not color:
                color = item.color

            item.set_label(text, text_pos, color)

            return item.get_label()


def check_polygon_item(item):
    point_mass = hf.convert_item_polygon_to_point_mass(item.polygon())

    if len(point_mass) > 2:
        return True

    return False


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
    Нормализация и ограничение прозрачности
    """
    if not color:
        if cls_num > len(config.COLORS) - 1:
            color_num = len(config.COLORS) % (cls_num + 1) - 1
            color = config.COLORS[color_num]
        else:
            color = config.COLORS[cls_num]

    alpha = hf.convert_percent_to_alpha(alpha, alpha_min=alpha_min, alpha_max=alpha_max)

    return (color[0], color[1], color[2], alpha)


class AiPoint(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, parent, line_width=12, is_positive=True):
        self.positive_point_pen = QPen(QColor(*config.POSITIVE_POINT_COLOR), line_width, QtCore.Qt.SolidLine)
        self.negative_point_pen = QPen(QColor(*config.NEGATIVE_POINT_COLOR), line_width, QtCore.Qt.SolidLine)

        self.positive_point_brush = QtGui.QBrush(QColor(*config.POSITIVE_POINT_COLOR), QtCore.Qt.SolidPattern)
        self.negative_point_brush = QtGui.QBrush(QColor(*config.NEGATIVE_POINT_COLOR), QtCore.Qt.SolidPattern)

        super().__init__(parent)

        if is_positive:
            self.setBrush(self.positive_point_brush)
            self.setPen(self.positive_point_pen)
        else:
            self.setBrush(self.negative_point_brush)
            self.setPen(self.negative_point_pen)

    def get_label(self):
        return None


class RulerLine(QtWidgets.QGraphicsLineItem):
    def __init__(self, parent, line_width=12, ruler_color=None):
        if not ruler_color:
            ruler_color = config.RULER_COLOR

        self.ruler_pen = QPen(QColor(*ruler_color), line_width, QtCore.Qt.SolidLine)

        super().__init__(parent)

        self.setPen(self.ruler_pen)

    def get_label(self):
        return None


class RulerPoint(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, parent, line_width=12, ruler_color=None):
        if not ruler_color:
            ruler_color = config.RULER_COLOR

        self.ruler_brush = QtGui.QBrush(QColor(*ruler_color), QtCore.Qt.SolidPattern)

        self.ruler_pen = QPen(QColor(*ruler_color), line_width, QtCore.Qt.SolidLine)

        super().__init__(parent)

        self.setBrush(self.ruler_brush)
        self.setPen(self.ruler_pen)

    def get_label(self):
        return None


class FatPoint(QtWidgets.QGraphicsEllipseItem):
    def __init__(self, parent):
        super().__init__(parent)

    def get_label(self):
        return None


class GrEllipsLabel(QtWidgets.QGraphicsEllipseItem):
    """
    Эллипс с поддержкой номера класса, цветов и т.д.
    """

    def __init__(self, parent=None, cls_num=0, color=None, alpha_percent=50, id=-1, text=None, text_pos=None,
                 alpha_edge=None):

        super().__init__(parent)
        self.cls_num = cls_num
        self.id = id
        self.alpha_percent = alpha_percent
        self.color = get_color(color, cls_num, alpha_percent)

        self.alpha_edge = alpha_edge
        self.edge_color = get_color(color, cls_num, alpha_edge)

        self.label = None

        self.old_color = None
        self.old_label_text_color = None

        self.text = text
        self.text_pos = text_pos

        if text and text_pos:
            self.label = make_label(text, text_pos, self.color)

    def set_cls_num(self, cls_num):
        self.cls_num = cls_num

    def get_label(self):
        return self.label

    def set_label(self, text, text_pos, color=None):
        if color:
            self.color = color
        self.label = make_label(text, text_pos, self.color)
        self.text = text
        self.text_pos = text_pos

    def hide_color(self):
        self.old_color = self.color
        self.color = self.color[0], self.color[1], self.color[2], 0
        self.setBrush(QtGui.QBrush(QColor(*self.color), QtCore.Qt.SolidPattern))
        self.setPen(QPen(QColor(*self.color), 1, QtCore.Qt.SolidLine))
        if self.label:
            self.old_label_text_color = self.label.defaultTextColor()
            self.label.setDefaultTextColor(QColor(*self.color))

    def get_color_back(self, line_width=5):
        if self.old_color:
            self.color = self.old_color
            self.setBrush(QtGui.QBrush(QColor(*self.color), QtCore.Qt.SolidPattern))
            self.setPen(QPen(QColor(*self.edge_color), line_width, QtCore.Qt.SolidLine))
            if self.label:
                self.label.setDefaultTextColor(QColor(*self.old_label_text_color))

    def set_color(self, color=None, cls_num=None, alpha_percent=None):
        if not color:
            return
        if not cls_num:
            cls_num = self.cls_num
        if not alpha_percent:
            alpha_percent = self.alpha_percent
        self.color = get_color(color, cls_num, alpha_percent)

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

    def is_self_intersected(self):
        pol = self.convert_to_polygon()
        pol = hf.convert_item_polygon_to_shapely(pol)
        if pol.is_valid:
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

        text = self.text
        point_mass = hf.convert_item_polygon_to_point_mass(polygon)
        text_pos = hf.calc_label_pos(point_mass)

        grpol = GrPolygonLabel(parent, color=self.color, cls_num=self.cls_num, alpha_percent=self.alpha_percent,
                               id=self.id, text=text, text_pos=text_pos, alpha_edge=self.alpha_edge)

        grpol.setBrush(QtGui.QBrush(QColor(*self.color), QtCore.Qt.SolidPattern))
        grpol.setPen(QPen(QColor(*self.edge_color), 5, QtCore.Qt.SolidLine))
        grpol.setPolygon(polygon)

        return grpol

    def polygon(self, points_count=30):
        rect = self.boundingRect()

        # взять кол-во точек. Определить делта-угол. Определить центр.
        # Посчитать точки и накопить полигон.
        polygon = QPolygonF()
        delta_angle_rad = 2 * math.pi / points_count
        tek_angle = 0
        for i in range(points_count):
            polygon.append(hf.calc_ellips_point_coords(rect, tek_angle))
            tek_angle += delta_angle_rad
        return polygon

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

    def __init__(self, parent=None, cls_num=0, color=None, alpha_percent=50, id=0, text=None, text_pos=None,
                 alpha_edge=None):

        super().__init__(parent)
        self.cls_num = cls_num
        self.id = id
        self.alpha_percent = alpha_percent
        self.color = get_color(color, cls_num, alpha_percent)

        self.alpha_edge = alpha_edge
        self.edge_color = get_color(color, cls_num, alpha_edge)

        self.label = None
        self.old_color = None
        self.old_label_text_color = None

        self.text = text
        self.text_pos = text_pos

        if text and text_pos:
            self.label = make_label(text, text_pos, self.color)

    def is_self_intersected(self):
        pol = self.polygon()
        pol = hf.convert_item_polygon_to_shapely(pol)
        if pol.is_valid:
            return False
        return True

    def get_label(self):
        return self.label

    def hide_color(self):
        self.old_color = self.color
        self.color = self.color[0], self.color[1], self.color[2], 0
        self.setBrush(QtGui.QBrush(QColor(*self.color), QtCore.Qt.SolidPattern))
        self.setPen(QPen(QColor(*self.color), 1, QtCore.Qt.SolidLine))
        if self.label:
            self.label.hide()

    def get_color_back(self, line_width=5):
        if self.old_color:
            self.color = self.old_color
            self.setBrush(QtGui.QBrush(QColor(*self.color), QtCore.Qt.SolidPattern))
            self.setPen(QPen(QColor(*self.edge_color), line_width, QtCore.Qt.SolidLine))
            if self.label:
                self.label.show()

    def set_color(self, color=None, cls_num=None, alpha_percent=None):
        if not color:
            return
        if not cls_num:
            cls_num = self.cls_num
        if not alpha_percent:
            alpha_percent = self.alpha_percent
        self.color = get_color(color, cls_num, alpha_percent)

    def set_cls_num(self, cls_num):
        self.cls_num = cls_num

    def set_label(self, text, text_pos, color=None):
        if color:
            self.color = color
        self.label = make_label(text, text_pos, self.color)
        self.text = text
        self.text_pos = text_pos

    def __str__(self):
        pol = self.polygon()
        txt = f"Polygon class = {self.cls_num} color = {self.color}\n\tpoints: "
        for p in pol:
            txt += f"({p.x()}, {p.y()}) "
        return txt


class ActiveHandler(list):
    def __init__(self, iterable):
        super().__init__(iterable)
        self.active_brush = None
        self.active_pen = None
        self.line_width = None

    def set_brush_pen_line_width(self, active_brush, active_pen, line_width=5):
        self.active_brush = active_brush
        self.active_pen = active_pen
        self.line_width = line_width

    def append(self, item):
        item.setBrush(self.active_brush)
        item.setPen(self.active_pen)
        super().append(item)

    def remove(self, item):
        item.setBrush(QtGui.QBrush(QColor(*item.color), QtCore.Qt.SolidPattern))
        item.setPen(QPen(QColor(*item.edge_color), self.line_width, QtCore.Qt.SolidLine))
        super().remove(item)

    def clear(self):
        for item in self:
            item.setBrush(QtGui.QBrush(QColor(*item.color), QtCore.Qt.SolidPattern))
            item.setPen(QPen(QColor(*item.edge_color), self.line_width, QtCore.Qt.SolidLine))
        super().clear()

    def reset_clicked_item(self, item_clicked, is_shift):
        if item_clicked in self:
            self.remove(item_clicked)
        else:
            if is_shift:
                self.append(item_clicked)
            else:
                self.clear()
                self.append(item_clicked)

    def merge_polygons_to_shapely_union(self):
        shapely_polygons = []
        if len(self) < 2:
            return shapely_polygons

        for item in self:
            pol = item.polygon()
            shapely_pol = Polygon([(p.x(), p.y()) for p in pol])
            shapely_polygons.append(shapely_pol)

        shapely_union = hf.merge_polygons(shapely_polygons)

        return shapely_union

    def is_all_actives_same_class(self):
        cls_nums = [item.cls_num for item in self]
        return len(set(cls_nums)) == 1
