import os

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPolygonF, QColor, QPen, QPainter, QPixmap, QFont
from PyQt5.QtWidgets import QAction, QMenu, QGraphicsItem, QGraphicsLineItem, QGraphicsSimpleTextItem
from PyQt5.QtWidgets import QApplication
from shapely import Polygon, Point, unary_union

from ui.grapic_group import GrGroup
from ui.polygons import GrPolygonLabel, GrEllipsLabel, ActiveHandler
from ui.signals_and_slots import PolygonDeleteConnection, ViewMouseCoordsConnection, PolygonPressedConnection, \
    PolygonEndDrawing, MaskEndDrawing, PolygonChangeClsNumConnection, LoadIdProgress, InfoConnection
from utils import config
from utils import help_functions as hf
from utils.ids_worker import IdsSetterWorker
from utils.settings_handler import AppSettings

import shapely


class GraphicsView(QtWidgets.QGraphicsView):
    """
    Сцена для отображения текущей картинки и полигонов
    """

    def __init__(self, parent=None, active_color=None, fat_point_color=None, on_rubber_band_mode=None,
                 ruler_color=None, ruler_text_color=None):
        """
        active_color - цвет активного полигона, по умолчанию config.ACTIVE_COLOR
        fat_point_color - цвет узлов активного полигона, по умолчанию config.FAT_POINT_COLOR
        """

        super().__init__(parent)
        scene = QtWidgets.QGraphicsScene(self)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        # SIGNALS
        self.polygon_clicked = PolygonPressedConnection()
        self.polygon_delete = PolygonDeleteConnection()
        self.polygon_cls_num_change = PolygonChangeClsNumConnection()
        self.load_ids_conn = LoadIdProgress()
        self.mouse_move_conn = ViewMouseCoordsConnection()
        self.info_conn = InfoConnection()

        if on_rubber_band_mode:
            # connect SIGNAL on_rubber_band_mode TO SLOT
            on_rubber_band_mode.connect(self.on_rb_mode_change)
        self.polygon_end_drawing = PolygonEndDrawing()
        self.mask_end_drawing = MaskEndDrawing()

        self.setScene(scene)

        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self._pixmap_item)

        self.buffer = None
        self.init_objects_and_params()

        if not active_color:
            self.active_color = config.ACTIVE_COLOR
        else:
            self.active_color = active_color

        if not fat_point_color:
            self.fat_point_color = config.FAT_POINT_COLOR
        else:
            self.fat_point_color = fat_point_color

        if not ruler_color:
            self.ruler_color = config.RULER_COLOR
        else:
            self.ruler_color = ruler_color

        if not ruler_text_color:
            self.ruler_text_color = config.RULER_TEXT_COLOR
        else:
            self.ruler_text_color = ruler_text_color

        # Кисти
        self.set_brushes()
        self.active_group = ActiveHandler([])

        # self.setMouseTracking(False)
        self.min_ellipse_size = 10
        self.fat_width_default_percent = 50

        self.create_actions()

        self.setRenderHint(QPainter.Antialiasing)

    def init_objects_and_params(self):
        """
        Создание объектов по умолчанию
        """
        self.is_rubber_mode = False
        self.is_drawing = False
        self.is_ruler_mode = False

        self.drawing_type = "Polygon"

        self.labels_ids = []
        self.last_label_id = -1

        # Ruler Items:
        self.ruler_points = []
        self.ruler_draw_points = []
        self.ruler_line = None
        self.ruler_lrm = None
        self.ruler_text = None

        self.setMouseTracking(True)
        self.fat_point = None
        self.drag_mode = "No"
        self.dragged_vertex = None
        self.ellipse_start_point = None
        self.box_start_point = None

        self.pressed_polygon = None

        self._zoom = 0

        self.negative_points = []
        self.positive_points = []
        self.right_clicked_points = []
        self.left_clicked_points = []
        self.groups = []
        self.segments = []

        self.last_added = []

    def start_circle_progress(self):
        w = self.scene().width()
        h = self.scene().height()
        icon_folder = self.settings.get_icon_folder()
        self.loading_circle_angle = 0
        pixmap = QPixmap(os.path.join(icon_folder, "loader_ring.png"))

        pixmap = pixmap.scaled(128, 128)

        self.loading_circle = QtWidgets.QGraphicsPixmapItem(pixmap)
        self.loading_circle.setPos(w / 2 - pixmap.width() / 2, h / 2 - pixmap.height() / 2)
        self.loading_circle.setTransformOriginPoint(pixmap.width() / 2, pixmap.height() / 2)
        self.scene().addItem(self.loading_circle)

        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.spin_circle_progress)
        self.timer.start(5)

    def spin_circle_progress(self):
        self.loading_circle_angle += 1
        self.loading_circle.setRotation(self.loading_circle_angle)

    def stop_circle_progress(self):
        if self.loading_circle:
            self.scene().removeItem(self.loading_circle)

    def set_brushes(self):
        # Кисти для активного элемента, узла, позитивного и негативного промпта SAM
        self.active_brush = QtGui.QBrush(QColor(*self.active_color), QtCore.Qt.SolidPattern)
        self.fat_point_brush = QtGui.QBrush(QColor(*self.fat_point_color), QtCore.Qt.SolidPattern)
        self.ruler_brush = QtGui.QBrush(QColor(*self.ruler_color), QtCore.Qt.SolidPattern)
        self.ruler_text_brush = QtGui.QBrush(QColor(*self.ruler_text_color), QtCore.Qt.SolidPattern)
        self.positive_point_brush = QtGui.QBrush(QColor(*config.POSITIVE_POINT_COLOR), QtCore.Qt.SolidPattern)
        self.negative_point_brush = QtGui.QBrush(QColor(*config.NEGATIVE_POINT_COLOR), QtCore.Qt.SolidPattern)

    def create_actions(self):
        self.delPolyAct = QAction("Удалить полигон", self, enabled=True, triggered=self.on_del_polygon)
        self.changeClsNumAct = QAction("Изменить имя метки", self, enabled=True, triggered=self.on_change_cls_num)
        self.mergeActivePolygons = QAction("Объединить выделенные полигоны", self, enabled=True,
                                           triggered=self.on_merge_polygons)

    def on_merge_polygons(self):

        shapely_union = self.active_group.merge_polygons_to_shapely_union()

        for shape in shapely_union:
            point_mass = list(shape.exterior.coords)
            # text_pos = hf.calc_label_pos(point_mass)

            cls_num = self.active_group[0].cls_num
            color = self.active_group[0].color
            alpha = self.active_group[0].alpha_percent
            text = self.active_group[0].text

            self.add_polygon_to_scene(cls_num, point_mass, color, alpha, text=text)

        for item in self.active_group:
            self.remove_item(item, is_delete_id=True)
        self.active_group.clear()

        self.polygon_end_drawing.on_end_drawing.emit(True)

    def on_change_cls_num(self):
        if len(self.active_group) == 0:
            return

        change_ids = []
        if not self.active_group.is_all_actives_same_class():
            return

        for item in self.active_group:
            change_ids.append(item.id)
            cls_num = item.cls_num

        self.polygon_cls_num_change.pol_cls_num_and_id.emit(cls_num, change_ids)

    def on_del_polygon(self):
        delete_ids = []
        for item in self.active_group:
            delete_ids.append(item.id)
            self.remove_item(item, is_delete_id=True)
        self.polygon_delete.id_delete.emit(delete_ids)
        self.active_group.clear()

    def del_pressed_polygon(self):
        if self.pressed_polygon:
            self.pressed_polygon = None

    @property
    def pixmap_item(self):
        return self._pixmap_item

    def setPixmap(self, pixmap):
        """
        Задать новую картинку
        """
        scene = QtWidgets.QGraphicsScene(self)
        self.setScene(scene)
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self._pixmap_item)
        self.pixmap_item.setPixmap(pixmap)

        self.init_objects_and_params()

        self.set_fat_width()
        self.set_pens()
        self.remove_fat_point_from_scene()
        self.clear_ai_points()

        self.active_group = ActiveHandler([])
        self.active_group.set_brush_pen_line_width(self.active_brush, self.active_pen, self.line_width)

        # Ruler items clear:
        self.on_ruler_mode_off()

        if self.is_rubber_mode:
            self.on_rb_mode_change(False)

    def add_segment_pixmap(self, segment_pixmap, opacity=0.5, z_value=100):
        segment = QtWidgets.QGraphicsPixmapItem()
        segment.setOpacity(opacity)
        segment.setZValue(z_value)

        self.scene().addItem(segment)
        segment.setPixmap(segment_pixmap)
        self.segments.append(segment)

    def remove_all_segments(self):
        for s in self.segments:
            self.remove_item(s)

    def clearScene(self):
        """
        Очистить сцену
        """
        scene = QtWidgets.QGraphicsScene(self)
        self.setScene(scene)
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self._pixmap_item)

    def activate_item_by_id(self, id_to_found):
        found_item = None
        for item in self.scene().items():
            # ищем полигон с заданным id
            try:
                if item.id == id_to_found:
                    found_item = item
                    break
            except:
                pass

        self.active_group.reset_clicked_item(found_item, False)

    def set_fat_width(self, fat_width_percent_new=None):
        """
        Определение и установка толщины граней активного полигона и эллипса узловой точки активного полигона
        """
        pixmap_width = self.pixmap_item.pixmap().width()
        scale = pixmap_width / 2000.0

        if fat_width_percent_new:
            fat_scale = 0.3 + fat_width_percent_new / 50.0
            self.fat_width_default_percent = fat_width_percent_new
        else:
            fat_scale = 0.3 + self.fat_width_default_percent / 50.0

        self.fat_width = fat_scale * scale * 12 + 1
        self.line_width = int(self.fat_width / 8) + 1

        self._zoom = 0

        return self.fat_width

    def set_pens(self):
        self.active_pen = QPen(QColor(*hf.set_alpha_to_max(self.active_color)), self.line_width, QtCore.Qt.SolidLine)
        self.fat_point_pen = QPen(QColor(*self.fat_point_color), self.line_width, QtCore.Qt.SolidLine)
        self.ruler_pen = QPen(QColor(*self.ruler_color), self.line_width, QtCore.Qt.SolidLine)
        self.positive_point_pen = QPen(QColor(*config.POSITIVE_POINT_COLOR), self.line_width, QtCore.Qt.SolidLine)
        self.negative_point_pen = QPen(QColor(*config.NEGATIVE_POINT_COLOR), self.line_width, QtCore.Qt.SolidLine)

    def is_close_to_fat_point(self, lp):
        """
        fat_point - Эллипс - узел полигона
        """
        if self.fat_point:
            scale = self._zoom / 3.0 + 1
            rect = self.fat_point.rect()
            width = abs(rect.topRight().x() - rect.topLeft().x())
            height = abs(rect.topRight().y() - rect.bottomRight().y())
            center = QtCore.QPointF(rect.topLeft().x() + width / 2, rect.topLeft().y() + height / 2)
            d = hf.distance(lp, center)
            if d < (self.fat_width / scale):
                return True

        return False

    def check_near_by_active_pressed(self, lp):
        for active_item in self.active_group:
            scale = self._zoom / 3.0 + 1
            pol = active_item.polygon()

            size = len(pol)
            for i in range(size - 1):
                p1 = pol[i]
                p2 = pol[i + 1]

                d = hf.distance_from_point_to_segment(lp, p1, p2)

                if d < self.fat_width / scale:
                    self.polygon_clicked.id_pressed.emit(active_item.id)
                    return True

        return False

    def check_active_pressed(self, pressed_point):
        for active_item in self.active_group:
            pol = active_item.polygon()
            shapely_pol = Polygon([(p.x(), p.y()) for p in pol])

            if shapely_pol.contains(Point(pressed_point.x(), pressed_point.y())):
                self.polygon_clicked.id_pressed.emit(active_item.id)
                return True

        return False

    def is_point_in_pixmap_size(self, point):
        is_in_range = True
        pixmap_width = self.pixmap_item.pixmap().width()
        pixmap_height = self.pixmap_item.pixmap().height()
        if point.x() > pixmap_width:
            is_in_range = False
        if point.x() < 0:
            is_in_range = False
        if point.y() > pixmap_height:
            is_in_range = False
        if point.y() < 0:
            is_in_range = False

        return is_in_range

    def crop_by_pixmap_size(self, item):
        """
        Обрезка полигона рабочей областью. Решается как пересечение двух полигонов - активного и рабочей области
        """
        # Активный полигон
        pol = item.polygon()
        # Полигон рабочей области
        pixmap_width = self.pixmap_item.pixmap().width()
        pixmap_height = self.pixmap_item.pixmap().height()

        if not hf.check_polygon_out_of_screen(pol, pixmap_width, pixmap_height):
            pol = hf.convert_item_polygon_to_shapely(pol)
            pixmap_box = hf.make_shapely_box(pixmap_width, pixmap_height)

            cropped_polygon = pixmap_box.intersection(pol)
            pol = hf.convert_shapely_to_item_polygon(cropped_polygon)

            item.setPolygon(pol)

    def get_pressed_polygon(self, pressed_point):
        """
        Ищем полигон под точкой lp,
        Найдем - возвращаем полигон, не найдем - None.
        """

        for item in self.scene().items():
            # ищем, не попали ли уже в нарисованный полигон
            try:
                pol = item.polygon()
                if pol.containsPoint(pressed_point, QtCore.Qt.OddEvenFill):
                    return item
            except:
                pass

        return None

    def set_ids_from_project(self, project_data, on_set_callback=None, percent_max=100):
        self.ids_worker = IdsSetterWorker(images_data=project_data['images'], percent_max=percent_max)
        self.on_set_callback = on_set_callback
        self.ids_worker.load_ids_conn.percent.connect(self.on_load_percent_change)

        self.ids_worker.finished.connect(self.on_ids_worker_finished)

        if not self.ids_worker.isRunning():
            self.ids_worker.start()

    def on_load_percent_change(self, percent):
        self.load_ids_conn.percent.emit(percent)

    def on_ids_worker_finished(self):
        self.labels_ids = self.ids_worker.get_labels_ids()
        if len(self.labels_ids) > 0:
            self.last_label_id = self.labels_ids[-1]
        else:
            self.last_label_id = 0

        if self.on_set_callback:
            self.on_set_callback()

    def get_unique_label_id(self):
        new_id = self.last_label_id + 1
        if new_id not in self.labels_ids:
            self.labels_ids.append(new_id)
            self.last_label_id = new_id
            return new_id

        id_tek = 0
        while id_tek in self.labels_ids:
            id_tek += 1
        self.last_label_id = id_tek
        self.labels_ids.append(self.last_label_id)
        return id_tek

    def remove_label_id(self, id):
        if id in self.labels_ids:
            self.labels_ids.remove(id)
            if id == self.last_label_id:
                self.last_label_id = len(self.labels_ids) - 1

    def remove_last_changes(self):
        for item_id in self.last_added:
            self.remove_shape_by_id(item_id)
        self.last_added = []

    def add_polygons_group_to_scene(self, cls_num, point_of_points_mass, color=None, alpha=50, text=None):
        self.last_added = []
        for points_mass in point_of_points_mass:
            id = self.get_unique_label_id()
            self.last_added.append(id)
            self.add_polygon_to_scene(cls_num, points_mass, color=color, alpha=alpha, id=id, is_save_last=False,
                                      text=text)

    def add_polygon_to_scene(self, cls_num, point_mass, color=None, alpha=50, id=None, is_save_last=True, text=None):
        """
        Добавление полигона на сцену
        color - цвет. Если None - будет выбран цвет, соответствующий номеру класса из config.COLORS
        alpha - прозрачность в процентах
        """

        if len(point_mass) < 3:
            return

        if id == None:
            id = self.get_unique_label_id()
            if is_save_last:
                self.last_added = []
                self.last_added.append(id)

        if id not in self.labels_ids:
            self.labels_ids.append(id)

        polygon_new = GrPolygonLabel(None, color=color, cls_num=cls_num, alpha_percent=alpha, id=id, text=text,
                                     text_pos=hf.calc_label_pos(point_mass))

        polygon_new.setBrush(QtGui.QBrush(QColor(*polygon_new.color), QtCore.Qt.SolidPattern))
        polygon_new.setPen(QPen(QColor(*hf.set_alpha_to_max(polygon_new.color)), self.line_width, QtCore.Qt.SolidLine))

        poly = QPolygonF()

        for p in point_mass:
            poly.append(QtCore.QPointF(p[0], p[1]))

        polygon_new.setPolygon(poly)

        self.scene().addItem(polygon_new)
        if text:
            self.scene().addItem(polygon_new.get_label())

    def add_point_to_active(self, lp):

        if len(self.active_group) == 1:
            active_item = self.active_group[0]
            poly = active_item.polygon()
            closest_pair = hf.find_nearest_edge_of_polygon(poly, lp)
            poly_new = QPolygonF()

            size = len(poly)
            for i in range(size):
                p1 = poly[i]
                if i == size - 1:
                    p2 = poly[0]
                else:
                    p2 = poly[i + 1]

                if closest_pair == (p1, p2):
                    poly_new.append(p1)
                    closest_point = hf.get_closest_to_line_point(lp, p1, p2)
                    if closest_point:
                        poly_new.append(closest_point)

                else:
                    poly_new.append(p1)

            active_item.setPolygon(poly_new)

    def remove_polygon_vertext(self, lp):

        if len(self.active_group) == 1:
            active_item = self.active_group[0]
            point_closed = self.get_point_near_by_active_polygon_vertex(lp)

            if point_closed:

                poly_new = QPolygonF()

                pol = active_item.polygon()
                for p in pol:
                    if p != point_closed:
                        poly_new.append(p)

                active_item.setPolygon(poly_new)



    def copy_active_item_to_buffer(self):
        self.buffer = []
        for active_item in self.active_group:

            active_cls_num = active_item.cls_num
            active_alpha = active_item.alpha_percent
            active_color = active_item.color
            text = active_item.text
            text_pos = active_item.text_pos

            copy_id = self.get_unique_label_id()

            polygon_new = GrPolygonLabel(None, color=active_color, cls_num=active_cls_num,
                                         alpha_percent=active_alpha, id=copy_id, text=text, text_pos=text_pos)
            polygon_new.setPen(self.active_pen)
            polygon_new.setBrush(self.active_brush)
            poly = QPolygonF()
            for point in active_item.polygon():
                poly.append(point)

            polygon_new.setPolygon(poly)

            self.buffer.append(polygon_new)

    def paste_buffer(self):
        if self.buffer:
            for active_item in self.buffer:
                pol = active_item.polygon()
                xs = []
                ys = []
                for point in pol:
                    xs.append(point.x())
                    ys.append(point.y())
                min_x = min(xs)
                max_x = max(xs)
                min_y = min(ys)
                max_y = max(ys)
                w = abs(max_x - min_x)
                h = abs(max_y - min_y)

                pol_new = QPolygonF()
                for point in pol:
                    pol_new.append(QtCore.QPointF(point.x() + w / 2, point.y() + h / 2))

                active_item.setPolygon(pol_new)

                self.scene().addItem(active_item)

    def mousePressEvent(self, event: QtGui.QMouseEvent) -> None:

        modifierPressed = QApplication.keyboardModifiers()
        modifierName = ''
        # if (modifierPressed & QtCore.Qt.AltModifier) == QtCore.Qt.AltModifier:
        #     modifierName += 'Alt'

        if (modifierPressed & QtCore.Qt.ControlModifier) == QtCore.Qt.ControlModifier:
            modifierName += 'Ctrl'

        if (modifierPressed & QtCore.Qt.ShiftModifier) == QtCore.Qt.ShiftModifier:
            is_shift = True

        else:
            is_shift = False

        if event.buttons() == QtCore.Qt.RightButton:
            modifierName += " Right Click"

        if event.buttons() == QtCore.Qt.LeftButton:
            modifierName += " Left Click"

        sp = self.mapToScene(event.pos())
        lp = self.pixmap_item.mapFromScene(sp)

        if self.is_rubber_mode:

            if not self.box_start_point:
                self.drag_mode = "RubberBandStartDrawMode"
                self.box_start_point = lp

            return

        if self.is_ruler_mode:

            if len(self.ruler_points) == 1:
                # уже есть первая точка
                distance = hf.distance(self.ruler_points[0], lp)

                self.ruler_points.append(lp)
                self.draw_ruler_point(lp)
                self.draw_ruler_line()

                lang = self.settings.read_lang()
                if not self.ruler_lrm:
                    text = f"Расстояние {distance:0.1f} px" if lang == 'RU' else f"Distance {distance:0.1f} px"
                else:
                    distance *= self.ruler_lrm
                    text = f"Расстояние {distance:0.1f} м" if lang == 'RU' else f"Distance {distance:0.1f} m"

                self.draw_ruler_text(text, lp)

            elif len(self.ruler_points) == 2:
                self.ruler_points.clear()
                for p in self.ruler_draw_points:
                    self.remove_item(p)
                self.delete_ruler_line()
                self.delete_ruler_text()

            else:
                self.ruler_points.append(lp)
                self.draw_ruler_point(lp)

            return

        if self.is_drawing:

            if self.drawing_type == "Polygon":

                # Режим рисования, добавляем точки к текущему полигону
                if len(self.active_group) == 1:
                    active_item = self.active_group[0]
                    poly = active_item.polygon()
                    poly.append(lp)
                    active_item.setPolygon(poly)

            elif self.drawing_type == "Ellips":

                # Режим рисования, добавляем точки к текущему полигону

                if not self.ellipse_start_point:
                    self.drag_mode = "EllipsStartDrawMode"
                    self.ellipse_start_point = lp

            elif self.drawing_type == "Box":

                if not self.box_start_point:
                    self.drag_mode = "BoxStartDrawMode"
                    self.box_start_point = lp

            elif self.drawing_type == "AiPoints":
                if 'Left Click' in modifierName:
                    if self.is_point_in_pixmap_size(lp):
                        self.left_clicked_points.append(lp)
                        self.add_positive_ai_point_to_scene(lp)
                elif 'Right Click' in modifierName:
                    if self.is_point_in_pixmap_size(lp):
                        self.right_clicked_points.append(lp)
                        self.add_negative_ai_point_to_scene(lp)

            elif self.drawing_type == "AiMask":
                if not self.box_start_point:
                    self.drag_mode = "BoxStartDrawMode"
                    self.box_start_point = lp

        else:

            if 'Right Click' in modifierName:
                return

            if self.check_near_by_active_pressed(lp):  # нажали рядом с активным полигоном

                if self.is_close_to_fat_point(lp):
                    # Нажали по узлу

                    if 'Ctrl' in modifierName:
                        # Зажали одновременно Ctrl - убираем узел
                        self.remove_polygon_vertext(lp)

                    else:
                        # Начинаем тянуть
                        self.drag_mode = "PolygonVertexMove"
                        self.dragged_vertex = lp

                else:
                    # Нажали по грани
                    if 'Ctrl' in modifierName:
                        # Добавляем узел
                        self.add_point_to_active(lp)

                    else:
                        self.active_group.clear()

            else:

                # нажали не рядом с активным полигоном

                if self.check_active_pressed(lp):

                    # нажали прямо по активному полигону, строго внутри
                    # Начать перемещение
                    self.drag_mode = "PolygonMoveMode"
                    self.start_point = lp

                else:
                    # кликнули не по активной. Если по какой-то другой - изменить активную
                    pressed_polygon = self.get_pressed_polygon(lp)
                    if pressed_polygon:
                        self.active_group.reset_clicked_item(pressed_polygon, is_shift)
                        self.polygon_clicked.id_pressed.emit(pressed_polygon.id)
                    else:
                        self.active_group.clear()

    def get_point_near_by_active_polygon_vertex(self, point):

        if len(self.active_group) == 1:
            active_item = self.active_group[0]
            scale = self._zoom / 3.0 + 1
            pol = active_item.polygon()
            for p in pol:
                if hf.distance(p, point) < self.fat_width / scale:
                    return p

        return None

    def add_fat_point_to_polygon_vertex(self, vertex):
        scale = self._zoom / 5.0 + 1
        self.fat_point = GrEllipsLabel()
        self.fat_point.setRect(vertex.x() - self.fat_width / (2 * scale),
                               vertex.y() - self.fat_width / (2 * scale),
                               self.fat_width / scale, self.fat_width / scale)
        self.fat_point.setPen(self.fat_point_pen)
        self.fat_point.setBrush(self.fat_point_brush)

        self.scene().addItem(self.fat_point)

    def add_positive_ai_point_to_scene(self, point):
        scale = self._zoom / 2.5 + 1
        positive_point = GrEllipsLabel()
        positive_point.setRect(point.x() - self.fat_width / (2 * scale),
                               point.y() - self.fat_width / (2 * scale),
                               self.fat_width / scale, self.fat_width / scale)
        positive_point.setPen(self.positive_point_pen)
        positive_point.setBrush(self.positive_point_brush)

        self.positive_points.append(positive_point)
        self.scene().addItem(positive_point)

    def add_negative_ai_point_to_scene(self, point):
        scale = self._zoom / 2.5 + 1
        negative_point = GrEllipsLabel()
        negative_point.setRect(point.x() - self.fat_width / (2 * scale),
                               point.y() - self.fat_width / (2 * scale),
                               self.fat_width / scale, self.fat_width / scale)
        negative_point.setPen(self.negative_point_pen)
        negative_point.setBrush(self.negative_point_brush)

        self.negative_points.append(negative_point)
        self.scene().addItem(negative_point)

    def clear_ai_points(self):

        for p in self.negative_points:
            self.remove_item(p, is_delete_id=False)
        for p in self.positive_points:
            self.remove_item(p, is_delete_id=False)
        self.positive_points.clear()
        self.negative_points.clear()
        self.right_clicked_points.clear()
        self.left_clicked_points.clear()

    def remove_fat_point_from_scene(self):
        if self.fat_point:
            self.remove_item(self.fat_point, is_delete_id=False)
            self.fat_point = None

    def wheelEvent(self, event: QtGui.QWheelEvent) -> None:
        sp = self.mapToScene(event.pos())
        lp = self.pixmap_item.mapFromScene(sp)

        modifierPressed = QApplication.keyboardModifiers()
        modifierName = ''
        if (modifierPressed & QtCore.Qt.AltModifier) == QtCore.Qt.AltModifier:
            modifierName += 'Alt'

        if (modifierPressed & QtCore.Qt.ControlModifier) == QtCore.Qt.ControlModifier:
            modifierName += 'Ctrl'

        if 'Ctrl' in modifierName:
            if event.angleDelta().y() > 0:
                factor = 1.25
                self._zoom += 1
            else:
                factor = 0.8
                self._zoom -= 1

            if self._zoom > 0:
                self.scale(factor, factor)
                self.centerOn(lp)

            elif self._zoom == 0:
                self.fitInView(self.pixmap_item, QtCore.Qt.KeepAspectRatio)
            else:
                self._zoom = 0

    def scaleView(self, scaleFactor):
        factor = self.transform().scale(scaleFactor, scaleFactor).mapRect(QtCore.QRectF(0, 0, 1, 1)).width()
        if factor < 0.07 or factor > 100:
            return
        self.scale(scaleFactor, scaleFactor)

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):

        sp = self.mapToScene(event.pos())
        lp = self.pixmap_item.mapFromScene(sp)
        self.mouse_move_conn.on_mouse_move.emit(lp.x(), lp.y())

        if self.is_rubber_mode:
            if self.drag_mode == "RubberBandStartDrawMode" or self.drag_mode == "RubberBandContinueDrawMode":
                width = abs(lp.x() - self.box_start_point.x())
                height = abs(lp.y() - self.box_start_point.y())

                polygon = QPolygonF()
                polygon.append(self.box_start_point)
                polygon.append(QtCore.QPointF(self.box_start_point.x() + width, self.box_start_point.y()))
                polygon.append(lp)
                polygon.append(QtCore.QPointF(self.box_start_point.x(), self.box_start_point.y() + height))
                if len(self.active_group) == 1:
                    active_item = self.active_group[0]
                    active_item.setPolygon(polygon)

                self.drag_mode = "RubberBandContinueDrawMode"

            return

        if len(self.active_group) != 0:
            active_item = self.active_group[0]

            if self.drag_mode == "EllipsStartDrawMode" or self.drag_mode == "EllipseContinueDrawMode":

                width = abs(lp.x() - self.ellipse_start_point.x())
                height = abs(lp.y() - self.ellipse_start_point.y())
                if len(self.active_group) == 1:
                    active_item.setRect(self.ellipse_start_point.x(), self.ellipse_start_point.y(), width, height)

                self.drag_mode = "EllipseContinueDrawMode"

            elif self.drag_mode == "BoxStartDrawMode" or self.drag_mode == "BoxContinueDrawMode":

                width = abs(lp.x() - self.box_start_point.x())
                height = abs(lp.y() - self.box_start_point.y())

                polygon = QPolygonF()
                polygon.append(self.box_start_point)
                polygon.append(QtCore.QPointF(self.box_start_point.x() + width, self.box_start_point.y()))
                polygon.append(lp)
                polygon.append(QtCore.QPointF(self.box_start_point.x(), self.box_start_point.y() + height))
                if len(self.active_group) == 1:
                    active_item = self.active_group[0]
                    active_item.setPolygon(polygon)

                self.drag_mode = "BoxContinueDrawMode"

            elif self.drag_mode == "PolygonMoveMode":

                delta_x = lp.x() - self.start_point.x()
                delta_y = lp.y() - self.start_point.y()

                for active_item in self.active_group:

                    poly = QPolygonF()
                    for point in active_item.polygon():
                        point_moved = QtCore.QPointF(point.x() + delta_x, point.y() + delta_y)
                        poly.append(point_moved)

                    label = active_item.get_label()
                    if label:
                        pos = label.pos()
                        label.setPos(pos.x() + delta_x, pos.y() + delta_y)

                    active_item.setPolygon(poly)

                self.start_point = lp
            else:
                # Если активная - отслеживаем ее узлы
                self.remove_fat_point_from_scene()  # сперва убираем предыдущую точку

                point_closed = self.get_point_near_by_active_polygon_vertex(lp)
                if point_closed:
                    self.add_fat_point_to_polygon_vertex(point_closed)

                else:
                    self.remove_fat_point_from_scene()

    def change_dragged_polygon_vertex_to_point(self, new_point):
        scale = self._zoom / 3.0 + 1

        if len(self.active_group) == 1:
            active_item = self.active_group[0]

            poly = QPolygonF()
            for point in active_item.polygon():
                if hf.distance(point, self.dragged_vertex) < self.fat_width / scale:
                    poly.append(new_point)
                else:
                    poly.append(point)

            active_item.setPolygon(poly)

    def get_rubber_band_polygon(self):
        if len(self.active_group) == 1:
            active_item = self.active_group[0]
            if self.is_rubber_mode and active_item.cls_num == -1:
                points = []
                for p in active_item.polygon():
                    points.append([int(p.x()), int(p.y())])
                return points

    def get_active_item_polygon(self):
        if len(self.active_group) == 1:
            active_item = self.active_group[0]
            points = []
            for p in active_item.polygon():
                points.append([int(p.x()), int(p.y())])
            return points

    def is_self_intersection(self, item):
        pol = hf.convert_item_polygon_to_shapely(item.polygon())
        if not pol.is_valid:
            return True
        return False

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):

        if self.is_rubber_mode:
            if self.drag_mode == "RubberBandContinueDrawMode":

                # self.setMouseTracking(True)
                self.is_drawing = False
                self.box_start_point = None
                self.drag_mode = "No"

                if len(self.active_group) == 1:
                    active_item = self.active_group[0]
                    if active_item.is_self_intersected():
                        self.info_conn.info_message.emit(
                            "Polygon self-intersected" if self.lang == 'ENG' else "Полигон не должен содержать самопересечений. Удален")
                        self.remove_item(active_item, is_delete_id=True)
                    else:
                        self.crop_by_pixmap_size(active_item)

            return

        if self.drag_mode == "PolygonVertexMove":

            if self.dragged_vertex:
                sp = self.mapToScene(event.pos())
                lp = self.pixmap_item.mapFromScene(sp)

                self.change_dragged_polygon_vertex_to_point(lp)
                if len(self.active_group) == 1:
                    active_item = self.active_group[0]
                    if active_item.is_self_intersected():
                        self.info_conn.info_message.emit(
                            "Polygon self-intersected" if self.lang == 'ENG' else "Полигон не должен содержать самопересечений. Удален")
                        self.remove_item(active_item, is_delete_id=True)

            self.drag_mode = "No"
            # self.setMouseTracking(True)

        elif self.drag_mode == "PolygonMoveMode":

            for active_item in self.active_group:
                sp = self.mapToScene(event.pos())
                lp = self.pixmap_item.mapFromScene(sp)

                delta_x = lp.x() - self.start_point.x()
                delta_y = lp.y() - self.start_point.y()

                poly = QPolygonF()

                for point in active_item.polygon():
                    point_moved = QtCore.QPointF(point.x() + delta_x, point.y() + delta_y)
                    poly.append(point_moved)

                active_item.setPolygon(poly)

                if active_item.is_self_intersected():
                    self.info_conn.info_message.emit(
                        "Polygon self-intersected" if self.lang == 'ENG' else "Полигон не должен содержать самопересечений. Удален")
                    self.remove_item(active_item, is_delete_id=True)
                else:
                    self.crop_by_pixmap_size(active_item)

            self.drag_mode = "No"

        elif self.drag_mode == "EllipseContinueDrawMode":
            # self.setMouseTracking(True)
            self.is_drawing = False
            self.ellipse_start_point = None
            self.drag_mode = "No"

            if len(self.active_group) == 1:
                active_item = self.active_group[0]
                polygon_new = active_item.convert_to_polygon(points_count=30)
                if active_item.is_self_intersected():
                    self.info_conn.info_message.emit(
                        "Polygon self-intersected" if self.lang == 'ENG' else "Полигон не должен содержать самопересечений. Удален")
                    self.remove_item(active_item, is_delete_id=True)
                else:
                    self.remove_item(active_item, is_delete_id=False)
                    self.scene().addItem(polygon_new)
                    self.toggle(polygon_new)
                    active_item = polygon_new

                    self.crop_by_pixmap_size(active_item)

                    self.polygon_end_drawing.on_end_drawing.emit(True)



        elif self.drag_mode == "BoxContinueDrawMode":

            # self.setMouseTracking(True)
            self.is_drawing = False
            self.box_start_point = None
            self.drag_mode = "No"

            if len(self.active_group) == 1:
                active_item = self.active_group[0]

                if self.drawing_type != "AiMask":
                    self.crop_by_pixmap_size(active_item)
                    self.polygon_end_drawing.on_end_drawing.emit(True)
                else:
                    self.mask_end_drawing.on_mask_end_drawing.emit(True)

    def remove_active(self, is_delete_id=True):
        for item in self.active_group:
            self.remove_item(item, is_delete_id)

    def remove_item(self, item, is_delete_id=False):

        if is_delete_id and item.id in self.labels_ids:
            self.labels_ids.remove(item.id)
            if item.id == self.last_label_id:
                self.last_label_id = self.last_label_id - 1

        text_label = item.get_label()
        if text_label:
            self.scene().removeItem(text_label)
        self.scene().removeItem(item)

    def remove_all_polygons(self):
        self.is_drawing = False
        for item in self.scene().items():
            # ищем, не попали ли уже в нарисованный полигон
            try:
                pol = item.polygon()
                if pol:
                    self.remove_item(item, is_delete_id=True)
            except:
                pass

    def get_shapes_by_cls_num(self, cls_num, is_filter=True):
        shapes = []
        for item in self.scene().items():
            # ищем, не попали ли уже в нарисованный полигон
            try:
                pol = item.polygon()

                if pol:
                    tek_cls = item.cls_num
                    if tek_cls == cls_num:
                        if is_filter and len(pol) < 3:
                            self.remove_item(item, is_delete_id=True)
                            continue

                        shape = {"cls_num": item.cls_num, "id": item.id}
                        points = []
                        for p in pol:
                            points.append([p.x(), p.y()])
                        shape["points"] = points
                        shapes.append(shape)
            except:
                pass

        return shapes

    def remove_shape_by_id(self, shape_id):
        for item in self.scene().items():
            # ищем, не попали ли уже в нарисованный полигон
            try:
                pol = item.polygon()

                if pol:
                    tek_id = item.id
                    if tek_id == shape_id:
                        self.remove_item(item, is_delete_id=True)
                        return True
            except:
                pass

        return False

    def remove_shapes_by_cls(self, cls_num, is_filter=True):
        removed_count = 0
        for item in self.scene().items():
            # ищем, не попали ли уже в нарисованный полигон
            try:
                pol = item.polygon()

                if pol:
                    tek_cls = item.cls_num
                    if tek_cls == cls_num:
                        self.remove_item(item, is_delete_id=True)
                        removed_count += 1
            except:
                pass

        return removed_count

    def get_shape_by_id(self, shape_id, is_filter=True):
        for item in self.scene().items():
            # ищем, не попали ли уже в нарисованный полигон
            try:
                pol = item.polygon()

                if pol:
                    tek_id = item.id
                    if tek_id == shape_id:
                        if is_filter and len(pol) < 3:
                            self.remove_item(item, is_delete_id=True)
                            continue

                        shape = {"cls_num": item.cls_num, "id": item.id}
                        points = []
                        for p in pol:
                            points.append([p.x(), p.y()])
                        shape["points"] = points
                        return shape
            except:
                pass

        return None

    def get_all_shapes(self, is_filter=True):
        shapes = []

        for item in self.scene().items():
            # ищем, не попали ли уже в нарисованный полигон
            try:
                pol = item.polygon()
                if pol:

                    if is_filter and len(pol) < 3:
                        self.remove_item(item, is_delete_id=True)
                        continue

                    shape = {"cls_num": item.cls_num, "id": item.id}
                    points = []
                    for p in pol:
                        points.append([p.x(), p.y()])
                    shape["points"] = points
                    shapes.append(shape)
            except:
                pass

        return shapes

    def add_item_to_scene_as_active(self, item):
        self.active_group.append(item)
        self.scene().addItem(item)

    def start_drawing(self, type="Polygon", cls_num=0, color=None, alpha=50, id=None):
        """
        Старт отрисовки фигуры, по умолчанию - полигона

        type - тип фигуры, по умолчанию - полигон
        cls_num - номер класса
        color - цвет. Если None - будет выбран цвет, соответствующий номеру класса из config.COLORS
        alpha - прозрачность в процентах
        """

        # self.setMouseTracking(False)
        is_drawing_old = self.is_drawing
        self.active_group.clear()
        self.is_drawing = True

        self.drawing_type = type

        if type in ["Polygon", "Box", "Ellips"]:

            if id == None:
                if is_drawing_old:
                    if len(self.active_group) != 0:
                        active_item = self.active_group[0]
                        id = active_item.id
                        self.remove_item(active_item)
                else:
                    id = self.get_unique_label_id()

            if type == "Polygon" or type == "Box":
                active_item = GrPolygonLabel(self._pixmap_item, color=color, cls_num=cls_num, alpha_percent=alpha,
                                             id=id)
            elif type == "Ellips":
                active_item = GrEllipsLabel(self._pixmap_item, color=color, cls_num=cls_num, alpha_percent=alpha,
                                            id=id)

            self.add_item_to_scene_as_active(active_item)


        elif type == "AiPoints":
            self.left_clicked_points = QPolygonF()
            self.right_clicked_points = QPolygonF()

            self.active_group.clear()

        elif type == "AiMask":

            active_item = GrPolygonLabel(None, color=color, cls_num=cls_num, alpha_percent=alpha,
                                         id=-1)

            self.add_item_to_scene_as_active(active_item)

        elif type == "RubberBand":
            active_item = GrPolygonLabel(None, color=color, cls_num=-1, alpha_percent=alpha,
                                         id=-1)
            self.add_item_to_scene_as_active(active_item)

    def get_sam_input_points_and_labels(self):
        if self.drawing_type == "AiPoints":
            input_point = []
            input_label = []
            for p in self.right_clicked_points:
                input_point.append([int(p.x()), int(p.y())])
                input_label.append(0)

            for p in self.left_clicked_points:
                input_point.append([int(p.x()), int(p.y())])
                input_label.append(1)

            input_point = np.array(input_point)
            input_label = np.array(input_label)

            return input_point, input_label

    def get_sam_mask_input(self):
        if self.drawing_type == "AiMask":

            if len(self.active_group) != 0:
                active_item = self.active_group[0]
                pol = active_item.polygon()
                if len(pol) == 4:
                    # только если бокс
                    left_top_point = pol[0]
                    right_bottom_point = pol[2]
                    input_box = np.array([int(left_top_point.x()), int(left_top_point.y()),
                                          int(right_bottom_point.x()), int(right_bottom_point.y())])

                    return input_box
                else:
                    self.remove_item(active_item)

        return []

    def break_drawing(self):
        self.is_drawing = False
        self.box_start_point = None
        self.drag_mode = "No"
        self.remove_active()
        self.remove_fat_point_from_scene()
        if self.drawing_type == "AiPoints":
            self.clear_ai_points()

    def end_drawing(self, text=None, cls_num=-1, color=None):
        self.is_drawing = False

        self.remove_fat_point_from_scene()

        if self.drawing_type == "AiPoints":
            self.clear_ai_points()

        if len(self.active_group) != 0:
            active_item = self.active_group[0]
            if not active_item:
                return

            if active_item.is_self_intersected():
                self.info_conn.info_message.emit(
                    "Polygon self-intersected" if self.lang == 'ENG' else "Полигон не должен содержать самопересечений. Удален")
                self.remove_item(active_item, is_delete_id=True)
                return

            self.crop_by_pixmap_size(active_item)

            if text and cls_num != -1:
                # post settings
                point_mass = hf.convert_item_polygon_to_point_mass(active_item.polygon())
                text_pos = hf.calc_label_pos(point_mass)
                active_item.set_label(text, text_pos)
                active_item.set_color(color)
                active_item.set_cls_num(cls_num)

                self.scene().addItem(active_item.get_label())

            self.active_group.clear()

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:

        if self.drawing_type == "AiPoints" and self.is_drawing:
            return

        if self._pixmap_item:
            sp = self.mapToScene(event.pos())
            lp = self.pixmap_item.mapFromScene(sp)

            pressed_polygon = self.get_pressed_polygon(lp)

            if pressed_polygon:
                menu = QMenu(self)

                if pressed_polygon not in self.active_group:
                    self.active_group.append(pressed_polygon)

                if self.active_group.is_all_actives_same_class():
                    if len(self.active_group) > 1:
                        menu.addAction(self.mergeActivePolygons)

                    if len(self.active_group) > 1:
                        self.changeClsNumAct.setText(
                            "Изменить имя меток" if self.lang == 'RU' else 'Change labels names')
                    else:
                        self.changeClsNumAct.setText("Изменить имя метки" if self.lang == 'RU' else 'Change label name')
                    menu.addAction(self.changeClsNumAct)

                if len(self.active_group) > 1:
                    self.delPolyAct.setText("Удалить полигоны" if self.lang == 'RU' else 'Delete polygons')
                else:
                    self.delPolyAct.setText("Удалить полигон" if self.lang == 'RU' else 'Delete polygon')
                menu.addAction(self.delPolyAct)

                menu.exec(event.globalPos())

    def on_rb_mode_change(self, is_active):
        self.is_rubber_mode = is_active
        if is_active:
            self.start_drawing(type='RubberBand')
        else:
            for item in self.active_group:
                if item.cls_num == -1:
                    self.remove_item(item)

    def add_group_of_points(self, points, cls_name, color, alpha):

        self.groups = []
        group = GrGroup(cls_name=cls_name, alpha_percent=alpha, color=color, group_id=len(self.groups) + 1)
        group.add_points(points)

        # group.setZValue(1)
        group.setOpacity(alpha)
        group.setFlag(QGraphicsItem.ItemIsMovable)
        self.groups.append(group)
        self.scene().addItem(group)

    def on_ruler_mode_on(self, ruler_lrm=None):
        self.is_ruler_mode = True
        self.ruler_lrm = ruler_lrm

    def on_ruler_mode_off(self):
        self.is_ruler_mode = False
        self.ruler_points.clear()
        for p in self.ruler_draw_points:
            self.remove_item(p)

        if self.ruler_line:
            self.delete_ruler_line()

        if self.ruler_text:
            self.delete_ruler_text()

    def draw_ruler_point(self, pressed_point):
        scale = self._zoom / 5.0 + 1

        ruler_point_width = self.fat_width / 2.0

        ruler_draw_point = GrEllipsLabel()
        ruler_draw_point.setRect(pressed_point.x() - ruler_point_width / (2 * scale),
                                 pressed_point.y() - ruler_point_width / (2 * scale),
                                 ruler_point_width / scale, ruler_point_width / scale)
        ruler_draw_point.setPen(self.ruler_pen)
        ruler_draw_point.setBrush(self.ruler_brush)

        self.scene().addItem(ruler_draw_point)
        self.ruler_draw_points.append(ruler_draw_point)

    def draw_ruler_line(self):
        if len(self.ruler_points) != 2:
            return
        p1 = self.ruler_points[0]
        p2 = self.ruler_points[1]
        self.ruler_line = QGraphicsLineItem()
        self.ruler_line.setLine(p1.x(), p1.y(), p2.x(), p2.y())
        self.ruler_line.setPen(self.ruler_pen)
        self.scene().addItem(self.ruler_line)

    def delete_ruler_line(self):
        if self.ruler_line:
            self.scene().removeItem(self.ruler_line)

    def draw_ruler_text(self, text, pos, pixel_size=None):

        if not pixel_size:
            # Размер шрифта в пикселях. Нужно вычислить от размера снимка
            # для 1280 pixel_size = 10 норм
            im_height = self.scene().height()
            pixel_size = max(10, int(im_height / 128.0))

        self.ruler_text = QGraphicsSimpleTextItem()
        self.ruler_text.setText(text)
        self.ruler_text.setBrush(self.ruler_text_brush)
        self.ruler_text.setPos(pos)
        font = QFont("Arial", pixel_size, QFont.Normal)
        self.ruler_text.setFont(font)
        self.scene().addItem(self.ruler_text)

    def delete_ruler_text(self):
        if self.ruler_text:
            self.scene().removeItem(self.ruler_text)
