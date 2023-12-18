import math
import os

import numpy as np
from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import QPolygonF, QColor, QPen, QPainter, QPixmap, QFont, QCursor
from PyQt5.QtWidgets import QAction, QMenu, QGraphicsItem, QGraphicsSimpleTextItem
from PyQt5.QtWidgets import QApplication
from shapely import Polygon, Point

from ui.polygons import GrPolygonLabel, GrEllipsLabel, ActiveHandler, AiPoint, FatPoint, RulerPoint, RulerLine, \
    set_item_label, check_polygon_item
from ui.signals_and_slots import PolygonDeleteConnection, ViewMouseCoordsConnection, PolygonPressedConnection, \
    PolygonEndDrawing, MaskEndDrawing, PolygonChangeClsNumConnection, LoadIdProgress, InfoConnection, \
    ListOfPolygonsConnection
from utils import config
from utils import help_functions as hf
from utils.ids_worker import IdsSetterWorker
from utils.settings_handler import AppSettings
from utils.labels_ids_handler import LabelsIds

from utils.states import ViewState, DrawState, DragState


class GraphicsView(QtWidgets.QGraphicsView):
    """
    Сцена для отображения текущей картинки и полигонов
    """

    def __init__(self, parent=None, active_color=None, fat_point_color=None, on_rubber_band_mode=None):
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
        self.list_of_polygons_conn = ListOfPolygonsConnection()

        if on_rubber_band_mode:
            # connect SIGNAL on_rubber_band_mode TO SLOT
            on_rubber_band_mode.connect(self.on_rb_mode_change)
        self.polygon_end_drawing = PolygonEndDrawing()
        self.mask_end_drawing = MaskEndDrawing()

        self.setScene(scene)

        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self._pixmap_item)

        self.buffer = []
        self.view_state = ViewState.normal
        self.init_objects_and_params()

        if not active_color:
            self.active_color = config.ACTIVE_COLOR
        else:
            self.active_color = active_color

        if not fat_point_color:
            self.fat_point_color = config.FAT_POINT_COLOR
        else:
            self.fat_point_color = fat_point_color

        self.ruler_text_color = config.RULER_TEXT_COLOR

        # Кисти
        self.set_brushes()
        self.active_group = ActiveHandler([])

        # self.setMouseTracking(False)
        self.min_ellipse_size = 10
        self.fat_width_default_percent = 50
        self._zoom = 0

        self.create_actions()

        self.setRenderHint(QPainter.Antialiasing)

    def init_objects_and_params(self):
        """
        Создание объектов по умолчанию
        """

        self.set_view_state(ViewState.normal)
        self.drag_state = DragState.no
        self.draw_state = DrawState.polygon

        # Ruler Items:
        self.ruler_points = []
        self.ruler_draw_points = []
        self.ruler_line = None
        self.ruler_lrm = None
        self.ruler_text = None

        self.setMouseTracking(True)
        self.fat_point = None
        self.dragged_vertex = None
        self.ellipse_start_point = None
        self.box_start_point = None
        self.hand_start_point = None

        self.pressed_polygon = None

        self.negative_points = []
        self.positive_points = []
        self.right_clicked_points = []
        self.left_clicked_points = []
        self.groups = []
        self.segments = []

        self.last_added = []

    def set_view_state(self, state):

        self.viewport().setCursor(QCursor(QtCore.Qt.ArrowCursor))

        if state == ViewState.hand_move:
            self.view_state = state
            self.hand_start_point = None
            self.drag_state = DragState.no
            self.clear_ai_points()
            self.remove_fat_point_from_scene()
            self.active_group.clear()
            self.polygon_clicked.id_pressed.emit(-1)
            self.viewport().setCursor(QCursor(QtCore.Qt.OpenHandCursor))
            return

        if state == ViewState.hide_polygons:
            # скрыть полигоны
            self.view_state = state
            self.active_group.clear()
            self.polygon_clicked.id_pressed.emit(-1)
            self.toggle_all_polygons_alpha(is_hide=True)
            return

        if self.view_state == ViewState.hide_polygons:
            # ранее они были скрыты - показать
            self.view_state = state
            self.toggle_all_polygons_alpha(is_hide=False)
            return

        self.view_state = ViewState.normal

    def toggle_all_polygons_alpha(self, is_hide=True):
        for item in self.scene().items():
            # ищем, не попали ли уже в нарисованный полигон
            try:
                pol = item.polygon()
                if pol:
                    item.hide_color() if is_hide else item.get_color_back(self.line_width)
            except:
                pass

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
        self.ruler_text_brush = QtGui.QBrush(QColor(*self.ruler_text_color), QtCore.Qt.SolidPattern)

    def create_actions(self):
        self.delPolyAct = QAction("Удалить полигон" if self.lang == 'RU' else 'Delete polygon', self, enabled=True,
                                  triggered=self.on_del_polygon)
        self.changeClsNumAct = QAction("Изменить имя метки" if self.lang == 'RU' else 'Change label name', self,
                                       enabled=True, triggered=self.on_change_cls_num)
        self.mergeActivePolygons = QAction("Объединить выделенные полигоны" if self.lang == 'RU' else "Merge polygons",
                                           self, enabled=True,
                                           triggered=self.on_merge_polygons)

    def on_merge_polygons(self):

        shapely_union = self.active_group.merge_polygons_to_shapely_union()

        for shape in shapely_union:
            point_mass = list(shape.exterior.coords)
            # text_pos = hf.calc_label_pos(point_mass)

            cls_num = self.active_group[0].cls_num
            color = self.active_group[0].color
            alpha = self.active_group[0].alpha_percent
            alpha_edge = self.active_group[0].alpha_edge
            text = self.active_group[0].text

            self.add_polygon_to_scene(cls_num, point_mass, color, alpha=alpha, text=text, alpha_edge=alpha_edge)

        self.remove_items_from_active_group()
        self.active_group.clear()
        self.polygon_clicked.id_pressed.emit(-1)

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

        self.remove_items_from_active_group()
        self.active_group.clear()
        self.polygon_clicked.id_pressed.emit(-1)

        self.polygon_delete.id_delete.emit(delete_ids)

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
        # scene = QtWidgets.QGraphicsScene(self)
        # self.setScene(scene)
        self.scene().clear()

        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        self.scene().addItem(self._pixmap_item)
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

        if self.view_state == ViewState.rubber_band:
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

        return self.fat_width

    def set_pens(self):
        alpha_edge = self.settings.read_edges_alpha()
        alpha_edge = hf.convert_percent_to_alpha(alpha_edge)
        pen_color = list(self.active_color)
        pen_color[-1] = alpha_edge
        self.active_pen = QPen(QColor(*pen_color), self.line_width, QtCore.Qt.SolidLine)
        self.fat_point_pen = QPen(QColor(*self.fat_point_color), self.line_width, QtCore.Qt.SolidLine)

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

            d = hf.calc_distance_to_nearest_edge(pol, lp)

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
            if cropped_polygon.geom_type == 'MultiPolygon':
                polygons = list(cropped_polygon.geoms)
                cropped_polygon = polygons[0]
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
        self.labels_ids_handler = LabelsIds(self.ids_worker.get_labels_size())

        if self.on_set_callback:
            self.on_set_callback()

    def remove_last_changes(self):
        for item_id in self.last_added:
            self.remove_shape_by_id(item_id)
        self.last_added = []

    def add_polygons_group_to_scene(self, cls_num, point_of_points_mass, color=None, alpha=50, text=None,
                                    alpha_edge=None):
        self.last_added = []
        for points_mass in point_of_points_mass:
            id = self.labels_ids_handler.get_unique_label_id()
            self.last_added.append(id)
            self.add_polygon_to_scene(cls_num, points_mass, color=color, alpha=alpha, id=id, is_save_last=False,
                                      text=text, alpha_edge=alpha_edge)

    def add_polygon_to_scene(self, cls_num, point_mass, color=None, alpha=50, id=None,
                             is_save_last=True, text=None, alpha_edge=None):
        """
        Добавление полигона на сцену
        color - цвет. Если None - будет выбран цвет, соответствующий номеру класса из config.COLORS
        alpha - прозрачность в процентах
        """

        if len(point_mass) < 3:
            return

        if id == None:
            id = self.labels_ids_handler.get_unique_label_id()
            if is_save_last:
                self.last_added = []
                self.last_added.append(id)

        polygon_new = GrPolygonLabel(None, color=color, cls_num=cls_num, alpha_percent=alpha, id=id, text=text,
                                     text_pos=hf.calc_label_pos(point_mass), alpha_edge=alpha_edge)

        polygon_new.setBrush(QtGui.QBrush(QColor(*polygon_new.color), QtCore.Qt.SolidPattern))
        polygon_new.setPen(QPen(QColor(*polygon_new.edge_color), self.line_width, QtCore.Qt.SolidLine))

        poly = QPolygonF()

        for p in point_mass:
            poly.append(QtCore.QPointF(p[0], p[1]))

        polygon_new.setPolygon(poly)

        self.crop_by_pixmap_size(polygon_new)

        self.scene().addItem(polygon_new)
        if text:
            self.scene().addItem(polygon_new.get_label())

        return id

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
                if len(pol) > 3:
                    for p in pol:
                        if p != point_closed:
                            poly_new.append(p)

                    active_item.setPolygon(poly_new)
                else:
                    if self.lang == 'ENG':
                        message = "Can't delete point. Polygon must have more then 3 vertices"
                    else:
                        message = "Не могу удалить точку. У полигона должно быть более трех вершинт"
                    self.info_conn.info_message.emit(message)

    def copy_active_item_to_buffer(self):
        self.buffer = []
        for active_item in self.active_group:

            active_cls_num = active_item.cls_num
            active_alpha = active_item.alpha_percent
            active_color = active_item.color
            alpha_edge = active_item.alpha_edge
            text = active_item.text
            text_pos = active_item.text_pos

            copy_id = self.labels_ids_handler.get_unique_label_id()

            polygon_new = GrPolygonLabel(None, color=active_color, cls_num=active_cls_num,
                                         alpha_percent=active_alpha, id=copy_id, text=text, text_pos=text_pos,
                                         alpha_edge=alpha_edge)
            polygon_new.setPen(self.active_pen)
            polygon_new.setBrush(self.active_brush)
            poly = QPolygonF()
            for point in active_item.polygon():
                poly.append(point)

            polygon_new.setPolygon(poly)

            self.buffer.append(polygon_new)

    def paste_buffer(self):
        if len(self.buffer) > 0:
            for item in self.buffer:
                pol = item.polygon()
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

                new_item = GrPolygonLabel(None, color=item.color, cls_num=item.cls_num,
                                          alpha_percent=item.alpha_percent, alpha_edge=item.alpha_edge,
                                          id=self.labels_ids_handler.get_unique_label_id(), text=item.text)

                new_item.setPolygon(pol_new)

                self.scene().addItem(new_item)

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

        if self.view_state == ViewState.hide_polygons:
            return

        if self.draw_state == DrawState.rubber_band:

            if not self.box_start_point:
                self.drag_state = DragState.start  # "RubberBandStartDrawMode"
                self.box_start_point = lp

            return

        if self.view_state == ViewState.ruler:

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

        if self.view_state == ViewState.hand_move:
            if not self.hand_start_point:
                self.hand_start_point = event.pos()
                self.drag_state = DragState.start
                self.viewport().setCursor(QCursor(QtCore.Qt.ClosedHandCursor))
            return

        if self.view_state == ViewState.draw:

            if self.draw_state == DrawState.polygon:

                # Режим рисования, добавляем точки к текущему полигону
                if len(self.active_group) == 1:
                    active_item = self.active_group[0]
                    poly = active_item.polygon()
                    poly.append(lp)
                    active_item.setPolygon(poly)

            elif self.draw_state == DrawState.ellipse:

                # Режим рисования, добавляем точки к текущему полигону

                if not self.ellipse_start_point:
                    self.drag_state = DragState.start
                    self.ellipse_start_point = lp

            elif self.draw_state == DrawState.box:

                if not self.box_start_point:
                    self.drag_state = DragState.start
                    self.box_start_point = lp

            elif self.draw_state == DrawState.ai_points:
                if 'Left Click' in modifierName:
                    if self.is_point_in_pixmap_size(lp):
                        self.left_clicked_points.append(lp)
                        self.add_positive_ai_point_to_scene(lp)
                elif 'Right Click' in modifierName:
                    if self.is_point_in_pixmap_size(lp):
                        self.right_clicked_points.append(lp)
                        self.add_negative_ai_point_to_scene(lp)

            elif self.draw_state == DrawState.ai_mask:
                if not self.box_start_point:
                    self.drag_state = DragState.start
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
                        self.view_state = ViewState.vertex_move
                        self.drag_state = DragState.start
                        self.dragged_vertex = lp

                else:
                    # Нажали по грани
                    if 'Ctrl' in modifierName:
                        # Добавляем узел
                        self.add_point_to_active(lp)

                    else:
                        self.active_group.clear()
                        self.polygon_clicked.id_pressed.emit(-1)

            else:

                # нажали не рядом с активным полигоном

                if self.check_active_pressed(lp):

                    # нажали прямо по активному полигону, строго внутри
                    # Начать перемещение
                    self.drag_state = DragState.start
                    self.view_state = ViewState.drag
                    self.start_point = lp

                else:
                    # кликнули не по активной. Если по какой-то другой - изменить активную
                    pressed_polygon = self.get_pressed_polygon(lp)
                    if pressed_polygon:
                        self.active_group.reset_clicked_item(pressed_polygon, is_shift)
                        self.polygon_clicked.id_pressed.emit(pressed_polygon.id)
                    else:
                        self.polygon_clicked.id_pressed.emit(-1)
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
        circle_width = max(1.0, self.fat_width * math.exp(-self._zoom * 0.3) + 1)  # self._zoom / 5.0 + 1
        self.fat_point = FatPoint(None)
        self.fat_point.setRect(vertex.x() - circle_width / 2,
                               vertex.y() - circle_width / 2,
                               circle_width, circle_width)
        self.fat_point.setPen(self.fat_point_pen)
        self.fat_point.setBrush(self.fat_point_brush)

        self.scene().addItem(self.fat_point)

    def add_positive_ai_point_to_scene(self, point):
        scale = self._zoom / 2.5 + 1
        positive_point = AiPoint(None, line_width=self.line_width, is_positive=True)
        positive_point.setRect(point.x() - self.fat_width / (2 * scale),
                               point.y() - self.fat_width / (2 * scale),
                               self.fat_width / scale, self.fat_width / scale)

        self.positive_points.append(positive_point)
        self.scene().addItem(positive_point)

    def add_negative_ai_point_to_scene(self, point):
        scale = self._zoom / 2.5 + 1
        negative_point = AiPoint(None, line_width=self.line_width, is_positive=False)
        negative_point.setRect(point.x() - self.fat_width / (2 * scale),
                               point.y() - self.fat_width / (2 * scale),
                               self.fat_width / scale, self.fat_width / scale)

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

        if self.view_state == ViewState.hide_polygons:
            return

        if self.view_state == ViewState.hand_move:
            self.drag_state = DragState.in_process
            return

        if self.draw_state == DrawState.rubber_band:
            if self.drag_state in [DragState.start, DragState.in_process]:
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

                self.drag_state = DragState.in_process

            return

        if len(self.active_group) != 0:
            active_item = self.active_group[0]

            if self.drag_state in [DragState.start, DragState.in_process] and self.draw_state == DrawState.ellipse:

                width = abs(lp.x() - self.ellipse_start_point.x())
                height = abs(lp.y() - self.ellipse_start_point.y())
                if len(self.active_group) == 1:
                    active_item.setRect(self.ellipse_start_point.x(), self.ellipse_start_point.y(), width, height)

                self.drag_state = DragState.in_process

            elif self.drag_state in [DragState.start, DragState.in_process] and self.draw_state in [DrawState.box,
                                                                                                    DrawState.ai_mask]:

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

                self.drag_state = DragState.in_process

            elif self.drag_state == DragState.start and self.view_state == ViewState.vertex_move:

                if self.move_drag_vertex(lp):
                    self.dragged_vertex = lp

            elif self.drag_state == DragState.start and self.view_state == ViewState.drag:

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

    def get_rubber_band_polygon(self):
        if len(self.active_group) == 1:
            active_item = self.active_group[0]
            if active_item.cls_num == -1:
                points = []
                for p in active_item.polygon():
                    points.append([int(p.x()), int(p.y())])

                self.remove_items_from_active_group()
                self.active_group.clear()
                self.polygon_clicked.id_pressed.emit(-1)

                return points

    def get_active_item_polygon(self):
        if len(self.active_group) == 1:
            active_item = self.active_group[0]
            self.active_group.clear()
            self.polygon_clicked.id_pressed.emit(-1)
            points = []
            for p in active_item.polygon():
                points.append([int(p.x()), int(p.y())])
            return points

    def create_dragged_vertex_polygon(self, new_point):
        scale = self._zoom / 3.0 + 1

        if len(self.active_group) == 1:
            active_item = self.active_group[0]

            poly = QPolygonF()
            for point in active_item.polygon():
                if hf.distance(point, self.dragged_vertex) < self.fat_width / scale:
                    poly.append(new_point)
                else:
                    poly.append(point)

            return poly

        return None

    def move_drag_vertex(self, lp):
        """
        Передвигаем узел полигона
        True - если без самопересечений и передвинуть узел удалось
        """
        if self.dragged_vertex:
            dragged_poly = self.create_dragged_vertex_polygon(lp)
            if dragged_poly and len(self.active_group) == 1:
                active_item = self.active_group[0]

                if not hf.is_polygon_self_intersected(dragged_poly):
                    # 1. Задаем новый полигон
                    active_item.setPolygon(dragged_poly)
                    # 2. Обрезаем его по сцене, если надо
                    self.crop_by_pixmap_size(active_item)

                    # 3. Перемещаем имя метки, если она есть
                    if active_item.label:
                        point_mass = hf.convert_item_polygon_to_point_mass(active_item.polygon())
                        text_pos = hf.calc_label_pos(point_mass)
                        active_item.label.setPos(text_pos[0], text_pos[1])

                    # 4. Перемещаем подсветку узла
                    if self.fat_point:
                        circle_width = max(1.0,
                                           self.fat_width * math.exp(-self._zoom * 0.25) + 1)  # self._zoom / 5.0 + 1
                        self.fat_point.setRect(lp.x() - circle_width / 2,
                                               lp.y() - circle_width / 2,
                                               circle_width, circle_width)

                    return True

        return False

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        sp = self.mapToScene(event.pos())
        lp = self.pixmap_item.mapFromScene(sp)

        if self.view_state == ViewState.hide_polygons:
            return

        if self.view_state == ViewState.hand_move and self.drag_state == DragState.in_process:
            lp = event.pos()
            dx = int(lp.x() - self.hand_start_point.x())
            dy = int(lp.y() - self.hand_start_point.y())

            hpos = self.horizontalScrollBar().value()
            self.horizontalScrollBar().setValue(hpos - dx)

            vpos = self.verticalScrollBar().value()
            self.verticalScrollBar().setValue(vpos - dy)

            self.hand_start_point = None
            self.viewport().setCursor(QCursor(QtCore.Qt.OpenHandCursor))

            self.drag_state = DragState.no

            return

        if self.draw_state == DrawState.rubber_band:
            if self.drag_state == DragState.in_process:

                self.box_start_point = None

                if len(self.active_group) == 1:
                    active_item = self.active_group[0]

                    self.crop_by_pixmap_size(active_item)

                    points = []
                    for p in active_item.polygon():
                        points.append([int(p.x()), int(p.y())])

                    self.remove_items_from_active_group()
                    self.active_group.clear()
                    self.polygon_clicked.id_pressed.emit(-1)

                    self.list_of_polygons_conn.list_of_polygons.emit(points)

                    # return points

            self.reset_draw_state()

        if self.drag_state in [DragState.start, DragState.in_process] and self.view_state == ViewState.vertex_move:

            self.move_drag_vertex(lp)
            self.reset_draw_state()

        elif self.drag_state in [DragState.start, DragState.in_process] and self.view_state == ViewState.drag:

            for active_item in self.active_group:

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
            self.reset_draw_state()

        elif self.drag_state in [DragState.start, DragState.in_process] and self.draw_state == DrawState.ellipse:
            # self.setMouseTracking(True)
            self.ellipse_start_point = None

            if len(self.active_group) == 1:
                active_item = self.active_group[0]

                self.active_group.clear()
                self.polygon_clicked.id_pressed.emit(-1)
                self.remove_item(active_item, is_delete_id=False)

                polygon_new = active_item.convert_to_polygon(points_count=30)

                label = polygon_new.get_label()

                if label:
                    self.scene().addItem(label)

                # Self-intersection можно не проверять. Это эллипс
                self.scene().addItem(polygon_new)
                self.crop_by_pixmap_size(polygon_new)

                self.polygon_end_drawing.on_end_drawing.emit(True)

            self.reset_draw_state()

        elif self.drag_state in [DragState.start, DragState.in_process] and self.draw_state in [DrawState.box,
                                                                                                DrawState.ai_mask]:

            # self.setMouseTracking(True)
            self.box_start_point = None

            if len(self.active_group) == 1:
                active_item = self.active_group[0]

                if self.draw_state == DrawState.box:
                    self.active_group.clear()  # Только здесь. Иначе AiMask превратится в Бокс !!!
                    self.polygon_clicked.id_pressed.emit(-1)
                    # Box doesn't have a label
                    text = active_item.text
                    if text:
                        label = set_item_label(active_item, text)
                        self.scene().addItem(label)

                    self.crop_by_pixmap_size(active_item)
                    self.polygon_end_drawing.on_end_drawing.emit(True)
                elif self.draw_state == DrawState.ai_mask:
                    self.mask_end_drawing.on_mask_end_drawing.emit(True)

            self.reset_draw_state()

        # Нельзя делать self.reset_draw_state() здесь, поскольку при рисовании полигона точками
        # view_state = ViewState.draw
        # draw_state = DrawState.polygon

    def reset_draw_state(self):
        self.set_view_state(ViewState.normal)
        self.drag_state = DragState.no
        self.draw_state = DrawState.no

    def remove_items_from_active_group(self, is_delete_id=True):
        for item in self.active_group:
            self.remove_item(item, is_delete_id)

    def remove_item(self, item, is_delete_id=False):

        if is_delete_id:
            self.labels_ids_handler.remove_label_id(item.id)

        text_label = item.get_label()
        if text_label:
            self.scene().removeItem(text_label)
        self.scene().removeItem(item)

    def remove_all_polygons(self):
        self.view_state = ViewState.normal
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

    def start_drawing(self, draw_type=DrawState.polygon, cls_num=0, color=None, alpha=50, id=None, text=None,
                      alpha_edge=None):
        """
        Старт отрисовки фигуры, по умолчанию - полигона

        type - тип фигуры, по умолчанию - полигон
        cls_num - номер класса
        color - цвет. Если None - будет выбран цвет, соответствующий номеру класса из config.COLORS
        alpha - прозрачность в процентах
        """

        # При повторном вызове могут быть недорисованные полигоны. Нужно их
        #   - сохранить на сцену (они и так сохранятся, поскольку уже добавлены)
        #   - приписать текст
        #   - очистить active_group
        self.draw_state = DrawState.no
        self.drag_state = DragState.no
        self.viewport().setCursor(QCursor(QtCore.Qt.ArrowCursor))

        for item in self.active_group:

            if check_polygon_item(item):
                if item.is_self_intersected():
                    self.info_conn.info_message.emit(
                        "Polygon self-intersected" if self.lang == 'ENG' else "Полигон не должен содержать самопересечений. Удален")
                    self.remove_item(item, is_delete_id=True)
                else:
                    label = set_item_label(item, text)
                    if label:
                        self.scene().addItem(label)
                        self.polygon_end_drawing.on_end_drawing.emit(True)
            else:
                self.remove_item(item, is_delete_id=True)

        self.active_group.clear()

        self.view_state = ViewState.draw
        self.draw_state = draw_type

        if draw_type in [DrawState.polygon, DrawState.box, DrawState.ellipse]:

            if id == None:
                id = self.labels_ids_handler.get_unique_label_id()

            if draw_type == DrawState.polygon or draw_type == DrawState.box:
                active_item = GrPolygonLabel(self._pixmap_item, color=color, cls_num=cls_num, alpha_percent=alpha,
                                             id=id, text=text, alpha_edge=alpha_edge)
            elif draw_type == DrawState.ellipse:
                active_item = GrEllipsLabel(self._pixmap_item, color=color, cls_num=cls_num, alpha_percent=alpha,
                                            id=id, text=text, alpha_edge=alpha_edge)

            self.add_item_to_scene_as_active(active_item)


        elif draw_type == DrawState.ai_points:
            self.left_clicked_points = QPolygonF()
            self.right_clicked_points = QPolygonF()

            self.remove_items_from_active_group()

        elif draw_type == DrawState.ai_mask:

            active_item = GrPolygonLabel(None, color=color, cls_num=cls_num, alpha_percent=alpha, alpha_edge=alpha_edge,
                                         id=-1)

            self.add_item_to_scene_as_active(active_item)

        elif draw_type == DrawState.rubber_band:
            active_item = GrPolygonLabel(None, color=color, cls_num=-1, alpha_percent=alpha, alpha_edge=alpha_edge,
                                         id=-1)
            self.add_item_to_scene_as_active(active_item)

    def get_sam_input_points_and_labels(self):
        if self.draw_state == DrawState.ai_points:
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
        if self.draw_state == DrawState.ai_mask:

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
        self.view_state = ViewState.normal
        self.box_start_point = None
        self.drag_state = DragState.no

        self.remove_items_from_active_group()
        self.active_group.clear()
        self.polygon_clicked.id_pressed.emit(-1)

        self.remove_fat_point_from_scene()
        if self.draw_state == DrawState.ai_points:
            self.clear_ai_points()

    def end_drawing(self, text=None, cls_num=-1, color=None, alpha_percent=None):
        self.view_state = ViewState.normal

        self.remove_fat_point_from_scene()

        if self.draw_state == DrawState.ai_points:
            self.clear_ai_points()

        if len(self.active_group) != 0:
            active_item = self.active_group[0]
            if not active_item:
                return

            if not check_polygon_item(active_item):
                self.remove_item(active_item, is_delete_id=True)
                self.active_group.clear()
                self.polygon_clicked.id_pressed.emit(-1)
                return

            if active_item.is_self_intersected():
                self.info_conn.info_message.emit(
                    "Polygon self-intersected" if self.lang == 'ENG' else "Полигон не должен содержать самопересечений. Удален")
                self.remove_item(active_item, is_delete_id=True)
                self.active_group.clear()
                self.polygon_clicked.id_pressed.emit(-1)
                return

            self.crop_by_pixmap_size(active_item)

            if text and cls_num != -1:

                active_item.set_color(color=color, alpha_percent=alpha_percent)
                active_item.set_cls_num(cls_num)

                label = set_item_label(active_item, text, color)
                if label:
                    self.scene().addItem(label)

            self.active_group.clear()
            self.polygon_clicked.id_pressed.emit(-1)

    def contextMenuEvent(self, event: QtGui.QContextMenuEvent) -> None:

        if self.draw_state == DrawState.ai_points and self.view_state == ViewState.draw:
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

        if self.view_state != ViewState.rubber_band and is_active:
            self.view_state = ViewState.rubber_band
            self.draw_state = DrawState.rubber_band
            self.start_drawing(draw_type=DrawState.rubber_band)

        else:
            for item in self.active_group:
                if item.cls_num == -1:
                    self.remove_item(item)

    def on_ruler_mode_on(self, ruler_lrm=None):
        self.view_state = ViewState.ruler
        self.ruler_lrm = ruler_lrm

    def on_ruler_mode_off(self):
        self.view_state = ViewState.normal
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

        ruler_draw_point = RulerPoint(None, self.line_width)
        ruler_draw_point.setRect(pressed_point.x() - ruler_point_width / (2 * scale),
                                 pressed_point.y() - ruler_point_width / (2 * scale),
                                 ruler_point_width / scale, ruler_point_width / scale)

        self.scene().addItem(ruler_draw_point)
        self.ruler_draw_points.append(ruler_draw_point)

    def draw_ruler_line(self):
        if len(self.ruler_points) != 2:
            return
        p1 = self.ruler_points[0]
        p2 = self.ruler_points[1]
        self.ruler_line = RulerLine(None, self.line_width)
        self.ruler_line.setLine(p1.x(), p1.y(), p2.x(), p2.y())
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
