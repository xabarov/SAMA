from PySide2.QtCore import Signal
import PySide2.QtCore as Ps2Core


class PolygonPressedConnection(Ps2Core.QObject):
    id_pressed = Signal(int)


class PolygonDeleteConnection(Ps2Core.QObject):
    id_delete = Signal(int)


class PolygonEndDrawing(Ps2Core.QObject):
    on_end_drawing = Signal(bool)


class MaskEndDrawing(Ps2Core.QObject):
    on_mask_end_drawing = Signal(bool)


class ThemeChangeConnection(Ps2Core.QObject):
    on_theme_change = Signal(str)


class ImagesPanelCountConnection(Ps2Core.QObject):
    on_image_count_change = Signal(int)


class LabelsPanelCountConnection(Ps2Core.QObject):
    on_labels_count_change = Signal(int)


class LoadPercentConnection(Ps2Core.QObject):
    percent = Signal(int)


class ErrorConnection(Ps2Core.QObject):
    error_message = Signal(str)


class InfoConnection(Ps2Core.QObject):
    info_message = Signal(str)


class PolygonChangeClsNumConnection(Ps2Core.QObject):
    pol_cls_num_and_id = Signal(int, int)


class LoadIdProgress(Ps2Core.QObject):
    percent = Signal(int)


class RubberBandModeConnection(Ps2Core.QObject):
    on_rubber_mode_change = Signal(bool)


class ViewMouseCoordsConnection(Ps2Core.QObject):
    on_mouse_move = Signal(float, float)


class ProjectSaveLoadConn(Ps2Core.QObject):
    on_finished = Signal(bool)
