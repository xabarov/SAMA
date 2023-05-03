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
