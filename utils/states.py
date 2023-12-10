from enum import Enum


class WindowState(Enum):
    normal = 1
    drawing = 2
    rubber_band = 3
    hided_polygons = 4


class ViewState(Enum):
    """
    Общее состояние поля View
    """
    normal = 1  # ожидаем клик по полигону
    draw = 2  # в процессе отрисовки
    rubber_band = 3  # выделение области изображения
    ruler = 4  # рулетка
    drag = 5  # перемещение полигона
    vertex_move = 6


class DrawState(Enum):
    """
    Вариант состояния ViewState.draw поля View
    """
    no = 0
    box = 1
    ellipse = 2
    polygon = 3
    ai_points = 4
    ai_mask = 5
    grounding_dino = 6
    rubber_band = 7



class DragState(Enum):
    """
    Вариант состояния ViewState.drag поля View
    """
    no = 0
    start = 1
    in_process = 2
    end = 3
