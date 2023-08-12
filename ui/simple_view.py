from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtWidgets import QApplication
from ui.signals_and_slots import ViewMouseCoordsConnection


class SimpleView(QtWidgets.QGraphicsView):
    """
    Сцена для отображения текущей картинки
    """

    def __init__(self, parent=None):

        super().__init__(parent)
        scene = QtWidgets.QGraphicsScene(self)

        # SIGNALS
        self.mouse_move_conn = ViewMouseCoordsConnection()

        self.setScene(scene)

        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self._pixmap_item)
        self._zoom = 0
        self.setMouseTracking(True)

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

    def clearScene(self):
        """
        Очистить сцену
        """
        scene = QtWidgets.QGraphicsScene(self)
        self.setScene(scene)
        self._pixmap_item = QtWidgets.QGraphicsPixmapItem()
        scene.addItem(self._pixmap_item)

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

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):

        sp = self.mapToScene(event.pos())
        lp = self.pixmap_item.mapFromScene(sp)
        self.mouse_move_conn.on_mouse_move.emit(lp.x(), lp.y())
