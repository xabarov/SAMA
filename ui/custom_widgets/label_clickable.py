from PyQt5.QtWidgets import QLabel

from ui.signals_and_slots import MouseClicked


class LabelClickable(QLabel):
    mouse_clicked_conn = MouseClicked()

    def mousePressEvent(self, ev) -> None:
        self.mouse_clicked_conn.on_mouse_clicked.emit(True)
