import os

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QApplication, QFrame, QSizePolicy, QVBoxLayout

from utils.settings_handler import AppSettings


class EnumerateCard(QFrame):
    def __init__(self, body, text="", num=1, is_number_flat=False, on_number_click=None, max_head_height=100):
        """
        Поле Edit с кнопкой
        """
        super().__init__(None)
        self.settings = AppSettings()

        self.layout = QVBoxLayout()
        self.head_layout = QHBoxLayout()

        self.head = QPushButton()
        pixmap_path = os.path.join(os.path.dirname(__file__), "..", "icons", "numbers", f"{num}.png")
        icon = QIcon(pixmap_path)
        self.head.setIcon(icon)
        self.head.setFlat(is_number_flat)
        self.head.setText(text)
        self.head.setMaximumHeight(max_head_height)
        if on_number_click:
            self.head.clicked.connect(on_number_click)

        self.head_layout.addWidget(self.head)
        self.layout.addLayout(self.head_layout)

        self.body = body
        self.layout.addWidget(self.body, stretch=5)

        self.body.hide()

        self.setFrameShape(QFrame.HLine)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setLineWidth(3)

        self.setLayout(self.layout)


if __name__ == '__main__':
    from qt_material import apply_stylesheet
    import sys
    from ui.dialogs.export_steps.set_export_path_widget import SetPathWidget

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    dialog = EnumerateCard(body=SetPathWidget(None, theme='dark_blue.xml'), num=1)
    dialog.show()
    sys.exit(app.exec_())
