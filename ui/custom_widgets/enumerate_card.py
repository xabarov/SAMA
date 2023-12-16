import os

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QHBoxLayout, QPushButton, QApplication, QLabel, QFrame, QSizePolicy

from utils.settings_handler import AppSettings


class EnumerateCard(QFrame):
    def __init__(self, widget, text="", num=1, is_number_flat=False, on_number_click=None):
        """
        Поле Edit с кнопкой
        """
        super().__init__(None)
        self.settings = AppSettings()

        self.layout = QHBoxLayout()

        self.label_num = QPushButton()
        pixmap_path = os.path.join(os.path.dirname(__file__), "..", "icons", "numbers", f"{num}.png")
        icon = QIcon(pixmap_path)
        self.label_num.setIcon(icon)
        self.label_num.setFlat(is_number_flat)
        if on_number_click:
            self.label_num.clicked.connect(on_number_click)

        self.label_text = QLabel(text)

        self.layout.addWidget(self.label_num, stretch=1)
        self.layout.addWidget(self.label_text, stretch=1)
        self.layout.addWidget(widget, stretch=5)

        self.label_num.setMaximumHeight(widget.height())

        self.setFrameShape(QFrame.HLine)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.setLineWidth(3)

        self.setLayout(self.layout)


if __name__ == '__main__':
    from qt_material import apply_stylesheet
    import sys
    from ui.dialogs.export_steps.train_test_splitter import TrainTestSplitter
    from ui.dialogs.export_steps.set_export_path_widget import SetPathWidget

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    dialog = EnumerateCard(widget=SetPathWidget(None, theme='dark_blue.xml'), num=1)
    dialog.show()
    sys.exit(app.exec_())
