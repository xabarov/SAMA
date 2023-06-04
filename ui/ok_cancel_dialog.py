from PyQt5.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout, QPushButton, QLabel
from PyQt5.QtCore import Qt
from utils import config

class OkCancelDialog(QWidget):
    def __init__(self, parent, title, text, on_ok=None, on_cancel=None):
        """
        """
        super().__init__(parent)
        self.setWindowFlag(Qt.Tool)

        self.setWindowTitle(title)

        layout = QVBoxLayout()

        self.label = QLabel(text)

        buttons_layout = QHBoxLayout()
        self.ok_button = QPushButton('Ок' if config.LANGUAGE == 'RU' else "OK", self)
        self.cancel_button = QPushButton('Отменить' if config.LANGUAGE == 'RU' else "Cancel", self)

        if not on_ok:
            self.ok_button.clicked.connect(self.on_ok_clicked)
        else:
            self.ok_button.clicked.connect(on_ok)

        if not on_cancel:
            self.cancel_button.clicked.connect(self.cancel_button_clicked)
        else:
            self.cancel_button.clicked.connect(on_cancel)

        buttons_layout.addWidget(self.ok_button)
        buttons_layout.addWidget(self.cancel_button)
        layout.addWidget(self.label)
        layout.addLayout(buttons_layout)
        self.setLayout(layout)

        self.show()

    def on_ok_clicked(self):
        self.close()

    def cancel_button_clicked(self):
        self.close()
