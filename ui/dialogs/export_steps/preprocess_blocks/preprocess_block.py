import os

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout

from utils.settings_handler import AppSettings


class PreprocessBlock(QWidget):

    def __init__(self, name, options=None, parent=None, title=None, min_width=780, minimum_expand_height=300,
                 icon_name=""):
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.minimum_expand_height = minimum_expand_height

        if title:
            self.setWindowTitle(title)
        else:
            self.setWindowTitle(name)

        if icon_name:
            icon_folder = os.path.join(os.path.dirname(__file__), "preprocess_icons")

            self.setWindowIcon(QIcon(icon_folder + f"/{icon_name}"))

        self.name = name
        if options:
            self.options = options
        else:
            self.options = {}

        self.layout = QVBoxLayout()

        self.setLayout(self.layout)
        self.setMinimumWidth(min_width)

    def get_options(self):
        return self.options

    def create_buttons(self, on_ok=None, on_cancel=None):
        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Принять' if self.lang == 'RU' else 'Apply', self)
        if on_ok:
            self.okBtn.clicked.connect(on_ok)
        else:
            self.okBtn.clicked.connect(self.hide)

        self.cancelBtn = QPushButton('Отменить' if self.lang == 'RU' else 'Cancel', self)

        if on_cancel:
            self.cancelBtn.clicked.connect(on_cancel)
        else:
            self.cancelBtn.clicked.connect(self.hide)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        self.layout.addLayout(btnLayout)
