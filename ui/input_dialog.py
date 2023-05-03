from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QPushButton, QLineEdit
from PyQt5.QtCore import Qt
from utils import config

class CustomInputDialog(QWidget):
    def __init__(self, parent, title_name, question_name, width=400, height=200):
        super().__init__(parent)
        self.setWindowTitle(f"{title_name}")
        self.setWindowFlag(Qt.Tool)

        self.label = QLabel(f"{question_name}")

        self.edit = QLineEdit()

        btnLayout = QVBoxLayout()

        self.okBtn = QPushButton('Ввести' if config.LANGUAGE == 'RU' else "OK", self)

        btnLayout.addWidget(self.okBtn)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.edit)
        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)

        self.resize(int(width), int(height))


    def getText(self):
        return self.edit.text()
