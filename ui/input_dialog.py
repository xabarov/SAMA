from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QPushButton, QLineEdit, QFormLayout, QComboBox, QProgressBar
from PyQt5.QtCore import Qt
from utils import config

import numpy as np

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


class PromptInputDialog(QWidget):
    def __init__(self, parent, width=400, height=200, class_names=None, on_ok_clicked=None):
        super().__init__(parent)
        self.setWindowTitle("Выделение объектов по ключевым словам" if
                            config.LANGUAGE == 'RU' else "Select objects by text prompt")
        self.setWindowFlag(Qt.Tool)

        prompt_layout = QFormLayout()

        self.prompt_label = QLabel("Что будем искать:" if config.LANGUAGE == 'RU' else "Prompt:")
        self.prompt_edit = QLineEdit()
        prompt_layout.addRow(self.prompt_label, self.prompt_edit)

        class_layout = QFormLayout()

        self.class_label = QLabel("Каким классом разметить" if config.LANGUAGE == 'RU' else "Select label name:")
        self.cls_combo = QComboBox()
        if not class_names:
            class_names = np.array(['no name'])
        self.cls_combo.addItems(np.array(class_names))
        self.cls_combo.setMinimumWidth(150)

        class_layout.addRow(self.class_label, self.cls_combo)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)

        btnLayout = QVBoxLayout()

        self.okBtn = QPushButton('Начать поиск' if config.LANGUAGE == 'RU' else "Run", self)

        btnLayout.addWidget(self.okBtn)
        self.on_ok_clicked = on_ok_clicked
        if on_ok_clicked:
            self.okBtn.clicked.connect(self.on_ok)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(prompt_layout)
        self.mainLayout.addLayout(class_layout)
        self.mainLayout.addLayout(btnLayout)
        self.mainLayout.addWidget(self.progress_bar)

        self.setLayout(self.mainLayout)

        self.resize(int(width), int(height))

    def on_ok(self):
        self.prompt_edit.setVisible(False)
        self.prompt_label.setVisible(False)
        self.cls_combo.setVisible(False)
        self.class_label.setVisible(False)
        self.okBtn.setEnabled(False)

        self.on_ok_clicked()
    def getPrompt(self):
        return self.prompt_edit.text()

    def getClsName(self):
        return self.cls_combo.currentText()

    def getClsNumber(self):
        return self.cls_combo.currentIndex()

    def set_progress(self, progress_value):
        if progress_value != 100:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(progress_value)
        else:
            self.progress_bar.setVisible(False)