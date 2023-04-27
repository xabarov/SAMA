from PyQt5.QtWidgets import QLabel, QCheckBox, QWidget, QGroupBox, QFormLayout, QComboBox, QSpinBox, QVBoxLayout, \
    QHBoxLayout, QPushButton, QDoubleSpinBox
from PyQt5.QtCore import Qt
import numpy as np


class AskDelWindow(QWidget):
    def __init__(self, parent, list_of_labels, cls_name):
        super().__init__(parent)
        self.setWindowTitle(f"Удаление масок для класса {cls_name}")
        self.setWindowFlag(Qt.Tool)

        self.list_of_labels = list_of_labels
        self.cls_name = cls_name

        self.what_to_do_ask = QLabel("Что следует сделать?")

        self.cls_combo = QComboBox()
        self.cls_labels = np.array([lbl for lbl in self.list_of_labels if lbl != cls_name])
        self.cls_combo.addItems(self.cls_labels)

        btnLayout = QVBoxLayout()

        self.del_all_btn = QPushButton(f'Удалить все маски с классом {cls_name}', self)
        self.change_btn = QPushButton(f'Заменить маски с классом {cls_name} на другой', self)
        self.cancelBtn = QPushButton('Отменить', self)
        self.okBtn = QPushButton('Применить', self)

        self.change_btn.clicked.connect(self.set_second_ask)

        btnLayout.addWidget(self.del_all_btn)
        btnLayout.addWidget(self.change_btn)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.what_to_do_ask)
        self.mainLayout.addWidget(self.cls_combo)
        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)

        self.cls_combo.setVisible(False)
        self.okBtn.setVisible(False)

        self.resize(400, 200)

    def set_second_ask(self):
        self.what_to_do_ask.setText("Выберите имя класса для замены:")
        self.cls_combo.setVisible(True)
        self.del_all_btn.setVisible(False)
        self.change_btn.setVisible(False)
        self.okBtn.setVisible(True)
