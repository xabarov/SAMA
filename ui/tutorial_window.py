from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QPushButton, QLineEdit, QTextEdit
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

from utils import config

def make_label(text, is_bold=False):
    label = QLabel(text)
    if is_bold:
        label.setFont(QFont('Times', 14, QFont.Bold))
    else:
        label.setFont(QFont('Times', 14, QFont.Normal))
    return label

class Tutorial(QWidget):
    def __init__(self, parent, width=600, height=500):
        super().__init__(parent)

        self.setWindowTitle(f"Горячие клавиши" if config.LANGUAGE == 'RU' else "Shortcuts")
        self.setWindowFlag(Qt.Tool)

        label_txt = ""
        if config.LANGUAGE == 'RU':
            header = "Общие"
        else:
            header = "General"

        label_txt += f"{header:^100s}\n\n"

        if config.LANGUAGE == 'RU':
            hot_keys = {
                "S": "нарисовать новую метку",
                "D": "удалить текущую метку",
                "Space": "закончить рисование текущей метки",
                "Ctrl + C": "копировать текущую метку",
                "Ctrl + V": "вставить текущую метку"
            }
        else:
            hot_keys = {
                "S": "нарисовать новую метку",
                "D": "удалить текущую метку",
                "Space": "закончить рисование текущей метки",
                "Ctrl + C": "копировать текущую метку",
                "Ctrl + V": "вставить текущую метку"
            }
        if config.LANGUAGE == 'RU':
            hot_keys2 = {
                "Левая кнопка мыши": "установить точку внутри сегмента",
                "Правая кнопка мыши": "установить точку снаружи сегмента (фон)",
                "Space": "запустить нейросеть SAM для отрисовки метки"
            }
        else:
            hot_keys2 = {
                "Левая кнопка мыши": "установить точку внутри сегмента",
                "Правая кнопка мыши": "установить точку снаружи сегмента (фон)",
                "Space": "запустить нейросеть SAM для отрисовки метки"
            }

        for key in hot_keys:
            label_txt += f"    {key:^30s} {hot_keys[key]:^100s}    \n"

        header = "В режиме сегментации с помощью нейросети" if config.LANGUAGE == 'RU' else "In SAM mode"
        label_txt += f"\n\n{header:^100s}\n\n"
        header = "1. Сегментация по точкам" if config.LANGUAGE == 'RU' else "SAM by points"
        label_txt += f"\n\n{header:^100s}\n\n"

        for key in hot_keys2:
            label_txt += f"    {key:^30s} {hot_keys2[key]:^100s}    \n"

        header = "2.Сегментация внутри бокса" if config.LANGUAGE == 'RU' else "SAM by box"
        label_txt += f"\n\n{header:^100s}\n\n"

        if config.LANGUAGE == 'RU':
            steps = ["1) Нарисуйте прямоугольную маску с областью для сегментации",
                     "2) Дождитесь появления метки"]
        else:
            steps = ["1) Нарисуйте прямоугольную маску с областью для сегментации",
                     "2) Дождитесь появления метки"]

        for s in steps:
            label_txt += f"    {s:<100s}\n"

        self.label = QTextEdit()
        self.label.setPlainText(label_txt)

        btnLayout = QVBoxLayout()

        self.okBtn = QPushButton('Ок', self)
        self.okBtn.clicked.connect(self.close)

        btnLayout.addWidget(self.okBtn)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.label)

        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)

        self.resize(int(width), int(height))
