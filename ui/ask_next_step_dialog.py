from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QPushButton
from PyQt5.QtCore import Qt
from utils import config

from utils.settings_handler import AppSettings


class AskNextStepDialog(QWidget):

    def __init__(self, parent, before_step_name, next_step_name, message, width_percent=0.2, height_percent=0.2):
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        if self.lang == 'RU':
            title = f'Завершен этап {before_step_name}'
        else:
            title = f'{before_step_name} finished'

        self.setWindowTitle(f"{title}")
        self.setWindowFlag(Qt.Tool)

        if self.lang == 'RU':
            label_text = f"Следующий шаг {next_step_name}"
        else:
            label_text = f"Next step is {next_step_name}"

        self.label = QLabel(f"{message}\n{label_text}")

        btnLayout = QVBoxLayout()

        if self.lang == 'RU':
            next_button_text = "Продолжить"
        else:
            next_button_text = "Next"

        if self.lang == 'RU':
            stop_button_text = "Стоп"
        else:
            stop_button_text = "Stop"

        self.nextBtn = QPushButton(next_button_text, self)
        self.stopBtn = QPushButton(stop_button_text, self)
        self.stopBtn.clicked.connect(self.hide)

        btnLayout.addWidget(self.nextBtn)
        btnLayout.addWidget(self.stopBtn)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.label)
        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)
        size, pos = self.settings.read_size_pos_settings()
        self.setMinimumWidth(int(size.width() * width_percent))
        self.setMinimumHeight(int(size.height() * height_percent))
