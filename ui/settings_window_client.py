from PyQt5.QtWidgets import QLabel, QFormLayout, QVBoxLayout

from ui.custom_widgets.styled_widgets import StyledEdit
from ui.settings_window import SettingsWindow
from utils.settings_handler import AppSettings


class SettingsWindowClient(SettingsWindow):
    def __init__(self, parent):
        super().__init__(parent)
        self.settings = AppSettings()

    def create_connection_layout(self):
        self.server_adress_edit = StyledEdit(self.settings.read_theme())
        self.server_adress_edit.setText(self.settings.read_server_name())
        # Настройки обнаружения

        self.connection_layout = QFormLayout()

        connection_label = QLabel("Адрес сервера:" if self.lang == 'RU' else "Server")

        self.connection_layout.addRow(connection_label, self.server_adress_edit)

    def stack_layouts(self):
        self.create_cnn_layout()
        self.create_connection_layout()

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.main_group)
        self.mainLayout.addWidget(self.labeling_group)
        self.mainLayout.addWidget(self.classifierGroupBox)
        self.mainLayout.addLayout(self.connection_layout)

        btnLayout = self.create_buttons()
        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)

    def on_ok_clicked(self):
        super(SettingsWindow, self).on_ok_clicked()

        self.settings.write_server_name(self.server_adress_edit.text())
