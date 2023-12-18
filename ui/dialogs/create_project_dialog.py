from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton

from PyQt5.QtCore import Qt

from ui.custom_widgets.edit_with_button import EditWithButton
from utils.settings_handler import AppSettings

class CreateProjectDialog(QWidget):
    def __init__(self, parent, width=480, height=200, on_ok_clicked=None,
                 theme='dark_blue.xml'):
        """
        Создание нового проекта
        """
        super().__init__(parent)
        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        
        self.setWindowTitle("Создание нового проекта" if self.lang == 'RU' else 'Create new project')
        self.setWindowFlag(Qt.Tool)

        # Images layout:
        placeholder = "Путь к изображениям" if self.lang == 'RU' else 'Path to images'

        self.images_edit_with_button = EditWithButton(None, theme=theme,
                                                      dialog_text=placeholder, is_dir=True,
                                                      placeholder=placeholder)

        # Project Name

        placeholder = 'Введите имя нового проекта...' if self.lang == 'RU' else "Set new project name..."
        self.project_name_edit_with_button = EditWithButton(None, theme=theme,
                                                            file_type='json',
                                                            dialog_text=placeholder,
                                                            placeholder=placeholder, is_dir=False, is_existing_file_only=False)

        # Buttons layout:
        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Создать' if self.lang == 'RU' else "Create", self)
        if on_ok_clicked:
            self.okBtn.clicked.connect(on_ok_clicked)

        self.cancelBtn = QPushButton('Отменить' if self.lang == 'RU' else 'Cancel', self)

        self.cancelBtn.clicked.connect(self.on_cancel_clicked)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        # Stack layers

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.images_edit_with_button)
        self.mainLayout.addWidget(self.project_name_edit_with_button)

        self.mainLayout.addLayout(btnLayout)

        self.setLayout(self.mainLayout)

        self.resize(int(width), int(height))

    def on_cancel_clicked(self):
        self.hide()

    def get_project_name(self):
        return self.project_name_edit_with_button.getEditText()

    def get_image_folder(self):
        return self.images_edit_with_button.getEditText()
