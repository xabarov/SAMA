from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton

from PyQt5.QtCore import Qt

from utils import config
from ui.edit_with_button import EditWithButton


class CreateProjectDialog(QWidget):
    def __init__(self, parent, width=480, height=200, on_ok_clicked=None,
                 theme='dark_blue.xml'):
        """
        Создание нового проекта
        """
        super().__init__(parent)
        self.setWindowTitle(f"Создание нового проекта" if config.LANGUAGE == 'RU' else 'Create new project')
        self.setWindowFlag(Qt.Tool)

        # Images layout:
        placeholder = "Путь к изображениям" if config.LANGUAGE == 'RU' else 'Path to images'

        self.images_edit_with_button = EditWithButton(None, theme=theme,
                                                      dialog_text=placeholder, start_folder='projects', is_dir=True,
                                                      placeholder=placeholder)

        # Project Name

        placeholder = 'Введите имя нового проекта...' if config.LANGUAGE == 'RU' else "Set new project name..."
        self.project_name_edit_with_button = EditWithButton(None, theme=theme,
                                                            file_type='json',
                                                            dialog_text=placeholder, start_folder='projects',
                                                            placeholder=placeholder, is_dir=False, is_existing_file_only=False)

        # Buttons layout:
        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Создать' if config.LANGUAGE == 'RU' else "Create", self)
        if on_ok_clicked:
            self.okBtn.clicked.connect(on_ok_clicked)

        self.cancelBtn = QPushButton('Отменить' if config.LANGUAGE == 'RU' else 'Cancel', self)

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
