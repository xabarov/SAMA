import os
import sys

import numpy as np
import ujson
import yaml
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QFormLayout, QCheckBox, QMessageBox
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QProgressBar, QApplication

from ui.custom_widgets.edit_with_button import EditWithButton
from ui.custom_widgets.styled_widgets import StyledComboBox
from utils.settings_handler import AppSettings


def show_message(text, title):
    msgbox = QMessageBox()
    msgbox.setIcon(QMessageBox.Information)
    msgbox.setText(text)
    msgbox.setWindowTitle(title)
    msgbox.exec()


class ImportLRMSDialog(QWidget):
    def __init__(self, parent, width=480, height=200, on_ok_clicked=None,
                 theme='dark_blue.xml'):
        super(ImportLRMSDialog, self).__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        self.lrms_data = {}  # данные о ЛРМ снимков

        if self.lang == 'RU':
            title = "Импорт данных о ЛРМ снимков из JSON-файла"
        else:
            title = "Linear ground resolution data import from JSON"
        self.setWindowTitle(title)
        self.setWindowFlag(Qt.Tool)

        # Yaml file layout:
        placeholder = "Путь к JSON файлу" if self.lang == 'RU' else 'Path to JSON file'
        dialog_text = 'Открытие файла в формате JSON' if self.lang == 'RU' else 'Open file in JSON format'

        self.json_edit_with_button = EditWithButton(None, theme=theme,
                                                    on_button_clicked_callback=self.on_json_button_clicked,
                                                    file_type='json',
                                                    dialog_text=dialog_text,
                                                    placeholder=placeholder)

        self.okBtn = QPushButton('Загрузить' if self.lang == 'RU' else "Load", self)
        if on_ok_clicked:
            self.okBtn.clicked.connect(on_ok_clicked)

        self.cancelBtn = QPushButton('Отменить' if self.lang == 'RU' else 'Cancel', self)

        self.cancelBtn.clicked.connect(self.hide)

        # Buttons layout:
        btnLayout = QHBoxLayout()

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        # Stack layers

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.json_edit_with_button)
        self.mainLayout.addLayout(btnLayout)

        self.setLayout(self.mainLayout)

        self.resize(int(width), int(height))

    def on_json_button_clicked(self):
        json_name = self.json_edit_with_button.getEditText()
        if json_name:
            with open(json_name, 'r') as f:
                self.lrms_data = ujson.load(f)


class ImportFromYOLODialog(QWidget):
    def __init__(self, parent, width=480, height=200, on_ok_clicked=None,
                 theme='dark_blue.xml', convert_to_mask=False):
        """
        Импорт разметки из YOLO
        """
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.setWindowTitle("Импорт разметки в формате YOLO" if self.lang == 'RU' else "Import labeling in YOLO format")
        self.setWindowFlag(Qt.Tool)

        # Yaml file layout:
        placeholder = "Путь к YAML файлу" if self.lang == 'RU' else 'Path to YAML file'
        dialog_text = 'Открытие файла в формате YAML' if self.lang == 'RU' else 'Open file in YAML format'

        self.form_layout = QFormLayout()

        self.yaml_edit_with_button = EditWithButton(None, theme=theme,
                                                    on_button_clicked_callback=self.on_yaml_button_clicked,
                                                    file_type='yaml',
                                                    dialog_text=dialog_text,
                                                    placeholder=placeholder)
        self.yaml_label = QLabel(placeholder)
        self.form_layout.addRow(self.yaml_label, self.yaml_edit_with_button)

        placeholder = "Имя нового проекта" if self.lang == 'RU' else 'New project name'

        self.import_project_edit_with_button = EditWithButton(None, theme=theme,
                                                              is_dir=False, file_type='json',
                                                              dialog_text=placeholder,
                                                              placeholder=placeholder, is_existing_file_only=False)
        self.import_project_label = QLabel(placeholder)
        self.form_layout.addRow(self.import_project_label, self.import_project_edit_with_button)
        # Dataset Combo layout:

        self.dataset_combo = StyledComboBox(None, theme=self.settings.read_theme())

        self.dataset_label = QLabel("Датасет" if self.lang == 'RU' else 'Dataset')

        self.form_layout.addRow(self.dataset_label, self.dataset_combo)

        self.dataset_label.setVisible(False)
        self.dataset_combo.setVisible(False)
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_change)

        # save images?
        self.is_copy_images = False

        self.save_images_checkbox = QCheckBox()
        self.save_images_checkbox.setChecked(False)
        self.save_images_checkbox.clicked.connect(self.on_checkbox_clicked)

        self.save_images_checkbox_label = QLabel('Копировать изображения' if self.lang == 'RU' else 'Copy images')
        self.form_layout.addRow(self.save_images_checkbox_label, self.save_images_checkbox)

        # save images edit + button

        placeholder = 'Путь для сохранения изображений...' if self.lang == 'RU' else "Path to save images..."
        dialog_text = 'Выберите папку для сохранения изображений' if self.lang == 'RU' else "Set images folder"
        self.save_images_edit_with_button = EditWithButton(None, theme=theme,
                                                           dialog_text=dialog_text,
                                                           placeholder=placeholder, is_dir=True)

        self.save_images_edit_with_button.setVisible(False)
        self.save_images_label = QLabel(placeholder)
        self.save_images_label.setVisible(False)
        self.form_layout.addRow(self.save_images_label, self.save_images_edit_with_button)

        self.convert_to_mask = convert_to_mask
        if convert_to_mask:
            self.convert_to_mask_checkbox = QCheckBox()
            self.convert_to_mask_checkbox.setChecked(False)
            self.convert_to_mask_label = QLabel(
                'Использовать SAM для конвертации боксов в сегменты' if self.lang == 'RU' else 'Use SAM to convert boxes to segments')
            self.form_layout.addRow(self.convert_to_mask_label, self.convert_to_mask_checkbox)

        # Buttons layout:
        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Импортировать' if self.lang == 'RU' else "Import", self)
        self.on_ok_clicked = on_ok_clicked
        if on_ok_clicked:
            self.okBtn.clicked.connect(self.on_ok)

        self.cancelBtn = QPushButton('Отменить' if self.lang == 'RU' else 'Cancel', self)

        self.cancelBtn.clicked.connect(self.on_cancel_clicked)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel(
            'Подождите, идет импорт датасета...' if self.lang == 'RU' else 'Please wait, the dataset is being imported...')
        self.progress_label.setVisible(False)
        self.form_layout.addRow(self.progress_label, self.progress_bar)

        # Stack layers

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.form_layout)

        self.mainLayout.addLayout(btnLayout)

        self.setLayout(self.mainLayout)

        self.data = {}

        self.resize(int(width), int(height))

    def on_dataset_change(self, dataset_type):
        self.data["selected_dataset"] = dataset_type

    def on_checkbox_clicked(self):
        self.is_copy_images = self.save_images_checkbox.isChecked()
        self.save_images_edit_with_button.setVisible(self.is_copy_images)
        self.save_images_label.setVisible(self.is_copy_images)

    def on_ok(self):

        if self.save_images_checkbox.isChecked():
            if self.save_images_edit_with_button.getEditText() == "":
                if self.lang == 'RU':
                    text = f"Задайте директорию для сохранения изображений"
                else:
                    text = f"Set the directory to save images"

                title = "Заполните все поля" if self.lang == 'RU' else "Fill in all the fields"
                show_message(text, title)
                return

        if self.yaml_edit_with_button.getEditText() == "":

            if self.lang == 'RU':
                text = f"Задайте имя YAML файла"
            else:
                text = f"Set the YAML file name"

            title = "Заполните все поля" if self.lang == 'RU' else "Fill in all the fields"
            show_message(text, title)
            return

        if self.import_project_edit_with_button.getEditText() == "":

            if self.lang == 'RU':
                text = f"Задайте имя нового проекта"
            else:
                text = f"Give the new project a name"

            title = "Заполните все поля" if self.lang == 'RU' else "Fill in all the fields"
            show_message(text, title)
            return

        for w in [self.import_project_label, self.import_project_edit_with_button, self.dataset_label,
                  self.dataset_combo, self.yaml_label, self.yaml_edit_with_button, self.save_images_edit_with_button,
                  self.save_images_label, self.cancelBtn, self.okBtn, self.save_images_checkbox_label,
                  self.save_images_checkbox]:
            w.hide()

        if self.convert_to_mask:
            self.convert_to_mask_label.hide()
            self.convert_to_mask_checkbox.hide()

        self.data['is_copy_images'] = self.is_copy_images
        self.data['save_images_dir'] = self.save_images_edit_with_button.getEditText()
        self.data['import_project_path'] = self.import_project_edit_with_button.getEditText()

        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.adjustSize()

        self.on_ok_clicked()

    def on_cancel_clicked(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.hide()

    def on_yaml_button_clicked(self):
        yaml_name = self.yaml_edit_with_button.getEditText()
        if yaml_name:
            with open(yaml_name, 'r') as f:
                yaml_data = yaml.safe_load(f)
                combo_vars = []
                for t in ["train", "val", 'test']:
                    if t in yaml_data:
                        combo_vars.append(t)

                self.dataset_combo.addItems(np.array(combo_vars))

                self.data = yaml_data
                self.data["selected_dataset"] = self.dataset_combo.currentText()
                self.data['yaml_path'] = yaml_name

                self.dataset_label.setVisible(True)
                self.dataset_combo.setVisible(True)

    def getData(self):
        return self.data

    def set_progress(self, progress_value):
        if progress_value != 100:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(progress_value)
        else:
            self.progress_bar.setVisible(False)


class ImportFromCOCODialog(QWidget):
    def __init__(self, parent, width=480, height=200, on_ok_clicked=None,
                 theme='dark_blue.xml'):
        """
        Импорт разметки из COCO
        """
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.setWindowTitle(
            "Импорт разметки в формате COCO" if self.lang == 'RU' else 'Import labels in COCO format')
        self.setWindowFlag(Qt.Tool)

        self.labels = []

        # COCO file layout:
        self.form_layout = QFormLayout()
        placeholder = "Путь к файлу с разметкой COCO" if self.lang == 'RU' else "Path to COCO file"

        dialog_text = 'Открытие файла в формате COCO' if self.lang == 'RU' else 'Open file in COCO format'
        self.coco_edit_with_button = EditWithButton(None, theme=theme,
                                                    on_button_clicked_callback=self.on_coco_button_clicked,
                                                    file_type='json',
                                                    dialog_text=dialog_text,
                                                    placeholder=placeholder)

        self.coco_path_label = QLabel(placeholder)
        self.form_layout.addRow(self.coco_path_label, self.coco_edit_with_button)

        placeholder = "Имя нового проекта" if self.lang == 'RU' else 'New project name'

        self.import_project_edit_with_button = EditWithButton(None, theme=theme,
                                                              is_dir=False, file_type='json',
                                                              dialog_text=placeholder,
                                                              placeholder=placeholder, is_existing_file_only=False)
        self.import_project_label = QLabel(placeholder)
        self.form_layout.addRow(self.import_project_label, self.import_project_edit_with_button)

        # save images?
        self.is_copy_images = False

        self.save_images_checkbox = QCheckBox()
        self.save_images_checkbox.setChecked(False)
        self.save_images_checkbox.clicked.connect(self.on_checkbox_clicked)

        self.save_images_checkbox_label = QLabel('Копировать изображения' if self.lang == 'RU' else 'Copy images')
        self.form_layout.addRow(self.save_images_checkbox_label, self.save_images_checkbox)

        # save images edit + button

        placeholder = 'Путь для сохранения изображений...' if self.lang == 'RU' else "Path to save images..."
        dialog_text = 'Выберите папку для сохранения изображений' if self.lang == 'RU' else "Set images folder"
        self.save_images_edit_with_button = EditWithButton(None, theme=theme,
                                                           dialog_text=dialog_text,
                                                           placeholder=placeholder, is_dir=True)

        self.save_images_edit_with_button.setVisible(False)
        self.save_images_label = QLabel(placeholder)
        self.save_images_label.setVisible(False)
        self.form_layout.addRow(self.save_images_label, self.save_images_edit_with_button)

        # label names?
        self.is_label_names = False

        self.label_names_checkbox = QCheckBox()
        self.label_names_checkbox.setChecked(False)
        self.label_names_checkbox.clicked.connect(self.on_label_names_checkbox_clicked)

        self.labels_names_label = QLabel(
            'Задать файл с именами классов' if self.lang == 'RU' else "Set labels names from txt file")
        self.form_layout.addRow(self.labels_names_label, self.label_names_checkbox)

        # label_names edit + button
        dialog_text = 'Открытие файла с именами классов' if self.lang == 'RU' else "Open file with label names"
        placeholder = 'Путь к txt-файлу с именами классов' if self.lang == 'RU' else "Path to txt file with labels names"
        self.label_names_edit_with_button = EditWithButton(None, theme=theme,
                                                           file_type='txt',
                                                           dialog_text=dialog_text,
                                                           placeholder=placeholder)

        self.label_names_edit_with_button.setVisible(False)

        # Buttons layout:
        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Импортировать' if self.lang == 'RU' else "Import", self)
        self.on_ok_clicked = on_ok_clicked
        self.okBtn.clicked.connect(self.on_ok)

        self.cancelBtn = QPushButton('Отменить' if self.lang == 'RU' else 'Cancel', self)

        self.cancelBtn.clicked.connect(self.on_cancel_clicked)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        # Stack layers
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)
        self.progress_label = QLabel(
            'Подождите, идет импорт датасета...' if self.lang == 'RU' else 'Please wait, the dataset is being imported...')
        self.progress_label.setVisible(False)
        self.form_layout.addRow(self.progress_label, self.progress_bar)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.form_layout)
        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)

        self.data = {}

        self.resize(int(width), int(height))

    def on_ok(self):

        if self.save_images_checkbox.isChecked():
            if self.save_images_edit_with_button.getEditText() == "":
                if self.lang == 'RU':
                    text = f"Задайте директорию для сохранения изображений"
                else:
                    text = f"Set the directory to save images"

                title = "Заполните все поля" if self.lang == 'RU' else "Fill in all the fields"
                show_message(text, title)
                return

        if self.coco_edit_with_button.getEditText() == "":

            if self.lang == 'RU':
                text = f"Задайте имя COCO файла"
            else:
                text = f"Set the COCO file name"

            title = "Заполните все поля" if self.lang == 'RU' else "Fill in all the fields"
            show_message(text, title)
            return

        if self.import_project_edit_with_button.getEditText() == "":

            if self.lang == 'RU':
                text = f"Задайте имя нового проекта"
            else:
                text = f"Give the new project a name"

            title = "Заполните все поля" if self.lang == 'RU' else "Fill in all the fields"
            show_message(text, title)
            return

        for w in [self.coco_edit_with_button, self.coco_path_label, self.save_images_edit_with_button,
                  self.save_images_label, self.save_images_checkbox, self.import_project_edit_with_button,
                  self.import_project_label, self.save_images_checkbox_label,
                  self.cancelBtn, self.okBtn, self.labels_names_label, self.label_names_checkbox,
                  self.label_names_edit_with_button]:
            w.hide()

        self.data['import_project_path'] = self.import_project_edit_with_button.getEditText()

        self.progress_bar.setVisible(True)
        self.progress_label.setVisible(True)
        self.adjustSize()

        self.on_ok_clicked()

    def get_copy_images_path(self):
        return self.save_images_edit_with_button.getEditText()

    def on_checkbox_clicked(self):
        self.is_copy_images = self.save_images_checkbox.isChecked()
        self.save_images_edit_with_button.setVisible(self.is_copy_images)

    def on_label_names_checkbox_clicked(self):
        self.is_label_names = self.label_names_checkbox.isChecked()
        self.label_names_edit_with_button.setVisible(self.is_label_names)

    def on_cancel_clicked(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.hide()

    def on_coco_button_clicked(self):
        """
        Задаем путь к файлу с разметкой в формате COCO
        """
        coco_name = self.coco_edit_with_button.getEditText()

        if coco_name:
            with open(coco_name, 'r') as f:

                data = ujson.load(f)
                if self.check_coco(data):
                    self.data['coco_json'] = data
                    self.data['coco_name'] = coco_name
                else:
                    self.data = None

    def get_coco_name(self):
        return self.coco_name

    def get_label_names(self):
        text = self.label_names_edit_with_button.getEditText()
        if text:
            if os.path.exists(text):
                with open(text, 'r') as f:
                    label_names = []
                    for line in f:
                        label_names.append(line.strip())

                    return label_names
        return

    def check_coco(self, data):
        coco_keys = ['info', 'licenses', 'images', 'annotations', 'categories']
        for key in coco_keys:
            if key not in data:
                return False

        return True

    def showEvent(self, event):
        for lbl in self.labels:
            lbl.setMaximumWidth(self.labels[0].width())

    def getData(self):
        return self.data

    def set_progress(self, progress_value):
        if progress_value != 100:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(progress_value)
        else:
            self.progress_bar.setVisible(False)


if __name__ == '__main__':
    def on_ok():
        print(dialog.getData())


    from qt_material import apply_stylesheet

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    labels = ['F-16', 'F-35', 'C-130', 'C-17']
    dialog = ImportFromCOCODialog(None, on_ok_clicked=on_ok)
    dialog.show()
    sys.exit(app.exec_())
