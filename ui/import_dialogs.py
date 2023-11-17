from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLineEdit, QCheckBox, \
    QProgressBar, \
    QComboBox
from PyQt5.QtCore import Qt

from utils import config
from ui.edit_with_button import EditWithButton

import numpy as np
import yaml
import ujson
import os
from utils.settings_handler import AppSettings


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
        self.setWindowTitle(f"Импорт разметки в формате YOLO")
        self.setWindowFlag(Qt.Tool)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        # Yaml file layout:
        placeholder = "Путь к YAML файлу" if self.lang == 'RU' else 'Path to YAML file'
        dialog_text = 'Открытие файла в формате YAML' if self.lang == 'RU' else 'Open file in YAML format'


        self.yaml_edit_with_button = EditWithButton(None, theme=theme,
                                                    on_button_clicked_callback=self.on_yaml_button_clicked,
                                                    file_type='yaml',
                                                    dialog_text=dialog_text,
                                                    placeholder=placeholder)

        # Dataset Combo layout:

        self.dataset_layout = QHBoxLayout()
        self.dataset_combo = QComboBox()

        self.dataset_label = QLabel("Датасет" if self.lang == 'RU' else 'Dataset')

        self.dataset_layout.addWidget(self.dataset_label)
        self.dataset_layout.addWidget(self.dataset_combo)
        self.dataset_label.setVisible(False)
        self.dataset_combo.setVisible(False)
        self.dataset_combo.currentTextChanged.connect(self.on_dataset_change)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)

        # save images?
        self.is_copy_images = False

        self.save_images_checkbox = QCheckBox()
        self.save_images_checkbox.setChecked(False)
        self.save_images_checkbox.clicked.connect(self.on_checkbox_clicked)

        save_images_layout = QHBoxLayout()
        save_images_layout.addWidget(QLabel('Копировать изображения'))
        save_images_layout.addWidget(self.save_images_checkbox)

        # save images edit + button

        placeholder = 'Путь для сохранения изображений...' if self.lang == 'RU' else "Path to save imageds..."
        dialog_text = 'Выберите папку для сохранения изображений' if self.lang == 'RU' else "Set images folder"
        self.save_images_edit_with_button = EditWithButton(None, theme=theme,
                                                           dialog_text=dialog_text,
                                                           placeholder=placeholder, is_dir=True)

        self.save_images_edit_with_button.setVisible(False)

        if convert_to_mask:
            self.convert_to_mask_checkbox = QCheckBox()
            self.convert_to_mask_checkbox.setChecked(False)
            convert_to_mask_layout = QHBoxLayout()
            convert_to_mask_layout.addWidget(QLabel('Использовать SAM для конвертации боксов в сегменты'))
            convert_to_mask_layout.addWidget(self.convert_to_mask_checkbox)

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

        # Stack layers

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.yaml_edit_with_button)
        self.mainLayout.addLayout(self.dataset_layout)
        self.mainLayout.addLayout(save_images_layout)

        self.mainLayout.addWidget(self.save_images_edit_with_button)

        if convert_to_mask:
            self.mainLayout.addLayout(convert_to_mask_layout)

        self.mainLayout.addLayout(btnLayout)

        self.mainLayout.addWidget(self.progress_bar)

        self.setLayout(self.mainLayout)

        self.data = {}

        self.resize(int(width), int(height))

    def on_dataset_change(self, dataset_type):
        self.data["selected_dataset"] = dataset_type

    def on_checkbox_clicked(self):
        self.is_copy_images = self.save_images_checkbox.isChecked()
        self.save_images_edit_with_button.setVisible(self.is_copy_images)

    def on_ok(self):
        self.yaml_edit_with_button.setEnabled(False)
        self.dataset_combo.setVisible(False)
        self.cancelBtn.setEnabled(False)
        self.okBtn.setEnabled(False)
        self.data['is_copy_images'] = self.is_copy_images
        self.data['save_images_dir'] = self.save_images_edit_with_button.getEditText()

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
        placeholder = "Путь к файлу с разметкой COCO" if self.lang == 'RU' else "Path to COCO file"

        dialog_text = 'Открытие файла в формате COCO' if self.lang == 'RU' else 'Open file in COCO format'
        self.coco_edit_with_button = EditWithButton(None, theme=theme,
                                                    on_button_clicked_callback=self.on_coco_button_clicked,
                                                    file_type='json',
                                                    dialog_text=dialog_text,
                                                    placeholder=placeholder)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)

        # save images?
        self.is_copy_images = False

        self.save_images_checkbox = QCheckBox()
        self.save_images_checkbox.setChecked(False)
        self.save_images_checkbox.clicked.connect(self.on_checkbox_clicked)

        save_images_layout = QFormLayout()
        save_images_layout.addRow(QLabel('Копировать изображения' if self.lang == 'RU' else "Copy images"),
                                  self.save_images_checkbox)

        # save images edit + button
        placeholder = 'Путь для сохранения изображений...' if self.lang == 'RU' else "Path to save imageds..."
        dialog_text = 'Выберите папку для сохранения изображений' if self.lang == 'RU' else "Set images folder"
        self.save_images_edit_with_button = EditWithButton(None, theme=theme,
                                                           dialog_text=dialog_text,
                                                           placeholder=placeholder, is_dir=True)

        self.save_images_edit_with_button.setVisible(False)

        # label names?
        self.is_label_names = False

        self.label_names_checkbox = QCheckBox()
        self.label_names_checkbox.setChecked(False)
        self.label_names_checkbox.clicked.connect(self.on_label_names_checkbox_clicked)

        label_names_layout = QHBoxLayout()
        label_names_layout.addWidget(
            QLabel('Задать файл с именами классов' if self.lang == 'RU' else "Set labels names from txt file"))
        label_names_layout.addWidget(self.label_names_checkbox)

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

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.coco_edit_with_button)

        self.mainLayout.addLayout(save_images_layout)
        self.mainLayout.addWidget(self.save_images_edit_with_button)

        self.mainLayout.addLayout(label_names_layout)
        self.mainLayout.addWidget(self.label_names_edit_with_button)

        self.mainLayout.addLayout(btnLayout)
        self.mainLayout.addWidget(self.progress_bar)
        self.setLayout(self.mainLayout)

        self.data = {}

        self.resize(int(width), int(height))

    def on_ok(self):
        self.coco_edit_with_button.setEnabled(False)
        self.cancelBtn.setEnabled(False)
        self.okBtn.setEnabled(False)
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
                    self.data = data
                    self.coco_name = coco_name
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
