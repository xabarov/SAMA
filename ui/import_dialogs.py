from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QPushButton, QLineEdit, QCheckBox, \
    QProgressBar, \
    QComboBox
from PyQt5.QtCore import Qt

from utils import config
from ui.edit_with_button import EditWithButton

import numpy as np
import yaml
import json
import os


class ImportFromYOLODialog(QWidget):
    def __init__(self, parent, width=480, height=200, on_ok_clicked=None,
                 theme='dark_blue.xml'):
        """
        Импорт разметки из YOLO
        """
        super().__init__(parent)
        self.setWindowTitle(f"Импорт разметки в формате YOLO")
        self.setWindowFlag(Qt.Tool)

        # Yaml file layout:
        placeholder = "Путь к YAML файлу" if config.LANGUAGE == 'RU' else 'Path to YAML file'
        dialog_text = 'Открытие файла в формате YAML' if config.LANGUAGE == 'RU' else 'Open file in YAML format'

        self.yaml_edit_with_button = EditWithButton(None, theme=theme,
                                                    on_button_clicked_callback=self.on_yaml_button_clicked,
                                                    file_type='yaml',
                                                    dialog_text=dialog_text, start_folder='projects',
                                                    placeholder=placeholder)

        # Dataset Combo layout:

        self.dataset_layout = QHBoxLayout()
        self.dataset_combo = QComboBox()

        self.dataset_label = QLabel("Датасет" if config.LANGUAGE == 'RU' else 'Dataset')

        self.dataset_layout.addWidget(self.dataset_label)
        self.dataset_layout.addWidget(self.dataset_combo)
        self.dataset_label.setEnabled(False)
        self.dataset_combo.setEnabled(False)

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

        placeholder = 'Путь для сохранения изображений...' if config.LANGUAGE == 'RU' else "Path to save imageds..."
        dialog_text = 'Выберите папку для сохранения изображений' if config.LANGUAGE == 'RU' else "Set images folder"
        self.save_images_edit_with_button = EditWithButton(None, theme=theme,
                                                           dialog_text=dialog_text, start_folder='projects',
                                                           placeholder=placeholder, is_dir=True)

        self.save_images_edit_with_button.setVisible(False)

        # Buttons layout:
        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Импортировать' if config.LANGUAGE == 'RU' else "Import", self)
        self.on_ok_clicked = on_ok_clicked
        if on_ok_clicked:
            self.okBtn.clicked.connect(self.on_ok)

        self.cancelBtn = QPushButton('Отменить' if config.LANGUAGE == 'RU' else 'Cancel', self)

        self.cancelBtn.clicked.connect(self.on_cancel_clicked)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        # Stack layers

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.yaml_edit_with_button)
        self.mainLayout.addLayout(save_images_layout)
        self.mainLayout.addWidget(self.save_images_edit_with_button)

        self.mainLayout.addLayout(btnLayout)

        self.mainLayout.addWidget(self.progress_bar)

        self.setLayout(self.mainLayout)

        self.data = {}

        self.resize(int(width), int(height))

    def on_checkbox_clicked(self):
        self.is_copy_images = self.save_images_checkbox.isChecked()
        self.save_images_edit_with_button.setVisible(self.is_copy_images)

    def on_ok(self):
        self.yaml_edit_with_button.setEnabled(False)
        self.dataset_combo.setEnabled(False)
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
                    if yaml_data[t]:
                        combo_vars.append(t)

                self.dataset_combo.addItems(np.array(combo_vars))

                self.data = yaml_data
                self.data["selected_dataset"] = self.dataset_combo.currentText()
                self.data['yaml_path'] = yaml_name

                self.dataset_label.setEnabled(True)
                self.dataset_combo.setEnabled(True)

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
        self.setWindowTitle(
            "Импорт разметки в формате COCO" if config.LANGUAGE == 'RU' else 'Import labels in COCO format')
        self.setWindowFlag(Qt.Tool)

        self.labels = []

        # COCO file layout:
        placeholder = "Путь к json файлу" if config.LANGUAGE == 'RU' else "Path to JSON file"

        dialog_text = 'Открытие файла в формате COCO' if config.LANGUAGE == 'RU' else 'Open file in COCO format'
        self.coco_edit_with_button = EditWithButton(None, theme=theme,
                                                    on_button_clicked_callback=self.on_coco_button_clicked,
                                                    file_type='json',
                                                    dialog_text=dialog_text, start_folder='projects',
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
        save_images_layout.addRow(QLabel('Копировать изображения' if config.LANGUAGE == 'RU' else "Copy images"),
                                  self.save_images_checkbox)

        # save images edit + button
        placeholder = 'Путь для сохранения изображений...' if config.LANGUAGE == 'RU' else "Path to save imageds..."
        dialog_text = 'Выберите папку для сохранения изображений' if config.LANGUAGE == 'RU' else "Set images folder"
        self.save_images_edit_with_button = EditWithButton(None, theme=theme,
                                                           dialog_text=dialog_text, start_folder='projects',
                                                           placeholder=placeholder, is_dir=True)

        self.save_images_edit_with_button.setVisible(False)

        # label names?
        self.is_label_names = False

        self.label_names_checkbox = QCheckBox()
        self.label_names_checkbox.setChecked(False)
        self.label_names_checkbox.clicked.connect(self.on_label_names_checkbox_clicked)

        label_names_layout = QHBoxLayout()
        label_names_layout.addWidget(
            QLabel('Задать файл с именами классов' if config.LANGUAGE == 'RU' else "Set labels names from txt file"))
        label_names_layout.addWidget(self.label_names_checkbox)

        # label_names edit + button
        dialog_text = 'Открытие файла с именами классов' if config.LANGUAGE == 'RU' else "Open file with label names"
        placeholder = 'Путь к txt-файлу с именами классов' if config.LANGUAGE == 'RU' else "Path to txt file with labels names"
        self.label_names_edit_with_button = EditWithButton(None, theme=theme,
                                                           file_type='txt',
                                                           dialog_text=dialog_text, start_folder='projects',
                                                           placeholder=placeholder)

        self.label_names_edit_with_button.setVisible(False)

        # Buttons layout:
        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Импортировать' if config.LANGUAGE == 'RU' else "Import", self)
        self.on_ok_clicked = on_ok_clicked
        self.okBtn.clicked.connect(self.on_ok)

        self.cancelBtn = QPushButton('Отменить' if config.LANGUAGE == 'RU' else 'Cancel', self)

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

                data = json.load(f)
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
