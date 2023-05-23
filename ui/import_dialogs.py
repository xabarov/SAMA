from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLineEdit, QCheckBox, QProgressBar, \
    QComboBox, QFileDialog
from PyQt5.QtCore import Qt
from utils import config
from PyQt5.QtGui import QIcon

import numpy as np
import yaml
import json
import os
import shutil


class ImportFromYOLODialog(QWidget):
    def __init__(self, parent, width=480, height=200, on_ok_clicked=None,
                 theme='dark_blue.xml'):
        """
        Импорт разметки из YOLO
        """
        super().__init__(parent)
        self.setWindowTitle(f"Импорт разметки в формате YOLO")
        self.setWindowFlag(Qt.Tool)

        self.labels = []

        layout = QVBoxLayout()

        # Yaml file layout:
        yaml_label = QLabel("Путь к YAML файлу")
        self.labels.append(yaml_label)

        import_layout = QHBoxLayout()

        self.yaml_edit = QLineEdit()
        self.yaml_button = QPushButton(self)

        theme_type = theme.split('.')[0]

        self.icon_folder = "ui/icons/" + theme_type

        self.yaml_button.setIcon(QIcon(self.icon_folder + "/folder.png"))

        self.yaml_button.clicked.connect(self.on_yaml_button_clicked)

        import_layout.addWidget(yaml_label)
        import_layout.addWidget(self.yaml_edit)
        import_layout.addWidget(self.yaml_button)
        layout.addLayout(import_layout)

        # Dataset Combo layout:

        self.dataset_layout = QHBoxLayout()
        self.dataset_combo = QComboBox()

        self.dataset_label = QLabel("Датасет" if config.LANGUAGE == 'RU' else 'Dataset')
        self.labels.append(self.dataset_label)

        self.dataset_layout.addWidget(self.dataset_label)
        self.dataset_layout.addWidget(self.dataset_combo)
        self.dataset_label.setEnabled(False)
        self.dataset_combo.setEnabled(False)

        layout.addLayout(self.dataset_layout)

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

        self.save_images_button = QPushButton(self)
        theme_type = theme.split('.')[0]
        self.icon_folder = "ui/icons/" + theme_type

        self.save_images_widgets = []

        self.save_images_edit_button_layout = QHBoxLayout()
        self.save_images_label = QLabel('Сохранить в...' if config.LANGUAGE == 'RU' else "Save to...")
        self.save_images_widgets.append(self.save_images_label)
        self.labels.append(self.save_images_label)

        self.save_images_edit = QLineEdit()
        self.save_images_widgets.append(self.save_images_edit)

        self.save_images_button.setIcon(QIcon(self.icon_folder + "/folder.png"))
        self.save_images_button.clicked.connect(self.on_save_images_button_clicked)
        self.save_images_widgets.append(self.save_images_button)

        save_images_edit_button_layout = QHBoxLayout()

        for widget in self.save_images_widgets:
            widget.setVisible(False)
            save_images_edit_button_layout.addWidget(widget)

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
        self.mainLayout.addLayout(layout)
        self.mainLayout.addLayout(save_images_layout)
        self.mainLayout.addLayout(save_images_edit_button_layout)

        self.mainLayout.addLayout(btnLayout)

        self.mainLayout.addWidget(self.progress_bar)

        self.setLayout(self.mainLayout)

        self.data = {}

        self.resize(int(width), int(height))

    def on_checkbox_clicked(self):
        self.is_copy_images = self.save_images_checkbox.isChecked()
        for widget in self.save_images_widgets:
            widget.setVisible(self.is_copy_images)

    def on_ok(self):
        self.yaml_edit.setEnabled(False)
        self.dataset_combo.setEnabled(False)
        self.cancelBtn.setEnabled(False)
        self.okBtn.setEnabled(False)
        self.yaml_button.setEnabled(False)
        self.data['is_copy_images'] = self.is_copy_images
        self.data['save_images_dir'] = self.save_images_edit.text()

        self.on_ok_clicked()

    def on_cancel_clicked(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.hide()

    def on_save_images_button_clicked(self):
        images_dir = QFileDialog.getExistingDirectory(self,
                                                      'Выберите папку для сохранения изображений' if config.LANGUAGE == 'RU' else "Set images folder",
                                                      'images')
        if images_dir:
            self.save_images_edit.setText(images_dir)

    def on_yaml_button_clicked(self):
        yaml_name, _ = QFileDialog.getOpenFileName(self,
                                                   'Открытие файла yaml датасета YOLO' if config.LANGUAGE == 'RU' else "Open YOLO yaml file",
                                                   'projects',
                                                   'YAML File (*.yaml)')
        if yaml_name:
            self.yaml_edit.setText(yaml_name)
            with open(yaml_name, 'r') as f:
                yaml_data = yaml.safe_load(f)
                combo_vars = []
                for t in ["train", "val", 'test']:
                    if yaml_data[t]:
                        combo_vars.append(t)

                self.dataset_combo.addItems(np.array(combo_vars))

                self.data = yaml_data
                self.data["selected_dataset"] = self.dataset_combo.currentText()
                self.data['yaml_path'] = self.yaml_edit.text()

                self.dataset_label.setEnabled(True)
                self.dataset_combo.setEnabled(True)

    def showEvent(self, event):
        self.yaml_button.setMaximumHeight(self.yaml_edit.height())
        self.dataset_combo.setMinimumWidth(self.yaml_edit.width() + self.yaml_button.width() + 8)
        self.save_images_button.setMaximumHeight(self.yaml_edit.height())
        self.save_images_button.setMaximumWidth(self.yaml_button.width())

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


class ImportFromCOCODialog(QWidget):
    def __init__(self, parent, width=480, height=200, on_ok_clicked=None,
                 theme='dark_blue.xml'):
        """
        Импорт разметки из COCO
        """
        super().__init__(parent)
        self.setWindowTitle(f"Импорт разметки в формате COCO")
        self.setWindowFlag(Qt.Tool)

        self.labels = []

        layout = QVBoxLayout()

        # COCO file layout:
        coco_label = QLabel("Путь к json файлу")
        self.labels.append(coco_label)

        import_layout = QHBoxLayout()

        self.coco_edit = QLineEdit()
        self.coco_button = QPushButton(self)

        theme_type = theme.split('.')[0]

        self.icon_folder = "ui/icons/" + theme_type

        self.coco_button.setIcon(QIcon(self.icon_folder + "/folder.png"))

        self.coco_button.clicked.connect(self.on_coco_button_clicked)

        import_layout.addWidget(coco_label)
        import_layout.addWidget(self.coco_edit)
        import_layout.addWidget(self.coco_button)
        layout.addLayout(import_layout)

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

        self.save_images_button = QPushButton(self)
        theme_type = theme.split('.')[0]
        self.icon_folder = "ui/icons/" + theme_type

        self.save_images_widgets = []

        self.save_images_edit_button_layout = QHBoxLayout()
        self.save_images_label = QLabel('Сохранить в...' if config.LANGUAGE == 'RU' else "Save to...")
        self.save_images_widgets.append(self.save_images_label)
        self.labels.append(self.save_images_label)

        self.save_images_edit = QLineEdit()
        self.save_images_widgets.append(self.save_images_edit)

        self.save_images_button.setIcon(QIcon(self.icon_folder + "/folder.png"))
        self.save_images_button.clicked.connect(self.on_save_images_button_clicked)
        self.save_images_widgets.append(self.save_images_button)

        save_images_edit_button_layout = QHBoxLayout()

        for widget in self.save_images_widgets:
            widget.setVisible(False)
            save_images_edit_button_layout.addWidget(widget)

        # label names?
        self.is_label_names = False

        self.label_names_checkbox = QCheckBox()
        self.label_names_checkbox.setChecked(False)
        self.label_names_checkbox.clicked.connect(self.on_label_names_checkbox_clicked)

        label_names_layout = QHBoxLayout()
        label_names_layout.addWidget(QLabel('Задать файл с именами классов'))
        label_names_layout.addWidget(self.label_names_checkbox)

        # label_names edit + button

        self.label_names_button = QPushButton(self)
        theme_type = theme.split('.')[0]
        self.icon_folder = "ui/icons/" + theme_type

        self.label_names_widgets = []

        self.label_names_edit_button_layout = QHBoxLayout()
        self.label_names_label = QLabel('Открыть файл' if config.LANGUAGE == 'RU' else "Open file")
        self.label_names_widgets.append(self.label_names_label)
        self.labels.append(self.label_names_label)

        self.label_names_edit = QLineEdit()
        self.label_names_widgets.append(self.label_names_edit)

        self.label_names_button.setIcon(QIcon(self.icon_folder + "/folder.png"))
        self.label_names_button.clicked.connect(self.on_label_names_button_clicked)
        self.label_names_widgets.append(self.label_names_button)

        label_names_edit_button_layout = QHBoxLayout()

        for widget in self.label_names_widgets:
            widget.setVisible(False)
            label_names_edit_button_layout.addWidget(widget)

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
        self.mainLayout.addLayout(layout)

        self.mainLayout.addLayout(save_images_layout)
        self.mainLayout.addLayout(save_images_edit_button_layout)

        self.mainLayout.addLayout(label_names_layout)
        self.mainLayout.addLayout(label_names_edit_button_layout)

        self.mainLayout.addLayout(btnLayout)
        self.mainLayout.addWidget(self.progress_bar)
        self.setLayout(self.mainLayout)

        self.data = {}

        self.resize(int(width), int(height))

    def on_ok(self):
        self.coco_edit.setEnabled(False)
        self.cancelBtn.setEnabled(False)
        self.okBtn.setEnabled(False)
        self.coco_button.setEnabled(False)

        if self.is_copy_images:
            # change paths
            images = []
            print(self.coco_name)

            for i, im in enumerate(self.data["images"]):
                im_copy = im
                # make sense copy from real folder, not from flickr_url
                if os.path.exists(im['flickr_url']):

                    shutil.copy(im['flickr_url'], os.path.join(self.save_images_edit.text(), im["file_name"]))

                elif os.path.exists(os.path.join(os.path.dirname(self.coco_name), im["file_name"])):
                    shutil.copy(os.path.join(os.path.dirname(self.coco_name), im["file_name"]),
                                os.path.join(self.save_images_edit.text(), im["file_name"]))
                else:
                    continue

                im_copy['flickr_url'] = os.path.join(self.save_images_edit.text(), im["file_name"])
                im_copy['coco_url'] = os.path.join(self.save_images_edit.text(), im["file_name"])
                images.append(im_copy)

                self.set_progress(int(i * 100.0 / len(self.data["images"])))

            self.data["images"] = images

        self.on_ok_clicked()

    def on_checkbox_clicked(self):
        self.is_copy_images = self.save_images_checkbox.isChecked()
        for widget in self.save_images_widgets:
            widget.setVisible(self.is_copy_images)

    def on_label_names_checkbox_clicked(self):
        self.is_label_names = self.label_names_checkbox.isChecked()
        for widget in self.label_names_widgets:
            widget.setVisible(self.is_label_names)

    def on_cancel_clicked(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.hide()

    def on_coco_button_clicked(self):
        """
        Задаем путь к файлу с разметкой в формате COCO
        """
        coco_name, _ = QFileDialog.getOpenFileName(self,
                                                   'Открытие файла json датасета COCO' if config.LANGUAGE == 'RU' else "Open COCO json file",
                                                   'projects',
                                                   'JSON File (*.json)')
        if coco_name:
            self.coco_edit.setText(coco_name)

            with open(coco_name, 'r') as f:

                data = json.load(f)
                if self.check_coco(data):
                    self.data = data
                    self.coco_name = coco_name
                else:
                    self.data = None

    def on_save_images_button_clicked(self):
        """
        Задаем путь куда будем сохранять изображения
        """
        images_dir = QFileDialog.getExistingDirectory(self,
                                                      'Выберите папку для сохранения изображений' if config.LANGUAGE == 'RU' else "Set images folder",
                                                      'images')
        if images_dir:
            self.save_images_edit.setText(images_dir)

    def on_label_names_button_clicked(self):
        """
        Задаем файл с именами
        """
        label_names_file, _ = QFileDialog.getOpenFileName(self,
                                                          'Открытие файла с именами классов' if config.LANGUAGE == 'RU' else "Open file with label names",
                                                          'projects',
                                                          'txt File (*.txt)')
        if label_names_file:
            self.label_names_edit.setText(label_names_file)

    def get_label_names(self):

        if self.label_names_edit.text():
            if os.path.exists(self.label_names_edit.text()):
                with open(self.label_names_edit.text(), 'r') as f:
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
        self.coco_button.setMaximumHeight(self.coco_edit.height())
        self.save_images_button.setMaximumHeight(self.coco_edit.height())
        self.save_images_button.setMaximumWidth(self.coco_button.width())

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
