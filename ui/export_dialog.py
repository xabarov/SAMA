import math

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


class ExportDialog(QWidget):
    def __init__(self, parent, width=800, height=200, on_ok_clicked=None, label_names=None,
                 theme='dark_blue.xml', export_format='yolo'):
        """
        Экспорт разметки
        """
        super().__init__(parent)
        self.setWindowTitle(f"Экспорт разметки")
        self.setWindowFlag(Qt.Tool)

        # Export dir layout:
        if export_format == 'yolo':
            placeholder = "Директория экспорта YOLO" if config.LANGUAGE == 'RU' else 'Path to export YOLO'

            self.export_edit_with_button = EditWithButton(None, theme=theme,
                                                               on_button_clicked_callback=self.on_export_button_clicked,
                                                               is_dir=True,
                                                               dialog_text=placeholder, start_folder='projects',
                                                               placeholder=placeholder)
        elif export_format == 'coco':
            placeholder = "Имя экспортируемого файла COCO" if config.LANGUAGE == 'RU' else 'Export filename'

            self.export_edit_with_button = EditWithButton(None, theme=theme,
                                                          on_button_clicked_callback=self.on_export_button_clicked,
                                                          is_dir=False, file_type='json',
                                                          dialog_text=placeholder, start_folder='projects',
                                                          placeholder=placeholder, is_existing_file_only=False)
        # Выбор меток
        # Чекбокс
        self.choose_labels_layout = QHBoxLayout()
        self.choose_labels_checkbox = QCheckBox(text='Выбрать имена классов для экспорта')
        self.choose_labels_checkbox.clicked.connect(self.on_choose_labels_checkbox_clicked)
        self.choose_labels_layout.addWidget(self.choose_labels_checkbox)

        # Список с именами классов

        self.labels_checkboxes = []
        if len(label_names) < 8:
            self.label_names_panel = QVBoxLayout()
            for name in label_names:
                checkbox = QCheckBox(text=name)
                checkbox.setVisible(False)
                self.label_names_panel.addWidget(checkbox)
                self.labels_checkboxes.append(checkbox)
        else:
            self.label_names_panel = QHBoxLayout()

            col_tek = QVBoxLayout()
            for i, name in enumerate(label_names):
                if i != 0 and i % 8 == 0:
                    self.label_names_panel.addLayout(col_tek)
                    col_tek = QVBoxLayout()
                checkbox = QCheckBox(text=name)
                checkbox.setChecked(True)
                checkbox.setVisible(False)
                col_tek.addWidget(checkbox)
                self.labels_checkboxes.append(checkbox)

            self.label_names_panel.addLayout(col_tek)

        # Разбиение на train, val

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)

        # Buttons layout:
        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Экспортировать' if config.LANGUAGE == 'RU' else "Import", self)
        self.on_ok_clicked = on_ok_clicked
        if on_ok_clicked:
            self.okBtn.clicked.connect(self.on_ok)

        self.cancelBtn = QPushButton('Отменить' if config.LANGUAGE == 'RU' else 'Cancel', self)

        self.cancelBtn.clicked.connect(self.on_cancel_clicked)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        # Stack layers

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.export_edit_with_button)
        self.mainLayout.addLayout(self.choose_labels_layout)
        self.mainLayout.addLayout(self.label_names_panel)

        self.mainLayout.addLayout(btnLayout)

        self.mainLayout.addWidget(self.progress_bar)

        self.setLayout(self.mainLayout)

        self.data = {}

        self.resize(int(width), int(height))

    def on_choose_labels_checkbox_clicked(self):
        is_checked = self.choose_labels_checkbox.isChecked()
        for checkbox in self.labels_checkboxes:
            checkbox.setVisible(is_checked)

        self.adjustSize()

    def get_export_path(self):
        return self.export_edit_with_button.getEditText()

    def get_export_filename(self):
        return self.export_edit_with_button.getEditText()

    def get_checked_names(self):
        checked_names = [checkbox.text() for checkbox in self.labels_checkboxes if checkbox.isChecked()]

        return checked_names

    def on_ok(self):
        self.export_edit_with_button.setEnabled(False)
        self.cancelBtn.setEnabled(False)
        self.okBtn.setEnabled(False)

        self.on_ok_clicked()

    def on_cancel_clicked(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.hide()

    def on_export_button_clicked(self):
        pass

    def set_progress(self, progress_value):
        if progress_value != 100:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(progress_value)
        else:
            self.progress_bar.setVisible(False)
