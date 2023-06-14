from PyQt5.QtWidgets import QLabel, QWidget, QGroupBox, QFormLayout, QComboBox, QVBoxLayout, \
    QHBoxLayout, QPushButton, QSlider, QDoubleSpinBox
from PyQt5.QtCore import Qt

from utils.settings_handler import AppSettings
from utils import cls_settings
from ui.settings_window_base import SettingsWindowBase

import numpy as np


class SettingsWindow(SettingsWindowBase):
    def __init__(self, parent):
        super().__init__(parent)

    def stack_layouts(self):
        self.create_cnn_layout()

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.main_group)
        self.mainLayout.addWidget(self.labeling_group)
        self.mainLayout.addWidget(self.classifierGroupBox)

        btnLayout = self.create_buttons()
        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)

    def create_cnn_layout(self):
        self.where_calc_combo = QComboBox()
        self.where_vars = np.array(["cpu", "cuda", 'Auto'])
        self.where_calc_combo.addItems(self.where_vars)
        where_label = QLabel("Платформа для ИИ" if self.lang == 'RU' else 'Platform for AI')

        self.where_calc_combo.setCurrentIndex(0)
        idx = np.where(self.where_vars == self.settings.read_platform())[0][0]
        self.where_calc_combo.setCurrentIndex(idx)

        # Настройки обнаружения
        self.classifierGroupBox = QGroupBox("Настройки классификации" if self.lang == 'RU' else 'Classifier')

        classifier_layout = QFormLayout()

        self.cnn_combo = QComboBox()
        cnn_list = list(cls_settings.CNN_DICT.keys())
        self.cnns = np.array(cnn_list)
        self.cnn_combo.addItems(self.cnns)
        cnn_label = QLabel("Модель СНС:" if self.lang == 'RU' else "Classifier model")

        classifier_layout.addRow(cnn_label, self.cnn_combo)
        classifier_layout.addRow(where_label, self.where_calc_combo)

        self.cnn_combo.setCurrentIndex(0)
        idx = np.where(self.cnns == self.settings.read_cnn_model())[0][0]
        self.cnn_combo.setCurrentIndex(idx)

        self.conf_thres_spin = QDoubleSpinBox()
        self.conf_thres_spin.setDecimals(3)
        conf_thres = self.settings.read_conf_thres()
        self.conf_thres_spin.setValue(float(conf_thres))

        self.conf_thres_spin.setMinimum(0.01)
        self.conf_thres_spin.setMaximum(1.00)
        self.conf_thres_spin.setSingleStep(0.01)
        classifier_layout.addRow(QLabel("Доверительный порог:" if self.lang == 'RU' else "Conf threshold"),
                                 self.conf_thres_spin)

        self.IOU_spin = QDoubleSpinBox()
        self.IOU_spin.setDecimals(3)
        iou_thres = self.settings.read_iou_thres()
        self.IOU_spin.setValue(float(iou_thres))

        self.IOU_spin.setMinimum(0.01)
        self.IOU_spin.setMaximum(1.00)
        self.IOU_spin.setSingleStep(0.01)
        classifier_layout.addRow(QLabel("IOU порог:" if self.lang == 'RU' else "IoU threshold"), self.IOU_spin)

        self.classifierGroupBox.setLayout(classifier_layout)

    def on_ok_clicked(self):
        super(SettingsWindow, self).on_ok_clicked()

        self.settings.write_platform(self.where_vars[self.where_calc_combo.currentIndex()])

        self.settings.write_cnn_model(self.cnns[self.cnn_combo.currentIndex()])
        self.settings.write_iou_thres(self.IOU_spin.value())
        self.settings.write_conf_thres(self.conf_thres_spin.value())
