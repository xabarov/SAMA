from PyQt5.QtWidgets import QLabel, QWidget, QGroupBox, QFormLayout, QComboBox, QVBoxLayout, \
    QHBoxLayout, QPushButton, QSlider, QDoubleSpinBox
from PyQt5.QtCore import Qt

from utils.settings_handler import AppSettings
from utils import cls_settings

import numpy as np


class SettingsWindow(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        self.setWindowTitle("Настройки приложения" if self.lang == 'RU' else 'Settings')
        self.setWindowFlag(Qt.Tool)

        # Настройки разметки
        self.formGroupBox = QGroupBox("Настройки разметки" if self.lang == 'RU' else 'Labeling')

        layout = QFormLayout()

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setValue(self.settings.read_alpha())

        layout.addRow(QLabel("Степень прозрачности масок" if self.lang == 'RU' else 'Label transparency'),
                      self.alpha_slider)

        self.fat_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.fat_width_slider.setValue(self.settings.read_fat_width())

        layout.addRow(QLabel("Толщина граней разметки" if self.lang == 'RU' else 'Edges width'),
                      self.fat_width_slider)

        self.where_calc_combo = QComboBox()
        self.where_vars = np.array(["cpu", "cuda", 'Auto'])
        self.where_calc_combo.addItems(self.where_vars)
        where_label = QLabel("Платформа для ИИ" if self.lang == 'RU' else 'Platform for AI')

        self.where_calc_combo.setCurrentIndex(0)
        idx = np.where(self.where_vars == self.settings.read_platform())[0][0]
        self.where_calc_combo.setCurrentIndex(idx)

        self.formGroupBox.setLayout(layout)

        # настройки темы

        self.formGroupBoxGlobal = QGroupBox(
            "Настройки приложения" if self.lang == 'RU' else 'Appearance')

        layout_global = QFormLayout()

        self.theme_combo = QComboBox()
        self.themes = np.array(['dark_amber.xml',
                                'dark_blue.xml',
                                'dark_cyan.xml',
                                'dark_lightgreen.xml',
                                'dark_pink.xml',
                                'dark_purple.xml',
                                'dark_red.xml',
                                'dark_teal.xml',
                                'dark_yellow.xml',
                                'light_amber.xml',
                                'light_blue.xml',
                                'light_blue_500.xml',
                                'light_cyan.xml',
                                'light_cyan_500.xml',
                                'light_lightgreen.xml',
                                'light_lightgreen_500.xml',
                                'light_orange.xml',
                                'light_pink.xml',
                                'light_pink_500.xml',
                                'light_purple.xml',
                                'light_purple_500.xml',
                                'light_red.xml',
                                'light_red_500.xml',
                                'light_teal.xml',
                                'light_teal_500.xml',
                                'light_yellow.xml'])

        if self.lang == 'RU':

            self.themes_for_display_names = np.array(['темно-янтарная',
                                                      'темно-синяя',
                                                      'темно-голубая',
                                                      'темно-светло-зеленая',
                                                      'темно-розовая',
                                                      'темно фиолетовая',
                                                      'темно-красная',
                                                      'темно-бирюзовая',
                                                      'темно-желтая',
                                                      'светлый янтарная',
                                                      'светло-синяя',
                                                      'светло-синяя-500',
                                                      'светло-голубая',
                                                      'светлый-голубая-500',
                                                      'светло-зеленая',
                                                      'светло-зеленая-500',
                                                      'светло-оранжевая',
                                                      'светло-розовая',
                                                      'светло-розовая-500',
                                                      'светло-фиолетовая',
                                                      'светло-фиолетовая-500',
                                                      'светло-красная',
                                                      'светло-красная-500',
                                                      'светло-бирюзовая',
                                                      'светло-бирюзовая-500',
                                                      'светло-желтый'])
        else:
            self.themes_for_display_names = [theme[0:-4] for theme in self.themes]

        self.theme_combo.addItems(self.themes_for_display_names)
        theme_label = QLabel("Тема приложения:" if self.lang == 'RU' else 'Theme:')
        layout_global.addRow(theme_label, self.theme_combo)

        density_label = QLabel('Плотность расположения инструментов:' if self.lang == 'RU' else 'Density:')
        self.density_slider = QSlider(Qt.Orientation.Horizontal)

        self.density_slider.setValue(self.settings.read_density())

        layout_global.addRow(density_label, self.density_slider)

        # lang_label = QLabel('Язык' if self.lang == 'RU' else 'Language')
        # self.language_combo = QComboBox()
        # self.language_vars = np.array(["RU", "ENG"])
        # self.language_combo.addItems(self.language_vars)
        #
        # if settings:
        #     self.language_combo.setCurrentIndex(0)
        #     idx = np.where(self.language_vars == settings["lang"])[0][0]
        #     self.language_combo.setCurrentIndex(idx)
        #
        # layout_global.addRow(lang_label, self.language_combo)

        idx = np.where(self.themes == self.settings.read_theme())[0][0]
        self.theme_combo.setCurrentIndex(idx)

        self.formGroupBoxGlobal.setLayout(layout_global)

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

        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Принять' if self.lang == 'RU' else 'Apply', self)
        self.okBtn.clicked.connect(self.on_ok_clicked)

        self.cancelBtn = QPushButton('Отменить' if self.lang == 'RU' else 'Cancel', self)
        self.cancelBtn.clicked.connect(self.on_cancel_clicked)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBoxGlobal)
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addWidget(self.classifierGroupBox)
        mainLayout.addLayout(btnLayout)
        self.setLayout(mainLayout)

        self.resize(500, 500)

    def on_ok_clicked(self):
        self.settings.write_theme(self.themes[self.theme_combo.currentIndex()])

        self.settings.write_platform(self.where_vars[self.where_calc_combo.currentIndex()])

        self.settings.write_alpha(100 - self.alpha_slider.value())  # Transparancy = 1 - alpha
        self.settings.write_fat_width(self.fat_width_slider.value())
        self.settings.write_density(self.density_slider.value())

        self.settings.write_cnn_model(self.cnns[self.cnn_combo.currentIndex()])
        self.settings.write_iou_thres(self.IOU_spin.value())
        self.settings.write_conf_thres(self.conf_thres_spin.value())

        self.close()

    def on_cancel_clicked(self):
        self.close()
