from PyQt5.QtWidgets import QLabel, QCheckBox, QWidget, QGroupBox, QFormLayout, QComboBox, QSpinBox, QVBoxLayout, \
    QHBoxLayout, QPushButton, QDoubleSpinBox, QSlider
from PyQt5.QtCore import Qt
import numpy as np
import utils.config as config


class SettingsWindow(QWidget):
    def __init__(self, parent, settings=None):
        super().__init__(parent)
        self.setWindowTitle("Настройки приложения" if settings['lang'] == 'RU' else 'Settings')
        self.setWindowFlag(Qt.Tool)

        # Настройки обнаружения
        self.formGroupBox = QGroupBox("Настройки разметки" if settings['lang'] == 'RU' else 'Labeling')

        self.settings = {}

        layout = QFormLayout()

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        if settings["alpha"]:
            self.alpha_slider.setValue(settings["alpha"])
        else:
            self.alpha_slider.setValue(50)

        layout.addRow(QLabel("Степень прозрачности масок" if settings['lang'] == 'RU' else 'Label transparency'),
                      self.alpha_slider)

        self.fat_width_slider = QSlider(Qt.Orientation.Horizontal)
        if settings["fat_width"]:
            self.fat_width_slider.setValue(settings["fat_width"])
        else:
            self.fat_width_slider.setValue(50)

        layout.addRow(QLabel("Толщина граней разметки" if settings['lang'] == 'RU' else 'Edges width'),
                      self.fat_width_slider)

        self.where_calc_combo = QComboBox()
        self.where_vars = np.array(["cpu", "cuda", 'Auto'])
        self.where_calc_combo.addItems(self.where_vars)
        where_label = QLabel("Платформа для вычислений" if settings['lang'] == 'RU' else 'Platform for SAM')

        if settings:
            self.where_calc_combo.setCurrentIndex(0)
            idx = np.where(self.where_vars == settings["platform"])[0][0]
            self.where_calc_combo.setCurrentIndex(idx)

        layout.addRow(where_label, self.where_calc_combo)

        self.formGroupBox.setLayout(layout)

        # настройки темы

        self.formGroupBoxGlobal = QGroupBox(
            "Настройки приложения" if settings['lang'] == 'RU' else 'Appearance')

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

        if settings['lang'] == 'RU':

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
        theme_label = QLabel("Тема приложения:" if settings['lang'] == 'RU' else 'Theme:')
        layout_global.addRow(theme_label, self.theme_combo)

        density_label = QLabel('Плотность расположения инструментов:' if settings['lang'] == 'RU' else 'Density:')
        self.density_slider = QSlider(Qt.Orientation.Horizontal)

        self.density_slider.setValue(settings["density"])

        layout_global.addRow(density_label, self.density_slider)

        # lang_label = QLabel('Язык' if settings['lang'] == 'RU' else 'Language')
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

        if settings:
            idx = np.where(self.themes == settings["theme"])[0][0]
            self.theme_combo.setCurrentIndex(idx)

        self.formGroupBoxGlobal.setLayout(layout_global)

        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Принять' if settings['lang'] == 'RU' else 'Apply', self)
        self.okBtn.clicked.connect(self.on_ok_clicked)

        self.cancelBtn = QPushButton('Отменить' if settings['lang'] == 'RU' else 'Cancel', self)
        self.cancelBtn.clicked.connect(self.on_cancel_clicked)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        mainLayout = QVBoxLayout()
        mainLayout.addWidget(self.formGroupBoxGlobal)
        mainLayout.addWidget(self.formGroupBox)
        mainLayout.addLayout(btnLayout)
        self.setLayout(mainLayout)

        self.resize(500, 500)

    def on_ok_clicked(self):
        self.settings['theme'] = self.themes[self.theme_combo.currentIndex()]

        self.settings['platform'] = self.where_vars[self.where_calc_combo.currentIndex()]
        self.settings['lang'] = config.LANGUAGE  # self.language_vars[self.language_combo.currentIndex()]

        self.settings["alpha"] = self.alpha_slider.value()
        self.settings["fat_width"] = self.fat_width_slider.value()
        self.settings['density'] = self.density_slider.value()
        self.close()

    def on_cancel_clicked(self):
        self.settings.clear()
        self.close()
