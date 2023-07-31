from PyQt5.QtWidgets import QLabel, QWidget, QGroupBox, QFormLayout, QComboBox, QVBoxLayout, \
    QHBoxLayout, QPushButton, QSlider
from PyQt5.QtCore import Qt

from utils.settings_handler import AppSettings

import numpy as np


class SettingsWindowBase(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.combos = []

        self.import_settings()

        self.create_labeling_group()
        self.create_main_group()

        self.stack_layouts()

        self.resize(500, 500)

        self.change_combos_text_color()

    def stack_layouts(self):
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.main_group)
        self.mainLayout.addWidget(self.labeling_group)

        btnLayout = self.create_buttons()
        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)

    def change_combos_text_color(self):

        theme = self.settings.read_theme()
        combo_box_color = "rgb(255,255,255)" if 'dark' in theme else " rgb(0,0,0)"

        for combo in self.combos:
            combo.setStyleSheet("QComboBox:items"
                                "{"
                                f"color: {combo_box_color};"
                                "}"
                                "QComboBox"
                                "{"
                                f"color: {combo_box_color};"
                                "}"
                                "QListView"
                                "{"
                                f"color: {combo_box_color};"
                                "}"
                                )

    def create_main_group(self):
        # настройки темы

        self.main_group = QGroupBox(
            "Настройки приложения" if self.lang == 'RU' else 'Appearance')

        layout_global = QFormLayout()

        self.theme_combo = QComboBox()
        self.combos.append(self.theme_combo)

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

        self.main_group.setLayout(layout_global)

    def create_labeling_group(self):
        # Настройки разметки
        self.labeling_group = QGroupBox("Настройки разметки" if self.lang == 'RU' else 'Labeling')

        layout = QFormLayout()

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setValue(self.settings.read_alpha())

        layout.addRow(QLabel("Степень непрозрачности масок" if self.lang == 'RU' else 'Label opaque'),
                      self.alpha_slider)

        self.fat_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.fat_width_slider.setValue(self.settings.read_fat_width())

        layout.addRow(QLabel("Толщина граней разметки" if self.lang == 'RU' else 'Edges width'),
                      self.fat_width_slider)

        self.labeling_group.setLayout(layout)

    def import_settings(self):
        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        self.setWindowTitle("Настройки приложения" if self.lang == 'RU' else 'Settings')
        self.setWindowFlag(Qt.Tool)

    def create_buttons(self):
        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Принять' if self.lang == 'RU' else 'Apply', self)
        self.okBtn.clicked.connect(self.on_ok_clicked)

        self.cancelBtn = QPushButton('Отменить' if self.lang == 'RU' else 'Cancel', self)
        self.cancelBtn.clicked.connect(self.on_cancel_clicked)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        return btnLayout

    def on_ok_clicked(self):

        self.settings.write_theme(self.themes[self.theme_combo.currentIndex()])

        self.settings.write_alpha(self.alpha_slider.value())  # Transparancy = 1 - alpha
        self.settings.write_fat_width(self.fat_width_slider.value())
        self.settings.write_density(self.density_slider.value())

        self.close()

    def on_cancel_clicked(self):
        self.close()
