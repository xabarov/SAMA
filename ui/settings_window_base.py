import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget, QGroupBox, QFormLayout, QVBoxLayout, \
    QHBoxLayout, QPushButton, QSlider, QTabWidget, QFontDialog, QCheckBox, QColorDialog
from PyQt5.QtGui import QPolygonF, QColor, QPen, QPainter, QPixmap, QFont, QIcon
from ui.combo_box_styled import StyledComboBox
from utils.settings_handler import AppSettings
from PyQt5 import QtWidgets
import sys
from qt_material import apply_stylesheet


class SettingsWindowBase(QWidget):
    def __init__(self, parent):
        super().__init__(parent)

        self.import_settings()

        self.create_labeling_group()
        self.create_main_group()

        self.stack_layouts()

        self.resize(500, 500)

    def stack_layouts(self):

        self.mainLayout = QVBoxLayout()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.main_group,
                         "Общие" if self.lang == 'RU' else 'General')  # addWidget(self.main_group)
        self.tabs.addTab(self.labeling_group,
                         "Разметка" if self.lang == 'RU' else 'Labeling')  # addWidget(self.labeling_group)
        self.mainLayout.addWidget(self.tabs)

        btns = self.create_buttons()
        self.mainLayout.addLayout(btns)
        self.setLayout(self.mainLayout)

    def create_main_group(self):
        # настройки темы

        self.main_group = QGroupBox()

        layout_global = QFormLayout()

        theme = self.settings.read_theme()

        self.theme_combo = StyledComboBox(self, theme=theme)

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
        self.labeling_group = QGroupBox()

        layout = QFormLayout()

        self.alpha_slider = QSlider(Qt.Orientation.Horizontal)
        self.alpha_slider.setValue(self.settings.read_alpha())

        layout.addRow(QLabel("Степень непрозрачности масок" if self.lang == 'RU' else 'Label opaque'),
                      self.alpha_slider)

        self.fat_width_slider = QSlider(Qt.Orientation.Horizontal)
        self.fat_width_slider.setValue(self.settings.read_fat_width())

        layout.addRow(QLabel("Толщина граней разметки" if self.lang == 'RU' else 'Edges width'),
                      self.fat_width_slider)

        label_text_params = self.settings.read_label_text_params()
        is_hide = label_text_params['hide']
        auto_color = label_text_params['auto_color']
        self.cur_font = label_text_params['font']
        self.default_color = label_text_params['default_color']

        # Скрывать ли метки

        self.font_hide_checkbox = QCheckBox()
        self.font_hide_checkbox.setChecked(is_hide)
        self.font_hide_checkbox.stateChanged.connect(self.font_hide_checkbox_clicked)
        layout.addRow(QLabel('Скрывать текст' if self.lang == 'RU' else 'Hide text'), self.font_hide_checkbox)

        # Шрифт Меток
        self.font_btn = QPushButton('Задать' if self.lang == 'RU' else 'Settings')
        self.font_btn.setIcon(QIcon(self.icon_folder + "/color.png"))
        self.font_btn.clicked.connect(self.on_font_clicked)

        self.font_label = QLabel()
        self.set_font_btn_label()

        layout.addRow(self.font_label, self.font_btn)

        # Цвет текста
        # Чекбокс
        self.autocolor_checkbox = QCheckBox()
        self.autocolor_checkbox.setChecked(auto_color)
        self.autocolor_checkbox.stateChanged.connect(self.autocolor_checkbox_clicked)
        self.autocolor_checkbox_label = QLabel(
            'Задать цвет текста как у метки' if self.lang == 'RU' else 'Set text color the same as label')
        layout.addRow(self.autocolor_checkbox_label,
                      self.autocolor_checkbox)

        # Палитра
        self.color_btn = QPushButton('Задать' if self.lang == 'RU' else 'Settings')

        self.color_btn.setIcon(QIcon(self.icon_folder + "/color.png"))
        self.color_btn.clicked.connect(self.on_color_clicked)

        self.color_label = QLabel()
        self.set_color_btn_label()

        layout.addRow(self.color_label, self.color_btn)

        # Положение текста относительно метки

        self.font_hide_checkbox_clicked()
        self.autocolor_checkbox_clicked()

        self.labeling_group.setLayout(layout)

    def autocolor_checkbox_clicked(self):
        auto_color = self.autocolor_checkbox.isChecked()
        if auto_color:
            self.color_btn.hide()
            self.color_label.hide()
        else:
            self.color_btn.show()
            self.color_label.show()

    def set_font_btn_label(self):
        font_btn_text = 'Шрифт меток: ' if self.lang == 'RU' else 'Labels  font: '
        font_btn_text += f"{self.cur_font.family()},{self.cur_font.pointSize()}"
        self.font_label.setText(font_btn_text)

    def set_color_btn_label(self):
        color_btn_text = 'Цвет шрифта: ' if self.lang == 'RU' else 'Font color: '
        color_btn_text += f"{self.default_color}"
        self.color_label.setText(color_btn_text)

    def font_hide_checkbox_clicked(self):
        hide = self.font_hide_checkbox.isChecked()
        if hide:
            self.font_btn.hide()
            self.font_label.hide()
            self.color_btn.hide()
            self.color_label.hide()
            self.autocolor_checkbox.hide()
            self.autocolor_checkbox_label.hide()
        else:
            self.font_btn.show()
            self.font_label.show()
            self.color_btn.show()
            self.color_label.show()
            self.autocolor_checkbox.show()
            self.autocolor_checkbox_label.show()

    def on_font_clicked(self):
        font_dialog = QFontDialog()
        font_dialog.setWindowTitle(
            f"Выберите шрифт для меток" if self.settings.read_lang() == 'RU' else f"Set font for labels")

        font_dialog.setCurrentFont(self.cur_font)
        font_dialog.setWindowIcon(QIcon(self.icon_folder + "/color.png"))

        font, ok = font_dialog.getFont()
        if ok:
            self.cur_font = font
            self.set_font_btn_label()

    def on_color_clicked(self):
        color_dialog = QColorDialog()

        color_dialog.setCurrentColor(QColor(*self.default_color))
        color_dialog.setWindowIcon(QIcon(self.icon_folder + "/color.png"))
        color_dialog.exec()
        rgb = color_dialog.selectedColor().getRgb()
        self.default_color = (rgb[0], rgb[1], rgb[2], 255)
        self.set_color_btn_label()

    def import_settings(self):
        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        self.setWindowTitle("Настройки приложения" if self.lang == 'RU' else 'Settings')
        self.icon_folder = self.settings.get_icon_folder()
        self.setWindowFlag(Qt.Tool)  # убрать при тесте

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

        self.settings.write_label_text_params(self.cur_font, hide=self.font_hide_checkbox.isChecked(),
                                              auto_color=self.autocolor_checkbox.isChecked(),
                                              default_color=self.default_color)

        self.close()

    def on_cancel_clicked(self):
        self.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    apply_stylesheet(app, theme='dark_blue.xml', invert_secondary=False)

    w = SettingsWindowBase(None)
    w.show()
    sys.exit(app.exec_())
