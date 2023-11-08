import os

import screeninfo
from PyQt5.QtCore import QSettings, QPoint, QSize

from utils import config
from utils.config import DOMEN_NAME

shortcuts = {'copy': {'appearance': 'Ctrl+C', 'modifier': ['Ctrl'], 'name_eng': 'Copy label',
                      'name_ru': 'Копировать выделенную метку', 'shortcut_key_eng': 67, 'shortcut_key_ru': 1057},
             'crop': {'appearance': 'Ctrl+I', 'modifier': ['Ctrl'], 'name_eng': 'Crop image',
                      'name_ru': 'Вырезать область', 'shortcut_key_eng': 73, 'shortcut_key_ru': 1064},
             'del': {'appearance': 'Delete', 'modifier': None, 'name_eng': 'Delete polygon',
                     'name_ru': 'Удаление полигона', 'shortcut_key_eng': 16777223, 'shortcut_key_ru': 16777223},
             'detect_single': {'appearance': 'Ctrl+Y', 'modifier': ['Ctrl'], 'name_eng': 'Detect object by one pass',
                               'name_ru': 'Обнаружить объекты за один проход', 'shortcut_key_eng': 89,
                               'shortcut_key_ru': 1053},
             'end_drawing': {'appearance': 'Space', 'modifier': None, 'name_eng': 'Finish label drawing',
                             'name_ru': 'Закончить создание метки', 'shortcut_key_eng': 32, 'shortcut_key_ru': 32},
             'fit': {'appearance': 'Ctrl+F', 'modifier': ['Ctrl'], 'name_eng': 'Fit image size',
                     'name_ru': 'Подогнать под размер окна', 'shortcut_key_eng': 70, 'shortcut_key_ru': 1040},
             'gd': {'appearance': 'Ctrl+G', 'modifier': ['Ctrl'], 'name_eng': 'Create labels by GroundingDINO+SAM',
                    'name_ru': 'Создание полигонов с помощью GroundingDINO+SAM', 'shortcut_key_eng': 71,
                    'shortcut_key_ru': 1055},
             'image_after': {'appearance': 'PERIOD', 'modifier': None, 'name_eng': 'Go to next image',
                             'name_ru': 'Следующее изображение', 'shortcut_key_eng': 46, 'shortcut_key_ru': 46},
             'image_before': {'appearance': ',', 'modifier': None, 'name_eng': 'Go to image before',
                              'name_ru': 'Предыдущее изображение', 'shortcut_key_eng': 44, 'shortcut_key_ru': 1041},
             'open_project': {'appearance': 'Ctrl+O', 'modifier': ['Ctrl'], 'name_eng': 'Open project',
                              'name_ru': 'Открыть проект', 'shortcut_key_eng': 79, 'shortcut_key_ru': 1065},
             'paste': {'appearance': 'Ctrl+V', 'modifier': ['Ctrl'], 'name_eng': 'Paste label',
                       'name_ru': 'Вставить выделенную метку', 'shortcut_key_eng': 86, 'shortcut_key_ru': 1052},
             'polygon': {'appearance': 'Ctrl+B', 'modifier': ['Ctrl'], 'name_eng': 'Create polygon manually',
                         'name_ru': 'Создание полигона  ручную', 'shortcut_key_eng': 66, 'shortcut_key_ru': 1048},
             'print': {'appearance': 'Ctrl+P', 'modifier': ['Ctrl'], 'name_eng': 'Print', 'name_ru': 'Печать',
                       'shortcut_key_eng': 80, 'shortcut_key_ru': 1047},
             'quit': {'appearance': 'Ctrl+Q', 'modifier': ['Ctrl'], 'name_eng': 'Quit', 'name_ru': 'Выйти',
                      'shortcut_key_eng': 81, 'shortcut_key_ru': 1049},
             'sam_box': {'appearance': 'Ctrl+M', 'modifier': ['Ctrl'], 'name_eng': 'Create polygon by SAM box',
                         'name_ru': 'Создание полигона с помощью бокса SAM', 'shortcut_key_eng': 77,
                         'shortcut_key_ru': 1068},
             'sam_points': {'appearance': 'Ctrl+A', 'modifier': ['Ctrl'], 'name_eng': 'Create polygon by SAM points',
                            'name_ru': 'Создание полигона с помощью точек SAM', 'shortcut_key_eng': 65,
                            'shortcut_key_ru': 1060},
             'save_project': {'appearance': 'Ctrl+S', 'modifier': ['Ctrl'], 'name_eng': 'Save project',
                              'name_ru': 'Сохранение проекта', 'shortcut_key_eng': 83, 'shortcut_key_ru': 1067},
             'settings': {'appearance': 'Ctrl+PERIOD', 'modifier': ['Ctrl'], 'name_eng': 'Settings',
                          'name_ru': 'Настройки', 'shortcut_key_eng': 46, 'shortcut_key_ru': 46},
             'start_drawing': {'appearance': 'S', 'modifier': None, 'name_eng': 'New label', 'name_ru': 'Новая метка',
                               'shortcut_key_eng': 83, 'shortcut_key_ru': 1067},
             'undo': {'appearance': 'Ctrl+Z', 'modifier': ['Ctrl'], 'name_eng': 'Undo', 'name_ru': 'Отменить',
                      'shortcut_key_eng': 90, 'shortcut_key_ru': 1071},
             'zoom_in': {'appearance': 'Ctrl++', 'modifier': ['Ctrl', ''], 'name_eng': 'Zoom In',
                         'name_ru': 'Увеличить масштаб', 'shortcut_key_eng': 61, 'shortcut_key_ru': 61},
             'zoom_out': {'appearance': 'Ctrl+-', 'modifier': ['Ctrl'], 'name_eng': 'Zoom Out',
                          'name_ru': 'Уменьшить масштаб', 'shortcut_key_eng': 45, 'shortcut_key_ru': 45}}


# print(list(shortcuts.keys()))

class AppSettings:
    def __init__(self, app_name=None):
        if not app_name:
            app_name = config.QT_SETTINGS_APP
        self.qt_settings = QSettings(config.QT_SETTINGS_COMPANY, app_name)
        self.write_lang(config.LANGUAGE)

    def write_sam_hq(self, use_hq):
        self.qt_settings.setValue("cnn/sam_hq", use_hq)

    def read_sam_hq(self):
        return self.qt_settings.value("cnn/sam_hq", True)

    def write_username(self, username):
        self.qt_settings.setValue("general/username", username)

    def read_username(self):
        return self.qt_settings.value("general/username", "no_name")

    def write_username_variants(self, username_variants):
        self.qt_settings.setValue("general/username_variants", username_variants)

    def read_username_variants(self):
        return self.qt_settings.value("general/username_variants", [self.read_username()])

    def write_shortcuts(self, shortcuts):
        self.qt_settings.setValue("general/shortcuts", shortcuts)

    def read_shortcuts(self):
        return self.qt_settings.value("general/shortcuts", shortcuts)

    def reset_shortcuts(self):
        self.qt_settings.setValue("general/shortcuts", shortcuts)

    def write_size_pos_settings(self, size, pos):
        self.qt_settings.beginGroup("main_window")
        self.qt_settings.setValue("size", size)
        self.qt_settings.setValue("pos", pos)
        self.qt_settings.endGroup()

    def read_size_pos_settings(self):

        self.qt_settings.beginGroup("main_window")
        size = self.qt_settings.value("size", QSize(1200, 800))
        pos = self.qt_settings.value("pos", QPoint(50, 50))
        self.qt_settings.endGroup()

        monitors = screeninfo.get_monitors()

        if len(monitors) == 1:
            m = monitors[0]
            width = m.width
            height = m.height
            if pos.x() > width * 0.7:
                pos.setX(0)
            if pos.y() > height * 0.7:
                pos.setY(0)

        return size, pos

    def write_lang(self, lang):
        self.qt_settings.setValue("main/lang", lang)

    def read_lang(self):
        return self.qt_settings.value("main/lang", 'ENG')

    def write_theme(self, theme):
        self.qt_settings.setValue("main/theme", theme)

    def read_theme(self):
        return self.qt_settings.value("main/theme", 'dark_blue.xml')

    def read_server_name(self):
        return self.qt_settings.value("main/server", DOMEN_NAME)

    def write_server_name(self, server_name):
        self.qt_settings.setValue("main/server", server_name)

    def get_icon_folder(self):
        theme_str = self.read_theme()
        theme_type = theme_str.split('.')[0]
        icon_folder = os.path.join("ui/icons/", theme_type)
        if not os.path.exists(icon_folder):
            return os.path.join("icons/", theme_type)
        return icon_folder

    def write_platform(self, platform):
        self.qt_settings.setValue("main/platform", platform)

    def read_platform(self):
        platform = self.qt_settings.value("main/platform", 'cuda')
        return platform

    def write_alpha(self, alpha):
        self.qt_settings.setValue("main/alpha", alpha)

    def read_alpha(self):
        return self.qt_settings.value("main/alpha", 50)

    def write_fat_width(self, fat_width):
        self.qt_settings.setValue("main/fat_width", fat_width)

    def read_fat_width(self):
        return self.qt_settings.value("main/fat_width", 50)

    def write_density(self, density):
        self.qt_settings.setValue("main/density", density)

    def read_density(self):
        return self.qt_settings.value("main/density", 50)

    def write_cnn_model(self, model_name):
        self.qt_settings.setValue("cnn/model_name", model_name)

    def read_cnn_model(self):
        return self.qt_settings.value("cnn/model_name", 'YOLOv8')

    def write_conf_thres(self, conf_thres):
        self.qt_settings.setValue("cnn/conf_thres", conf_thres)

    def read_conf_thres(self):
        return self.qt_settings.value("cnn/conf_thres", 0.5)

    def write_simplify_factor(self, simplify_factor):
        self.qt_settings.setValue("cnn/simplify_factor", simplify_factor)

    def read_simplify_factor(self):
        return self.qt_settings.value("cnn/simplify_factor", 1.0)

    def write_iou_thres(self, iou_thres):
        self.qt_settings.setValue("cnn/iou_thres", iou_thres)

    def read_iou_thres(self):
        return self.qt_settings.value("cnn/iou_thres", 0.5)
