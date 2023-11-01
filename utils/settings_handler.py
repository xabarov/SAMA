import os

import screeninfo
from PyQt5.QtCore import QSettings, QPoint, QSize

from utils import config
from utils.config import DOMEN_NAME


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
