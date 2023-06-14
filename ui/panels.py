from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QPushButton
from PyQt5.QtGui import QIcon

from utils import config


class ImagesPanel(QWidget):
    def __init__(self, parent, add_image_to_projectAct, del_image_from_projectAct, icon_folder, button_size=30,
                 on_color_change_signal=None, on_images_list_change=None):
        super().__init__(parent)

        # Панель изображений - заголовок
        images_header = QHBoxLayout()
        images_header.addWidget(QLabel("Список изображений" if config.LANGUAGE == 'RU' else "Images list"))

        self.add_im_button = QPushButton()
        self.add_im_button.clicked.connect(add_image_to_projectAct)
        images_header.addWidget(self.add_im_button)

        self.del_im_button = QPushButton()
        self.del_im_button.setEnabled(False)
        self.del_im_button.clicked.connect(del_image_from_projectAct)
        images_header.addWidget(self.del_im_button)

        self.add_im_button.setIcon((QIcon(icon_folder + "/add.png")))
        self.del_im_button.setIcon((QIcon(icon_folder + "/del.png")))

        self.add_im_button.setFixedHeight(button_size)
        self.del_im_button.setFixedHeight(button_size)

        self.add_im_button.setFixedWidth(button_size)
        self.del_im_button.setFixedWidth(button_size)

        self.add_im_button.setStyleSheet("border: none;")
        self.del_im_button.setStyleSheet("border: none;")

        if on_color_change_signal:
            on_color_change_signal.connect(self.on_color_change)

        if on_images_list_change:
            on_images_list_change.connect(self.on_im_list_change)

        self.setLayout(images_header)

    def on_color_change(self, icon_folder):
        self.del_im_button.setIcon((QIcon(icon_folder + "/del.png")))
        self.add_im_button.setIcon((QIcon(icon_folder + "/add.png")))

    def on_im_list_change(self, images_size):
        if images_size > 0:
            self.del_im_button.setEnabled(True)
            return
        self.del_im_button.setEnabled(False)


class LabelsPanel(QWidget):
    def __init__(self, parent, del_label_from_image_act, icon_folder, button_size=30, on_color_change_signal=None,
                 on_labels_count_change=None):
        super().__init__(parent)

        # Панель изображений - заголовок
        header = QHBoxLayout()
        header.addWidget(QLabel("Список меток" if config.LANGUAGE == 'RU' else "Labels list"))

        self.del_im_button = QPushButton()
        self.del_im_button.setEnabled(False)
        self.del_im_button.clicked.connect(del_label_from_image_act)
        header.addWidget(self.del_im_button)

        self.del_im_button.setIcon((QIcon(icon_folder + "/del.png")))
        if on_color_change_signal:
            on_color_change_signal.connect(self.on_color_change)

        if on_labels_count_change:
            on_labels_count_change.connect(self.on_lbs_count_change)

        self.del_im_button.setFixedHeight(button_size)

        self.del_im_button.setFixedWidth(button_size)

        self.del_im_button.setStyleSheet("border: none;")

        self.setLayout(header)

    def on_color_change(self, icon_folder):
        self.del_im_button.setIcon((QIcon(icon_folder + "/del.png")))

    def on_lbs_count_change(self, count):
        if count > 0:
            self.del_im_button.setEnabled(True)
            return
        self.del_im_button.setEnabled(False)
