from PyQt5.QtWidgets import QLabel, QWidget, QHBoxLayout, QPushButton
from PyQt5.QtGui import QIcon

class ImagesPanel(QWidget):
    def __init__(self, parent, add_image_to_projectAct, del_image_from_projectAct, icon_folder, button_size=30, on_color_change_signal=None):
        super().__init__(parent)

        # Панель изображений - заголовок
        images_header = QHBoxLayout()
        images_header.addWidget(QLabel("Список изображений"))

        self.add_im_button = QPushButton()
        self.add_im_button.clicked.connect(add_image_to_projectAct)
        images_header.addWidget(self.add_im_button)

        self.del_im_button = QPushButton()
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

        self.setLayout(images_header)

    def on_color_change(self, icon_folder):
        self.del_im_button.setIcon((QIcon(icon_folder + "/del.png")))
        self.add_im_button.setIcon((QIcon(icon_folder + "/add.png")))

class LabelsPanel(QWidget):
    def __init__(self, parent, del_label_from_image_act, icon_folder, button_size=30, on_color_change_signal=None):
        super().__init__(parent)

        # Панель изображений - заголовок
        header = QHBoxLayout()
        header.addWidget(QLabel("Список меток"))

        self.del_im_button = QPushButton()
        self.del_im_button.clicked.connect(del_label_from_image_act)
        header.addWidget(self.del_im_button)

        self.del_im_button.setIcon((QIcon(icon_folder + "/del.png")))
        if on_color_change_signal:
            on_color_change_signal.connect(self.on_color_change)

        self.del_im_button.setFixedHeight(button_size)

        self.del_im_button.setFixedWidth(button_size)

        self.del_im_button.setStyleSheet("border: none;")

        self.setLayout(header)

    def on_color_change(self, icon_folder):
        self.del_im_button.setIcon((QIcon(icon_folder + "/del.png")))
