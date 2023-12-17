import os

from PyQt5.QtGui import QPixmap, QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QLabel, QHBoxLayout
from qtpy.QtWidgets import QApplication

from utils.settings_handler import AppSettings


class Card(QWidget):

    def __init__(self, parent, text="", sub_text="", path_to_img=None, min_width=100, min_height=100,
                 theme='dark_blue.xml',
                 on_edit_clicked=None, is_del_button=True, is_edit_button=True, on_del=None):
        """
        Поле с текстом + Картинкой + Кнопками Правка и Удалить
        """
        super().__init__(parent)

        self.layout = QHBoxLayout()

        # left part - Text + Image
        self.image_text_lay = QVBoxLayout()
        self.text_label = QLabel(text)
        self.text_label.setAlignment(Qt.AlignCenter)

        self.img_label = QLabel()
        if path_to_img:
            pixmap = QPixmap(path_to_img)
            self.img_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))

        self.image_text_lay.addWidget(self.text_label, stretch=1)
        self.image_text_lay.addWidget(self.img_label, stretch=4)

        if sub_text != "":
            self.sub_text_label = QLabel(sub_text)
            self.sub_text_label.setAlignment(Qt.AlignCenter)
            self.image_text_lay.addWidget(self.sub_text_label, stretch=1)
        else:
            self.sub_text_label = None

        # right = buttons
        path_to_icons = os.path.join(os.path.dirname(__file__), "..", "icons", theme.split('.')[0])

        self.buttons_lay = QVBoxLayout()

        # Delete Button

        if is_del_button:
            self.delete_button = QPushButton()
            if on_del:
                self.delete_button.clicked.connect(on_del)
            else:
                self.delete_button.clicked.connect(self.hide)
            self.delete_button.setIcon(QIcon(os.path.join(path_to_icons, 'del.png')))
            self.buttons_lay.addWidget(self.delete_button, stretch=1)

        if is_edit_button:
            self.edit_button = QPushButton()
            self.edit_button.setIcon(QIcon(os.path.join(path_to_icons, 'edit.png')))
            if on_edit_clicked:
                self.edit_button.clicked.connect(on_edit_clicked)

            self.buttons_lay.addWidget(self.edit_button, stretch=1)

        self.main_lay = QHBoxLayout()
        self.main_lay.addLayout(self.image_text_lay, stretch=4)
        self.main_lay.addLayout(self.buttons_lay, stretch=1)

        self.setLayout(self.main_lay)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.setMinimumWidth(min_width)
        self.setMinimumHeight(min_height)

    def set_sub_text(self, sub_text):
        if not self.sub_text_label:
            self.sub_text_label = QLabel(sub_text)
            self.sub_text_label.setAlignment(Qt.AlignCenter)
            self.image_text_lay.addWidget(self.sub_text_label, stretch=1)
        else:
            self.sub_text_label.setText(sub_text)


if __name__ == '__main__':
    from qt_material import apply_stylesheet

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    slider = Card(None, text="Title", path_to_img="cat.jpg", is_del_button=False)
    slider.show()

    app.exec_()
