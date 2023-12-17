from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QLabel, QGridLayout, QFrame, QHBoxLayout, QPushButton
from qtpy.QtWidgets import QApplication

from ui.signals_and_slots import InfoConnection
from utils.help_functions import calc_rows_cols
from utils.settings_handler import AppSettings


class ButtonOptions(QWidget):
    def __init__(self, parent, buttons=None, on_ok=None, on_cancel=None):
        """
        Поле с текстом + Картинкой + Кнопками Правка и Удалить
        """
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        self.buttons = []

        self.grid = QGridLayout()
        self.r = 0
        self.c = 0

        if buttons:
            size = len(buttons)
            rows, cols = calc_rows_cols(size)
            r, c = 0, 0

            for b in buttons:
                self.grid.addWidget(b, r, c, alignment=Qt.AlignHCenter)
                self.buttons.append(b)
                c += 1
                if c == cols:
                    c = 0
                    r += 1
            self.r = r
            self.c = c

        self.layout = QVBoxLayout()

        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Принять' if self.lang == 'RU' else 'Apply', self)
        if on_ok:
            self.okBtn.clicked.connect(on_ok)
        else:
            self.okBtn.clicked.connect(self.hide)

        self.cancelBtn = QPushButton('Отменить' if self.lang == 'RU' else 'Cancel', self)

        if on_cancel:
            self.cancelBtn.clicked.connect(on_cancel)
        else:
            self.cancelBtn.clicked.connect(self.hide)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        self.layout.addLayout(self.grid)
        self.layout.addLayout(btnLayout)
        self.setLayout(self.layout)

    def add_button(self, button):
        r, c = self.r, self.c

        self.grid.addWidget(button, self.r, self.c, alignment=Qt.AlignHCenter)
        self.buttons.append(button)
        if c == 2:
            self.c = 0
            self.r += 1
        else:
            self.c += 1

    def set_active_by_option(self, option_name):
        for b in self.buttons:
            if b.option == option_name:
                b.activate()

    def reset(self):
        for b in self.buttons:
            b.reset()

    def get_options(self):
        options = []
        for b in self.buttons:
            if b.is_pressed:
                options.append(b.option)
        return options


class ButtonOption(QFrame):

    def __init__(self, parent, option_text="", option_name="", path_to_img=None, min_width=100, min_height=100):
        """
        Поле с текстом + Картинкой + Кнопками Правка и Удалить
        """
        super().__init__(parent)

        self.conn = InfoConnection()
        self.option = option_name
        self.is_pressed = False
        self.settings = AppSettings()
        self.theme = self.settings.read_theme()

        # left part - Text + Image
        self.image_text_lay = QVBoxLayout()
        self.text_label = QLabel(option_text)
        self.text_label.setAlignment(Qt.AlignCenter)

        self.img_label = QLabel()
        if path_to_img:
            pixmap = QPixmap(path_to_img)
            self.img_label.setPixmap(pixmap.scaled(150, 150, Qt.KeepAspectRatio))

        self.image_text_lay.addWidget(self.text_label, stretch=1)
        self.image_text_lay.addWidget(self.img_label, stretch=4)

        self.setLayout(self.image_text_lay)

        self.setMinimumWidth(min_width)
        self.setMinimumHeight(min_height)

    def reset(self):

        if self.is_pressed:
            if 'dark' in self.theme:
                self.setStyleSheet("QLabel {border: 0px solid LightGray; padding:3px;}")
            else:
                self.setStyleSheet("QLabel {border: 0px solid LightGray; padding:3px;}")

            self.is_pressed = False

    def activate(self):
        if 'dark' in self.theme:
            self.setStyleSheet("QLabel {border: 1px solid LightGray; padding:3px;}")
        else:
            self.setStyleSheet("QLabel {border: 1px solid LightGray; padding:3px;}")
        self.is_pressed = True

    def mousePressEvent(self, a0) -> None:
        self.conn.info_message.emit(self.option)
        if self.is_pressed:
            if 'dark' in self.theme:
                self.setStyleSheet("QLabel {border: 0px solid LightGray; padding:3px;}")
            else:
                self.setStyleSheet("QLabel {border: 0px solid LightGray; padding:3px;}")

            self.is_pressed = False
        else:
            if 'dark' in self.theme:
                self.setStyleSheet("QLabel {border: 1px solid LightGray; padding:3px;}")
            else:
                self.setStyleSheet("QLabel {border: 1px solid LightGray; padding:3px;}")
            self.is_pressed = True


if __name__ == '__main__':
    from qt_material import apply_stylesheet


    def print_options():
        print(w.get_options())


    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    w = ButtonOptions(None, on_ok=print_options)
    for i in range(10):
        b = ButtonOption(None, option_name=f"Option {i}", option_text=f"Option {i}", path_to_img='cat.jpg')
        w.add_button(b)

    w.show()

    app.exec_()
