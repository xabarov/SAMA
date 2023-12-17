import os

from PyQt5.QtGui import QIcon
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QPushButton, QHBoxLayout, QLabel
from qtpy.QtWidgets import QApplication

from utils.settings_handler import AppSettings
from ui.custom_widgets.card import Card


class CardsField(QWidget):
    """
        Поле для размещения карточек
    """

    def __init__(self, parent, cards=None, theme='dark_blue.xml', on_add_clicked=None, block_width=150,
                 block_height=150, label_text=None):
        super().__init__(parent)

        self.layout = QHBoxLayout()
        self.theme = theme

        # ADD Card button:
        self.add_button = QPushButton()
        path_to_icons = os.path.join(os.path.dirname(__file__), "..", "icons", theme.split('.')[0])
        self.add_button.setIcon(QIcon(os.path.join(path_to_icons, 'add.png')))

        self.block_height = block_height
        self.block_width = block_width
        self.add_button.setMinimumWidth(block_width)
        self.add_button.setMinimumHeight(block_height)
        if on_add_clicked:
            self.add_button.clicked.connect(on_add_clicked)

        self.layout.addWidget(self.add_button, stretch=1)

        if label_text:
            # Добавить еще текст. Будет скрыт, если есть хотя бы один элемент
            self.text_label = QLabel(label_text)
            self.text_label.setAlignment(Qt.AlignHCenter)
            self.layout.addWidget(self.text_label, stretch=1)
        else:
            self.text_label = None

        self.cards = []
        if cards:
            for card in cards:
                self.cards.append(card)
                self.layout.addWidget(card, stretch=1)

        self.setLayout(self.layout)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

    def add_card(self, title, path_to_img, min_width=100, min_height=100, on_edit=None):
        card = Card(None, text=title, path_to_img=path_to_img, min_width=min_width, min_height=min_height,
                    theme=self.theme,
                    on_edit_clicked=on_edit)
        self.cards.append(card)
        self.layout.addWidget(card)
        self.adjustSize()
        if self.text_label:
            self.text_label.hide()

    def add_exist_card(self, card):
        self.cards.append(card)
        self.layout.addWidget(card)
        self.adjustSize()
        if self.text_label:
            self.text_label.hide()

    def delete_card_by_idx(self, idx):
        if len(self.cards) > idx:
            self.cards[idx].hide()
        if len(self.cards) == 0:
            if self.text_label:
                self.text_label.show()


if __name__ == '__main__':
    from qt_material import apply_stylesheet

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    w = CardsField(None)
    for i in range(3):
        w.add_card(f"Cat {i}", path_to_img='cat.jpg')
    w.show()

    app.exec_()
