from PyQt5.QtWidgets import QComboBox


class StyledComboBox(QComboBox):
    def __init__(self, parent=None, theme='dark_blue.xml', dark_color=(255, 255, 255), light_color=(0, 0, 0)):
        super().__init__(parent)

        self.light_color = light_color
        self.dark_color = dark_color

        self.change_theme(theme)

    def change_theme(self, theme):
        self.theme = theme
        combo_box_color = f"rgb{self.dark_color}" if 'dark' in theme else f"rgb{self.light_color}"

        self.setStyleSheet("QComboBox:items"
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