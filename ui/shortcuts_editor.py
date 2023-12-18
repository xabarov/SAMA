import sys

from PyQt5 import QtCore, QtGui
from PyQt5 import QtWidgets
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QWidget, QPushButton, QVBoxLayout, QHBoxLayout, \
    QLabel, QMessageBox, QHeaderView
from qt_material import apply_stylesheet

from ui.custom_widgets.styled_widgets import StyledEdit

from utils.settings_handler import AppSettings

ru_text_to_key = {'й': 1049, 'ц': 1062, 'у': 1059, 'к': 1050, 'е': 1045, 'н': 1053, 'г': 1043,
                  'ш': 1064, 'щ': 1065, 'з': 1047, 'х': 1061, 'ъ': 1066, 'ф': 1060, 'ы': 1067, 'в': 1042, 'а': 1040,
                  'п': 1055, 'р': 1056, 'о': 1054, 'л': 1051, 'д': 1044, 'ж': 1046, 'э': 1069, 'я': 1071, 'ч': 1063,
                  'с': 1057, 'м': 1052, 'и': 1048, 'т': 1058, 'ь': 1068, 'б': 1041, 'ю': 1070}

eng_text_to_key = {'q': 81, 'w': 87, 'e': 69, 'r': 82, 't': 84, 'y': 89, 'u': 85, 'i': 73, 'o': 79,
                   'p': 80, '[': 91, ']': 93, 'a': 65, 's': 83, 'd': 68, 'f': 70, 'g': 71, 'h': 72, 'j': 74, 'k': 75,
                   'l': 76, ';': 59, "'": 39, 'z': 90, 'x': 88, 'c': 67, 'v': 86, 'b': 66, 'n': 78, 'm': 77, ',': 44,
                   '.': 46}

ru_to_eng = {k: v for k, v in zip(ru_text_to_key.keys(), eng_text_to_key.keys())}


def show_message(title, text):
    msgbox = QMessageBox()
    msgbox.setIcon(QMessageBox.Information)
    msgbox.setText(text)
    msgbox.setWindowTitle(title)
    msgbox.exec()


class ShortCutEdit(StyledEdit):
    def __init__(self, theme):
        super(ShortCutEdit, self).__init__(theme=theme)
        self.keymap = {}
        for key, value in vars(QtCore.Qt).items():
            if isinstance(value, QtCore.Qt.Key):
                self.keymap[value] = key.partition('_')[2]

        self.modmap = {
            QtCore.Qt.ControlModifier: self.keymap[QtCore.Qt.Key_Control],
            QtCore.Qt.AltModifier: self.keymap[QtCore.Qt.Key_Alt],
            QtCore.Qt.ShiftModifier: self.keymap[QtCore.Qt.Key_Shift]
        }

        self.command = None
        self.ru_text_to_key = {}
        self.eng_text_to_key = {}

    def keyevent_to_string(self, event, convert_ru_to_eng=False):
        sequence = []
        modifiers = []
        for modifier, cmd in self.modmap.items():
            if event.modifiers() & modifier:
                cmd_converted = self.convert_shortcut_to_pyqt(cmd)
                sequence.append(cmd_converted)
                modifiers.append(cmd_converted)

        key = self.keymap.get(event.key(), event.text())
        key = self.convert_shortcut_to_pyqt(key)

        e_key = None
        if key not in sequence:
            if key.lower() in ru_text_to_key and convert_ru_to_eng:
                ru_key = key.lower()
                eng_key = ru_to_eng[ru_key]
                sequence.append(eng_key.upper())
                e_key = eng_text_to_key[eng_key]
            else:
                sequence.append(key)
                e_key = event.key()

        appearance = '+'.join(sequence)
        self.command = {'appearance': appearance, 'key': e_key, 'modifiers': modifiers}
        return appearance

    def convert_shortcut_to_pyqt(self, sc):

        maps = {'Control': 'Ctrl', 'Period': '.', 'Comma': ',', 'Minus': '-', 'Equal': '+', 'Plus': '+',
                'BraceLeft': '[', 'BraceRight': ']', 'PageUp': 'PgUp', 'PageDown': 'PgDown',
                'Slash': '/', 'Apostrophe': "'", 'Semicolon': ";"}
        for k, v in maps.items():
            if k in sc:
                sc = sc.replace(k, v)

        return sc

    def append_to_ru(self, text, key):
        ru_letters = set('Ё!"№;%:?*()_+йцукенгшщзхъфывапролджэячсмитьбю.')
        if text in ru_letters and text not in self.ru_text_to_key:
            self.ru_text_to_key[text] = key

    def append_to_eng(self, text, key):
        eng_letters = set("~!@#$%^&*()_+qwertyuiop[]asdfghjkl;'zxcvbnm,./")
        if text in eng_letters and text not in self.eng_text_to_key:
            self.eng_text_to_key[text] = key

    def keyPressEvent(self, e: QtGui.QKeyEvent) -> None:
        self.setText(self.keyevent_to_string(e, convert_ru_to_eng=True))

    def get_command(self):
        return self.command


class ShortCutInputDialog(QWidget):

    def __init__(self, parent, title_name, question_name, placeholder="", min_width=300):
        super().__init__(parent)
        self.setWindowTitle(f"{title_name}")
        self.setWindowFlag(QtCore.Qt.Tool)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.label = QLabel(f"{question_name}")
        self.edit = ShortCutEdit(theme=self.settings.read_theme())
        self.edit.setPlaceholderText(placeholder)

        btnLayout = QVBoxLayout()

        self.okBtn = QPushButton('Ввести' if self.lang == 'RU' else "OK", self)

        btnLayout.addWidget(self.okBtn)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.label)
        self.mainLayout.addWidget(self.edit)
        self.mainLayout.addLayout(btnLayout)
        self.setLayout(self.mainLayout)
        self.setMinimumWidth(min_width)

    def getText(self):
        return self.edit.text()

    def getCommand(self):
        return self.edit.get_command()


class ShortCutsTable(QTableWidget):

    def __init__(self, parent=None, lang='RU', min_width=400, shortcuts=None):
        super(ShortCutsTable, self).__init__(parent)

        if shortcuts:
            self.setColumnCount(3)
            self.setRowCount(len(shortcuts))
            if lang == 'RU':
                self.setHorizontalHeaderLabels(['Key', 'Имя команды', 'Горячие\nклавиши'])
            else:
                self.setHorizontalHeaderLabels(['Key', 'Command', 'Shortcut'])
            self.horizontalHeader().setSectionResizeMode(1, QHeaderView.Stretch)
            row = 0
            for k in shortcuts:
                self.setItem(row, 0, QTableWidgetItem(k))
                name_item = QTableWidgetItem(shortcuts[k]['name_ru'])
                name_item.setFlags(name_item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.setItem(row, 1, name_item)
                appearance_item = QTableWidgetItem(shortcuts[k]['appearance'])
                appearance_item.setFlags(appearance_item.flags() & ~QtCore.Qt.ItemIsEditable)
                self.setItem(row, 2, appearance_item)
                row += 1

            self.setMinimumWidth(min_width)
            self.setColumnHidden(0, True)


class ShortCutsEditor(QWidget):

    def __init__(self, parent=None, on_ok_act=None, width_percent=0.5, height_percent=0.6):
        super(ShortCutsEditor, self).__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        self.icon_folder = self.settings.get_icon_folder()

        self.setWindowTitle("Настройки горячих клавиш" if self.lang == 'RU' else 'Shortcuts Settings')
        self.setWindowIcon(QIcon(self.icon_folder + "/keyboard.png"))
        self.setWindowFlag(QtCore.Qt.Window)

        self.shortcuts = self.settings.read_shortcuts()

        self.create_actions()
        self.create_toolbar()
        self.table = ShortCutsTable(lang=self.lang, shortcuts=self.shortcuts)

        mainlayout = QVBoxLayout()

        self.tablelay = QVBoxLayout()
        self.tablelay.addLayout(self.toolbar)
        self.tablelay.addWidget(self.table)

        buttons_lay = QHBoxLayout()
        ok_button = QPushButton('Применить' if self.lang == 'RU' else 'Apply')
        ok_button.clicked.connect(self.on_ok)
        if on_ok_act:
            ok_button.clicked.connect(on_ok_act)

        cancel_button = QPushButton('Отменить' if self.lang == 'RU' else 'Cancel')
        cancel_button.clicked.connect(self.hide)

        reset_all_button = QPushButton(" Вернуть к исходным" if self.lang == 'RU' else ' Reset')
        # reset_all_button.setIcon(QIcon(self.icon_folder + "/circle.png"))
        reset_all_button.clicked.connect(self.on_reset_all)

        buttons_lay.addWidget(ok_button)
        buttons_lay.addWidget(cancel_button)
        buttons_lay.addWidget(reset_all_button)

        mainlayout.addLayout(self.tablelay)

        mainlayout.addLayout(buttons_lay)

        self.setLayout(mainlayout)
        size, pos = self.settings.read_size_pos_settings()
        self.setMinimumWidth(int(size.width()*width_percent))
        self.setMinimumHeight(int(size.height()*height_percent))

    def on_ok(self):
        self.settings.write_shortcuts(self.shortcuts)
        self.hide()

    def create_actions(self):
        self.edit = QPushButton(" Правка" if self.lang == 'RU' else ' Edit')
        self.edit.setIcon(QIcon(self.icon_folder + "/edit.png"))
        self.edit.clicked.connect(self.on_edit)

        self.reset = QPushButton(" Удалить" if self.lang == 'RU' else ' Clear')
        self.reset.setIcon(QIcon(self.icon_folder + "/clear.png"))
        self.reset.clicked.connect(self.on_reset)

    def create_toolbar(self):
        self.toolbar = QHBoxLayout()
        self.toolbar.addWidget(self.edit, stretch=1)
        self.toolbar.addWidget(self.reset, stretch=1)

    def on_reset(self):
        self.edit_row = self.table.currentRow()
        if self.edit_row == -1:
            return
        k = self.table.item(self.edit_row, 0).text()
        shortcut_apperance = self.shortcuts[k]['appearance']
        if shortcut_apperance:
            self.shortcuts[k]['appearance'] = ""
            self.shortcuts[k]['modifier'] = None
            self.shortcuts[k]['shortcut_key_ru'] = None,
            self.shortcuts[k]['shortcut_key_eng'] = None
        appearance_item = self.table.item(self.edit_row, 2)
        appearance_item.setText(self.shortcuts[k]['appearance'])

    def on_reset_all(self):
        self.settings.reset_shortcuts()
        self.shortcuts = self.settings.read_shortcuts()
        self.table.close()
        self.table = ShortCutsTable(lang=self.lang, shortcuts=self.shortcuts)
        self.tablelay.addWidget(self.table)
        self.update()

    def on_edit(self):
        self.edit_row = self.table.currentRow()
        if self.edit_row == -1:
            return
        self.selected_k = self.table.item(self.edit_row, 0).text()
        cmd = self.table.item(self.edit_row, 1).text()
        shortcut_apperance = self.shortcuts[self.selected_k]['appearance']

        self.input_dialog = ShortCutInputDialog(self,
                                                title_name=f"Редактирование комманды {cmd}" if self.lang == 'RU' else f"Edit shortcut {cmd}",
                                                question_name="Введите новую комманду:" if self.lang == 'RU' else "Enter new shortcut:",
                                                placeholder=shortcut_apperance, min_width=400)
        self.input_dialog.okBtn.clicked.connect(self.on_edit_shortcut_ok)
        self.input_dialog.show()
        self.input_dialog.edit.setFocus()

    def check_shortcut(self, sc, selected_key):

        for k in self.shortcuts:
            if self.shortcuts[k]['appearance'] == sc['appearance']:
                if k != selected_key:
                    return False
        return True

    def on_edit_shortcut_ok(self):
        shortcut_command = self.input_dialog.getCommand()  # {'appearance': appearance, 'key': e_key, 'modifiers': modifiers}

        if not shortcut_command:
            if self.lang == 'RU':
                text = f"Введите команду для {self.table.item(self.edit_row, 1).text()}"
                title = "Не задана команда"
            else:
                text = f"Please input shortcut for {self.table.item(self.edit_row, 1).text()}"
                title = "Shortcut is empty"
            show_message(title, text)
            return
        if not self.check_shortcut(shortcut_command, self.selected_k):
            if self.lang == 'RU':
                text = "Команда уже занята"
                title = "Не верно задана команда"
            else:
                text = f"This shortcut is set"
                title = "Wrong shortcut"
            show_message(title, text)
            return

        appearance_item = self.table.item(self.edit_row, 2)
        k = self.table.item(self.edit_row, 0).text()
        shortcut_key = shortcut_command['key']
        ru_key_to_text = {k: v for v, k in ru_text_to_key.items()}

        if shortcut_key in ru_key_to_text:
            # Перевести RU -> ENG
            text = shortcut_command['appearance']
            modifiers = None
            if '+' in text:
                text_splitted = text.split('+')
                modifiers = text_splitted[:-1]
                symbol_big = text_splitted[-1]

                symbol_low = symbol_big.lower()
                eng_text = ru_to_eng.get(symbol_low, symbol_low)

                self.shortcuts[k]['appearance'] = '+'.join(modifiers) + '+' + eng_text.upper()

            else:

                symbol_low = text.lower()
                eng_text = ru_to_eng.get(symbol_low, symbol_low)
                self.shortcuts[k]['appearance'] = eng_text.upper()

            eng_shortcut_key = eng_text_to_key.get(eng_text, shortcut_key)
            self.shortcuts[k]['shortcut_key_eng'] = eng_shortcut_key
            self.shortcuts[k]['shortcut_key_ru'] = shortcut_key
            self.shortcuts[k]['modifier'] = modifiers

        else:
            text = shortcut_command['appearance']
            modifiers = None
            if '+' in text:
                text_splitted = text.split('+')
                modifiers = text_splitted[:-1]
                text = text_splitted[-1]
                self.shortcuts[k]['appearance'] = '+'.join(modifiers) + '+' + text
                text = text.lower()

            else:
                self.shortcuts[k]['appearance'] = text
                text = text.lower()

            eng_to_ru = {k: v for v, k in ru_to_eng.items()}
            if text in eng_to_ru:
                ru_text = eng_to_ru[text]
                ru_shortcut_key = ru_text_to_key.get(ru_text, shortcut_key)
                self.shortcuts[k]['shortcut_key_ru'] = ru_shortcut_key
            else:
                self.shortcuts[k]['shortcut_key_ru'] = shortcut_key
            self.shortcuts[k]['shortcut_key_eng'] = shortcut_key
            self.shortcuts[k]['modifier'] = modifiers

        appearance_item.setText(self.shortcuts[k]['appearance'])
        self.input_dialog.close()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    apply_stylesheet(app, theme='dark_blue.xml', invert_secondary=False)

    w = ShortCutsEditor()
    w.show()
    sys.exit(app.exec_())
