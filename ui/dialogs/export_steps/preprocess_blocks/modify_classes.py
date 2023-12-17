import os

from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QPushButton, QHBoxLayout

from ui.dialogs.export_steps.export_labels_list_view import ExportLabelsList
from utils.settings_handler import AppSettings


class ModifyClassesStep(QWidget):

    def __init__(self, parent, label_names, min_width=780, minimum_expand_height=300, theme='dark_blue.xml', on_ok=None,
                 on_cancel=None):
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.preprocess_steps = []

        self.minimum_expand_height = minimum_expand_height

        self.layout = QVBoxLayout()

        title = 'Выбрать имена классов для экспорта' if self.lang == 'RU' else 'Choose labels for export'
        self.setWindowTitle(title)

        icon_folder = os.path.join(os.path.dirname(__file__), "preprocess_icons")

        self.setWindowIcon(QIcon(icon_folder + "/list.png"))

        del_name = 'Удалить' if self.lang == 'RU' else 'Delete'
        blur_name = 'Размыть' if self.lang == 'RU' else 'Blur'
        headers = ('Метка', 'Заменить на') if self.lang == 'RU' else ('Label', 'Replace to')
        self.export_labels_list = ExportLabelsList(labels=label_names, theme=theme, del_name=del_name,
                                                   blur_name=blur_name, headers=headers)

        self.layout.addWidget(self.export_labels_list)

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

        self.layout.addLayout(btnLayout)
        self.setLayout(self.layout)
        self.setMinimumWidth(min_width)

    def get_labels_map(self):
        return self.export_labels_list.get_labels_map()


if __name__ == '__main__':
    from qt_material import apply_stylesheet
    from qtpy.QtWidgets import QApplication


    def on_ok_test():
        print(w.get_labels_map())


    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    labels = ['F-16', 'F-35', 'C-130', 'C-17']

    w = ModifyClassesStep(None, labels, on_ok=on_ok_test)

    w.show()

    app.exec_()
