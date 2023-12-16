from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QSlider
from qtpy.QtWidgets import QApplication
from ui.export_labels_list_view import ExportLabelsList
from ui.custom_widgets.two_handle_splitter import TwoHandleSplitter, part_colors
from utils.settings_handler import AppSettings


class PreprocessStep(QWidget):

    def __init__(self, parent, label_names=None, minimum_expand_height=300, theme='dark_blue.xml'):
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.minimum_expand_height = minimum_expand_height

        self.choose_labels_layout = QVBoxLayout()
        checkbox_text = 'Выбрать имена классов для экспорта' if self.lang == 'RU' else 'Choose labels for export'
        self.choose_labels_checkbox = QCheckBox(text=checkbox_text)
        self.choose_labels_checkbox.clicked.connect(self.on_choose_labels_checkbox_clicked)
        self.choose_labels_layout.addWidget(self.choose_labels_checkbox)

        del_name = 'Удалить' if self.lang == 'RU' else 'Delete'
        blur_name = 'Размыть' if self.lang == 'RU' else 'Blur'
        headers = ('Метка', 'Заменить на') if self.lang == 'RU' else ('Label', 'Replace to')
        self.export_labels_list = ExportLabelsList(labels=label_names, theme=theme, del_name=del_name,
                                                   blur_name=blur_name, headers=headers)
        self.export_labels_list.setVisible(False)
        self.choose_labels_layout.addWidget(self.export_labels_list)

        self.setLayout(self.choose_labels_layout)

    def on_choose_labels_checkbox_clicked(self):
        is_checked = self.choose_labels_checkbox.isChecked()
        self.export_labels_list.setVisible(is_checked)
        if is_checked:
            self.setMinimumHeight(self.minimum_expand_height)
            self.adjustSize()
        else:
            self.setMinimumHeight(100)
            self.adjustSize()


if __name__ == '__main__':
    from qt_material import apply_stylesheet

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')
    labels = ['F-16', 'F-35', 'C-130', 'C-17']

    slider = PreprocessStep(None, label_names=labels)
    slider.show()

    app.exec_()
