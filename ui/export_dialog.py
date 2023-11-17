import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QCheckBox, \
    QProgressBar, QApplication, QLabel

from ui.edit_with_button import EditWithButton
from ui.export_labels_list_view import ExportLabelsList
from utils import config


class ExportDialog(QWidget):
    def __init__(self, parent, width=800, height=200, on_ok_clicked=None, label_names=None,
                 theme='dark_blue.xml', export_format='yolo'):
        """
        Экспорт разметки
        """
        super().__init__(parent)
        self.setWindowTitle(f"Экспорт разметки")
        self.setWindowFlag(Qt.Tool)
        self.set_width = width
        self.set_height = height

        # Export dir layout:
        if export_format == 'yolo':
            placeholder = "Директория экспорта YOLO" if config.LANGUAGE == 'RU' else 'Path to export YOLO'

            self.export_edit_with_button = EditWithButton(None, theme=theme,
                                                          on_button_clicked_callback=self.on_export_button_clicked,
                                                          is_dir=True,
                                                          dialog_text=placeholder,
                                                          placeholder=placeholder)
        elif export_format == 'coco':
            placeholder = "Имя экспортируемого файла COCO" if config.LANGUAGE == 'RU' else 'Export filename'

            self.export_edit_with_button = EditWithButton(None, theme=theme,
                                                          on_button_clicked_callback=self.on_export_button_clicked,
                                                          is_dir=False, file_type='json',
                                                          dialog_text=placeholder,
                                                          placeholder=placeholder, is_existing_file_only=False)
        # Выбор меток
        # Чекбокс
        self.choose_labels_layout = QVBoxLayout()
        self.choose_labels_checkbox = QCheckBox(text='Выбрать имена классов для экспорта')
        self.choose_labels_checkbox.clicked.connect(self.on_choose_labels_checkbox_clicked)
        self.choose_labels_layout.addWidget(self.choose_labels_checkbox)

        self.export_labels_list = ExportLabelsList(labels=label_names, theme=theme)
        self.export_labels_list.setVisible(False)
        self.choose_labels_layout.addWidget(self.export_labels_list)
        self.message = QLabel()
        self.choose_labels_layout.addWidget(self.message)

        # Разбиение на train, val

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)

        # Buttons layout:
        btnLayout = QHBoxLayout()

        self.okBtn = QPushButton('Экспортировать' if config.LANGUAGE == 'RU' else "Import", self)
        self.on_ok_clicked = on_ok_clicked
        if on_ok_clicked:
            self.okBtn.clicked.connect(self.on_ok)

        self.cancelBtn = QPushButton('Отменить' if config.LANGUAGE == 'RU' else 'Cancel', self)

        self.cancelBtn.clicked.connect(self.on_cancel_clicked)

        btnLayout.addWidget(self.okBtn)
        btnLayout.addWidget(self.cancelBtn)

        # Stack layers

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.export_edit_with_button)
        self.mainLayout.addLayout(self.choose_labels_layout)

        self.mainLayout.addLayout(btnLayout)

        self.mainLayout.addWidget(self.progress_bar)

        self.setLayout(self.mainLayout)

        self.data = {}

        self.resize(int(width), int(height))

    def on_choose_labels_checkbox_clicked(self):
        is_checked = self.choose_labels_checkbox.isChecked()
        self.export_labels_list.setVisible(is_checked)

        self.adjustSize()
        self.setMinimumWidth(self.set_width)

    def get_export_path(self):
        return self.export_edit_with_button.getEditText()

    def get_export_filename(self):
        return self.export_edit_with_button.getEditText()

    def get_labels_map(self):

        return self.export_labels_list.get_labels_map()

    def on_ok(self):
        if self.get_export_path() != "":
            self.export_edit_with_button.setEnabled(False)
            self.cancelBtn.setVisible(False)

            self.okBtn.setVisible(False)
            self.message.setText("")

            print(self.export_labels_list.get_labels_map())

            self.export_labels_list.setVisible(False)
            self.choose_labels_checkbox.setVisible(False)
            # self.set_progress(10)

            self.adjustSize()
            self.setMinimumWidth(self.set_width)

            self.on_ok_clicked()
        else:
            self.message.setText("Укажите имя директории для экспорта")


    def on_cancel_clicked(self):
        self.progress_bar.setVisible(False)
        self.progress_bar.setValue(0)
        self.hide()

    def on_export_button_clicked(self):
        if self.get_export_path() == "":
            self.message.setText("Укажите имя директории для экспорта")
        else:
            self.message.setText("")

    def set_progress(self, progress_value):
        if progress_value != 100:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(progress_value)
        else:
            self.progress_bar.setVisible(False)


if __name__ == '__main__':

    def on_ok():
        print("Ok!")

    app = QApplication(sys.argv)
    labels = ['F-16', 'F-35', 'C-130', 'C-17']
    dialog = ExportDialog(None, label_names=labels, on_ok_clicked=on_ok)
    dialog.show()
    sys.exit(app.exec_())
