import sys

from PyQt5.QtWidgets import QWidget, QVBoxLayout, QApplication

from ui.edit_with_button import EditWithButton
from utils.settings_handler import AppSettings


class SetPathWidget(QWidget):
    def __init__(self, parent, theme='dark_blue.xml', export_format='yolo'):
        """
        Экспорт разметки
        """
        super().__init__(parent)
        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        # Export dir layout:
        if 'yolo' in export_format:
            placeholder = "Директория экспорта YOLO" if self.lang == 'RU' else 'Path to export YOLO'

        elif export_format == 'coco':
            placeholder = "Директория экспорта COCO" if self.lang == 'RU' else 'Path to export COCO'

        self.export_edit_with_button = EditWithButton(None, theme=theme,
                                                      is_dir=True,
                                                      dialog_text=placeholder,
                                                      placeholder=placeholder)
        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.export_edit_with_button)

        self.setLayout(self.mainLayout)

    def get_export_path(self):
        return self.export_edit_with_button.getEditText()


if __name__ == '__main__':
    from qt_material import apply_stylesheet

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    dialog = SetPathWidget(None)
    dialog.show()
    sys.exit(app.exec_())
