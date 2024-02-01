import sys

import numpy as np
from PyQt5.QtWidgets import QWidget, QApplication, QFormLayout, QLabel

from ui.custom_widgets.styled_widgets import StyledComboBox
from ui.custom_widgets.edit_with_button import EditWithButton
from utils.settings_handler import AppSettings


class SetPathWidget(QWidget):
    def __init__(self, parent, theme='dark_blue.xml'):
        """
        Экспорт разметки
        """
        super().__init__(parent)
        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        # 1. Export format Row
        self.export_label = QLabel("Формат:" if self.lang == 'RU' else "Format:")

        self.export_format_combo = StyledComboBox(self, theme=theme)
        self.export_format_vars = np.array(["YOLO Seg", "YOLO Box", 'COCO', 'MMSegmentation'])
        self.export_format_combo.addItems(self.export_format_vars)
        self.export_format_combo.currentTextChanged.connect(self.on_export_format_combo_change)

        # Export dir layout:
        placeholder = "Директория экспорта YOLO" if self.lang == 'RU' else 'Path to export YOLO'

        self.export_path_label = QLabel("Путь:" if self.lang == 'RU' else "Path:")

        self.export_edit_with_button = EditWithButton(None, theme=theme,
                                                      is_dir=True,
                                                      dialog_text=placeholder,
                                                      placeholder=placeholder)
        self.mainLayout = QFormLayout()

        self.mainLayout.addRow(self.export_label, self.export_format_combo)
        self.mainLayout.addRow(self.export_path_label, self.export_edit_with_button)

        self.setLayout(self.mainLayout)

    def get_export_format(self):
        # 0 - "YOLO Seg", 1 - "YOLO Box", 2 - 'COCO', 3 - 'MM Segmentation'
        export_idx = self.export_format_combo.currentIndex()

        export_formats = ["yolo_seg", "yolo_box", "coco", "mm_seg"]
        return export_formats[export_idx]

    def get_export_path(self):
        return self.export_edit_with_button.getEditText()

    def on_export_format_combo_change(self, text):

        placeholder = ""

        if 'YOLO' in text:
            placeholder = "Директория экспорта YOLO" if self.lang == 'RU' else 'Path to export YOLO'

        elif text == 'COCO':
            placeholder = "Директория экспорта COCO" if self.lang == 'RU' else 'Path to export COCO'

        elif text == 'MMSegmentation':
            placeholder = "Директория экспорта MMSegmentation" if self.lang == 'RU' else 'Path to export MMSegmentation'

        self.export_edit_with_button.setPlaceholderText(placeholder)


if __name__ == '__main__':
    from qt_material import apply_stylesheet

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    dialog = SetPathWidget(None)
    dialog.show()
    sys.exit(app.exec_())
