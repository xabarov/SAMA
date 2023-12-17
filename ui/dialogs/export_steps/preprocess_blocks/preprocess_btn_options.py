import os

from PyQt5.QtGui import QIcon

from ui.custom_widgets.button_options import ButtonOptions, ButtonOption


class PreprocessBtnOptions(ButtonOptions):
    def __init__(self, parent, buttons=None, on_ok=None, on_cancel=None):
        super(PreprocessBtnOptions, self).__init__(parent, buttons=buttons, on_ok=on_ok, on_cancel=on_cancel)

        title = "Выберите вариант обработки" if self.lang == 'RU' else "Choose preprocess options"
        self.setWindowTitle(title)

        icon_folder = os.path.join(os.path.dirname(__file__), "preprocess_icons")

        self.setWindowIcon(QIcon(icon_folder + "/process.png"))

        self.option_names = ["modify_classes", "resize", "filter_null", "tile", "auto_contrast", "grayscale"]
        if self.lang == 'RU':
            txt = ["Изменить классы", "Изменить размер изображений", "Удалить неразмеченные", "Разбить на фрагменты",
                   "Авто контраст", "Преобразовать в серые тона"]
        else:
            txt = ["Modify Classes", "Resize", "Filter Null", "Tile", "Auto Contrast", "GrayScale"]

        paths = ["list", "resize", "null", "tile", "contrast", "grayscale"]

        icons_path = os.path.join(os.path.dirname(__file__), 'preprocess_icons')
        print(icons_path)

        for name, text, path in zip(self.option_names, txt, paths):
            self.add_button(ButtonOption(None, option_name=name, option_text=text,
                                         path_to_img=os.path.join(icons_path, f'{path}.png')))


if __name__ == '__main__':
    from qt_material import apply_stylesheet
    from qtpy.QtWidgets import QApplication


    def print_options():
        print(w.get_options())


    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    w = PreprocessBtnOptions(None, on_ok=print_options)

    w.show()

    app.exec_()
