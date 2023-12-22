import os

from PIL import Image, ImageQt
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QVBoxLayout, QHBoxLayout, QLabel, QCheckBox, QFormLayout

from ui.custom_widgets.card import Card
from ui.custom_widgets.styled_widgets import StyledSpinBox
from ui.dialogs.export_steps.preprocess_blocks.preprocess_block import PreprocessBlock
from utils.settings_handler import AppSettings


class ResizeCard(Card):
    """
    Карточка этапа обработки "Resize".
    Картинка -  resize.png
    Две кнопки - удалить и редактировать
    По нажатии на кнопку редактировать открывается окно редактирования размера изображений
    """

    def __init__(self, test_image_path, min_width=250, min_height=150, max_width=100, on_ok=None, on_delete=None):

        name = "Resize"
        self.icons_path = os.path.join(os.path.dirname(__file__), 'preprocess_icons')

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        if self.lang == "RU":
            sub_text = "Изменить размер\nизображений"
        else:
            sub_text = "Resize"

        self.on_ok = on_ok
        self.test_image_path = test_image_path

        super(ResizeCard, self).__init__(None, text=name,
                                         path_to_img=os.path.join(self.icons_path, 'resize.png'),
                                         min_width=min_width, min_height=min_height, max_width=max_width,
                                         on_edit_clicked=self.on_resize_edit, is_del_button=True, is_edit_button=True,
                                         on_del=on_delete, sub_text=sub_text)

        self.new_size = None

    def get_new_size(self):
        """
        Возвращает новый размер изображений
        """
        return self.new_size

    def on_resize_edit(self):
        """
        При нажатии на кнопку Edit карточки "Resize"
        Открытие окна редактирования размера изображений
        """

        self.resize_widget = ResizeStep(self.test_image_path, on_ok=self.on_resize_ok)
        self.resize_widget.show()

    def on_resize_ok(self):

        self.new_size = self.resize_widget.get_options()

        if self.lang == 'RU':
            sub_text = f"Размер изображений: {self.new_size}"
        else:
            sub_text = f"Images size: {self.new_size}"

        self.set_sub_text(sub_text)

        self.resize_widget.hide()

        if self.on_ok:
            # Если в конструкторе передан хук
            self.on_ok()


class ResizeStep(PreprocessBlock):
    """
    Окно редактирования размера изображений.
    """

    def __init__(self, test_image_path, min_width=780, minimum_expand_height=300, on_ok=None,
                 on_cancel=None, thumbnail_size=400):
        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.test_img_path = test_image_path
        name = "resize"
        title = "Изменить размер изображений" if self.lang == 'RU' else 'Resize images'

        super().__init__(name, options=None, parent=None, title=title, min_width=min_width,
                         minimum_expand_height=minimum_expand_height, icon_name="resize.png")

        self.img = Image.open(test_image_path)
        self.img_width, self.img_height = self.img.size
        self.img = ImageQt.ImageQt(self.img)

        scale = 1.0 * self.img_width / thumbnail_size
        self.thumbnail_size = thumbnail_size

        self.thumbnail_width = thumbnail_size
        self.thumbnail_height = int(self.img_height / scale)

        self.create_resize_layout()
        self.create_images_layout()

        self.create_buttons(on_ok, on_cancel)
        self.setMinimumWidth(min_width)

    def create_images_layout(self):
        self.images_layout = QHBoxLayout()
        self.pix_original = QPixmap(self.test_img_path)
        self.pix_original = self.pix_original.scaled(self.thumbnail_width, self.thumbnail_height)

        self.label_img_original = QLabel()
        self.label_img_original.setPixmap(self.pix_original)
        self.pix_resized = QPixmap.fromImage(self.img)
        self.pix_resized = self.pix_resized.scaled(self.thumbnail_width, self.thumbnail_height)

        self.label_img_resized = QLabel()
        self.label_img_resized.setPixmap(self.pix_resized)
        self.images_layout.addWidget(self.label_img_original)
        self.images_layout.addWidget(self.label_img_resized)
        self.layout.addLayout(self.images_layout)

    def create_resize_layout(self):
        self.resize_layout = QVBoxLayout()
        header_text = "Введите новый размер изображений" if self.lang == 'RU' else "Input new images size"
        self.header = QLabel(header_text)
        self.header.setAlignment(Qt.AlignVCenter)

        form_lay = QFormLayout()
        checkbox_text = "Сохранять пропорции" if self.lang == 'RU' else "Auto scale"
        self.auto_checkbox = QCheckBox()
        self.auto_checkbox.setChecked(True)
        self.auto_checkbox.clicked.connect(self.on_auto_checkbox_change)
        form_lay.addRow(QLabel(checkbox_text), self.auto_checkbox)
        width_text = "Ширина, px" if self.lang == 'RU' else "Width, px"
        height_text = "Высота, px" if self.lang == 'RU' else "Height, px"
        self.width_spin = StyledSpinBox(theme=self.settings.read_theme())

        self.width_spin.setMinimum(16)
        self.width_spin.setMaximum(10000)

        self.width_spin.setValue(self.img_width)
        self.width_spin.valueChanged.connect(self.on_width_change)

        self.height_spin = StyledSpinBox(theme=self.settings.read_theme())

        self.height_spin.setMinimum(16)
        self.height_spin.setMaximum(10000)

        self.height_spin.setValue(self.img_height)
        self.height_spin.valueChanged.connect(self.on_height_change)
        self.height_spin.setEnabled(False)

        form_lay.addRow(QLabel(width_text), self.width_spin)
        form_lay.addRow(QLabel(height_text), self.height_spin)

        self.resize_layout.addWidget(self.header)
        self.resize_layout.addLayout(form_lay)

        self.layout.addLayout(self.resize_layout)

    def on_width_change(self, new_width):
        if self.auto_checkbox.isChecked():
            scale = 1.0 * new_width / self.img_width
            new_height = int(self.img_height * scale)
            self.fit_thumbnail(new_width, new_height)
        else:
            self.fit_thumbnail(new_width, self.img_height)

    def fit_thumbnail(self, new_width, new_height):

        self.width_spin.setValue(new_width)
        self.height_spin.setValue(new_height)

        params = [self.img_width, self.img_height, new_width, new_height]
        p_max = max(params)
        params = [int(self.thumbnail_size * p / p_max) for p in params]

        pix_resized = self.pix_resized.copy().scaled(params[2], params[3])
        self.label_img_resized.setPixmap(pix_resized)

        pix_original = self.pix_original.copy().scaled(params[0], params[1])
        self.label_img_original.setPixmap(pix_original)

    def on_height_change(self, new_height):
        if self.auto_checkbox.isChecked():
            return

        self.fit_thumbnail(self.img_width, new_height)

    def on_auto_checkbox_change(self):
        self.height_spin.setEnabled(not self.auto_checkbox.isChecked())
        if self.auto_checkbox.isChecked():
            self.on_width_change(self.img_width)

    def get_options(self):
        return self.width_spin.value(), self.height_spin.value()


if __name__ == '__main__':
    from qt_material import apply_stylesheet
    from PyQt5.QtWidgets import QApplication


    def on_ok_test():
        print(w.get_new_size())


    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    image_path = "cat.jpg"

    w = ResizeCard(image_path, on_ok=on_ok_test)

    w.show()

    app.exec_()
