import os

from PyQt5.QtWidgets import QApplication

from ui.custom_widgets.cards_field import CardsField
from ui.dialogs.export_steps.preprocess_blocks.modify_classes import ModifyCard
from ui.dialogs.export_steps.preprocess_blocks.filter_null import FilterNullCard
from ui.dialogs.export_steps.preprocess_blocks.resize import ResizeCard
from ui.dialogs.export_steps.preprocess_blocks.preprocess_btn_options import PreprocessBtnOptions
from utils.settings_handler import AppSettings


class PreprocessStep(CardsField):
    """
    Окно добавления и корректировки этапов предобработки данных.
    Представляет собой кнопку "+" и карточки этапов обработки
    """

    def __init__(self, labels, test_image_path, cards=None, block_width=150,
                 block_height=150, min_height=260):

        """
        labels - список имен меток
        cards - предварительно добавленные карточки предоработки. По умолчанию - без карточек

        block_width, block_height     - максимальные ширина и высота корточки
        min_height - минимальноя высота всего блока

        """

        settings = AppSettings()
        lang = settings.read_lang()

        if lang == 'RU':
            label_text = "Нажмите кнопку, чтобы добавить предобработку"
        else:
            label_text = "Click button to add preprocess option"

        super(PreprocessStep, self).__init__(cards=cards, block_width=block_width,
                                             block_height=block_height, label_text=label_text)

        self.labels = labels  # Имена меток
        self.test_image_path = test_image_path
        self.icons_path = os.path.join(os.path.dirname(__file__), 'preprocess_blocks', 'preprocess_icons')

        # Кнопка +
        self.add_button.clicked.connect(self.on_add)

        # Окно с выбором этапов обработки
        self.preprocess_options = PreprocessBtnOptions(None, on_ok=self.on_preprocess_ok)

        # "modify_classes", "resize", "filter_null", "tile", "auto_contrast", "grayscale"
        self.option_names = self.preprocess_options.option_names

        # Параметры выбранных этапов обработки
        self.option_parameters = {}

        # Кол-во отображаемых этапов
        self.num_of_options = 0

        self.setMinimumHeight(min_height)

    def on_add(self):
        """
        По нажатии на кнопку +
        Открываем окно с выбором этапов
        """
        self.preprocess_options.reset()
        for op in self.option_parameters.keys():
            self.preprocess_options.set_active_by_option(op)
        self.preprocess_options.show()

    def on_preprocess_ok(self):
        """
        При нажатии ОК в выборе этапов обработки
        """
        new_options_names = self.preprocess_options.get_options()
        self.preprocess_options.hide()

        new_names = set(new_options_names) - set(self.option_parameters.keys())

        for name in new_names:
            if name == "modify_classes":
                self.add_modify_card()
                self.option_parameters["modify_classes"] = {}
                self.num_of_options += 1
            if name == "filter_null":
                self.add_filter_null_card()
                self.option_parameters["filter_null"] = {}
                self.num_of_options += 1
            if name == "resize":
                self.add_resize_card()
                self.option_parameters["resize"] = {}
                self.num_of_options += 1

    def add_resize_card(self):
        self.resize_card = ResizeCard(test_image_path=self.test_image_path, on_ok=self.get_resize_options,
                                      on_delete=self.delete_resize)
        self.add_exist_card(self.resize_card)

    def get_resize_options(self):
        self.option_parameters["resize"] = self.resize_card.get_new_size()

    def delete_resize(self):
        if "resize" in self.option_parameters.keys():
            del self.option_parameters["resize"]
            self.resize_card.hide()
            self.num_of_options -= 1
            self.show_text_label_if_needed()

        print(self.option_parameters)

    def add_filter_null_card(self):
        self.filter_null_card = FilterNullCard(on_delete=self.delete_filter_null)
        self.option_parameters["filter_null"] = True
        self.add_exist_card(self.filter_null_card)

    def delete_filter_null(self):
        """
        При нажатии на кнопку DEL карточки "Filter Null"
        """

        if "filter_null" in self.option_parameters.keys():
            del self.option_parameters["filter_null"]
            self.filter_null_card.hide()
            self.num_of_options -= 1
            self.show_text_label_if_needed()

        print(self.option_parameters)

    def add_modify_card(self):
        """
        Добавление карточки редактирования экспортируемых классов
        """
        self.modify_card = ModifyCard(self.labels, on_delete=self.delete_modify, on_ok=self.get_modify_options)
        self.add_exist_card(self.modify_card)

    def get_modify_options(self):
        self.option_parameters["modify_classes"] = self.modify_card.get_labels_map()

    def delete_modify(self):
        """
        При нажатии на кнопку DEL карточки "Modify Classes"
        """

        if "modify_classes" in self.option_parameters.keys():
            del self.option_parameters["modify_classes"]
            self.modify_card.hide()
            self.num_of_options -= 1
            self.show_text_label_if_needed()

        print(self.option_parameters)

    def show_text_label_if_needed(self):
        """
        Показать текст "Нажмите кнопку, чтобы добавить предобработку", если нет выбранных карточек
        """
        if self.num_of_options == 0:
            self.text_label.show()

    def get_params(self):
        return self.option_parameters


if __name__ == '__main__':
    from qt_material import apply_stylesheet

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')
    labels = ['F-16', 'F-35', 'C-130', 'C-17']

    slider = PreprocessStep(labels)
    slider.show()

    app.exec_()
