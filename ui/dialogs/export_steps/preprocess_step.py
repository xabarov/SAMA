from PyQt5.QtWidgets import QWidget, QVBoxLayout, QCheckBox, QPushButton
from qtpy.QtWidgets import QApplication

from ui.dialogs.export_steps.export_labels_list_view import ExportLabelsList
from utils.settings_handler import AppSettings
from ui.custom_widgets.cards_field import CardsField
from ui.custom_widgets.card import Card
from ui.dialogs.export_steps.preprocess_blocks.preprocess_btn_options import PreprocessBtnOptions
import os
from ui.dialogs.export_steps.preprocess_blocks.modify_classes import ModifyClassesStep


class PreprocessStep(CardsField):
    def __init__(self, parent, labels, cards=None, theme='dark_blue.xml', block_width=150,
                 block_height=150):

        settings = AppSettings()
        lang = settings.read_lang()

        if lang == 'RU':
            label_text = "Нажмите кнопку, чтобы добавить предобработку"
        else:
            label_text = "Click button to add preprocess option"

        super(PreprocessStep, self).__init__(parent, cards=cards, theme=theme, block_width=block_width,
                                             block_height=block_height, label_text=label_text)

        self.labels = labels
        self.icons_path = os.path.join(os.path.dirname(__file__), 'preprocess_blocks', 'preprocess_icons')

        self.add_button.clicked.connect(self.on_add)
        self.preprocess_options = PreprocessBtnOptions(None, on_ok=self.on_preprocess_ok)

        # "modify_classes", "resize", "filter_null", "tile", "auto_contrast", "grayscale"
        self.option_names = self.preprocess_options.option_names

        self.option_parameters = {}
        self.num_of_options = 0

    def on_add(self):
        self.preprocess_options.reset()
        for op in self.option_parameters.keys():
            self.preprocess_options.set_active_by_option(op)
        self.preprocess_options.show()

    def add_modify_card(self):
        if self.lang == "RU":
            sub_text = "Отредактируйте экспортируемые классы"
        else:
            sub_text = "Press edit button to change export labels"

        self.modify_card = Card(None, text="Modify Classes", path_to_img=os.path.join(self.icons_path, 'list.png'),
                                min_width=250, min_height=150, theme=self.theme,
                                on_edit_clicked=self.on_modify_class, is_del_button=True, is_edit_button=True,
                                on_del=self.delete_modify, sub_text=sub_text)
        self.add_exist_card(self.modify_card)

    def on_preprocess_ok(self):
        new_options_names = self.preprocess_options.get_options()
        self.preprocess_options.hide()

        new_names = set(new_options_names) - set(self.option_parameters.keys())

        for name in new_names:
            if name == "modify_classes":
                self.add_modify_card()
                self.option_parameters["modify_classes"] = {}
                self.num_of_options += 1

    def on_modify_class(self):
        self.modify_class_widget = ModifyClassesStep(None, self.labels, min_width=780, minimum_expand_height=300,
                                                     theme=self.theme, on_ok=self.on_modify_classes_ok,
                                                     on_cancel=None)
        self.modify_class_widget.show()

    def show_text_label_if_needed(self):
        if self.num_of_options == 0:
            self.text_label.show()

    def delete_modify(self):
        if "modify_classes" in self.option_parameters.keys():
            del self.option_parameters["modify_classes"]
            self.modify_card.hide()
            self.num_of_options -= 1
            self.show_text_label_if_needed()

        print(self.option_parameters)

    def on_modify_classes_ok(self):
        labels_map = self.modify_class_widget.get_labels_map()
        self.option_parameters["modify_classes"] = labels_map
        self.modify_class_widget.hide()

        del_num = 0
        blur_num = 0
        change_num = 0
        tek_num = 0
        for label, value in labels_map.items():
            if value == 'del':
                del_num += 1
                continue
            elif value == 'blur':
                blur_num += 1
                continue
            elif tek_num != value:
                change_num += 1
            tek_num += 1

        if change_num == 0 and blur_num == 0 and del_num == 0:
            is_change = False
        else:
            is_change = True

        if self.lang == 'RU':
            if not is_change:
                sub_text = f"Правки отсутствуют"
            else:
                sub_text = ""
                if change_num:
                    sub_text += f" Число замен: {change_num}. "
                if del_num:
                    sub_text += f" Удалено: {del_num}. "
                if blur_num:
                    sub_text += f" Заблюрено: {blur_num}. "
        else:
            if not is_change:
                sub_text = f"No changes"
            else:
                sub_text = ""
                if change_num:
                    sub_text += f" Changed: {change_num}. "
                if del_num:
                    sub_text += f" Deleted: {del_num}. "
                if blur_num:
                    sub_text += f" Blurred: {blur_num}. "

        self.modify_card.set_sub_text(sub_text)

        print(self.option_parameters)

    def get_params(self):
        return self.option_parameters


if __name__ == '__main__':
    from qt_material import apply_stylesheet

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')
    labels = ['F-16', 'F-35', 'C-130', 'C-17']

    slider = PreprocessStep(None, labels)
    slider.show()

    app.exec_()
