import os

from ui.custom_widgets.card import Card
from ui.dialogs.export_steps.export_labels_list_view import ExportLabelsList
from ui.dialogs.export_steps.preprocess_blocks.preprocess_block import PreprocessBlock
from utils.settings_handler import AppSettings


class ModifyCard(Card):
    """
    Карточка этапа обработки "Изменение классов".
    Картинка -  list.png
    Две кнопки - удалить и редактировать
    По нажатии на кнопку редактировать открывается окно редактирования классов (виджет ModifyClassesStep)
    """

    def __init__(self, labels, min_width=250, min_height=150, max_width=100, on_ok=None, on_delete=None):

        """
        labels - имена клоссов
        """

        name = "Modify Classes"
        self.icons_path = os.path.join(os.path.dirname(__file__), 'preprocess_icons')

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        self.labels = labels

        if self.lang == "RU":
            sub_text = "Отредактируйте\nклассы"
        else:
            sub_text = "Change\nexport labels"

        self.on_ok = on_ok

        super(ModifyCard, self).__init__(None, text=name,
                                         path_to_img=os.path.join(self.icons_path, 'list.png'),
                                         min_width=min_width, min_height=min_height, max_width=max_width,
                                         on_edit_clicked=self.on_modify_class, is_del_button=True, is_edit_button=True,
                                         on_del=on_delete, sub_text=sub_text)

        self.labels_map = None

    def get_labels_map(self):
        """
        Возвращает словарь с именами меток
        """
        return self.labels_map

    def on_modify_class(self):
        """
        При нажатии на кнопку Edit карточки "Modify Classes"
        Открытие окна редактирования классов
        """

        self.modify_class_widget = ModifyClassesStep(self.labels, min_width=780, minimum_expand_height=300,
                                                     theme=self.theme, on_ok=self.on_modify_classes_ok,
                                                     on_cancel=None)
        self.modify_class_widget.show()

    def on_modify_classes_ok(self):
        labels_map = self.modify_class_widget.get_options()
        self.labels_map = labels_map
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

        self.set_sub_text(sub_text)

        if self.on_ok:
            # Если в конструкторе передан хук
            self.on_ok()


class ModifyClassesStep(PreprocessBlock):
    """
    Окно редактирования экспортируемых классов.
    Есть возможность заменить класс на другой, удалить или заблюрить
    """

    def __init__(self, label_names, min_width=780, minimum_expand_height=300, theme='dark_blue.xml', on_ok=None,
                 on_cancel=None):
        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        name = "modify_classes"
        title = 'Выбрать имена классов для экспорта' if self.lang == 'RU' else 'Choose labels for export'

        super().__init__(name, options=None, parent=None, title=title, min_width=min_width,
                         minimum_expand_height=minimum_expand_height, icon_name="list.png")

        del_name = 'Удалить' if self.lang == 'RU' else 'Delete'
        blur_name = 'Размыть' if self.lang == 'RU' else 'Blur'
        headers = ('Метка', 'Заменить на') if self.lang == 'RU' else ('Label', 'Replace to')
        self.export_labels_list = ExportLabelsList(labels=label_names, theme=theme, del_name=del_name,
                                                   blur_name=blur_name, headers=headers)

        self.layout.addWidget(self.export_labels_list)

        self.create_buttons(on_ok, on_cancel)
        self.setMinimumWidth(min_width)

    def get_options(self):
        return self.export_labels_list.get_labels_map()


if __name__ == '__main__':
    from qt_material import apply_stylesheet
    from PyQt5.QtWidgets import QApplication


    def on_ok_test():
        print(w.get_labels_map())


    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')

    labels = ['F-16', 'F-35', 'C-130', 'C-17']

    w = ModifyClassesStep(labels, on_ok=on_ok_test)

    w.show()

    app.exec_()
