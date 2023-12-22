import os

from ui.custom_widgets.card import Card
from utils.settings_handler import AppSettings


class FilterNullCard(Card):
    """
    Карточка этапа обработки "Удалить неразмеченные".
    Картинка -  null.png
    Одна кнопка - удалить
    """

    def __init__(self, min_width=250, min_height=150, max_width=100, on_delete=None):

        """
        labels - имена клоссов
        """

        name = "Filter Null"
        self.icons_path = os.path.join(os.path.dirname(__file__), 'preprocess_icons')

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        if self.lang == "RU":
            sub_text = "Удалить\nнеразмеченные"
        else:
            sub_text = "Filter Null"

        super(FilterNullCard, self).__init__(None, text=name,
                                             path_to_img=os.path.join(self.icons_path, 'null.png'),
                                             min_width=min_width, min_height=min_height, max_width=max_width,
                                             is_del_button=True, is_edit_button=False, on_del=on_delete,
                                             sub_text=sub_text)
