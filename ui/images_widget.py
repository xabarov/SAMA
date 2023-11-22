from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QListWidget, QListWidgetItem
import os
import re
from difflib import SequenceMatcher


class ListWidgetItemCustomSort(QListWidgetItem):
    def __init__(self, parent):
        super().__init__(parent)

    def get_first_name_part(self, t, delimiter='.'):
        return t.split(delimiter)[0]

    def get_numbers(self, s, with_boundaries=False):
        if with_boundaries:
            return re.findall(r'\b\d+\b', s)
        return re.findall(r'\d+', s)

    def match_size(self, s1, s2):

        match = SequenceMatcher(None, s1, s2).find_longest_match()

        return match.size

    def low_then(self, numbers1, numbers2):

        numbers1 = [int(n) for n in numbers1]
        numbers2 = [int(n) for n in numbers2]

        all_numbers = set(numbers1 + numbers2)
        unique1 = all_numbers - set(numbers2)  # can be [], [3], [2017, 3]
        unique2 = all_numbers - set(numbers1)  # can be [], [4], [2015, 4]

        l1 = len(unique1)
        l2 = len(unique2)
        if l1 == 0:
            return True
        if l2 == 0:
            return False
        if l1 != l2:
            return l1 < l2

        # Одинаковой длины и не пустые:

        return max(unique1) < max(unique2)

    def cut_off_nubmers(self, s, numbers):
        for n in numbers:
            s = s.replace(n, "")

        return s

    def __lt__(self, other):
        try:
            txt_self = self.get_first_name_part(self.text())
            txt_other = self.get_first_name_part(other.text())

            self_numbers = self.get_numbers(txt_self)
            other_numbers = self.get_numbers(txt_other)  # can be ['2017', '3'], ['3'], ['2', '23']

            txt_self_wo_numbers = self.cut_off_nubmers(txt_self, self_numbers).strip(' _')  
            txt_other_wo_numbers = self.cut_off_nubmers(txt_other, other_numbers).strip(' _')

            if txt_other_wo_numbers != txt_self_wo_numbers:
                # Разные основания, например canada_roffle_1, canada_spirit_3
                return QListWidgetItem.__lt__(self, other)

            return self.low_then(self_numbers, other_numbers)

        except:
            return QListWidgetItem.__lt__(self, other)


class ImagesWidget(QListWidget):

    def __init__(self, parent, icon_folder):
        super(ImagesWidget, self).__init__(parent)

        self.icon_folder = os.path.join(icon_folder, '..', 'image_status')
        # self.setMouseTracking(True)

    def addItem(self, text, status=None) -> None:
        """
        Три варианта статуса
            'empty' - еще не начинали
            'in_work' - в работе
            'approve' - завершена работа
        """

        item = ListWidgetItemCustomSort(text)
        if not status or status == 'empty':
            item.setIcon(QIcon(self.icon_folder + "/empty.png"))
        elif status == "in_work":
            item.setIcon(QIcon(self.icon_folder + "/in_work.png"))
        elif status == "approve":
            item.setIcon(QIcon(self.icon_folder + "/approve.png"))
        super().addItem(item)

    def set_status(self, status):
        item = self.currentItem()
        if not status or status == 'empty':
            item.setIcon(QIcon(self.icon_folder + "/empty.png"))
        elif status == "in_work":
            item.setIcon(QIcon(self.icon_folder + "/in_work.png"))
        elif status == "approve":
            item.setIcon(QIcon(self.icon_folder + "/approve.png"))

    def get_next_idx(self):
        if self.count() == 0:
            return -1
        current_idx = self.currentRow()
        return current_idx + 1 if current_idx < self.count() - 1 else 0

    def get_next_name(self):
        next_idx = self.get_next_idx()
        if next_idx == -1:
            return
        return self.item(next_idx).text()

    def get_last_name(self):
        next_idx = self.count() - 1
        if next_idx == -1:
            return
        return self.item(next_idx).text()

    def move_next(self):
        next_idx = self.get_next_idx()
        if next_idx == -1:
            return
        self.setCurrentRow(next_idx)

    def move_last(self):
        next_idx = self.count() - 1
        if next_idx == -1:
            return
        self.setCurrentRow(next_idx)

    def move_to(self, index):
        self.setCurrentRow(index)

    def get_idx_before(self):
        if self.count() == 0:
            return -1
        current_idx = self.currentRow()
        return current_idx - 1 if current_idx > 0 else self.count() - 1

    def get_before_name(self):
        before_idx = self.get_idx_before()
        if before_idx == -1:
            return
        return self.item(before_idx).text()

    def move_before(self):
        before_idx = self.get_idx_before()
        if before_idx == -1:
            return
        self.setCurrentRow(before_idx)
