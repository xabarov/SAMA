import re
from difflib import SequenceMatcher

from PyQt5.QtWidgets import QListWidgetItem


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