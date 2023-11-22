from PyQt5.QtWidgets import QListWidget

from ui.list_widgets.list_item_custom import ListWidgetItemCustomSort


class LabelsWidget(QListWidget):

    def __init__(self, parent):
        super(LabelsWidget, self).__init__(parent)

    def addItem(self, text, status=None) -> None:
        item = ListWidgetItemCustomSort(text)
        super().addItem(item)

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
