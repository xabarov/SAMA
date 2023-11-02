from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QListWidget, QListWidgetItem


class ImagesWidget(QListWidget):

    def __init__(self, parent, icon_folder):
        super(ImagesWidget, self).__init__(parent)
        self.icon_folder = icon_folder
        # self.setMouseTracking(True)

    def addItem(self, text, status=None) -> None:
        """
        Три варианта статуса
            'empty' - еще не начинали
            'in_work' - в работе
            'approve' - завершена работа
        """

        item = QListWidgetItem(text)
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

    def move_next(self):
        next_idx = self.get_next_idx()
        if next_idx == -1:
            return
        self.setCurrentRow(next_idx)

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
