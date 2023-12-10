import os

from PyQt5 import QtCore, QtGui
from PyQt5.QtGui import QColor
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QListWidget

from ui.list_widgets.list_item_custom import ListWidgetItemCustomSort


class ImagesWidget(QListWidget):

    def __init__(self, parent, icon_folder):
        super(ImagesWidget, self).__init__(parent)

        self.icon_folder = os.path.join(icon_folder, '..', 'image_status')
        # self.setMouseTracking(True)

        self.icons = {'empty': QIcon(self.icon_folder + "/empty.png"),
                      'in_work': QIcon(self.icon_folder + "/in_work.png"),
                      'approve': QIcon(self.icon_folder + "/approve.png")}

    def addItem(self, text, status=None) -> None:
        """
        Три варианта статуса
            'empty' - еще не начинали
            'in_work' - в работе
            'approve' - завершена работа
        """

        item = ListWidgetItemCustomSort(text)
        if not status:
            status = 'empty'
        item.setIcon(self.icons[status])

        super().addItem(item)

    def set_status(self, status):
        item = self.currentItem()
        if not status:
            status = 'empty'
        item.setIcon(self.icons[status])

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

    def move_to_image_name(self, name):
        for i in range(self.count()):
            if self.item(i).text() == name:
                self.setCurrentRow(i)
                return

    def take_item_by_name(self, name):
        for i in range(self.count()):
            if self.item(i).text() == name:
                return self.takeItem(i)

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
