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


