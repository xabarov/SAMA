import copy
import sys

import numpy as np
from PyQt5 import QtCore
from PyQt5.QtGui import QIcon
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QTableWidget, QTableWidgetItem, QHeaderView, QWidget, QHBoxLayout
from qt_material import apply_stylesheet

from utils.settings_handler import AppSettings


class BalanceTable(QWidget):

    def __init__(self, parent=None, width_percent=0.6, height_percent=0.5, project_data=None, test_mode=False):
        super(BalanceTable, self).__init__(parent)
        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        self.icon_folder = self.settings.get_icon_folder()

        self.setWindowIcon(QIcon(self.icon_folder + "/bar-chart.png"))

        self.table = QTableWidget()
        self.layout = QHBoxLayout()

        if not test_mode:
            self.setWindowFlag(QtCore.Qt.Tool)

        if self.lang == 'RU':
            title = "Информация о разметке"
        else:
            title = "Labeling info"
        self.setWindowTitle(title)

        if not project_data:
            return

        self.table.setColumnCount(5)

        self.project_data = copy.copy(project_data)
        classes_num = len(project_data['labels'])

        self.table.setRowCount(classes_num)
        if self.lang == 'RU':
            self.table.setHorizontalHeaderLabels(
                ['Имя метки', 'Кол-во меток', '% от общего числа', 'Средний размер, px',
                 'СКО размера, px'])
        else:
            self.table.setHorizontalHeaderLabels(['Label name', 'Count', '%', 'Mean size, px',
                                                  'Std size, px'])
        for col in range(5):
            self.table.horizontalHeader().setSectionResizeMode(col, QHeaderView.Stretch)

        stat = self.calc_stat()

        row = 0
        for lbl in stat:
            self.add_not_editable(lbl, row, 0)
            self.add_not_editable(str(stat[lbl]['count']), row, 1)
            self.add_not_editable(f"{stat[lbl]['percent']:0.3f}", row, 2)
            self.add_not_editable(f"{stat[lbl]['size']['mean']:0.3f}", row, 3)
            self.add_not_editable(f"{stat[lbl]['size']['std']:0.3f}", row, 4)
            row += 1

        size, pos = self.settings.read_size_pos_settings()
        self.setMinimumWidth(int(size.width() * width_percent))
        self.setMinimumHeight(int(size.height() * height_percent))
        self.layout.addWidget(self.table)
        self.setLayout(self.layout)

    def add_not_editable(self, text, row, col):
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~QtCore.Qt.ItemIsEditable)
        self.table.setItem(row, col, item)

    def calc_stat(self):

        labels = self.project_data['labels']
        stats = {lbl: {'count': 0, 'percent': 0, 'size': {'mean': 0, 'second': 0, 'std': 0}} for lbl in labels}

        total_label_count = 0
        size_mass = {lbl: [] for lbl in labels}

        for im in self.project_data['images'].values():
            for shape in im['shapes']:
                cls_num = shape['cls_num']
                if cls_num > len(labels) - 1:
                    continue
                label_name = labels[cls_num]

                xs = [x for x, y in shape['points']]
                ys = [y for x, y in shape['points']]
                width = max(xs) - min(xs)
                height = max(ys) - min(ys)
                size = max(width, height)

                size_mass[label_name].append(size)

                stats[label_name]['count'] += 1
                total_label_count += 1

        for label_name in labels:
            stats[label_name]['percent'] = float(stats[label_name]['count']) / total_label_count
            mass = np.array(size_mass[label_name])
            if len(mass) > 0:
                stats[label_name]['size']['mean'] = mass.mean()
                stats[label_name]['size']['std'] = mass.std()
        return stats


if __name__ == '__main__':
    import ujson


    def load(json_path):
        with open(json_path, 'r', encoding='utf8') as f:
            return ujson.load(f)


    app = QtWidgets.QApplication(sys.argv)

    apply_stylesheet(app, theme='dark_blue.xml', invert_secondary=False)

    data = load("D:\python\\aia_git\\ai_annotator\projects\\fair1m_airplanes\\train_proj2.json")
    w = BalanceTable(project_data=data, test_mode=True)
    w.show()
    sys.exit(app.exec_())
