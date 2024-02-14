import sys

import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QTableWidget, \
    QTableWidgetItem

from ui.custom_widgets.styled_widgets import StyledComboBox

LastStateRole = Qt.UserRole


class CustomItem(QTableWidgetItem):
    def __init__(self, value, is_checked=True):
        super(QTableWidgetItem, self).__init__(value)
        if is_checked:
            self.setCheckState(Qt.Checked)
            self.setData(LastStateRole, self.checkState())


class ExportLabelsList(QWidget):
    def __init__(self, labels, del_name='Удалить', blur_name='Размыть', headers=('Метка', 'Заменить на'),
                 theme='dark_blue.xml', col_width=300):
        super(ExportLabelsList, self).__init__()
        mainlayout = QVBoxLayout()
        self.labels = labels
        self.export_labels = self.labels
        self.del_name = del_name
        self.blur_name = blur_name
        self.headers = headers
        self.setMinimumWidth(400)
        self.theme = theme
        self.col_width = col_width

        self.create_table()
        mainlayout.addWidget(self.table)

        self.setLayout(mainlayout)

    def create_table(self):
        self.table = QTableWidget(len(self.labels), 2)
        for i, label in enumerate(self.labels):
            item = CustomItem(label)
            self.table.setItem(i, 0, item)

            labels = [""]
            self.crete_combo(i, 1, labels, is_active=False)

        self.table.setColumnWidth(0, self.col_width)
        self.table.setColumnWidth(1, self.col_width)
        self.table.setHorizontalHeaderLabels(list(self.headers))
        self.table.cellClicked.connect(self.cell_changed)

    def cell_clicked(self, row, col):
        print(self.get_labels_map())

    def crete_combo(self, row, col, items, is_active=True, current_state=0):
        combo_box = StyledComboBox(theme=self.theme)
        labels = np.array(items)
        combo_box.addItems(labels)
        combo_box.setEnabled(is_active)
        combo_box.setCurrentIndex(current_state)
        self.table.setCellWidget(row, col, combo_box)

    def get_labels_map(self):
        labels_map = {}
        export_labels = []
        for i, label in enumerate(self.labels):

            item = self.table.cellWidget(i, 1)
            cell_text = item.currentText()
            if cell_text == self.del_name:
                labels_map[label] = 'del'
            elif cell_text == self.blur_name:
                labels_map[label] = 'blur'
            elif cell_text == "":
                export_labels.append(label)

        for i, label in enumerate(export_labels):
            labels_map[label] = i

        for i, label in enumerate(self.labels):
            item = self.table.cellWidget(i, 1)
            cell_text = item.currentText()
            if label not in export_labels and cell_text != self.del_name and cell_text != self.blur_name:
                labels_map[label] = export_labels.index(cell_text)

        return labels_map

    def update_cells(self):
        for i, label in enumerate(self.labels):
            if label not in self.export_labels:
                item = self.table.cellWidget(i, 1)
                label_name = item.currentText()
                labels = [self.del_name, self.blur_name]
                labels.extend(self.export_labels)
                current_state = 0
                if label_name in self.export_labels:
                    current_state = np.where(np.array(self.export_labels) == label_name)[0][0] + 2
                elif label_name == self.blur_name:
                    current_state = 1

                self.crete_combo(i, 1, labels, is_active=True, current_state=current_state)


    def cell_changed(self, row, col):
        # clicked on label
        item = self.table.item(row, 0)
        label_name = item.text()
        last_state = item.data(LastStateRole)
        current_state = item.checkState()
        if current_state != last_state:
            if current_state == Qt.Checked:
                self.export_labels.append(label_name)
                labels = [""]
                self.crete_combo(row, 1, labels, is_active=False, current_state=0)

            else:
                self.export_labels = [label for label in self.export_labels if label != label_name]

            self.update_cells()
            item.setData(LastStateRole, current_state)
        print(self.get_labels_map())

if __name__ == '__main__':
    app = QApplication(sys.argv)
    labels = [x for x in 'ABCDEFG']
    export_list = ExportLabelsList(labels)
    export_list.show()

    sys.exit(app.exec_())
