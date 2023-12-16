from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout, QCheckBox, QSlider
from qtpy.QtWidgets import QApplication

from ui.custom_widgets.two_handle_splitter import TwoHandleSplitter, part_colors
from utils.settings_handler import AppSettings


class TrainTestSplitter(QWidget):

    def __init__(self, parent, width_percent=0.2, colors=part_colors):
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()

        # self.setWindowFlag(Qt.Tool)

        self.use_test = QCheckBox("Использовать test" if self.lang == 'RU' else "Use test")
        self.use_test.setChecked(True)
        self.use_test.clicked.connect(self.on_use_test_clicked)

        self.splitter1 = QSlider(Qt.Orientation.Horizontal)
        self.splitter1.setValue(80)
        self.splitter1.valueChanged.connect(self.on_splitter1_changed)

        self.splitter2 = TwoHandleSplitter(None, colors=colors)
        self.splitter2.valueChanged.splits.connect(self.on_splitter2_changed)

        self.splitter2.slider1.setValue(88)
        self.splitter2.slider2.setValue(28)

        self.label_layout = QHBoxLayout()
        self.label_layout.setContentsMargins(0, 0, 0, 0)
        self.label_layout.setSpacing(0)

        label = QLabel("Задайте соотношения:" if self.lang == 'RU' else "Set proportions:")
        self.label_layout.addWidget(label, stretch=2)

        self.labels = []
        for text, color, split in zip(["Train", "Val", "Test"], colors, self.splitter2.get_splits()):
            label = QLabel(f"{text}: {split:0.1f}")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(f"QLabel {{ background-color: {color}; color: black}}")
            self.labels.append(label)
            self.label_layout.addWidget(label, stretch=1)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addWidget(self.use_test)
        self.mainLayout.addLayout(self.label_layout)
        self.mainLayout.addWidget(self.splitter1)
        self.mainLayout.addWidget(self.splitter2)
        self.splitter1.hide()
        self.setLayout(self.mainLayout)
        size, pos = self.settings.read_size_pos_settings()
        self.setMinimumWidth(int(size.width() * width_percent))

    def on_use_test_clicked(self):
        if self.use_test.isChecked():
            self.labels[-1].show()
            self.splitter1.hide()
            self.splitter2.show()
            self.on_splitter2_changed(self.splitter2.get_splits())
        else:
            self.labels[-1].hide()
            self.splitter1.show()
            self.on_splitter1_changed(self.splitter1.value())
            self.splitter2.hide()

        print(self.get_splits())

    def get_splits(self):
        if self.use_test.isChecked():
            return self.splitter2.get_splits()
        return self.splitter1.value(), 100 - self.splitter1.value()

    def on_splitter2_changed(self, splits):
        for text, label, split in zip(["Train", "Val", "Test"], self.labels, splits):
            label.setText(f"{text}: {split: 0.1f}")

    def on_splitter1_changed(self, value):
        splits = value, 100 - value
        for text, label, split in zip(["Train", "Val"], self.labels, splits):
            label.setText(f"{text}: {split: 0.1f}")


if __name__ == '__main__':
    from qt_material import apply_stylesheet

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')
    slider = TrainTestSplitter(None)
    slider.show()

    app.exec_()
