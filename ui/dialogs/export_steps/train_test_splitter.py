import numpy as np
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QLabel, QWidget, QVBoxLayout, QHBoxLayout, QSlider
from PyQt5.QtWidgets import QApplication

from ui.custom_widgets.styled_widgets import StyledComboBox
from ui.custom_widgets.two_handle_splitter import TwoHandleSplitter, part_colors
from utils.settings_handler import AppSettings


class TrainTestSplitter(QWidget):

    def __init__(self, parent, width_percent=0.2, colors=part_colors):
        super().__init__(parent)

        self.settings = AppSettings()
        self.lang = self.settings.read_lang()
        theme = self.settings.read_theme()

        # self.setWindowFlag(Qt.Tool)

        self.variants_combo = StyledComboBox(theme=theme)
        if self.lang == 'RU':
            self.variants = np.array(
                ["разбить на train/val/test", "разбить на train/val", "только в train", "только в val",
                 "только в test"])
        else:
            self.variants = np.array(["split into train/val/test", "split into train/val", "only train", "only val",
                                      "only test"])
        self.variants_combo.addItems(self.variants)
        self.variants_combo.currentIndexChanged.connect(self.on_variant_change)

        # способ группировки
        self.group_combo = StyledComboBox(theme=theme)
        if self.lang == 'RU':
            self.group_variants = np.array(
                ["случайно", "по близости имен", "по близости изображений"])
        else:
            self.group_variants = np.array(["random", "by names similarity", "by images similarity"])

        self.group_combo.addItems(self.group_variants)
        self.group_combo.currentIndexChanged.connect(self.on_group_combo_change)

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

        self.proportions_label = QLabel("Задайте соотношения:" if self.lang == 'RU' else "Set proportions:")
        self.label_layout.addWidget(self.proportions_label, stretch=2)

        self.labels = []

        for text, color, split in zip(["Train", "Val", "Test"], colors, self.splitter2.get_splits()):
            label = QLabel(f"{text}: {split:0.1f}")
            label.setAlignment(Qt.AlignCenter)
            label.setStyleSheet(f"QLabel {{ background-color: {color}; color: black}}")
            self.labels.append(label)
            self.label_layout.addWidget(label, stretch=1)

        self.mainLayout = QVBoxLayout()

        combo_lay = QHBoxLayout()
        combo_lay.addWidget(self.variants_combo)
        combo_lay.addWidget(self.group_combo)
        self.mainLayout.addLayout(combo_lay)

        self.mainLayout.addLayout(self.label_layout)
        self.mainLayout.addWidget(self.splitter1)
        self.mainLayout.addWidget(self.splitter2)
        self.splitter1.hide()
        self.setLayout(self.mainLayout)
        size, pos = self.settings.read_size_pos_settings()
        self.setMinimumWidth(int(size.width() * width_percent))

    def on_variant_change(self, idx):
        if idx == 0:  # train/val/test
            for lbl in self.labels:
                lbl.show()
            self.proportions_label.show()
            self.splitter1.hide()
            self.splitter2.show()
            self.on_splitter2_changed(self.splitter2.get_splits())
        elif idx == 1:  # train/val
            for lbl in self.labels:
                lbl.show()
            self.proportions_label.show()
            self.labels[-1].hide()
            self.splitter1.show()
            self.on_splitter1_changed(self.splitter1.value())
            self.splitter2.hide()
        else:
            for lbl in self.labels:
                lbl.hide()
            self.proportions_label.hide()
            self.splitter1.hide()
            self.splitter2.hide()

    def on_group_combo_change(self):
        print(f"Similarity {self.group_combo.currentText()}")

    def get_idx_text_variant(self):
        return self.variants_combo.currentIndex(), self.variants_combo.currentText()

    def get_idx_text_sim(self):
        """Idx and name of similarity"""
        return self.group_combo.currentIndex(), self.group_combo.currentText()

    def get_splits(self):
        idx = self.variants_combo.currentIndex()
        if idx == 0:
            return self.splitter2.get_splits()
        elif idx == 1:
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
