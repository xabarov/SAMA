from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QWidget, QVBoxLayout, QHBoxLayout
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QSlider

from utils.settings_handler import AppSettings
from ui.signals_and_slots import SliderSplitsConnection

part_colors = ('#FD5B03',
               '#92F22A',
               '#83D6DE')


class TwoHandleSplitter(QWidget):

    def __init__(self, parent, colors=part_colors):
        super().__init__(parent)

        self.settings = AppSettings()

        self.valueChanged = SliderSplitsConnection()

        self.slider1_stretch = 80
        self.slider2_stretch = 20

        self.sliders_layout = QHBoxLayout()
        self.slider1 = QSlider(Qt.Orientation.Horizontal)
        self.slider1.setRange(0, 100)
        self.slider1.valueChanged.connect(self.value_changed)

        QSS1 = f"QSlider {{ margin: 0; padding: 0; }} \n QSlider::add-page:horizontal {{ background: {colors[1]};}}\n"
        QSS1 += f"QSlider::sub-page:horizontal {{ background: {colors[0]}; }}"

        QSS2 = f"QSlider {{ margin: 0; padding: 0; }} \n QSlider::add-page:horizontal {{ background: {colors[2]};}}\n"
        QSS2 += f"QSlider::sub-page:horizontal {{ background: {colors[1]}; }}"

        self.slider1.setStyleSheet(QSS1)

        self.slider1.setSingleStep(1)

        self.slider2 = QSlider(Qt.Orientation.Horizontal)
        self.slider2.setRange(0, 100)
        self.slider2.setValue(1)
        self.slider2.setSingleStep(1)
        self.slider2.setStyleSheet(QSS2)

        self.slider2.valueChanged.connect(self.value_changed)
        self.sliders_layout.addWidget(self.slider1, stretch=self.slider1_stretch)
        self.sliders_layout.addWidget(self.slider2, stretch=self.slider2_stretch)
        self.sliders_layout.setContentsMargins(0, 0, 0, 0)
        self.sliders_layout.setSpacing(0)
        self.slider1.setValue(88)
        self.slider2.setValue(28)

        self.mainLayout = QVBoxLayout()
        self.mainLayout.addLayout(self.sliders_layout)
        self.setLayout(self.mainLayout)

    def calc_splits(self):
        s1 = self.slider1.value()
        s2 = self.slider2.value()
        self.first = s1 * self.slider1_stretch / 100
        self.second = (100 - s1) * self.slider1_stretch / 100 + s2 * self.slider2_stretch / 100
        self.third = (100 - s2) * self.slider2_stretch / 100

    def get_splits(self):
        return self.first, self.second, self.third


    def on_slider1_max(self):
        self.slider1_stretch += 1
        self.slider2_stretch -= 1
        self.sliders_layout.setStretch(0, self.slider1_stretch)
        self.sliders_layout.setStretch(1, self.slider2_stretch)
        self.slider1.setValue(self.slider1.value() - 1)
        self.slider2.setValue(self.slider2.value() + 1)

    def on_slider2_min(self):
        self.slider1_stretch -= 1
        self.slider2_stretch += 1
        self.sliders_layout.setStretch(0, self.slider1_stretch)
        self.sliders_layout.setStretch(1, self.slider2_stretch)
        self.slider2.setValue(self.slider2.value() + 1)

    def value_changed(self, value):
        self.calc_splits()
        s1 = self.slider1.value()
        s2 = self.slider2.value()
        if s1 == 100:
            self.on_slider1_max()
        elif s2 == 0:
            self.on_slider2_min()

        self.valueChanged.splits.emit(self.get_splits())
        # summ = self.first + self.second + self.third
        # print(f"Train {self.first :0.2f} test {self.second :0.2f} val {self.third :0.2f}. Summ {summ:0.2f}")
        # print(self.get_splits())


if __name__ == '__main__':
    from qt_material import apply_stylesheet

    app = QApplication([])
    apply_stylesheet(app, theme='dark_blue.xml')
    slider = TwoHandleSplitter(None)
    slider.show()

    app.exec_()
