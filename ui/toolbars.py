from PyQt5.QtWidgets import QProgressBar, QLabel, QHBoxLayout, QWidget


class ProgressBarToolbar(QWidget):

    def __init__(self, parent, signal=None, right_padding=0):
        super(ProgressBarToolbar, self).__init__(parent)

        self.right_padding = right_padding
        self.on_percent_change = signal
        layout = QHBoxLayout()

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setMinimumWidth(100)

        if self.on_percent_change:
            self.on_percent_change.connect(self.set_percent)

        layout.addWidget(self.progress_bar)
        self.padding_label = QLabel(' ' * self.right_padding)
        layout.addWidget(self.padding_label)
        self.setLayout(layout)

        self.hide_progressbar()

    def set_percent(self, percent):
        self.progress_bar.setValue(percent)
        print(percent)
        if percent >= 100:
            self.hide_progressbar()

    def set_signal(self, signal):
        if signal:
            signal.connect(self.set_percent)

    def get_percent(self):
        return self.progress_bar.value()

    def hide_progressbar(self):
        self.progress_bar.hide()
        self.padding_label.hide()

    def show_progressbar(self):
        self.progress_bar.show()
        self.padding_label.show()
