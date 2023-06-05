from PyQt5.QtWidgets import QWidget, QProgressBar, QVBoxLayout
from PyQt5.QtCore import Qt


class ProgressWindow(QWidget):
    def __init__(self, parent, title=None, width=300, height=80):
        """
        Progress bar
        """
        super().__init__(parent)

        if title:
            self.setWindowTitle(title)

        self.setWindowFlag(Qt.Tool)

        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setVisible(False)

        self.mainLayout = QVBoxLayout()

        self.mainLayout.addWidget(self.progress_bar)

        self.setLayout(self.mainLayout)

        self.resize(int(width), int(height))
        self.show()

    def set_progress(self, progress_value):
        if progress_value != 100:
            self.progress_bar.setVisible(True)
            self.progress_bar.setValue(progress_value)
        else:
            self.progress_bar.setVisible(False)
            self.close()
