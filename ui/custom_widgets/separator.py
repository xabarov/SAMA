from PyQt5.QtWidgets import QFrame, QSizePolicy


class Separator(QFrame):
    def __init__(self):
        super(Separator, self).__init__(None)
        self.setFrameShape(QFrame.HLine)
        self.setSizePolicy(QSizePolicy.Minimum, QSizePolicy.Expanding)
        self.setLineWidth(3)
