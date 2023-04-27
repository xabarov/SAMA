from PyQt5 import QtCore
from PyQt5.QtWidgets import QLabel


def createLabel(text, heigh=50):
    label = QLabel(text)
    label.setMaximumHeight(heigh)
    label.setAlignment(QtCore.Qt.AlignCenter)
    return label
