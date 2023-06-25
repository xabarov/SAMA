from PyQt5 import QtWidgets
import sys
from qt_material import apply_stylesheet

if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    apply_stylesheet(app, theme='dark_blue.xml')

    w = DetectorClient()
    w.show()
    sys.exit(app.exec_())