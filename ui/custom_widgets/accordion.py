from PyQt5.QtWidgets import QWidget, QVBoxLayout


class Accordion(QWidget):

    def __init__(self, widgets):
        """
        Аккордион
        """
        super().__init__(None)
        self.widgets = widgets

        self.layout = QVBoxLayout()
        for w in widgets:
            self.layout.addWidget(w, stretch=1)

        self.show_only_selected_idx(0)
        self.set_hide_show()

        self.setLayout(self.layout)

    def set_hide_show(self):
        for i, widget in enumerate(self.widgets):
            widget.head.clicked.connect(lambda state, x=i: self.show_only_selected_idx(x))

    def show_only_selected_idx(self, idx):
        for i in range(len(self.widgets)):
            if i == idx:
                self.widgets[i].body.show()
                self.widgets[i].setMaximumHeight(400)
            else:
                self.widgets[i].body.hide()
                self.widgets[i].setMaximumHeight(100)
