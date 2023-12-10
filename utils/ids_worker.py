from PySide2 import QtCore
from ui.signals_and_slots import LoadIdProgress


class IdsSetterWorker(QtCore.QThread):

    def __init__(self, images_data, percent_max=100):
        super(IdsSetterWorker, self).__init__()
        self.load_ids_conn = LoadIdProgress()
        self.images_data = images_data
        self.percent_max = percent_max
        self.labels_size = 0

    def run(self):
        self.load_ids_conn.percent.emit(0)
        i = 0
        self.labels_size = 0
        for im_name, im in self.images_data.items():
            self.labels_size += len(im['shapes'])
            i += 1
            self.load_ids_conn.percent.emit(int(self.percent_max * (i + 1) / len(self.images_data)))
        self.load_ids_conn.percent.emit(100)

    def get_labels_size(self):
        return self.labels_size
