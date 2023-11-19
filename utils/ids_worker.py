from PySide2 import QtCore
from ui.signals_and_slots import LoadIdProgress


class IdsSetterWorker(QtCore.QThread):

    def __init__(self, images_data, percent_max=100):
        super(IdsSetterWorker, self).__init__()
        self.load_ids_conn = LoadIdProgress()
        self.labels_ids = []
        self.images_data = images_data
        self.percent_max = percent_max

    def run(self):
        self.load_ids_conn.percent.emit(0)
        i = 0
        shapes_num = 0
        for im_name, im in self.images_data.items():
            shapes_num += len(im['shapes'])
            i += 1
            self.load_ids_conn.percent.emit(int(self.percent_max * (i + 1) / len(self.images_data)))
        self.labels_ids = [idx for idx in range(shapes_num)]
        self.load_ids_conn.percent.emit(100)

    def get_labels_ids(self):
        return self.labels_ids
