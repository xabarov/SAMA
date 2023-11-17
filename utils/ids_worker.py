from PySide2 import QtCore
from ui.signals_and_slots import LoadIdProgress


class IdsSetterWorker(QtCore.QThread):

    def __init__(self, images_data):
        super(IdsSetterWorker, self).__init__()
        self.load_ids_conn = LoadIdProgress()
        self.labels_ids = []
        self.images_data = images_data

    def run(self):
        self.labels_ids = []
        self.load_ids_conn.percent.emit(0)
        i = 0
        for im_name, im in self.images_data.items():
            for shape in im['shapes']:
                if shape['id'] not in self.labels_ids:
                    self.labels_ids.append(shape['id'])
            i += 1
            self.load_ids_conn.percent.emit(int(100 * (i + 1) / len(self.images_data)))

        self.load_ids_conn.percent.emit(100)

    def get_labels_ids(self):
        return self.labels_ids
