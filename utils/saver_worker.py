from PySide2 import QtCore
import json

from collections import namedtuple

SavedData = namedtuple('SavedData', ('filename', 'json_data'))


class SaverWorker(QtCore.QThread):

    def __init__(self):
        super(SaverWorker, self).__init__()
        self.queue = []

    def enqueue(self, save_name, json_data):
        s = SavedData(filename=save_name, json_data=json_data)
        self.queue.append(s)

    def run(self):
        if len(self.queue) > 0:
            last_data = self.queue[-1]
            self.queue.clear()

            with open(last_data.filename, 'w', encoding='utf8') as f:
                json.dump(last_data.json_data, f)
