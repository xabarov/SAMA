from PySide2 import QtCore
import json

from collections import namedtuple
from utils.help_functions import is_dicts_equals

SavedData = namedtuple('SavedData', ('filename', 'json_data'))
from ui.signals_and_slots import ProjectSaveLoadConn


class SaverLoaderWorker(QtCore.QThread):

    def __init__(self):
        super(SaverLoaderWorker, self).__init__()
        self.queue_save = []
        self.queue_load = []
        self.last_version = SavedData(filename='', json_data={})
        self.mode = 'save'
        self.on_save = ProjectSaveLoadConn()
        self.on_load = ProjectSaveLoadConn()

    def enqueue_save(self, save_name, json_data):
        s = SavedData(filename=save_name, json_data=json_data)
        self.queue_save.append(s)

    def enqueue_load(self, json_name):
        self.queue_load.append(json_name)

    def change_mode(self, mode):
        # 'save' or 'load'
        self.mode = mode

    def run(self):
        if self.mode == 'save':
            if len(self.queue_save) > 0:
                last_data = self.queue_save[-1]
                self.queue_save.clear()

                self.last_version = last_data
                with open(last_data.filename, 'w', encoding='utf8') as f:
                    json.dump(last_data.json_data, f)
                self.on_save.on_finished.emit(True)

        else:
            if len(self.queue_load) > 0:
                last_json = self.queue_load[-1]
                self.queue_load.clear()

                with open(last_json, 'r', encoding='utf8') as f:
                    self.last_version = SavedData(filename=last_json, json_data=json.load(f))

                self.on_load.on_finished.emit(True)
