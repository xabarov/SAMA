from PySide2 import QtCore
import requests
import json


class SAMImageSetterClient(QtCore.QThread):
    def __init__(self, server):
        super(SAMImageSetterClient, self).__init__()
        self.server = server

    def set_image(self, image_path):
        self.image_path = image_path

    def run(self):
        url = f'{self.server}/sam_set_image'
        files = {'file': open(self.image_path, 'rb')}

        self.response = requests.post(url, files=files)


class SAMPredictByPointsClient(QtCore.QThread):
    def __init__(self, server):
        super(SAMPredictByPointsClient, self).__init__()
        self.server = server
        self.points_mass = None
        self.points = None
        self.labels = None

    def set_inputs(self, points, labels):
        self.points = points.tolist()
        self.labels = labels.tolist()

    def run(self):
        url = f'{self.server}/sam_points'

        sam_res = requests.post(url, json={'input_points': self.points, 'input_labels': self.labels})

        self.points_mass = json.loads(sam_res.text)


class SAMPredictByMaskClient(QtCore.QThread):
    def __init__(self, server):
        super(SAMPredictByMaskClient, self).__init__()
        self.server = server
        self.points_mass = None
        self.input_box = None

    def set_input_box(self, input_box):
        self.input_box = input_box.tolist()

    def run(self):
        url = f'{self.server}/sam_box'

        sam_res = requests.post(url, json={'input_box': self.input_box})

        self.points_mass = json.loads(sam_res.text)
