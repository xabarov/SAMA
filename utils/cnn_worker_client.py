from PySide2 import QtCore
import requests
import json


def decode(mask_results):
    result = []
    for mask_res in mask_results:
        # {'cls_num': cls_num, 'points': points, 'conf': conf}
        cls_num = int(mask_res['cls_num'])
        conf = float(mask_res['conf'])
        points = []
        for p in mask_res['points']:
            p_floats = [float(p[0]), float(p[1])]
            points.append(p_floats)
        result.append({'cls_num': cls_num, 'points': points, 'conf': conf})
    return result


class CNNWorkerClient(QtCore.QThread):

    def __init__(self, server, image_path=None, conf_thres=0.5, iou_thres=0.5, lrm=0):
        super(CNNWorkerClient, self).__init__()
        self.server = server
        self.image_path = image_path
        self.mask_results = None
        self.conf = conf_thres
        self.iou = iou_thres
        self.lrm = lrm

    def set_image(self, image_path):
        self.image_path = image_path

    def run(self):
        url = f'{self.server}/detect?conf={self.conf}&iou={self.iou}&lrm={self.lrm}'
        files = {'file': open(self.image_path, 'rb')}
        response = requests.post(url, files=files)

        self.mask_results = decode(json.loads(response.text))
