from PySide2 import QtCore
from . import cls_settings
from .detect_yolo8 import predict_and_return_masks

import os
import torch


class PercentConnection(QtCore.QObject):
    percent = QtCore.Signal(int)


class CNN_worker(QtCore.QThread):

    def __init__(self, model, cnn_name=cls_settings.CNN_DEFAULT, conf_thres=0.7, iou_thres=0.5,
                 img_name="selected_area.png", img_path=None,
                 scanning=False, device='GPU', linear_dim=0.0923, simplify=3):

        super(CNN_worker, self).__init__()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.simplify = simplify
        self.model = model

        self.cnn_name = cnn_name
        self.cnn_type = cls_settings.get_cnn_type(cnn_name)
        cfg_path, weights_path = cls_settings.get_cfg_and_weights_by_cnn_name(self.cnn_name)
        self.cfg_path = os.path.join(os.getcwd(), cfg_path)
        self.weights_path = os.path.join(os.getcwd(), weights_path)

        self.img_name = img_name
        self.img_path = img_path
        self.scanning = scanning
        self.device = device

        self.img_ld = linear_dim
        self.train_ld = 0.12
        self.train_img_px = 8000  # в реальности - 1280, но это ужатые 8000 с ld = 0.0923

        self.psnt_connection = PercentConnection()

    def run(self):

        torch.cuda.empty_cache()

        if self.img_path == None:
            print("img_path doesn't set. Stop detection")
            return

        else:
            img_path_full = os.path.join(self.img_path, self.img_name)

        if self.cnn_type == "YOLO8":
            self.run_yolo8(img_path_full, self.scanning)

    def run_yolo8(self, img_path_full, is_scanning):

        if is_scanning:

            # Steps:
            # 1. Split image on fragments
            # 2. Collect mask on each fragment. Use `predict_and_return_masks` with source = fragmnets_paths_list
            # 3. Fit all fragment masks to image size
            # 4. Use IoU for filtering mask
            # 5. Return filtered results
            pass

        else:

            dev_set = 'cpu'
            if self.device == "cuda":
                dev_set = 0

            self.mask_results = predict_and_return_masks(self.model, img_path_full, self.weights_path, config_path=self.cfg_path, conf=self.conf_thres,
                                                         iou=self.iou_thres, save_txt=False, device=dev_set)
