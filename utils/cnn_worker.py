from PySide2 import QtCore
from . import cls_settings
from .detect_yolo8 import predict_and_return_masks

import os
import torch
import cv2
import help_functions as hf


class PercentConnection(QtCore.QObject):
    percent = QtCore.Signal(int)


class CNN_worker(QtCore.QThread):

    def __init__(self, model, conf_thres=0.7, iou_thres=0.5,
                 img_name="selected_area.png", img_path=None,
                 scanning=False, linear_dim=0.0923):

        super(CNN_worker, self).__init__()

        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.model = model

        self.img_name = img_name
        self.img_path = img_path
        self.scanning = scanning

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

        self.run_yolo8(img_path_full, self.scanning)

    def run_yolo8(self, img_path_full, is_scanning):

        if is_scanning:

            # Steps:
            frag_size = int(self.train_img_px * self.train_ld / self.img_ld)
            img = cv2.imread(img_path_full)
            parts = hf.split_into_fragments(img, frag_size)
            self.mask_results = predict_and_return_masks(self.model, parts, conf=self.conf_thres,
                                                         iou=self.iou_thres, save_txt=False)

            # 3. Fit all fragment masks to image size
            # 4. Use IoU for filtering mask
            # 5. Return filtered results
            pass

        else:

            self.mask_results = predict_and_return_masks(self.model, img_path_full, conf=self.conf_thres,
                                                         iou=self.iou_thres, save_txt=False)
