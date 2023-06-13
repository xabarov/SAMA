from PySide2 import QtCore
from . import cls_settings
from .detect_yolo8 import predict_and_return_masks
from utils.edges_from_mask import yolo8masks2points

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

        self.mask_results = []

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
        img = cv2.imread(img_path_full)
        shape = img.shape

        if is_scanning:

            frag_size = 1000  # int(self.train_img_px * self.train_ld / self.img_ld)

            parts = hf.split_into_fragments(img, frag_size)
            crop_x_y_sizes, x_parts_num, y_parts_num = hf.calc_parts(shape[1], shape[0], frag_size)

            scanning_results = []

            part_tek = 0
            for part, part_size in zip(parts, crop_x_y_sizes):

                part_mask_results = predict_and_return_masks(self.model, part, conf=self.conf_thres,
                                                             iou=self.iou_thres, save_txt=False)
                x_min, x_max = part_size[0]
                y_min, y_max = part_size[1]

                for res in part_mask_results:
                    for i, mask in enumerate(res['masks']):
                        points = yolo8masks2points(mask, simplify_factor=3, width=x_max - x_min, height=y_max - y_min)
                        if not points:
                            continue
                        points_shifted = []
                        for x, y in points:
                            points_shifted.append([x + x_min, y + y_min])
                        cls_num = res['classes'][i]
                        conf = res['confs'][i]
                        scanning_results.append({'cls_num': cls_num, 'points': points_shifted, 'conf': conf})

                part_tek += 1

            self.mask_results = hf.filter_masks(scanning_results)

        else:

            results = predict_and_return_masks(self.model, img_path_full, conf=self.conf_thres,
                                               iou=self.iou_thres, save_txt=False)
            
            mask_results = []
            for res in results:
                for i, mask in enumerate(res['masks']):
                    points = yolo8masks2points(mask, simplify_factor=3, width=shape[1], height=shape[0])
                    if not points:
                        continue
                    cls_num = res['classes'][i]
                    conf = res['confs'][i]
                    mask_results.append({'cls_num': cls_num, 'points': points, 'conf': conf})

            self.mask_results = mask_results
