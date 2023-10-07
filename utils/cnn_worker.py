from PySide2 import QtCore
from .detect_yolo8 import predict_and_return_masks, predict_and_return_mask_openvino
from utils.edges_from_mask import yolo8masks2points
from ui.signals_and_slots import LoadPercentConnection

import os
import torch
import cv2
from . import help_functions as hf
import math
from copy import deepcopy


class CNN_worker(QtCore.QThread):

    def __init__(self, model, conf_thres=0.7, iou_thres=0.5,
                 img_name="selected_area.png", img_path=None,
                 scanning=False, linear_dim=0.0923, images_list=None, simplify_factor=1.0, is_openvino=False, nc=12):

        super(CNN_worker, self).__init__()

        self.conf_thres = float(conf_thres)
        self.iou_thres = float(iou_thres)
        self.simplify_factor = float(simplify_factor)
        self.model = model

        self.img_name = img_name
        self.img_path = img_path
        self.image_list = images_list
        self.scanning = scanning

        self.mask_results = []
        self.image_list_results = []

        self.img_ld = linear_dim
        self.train_ld = 0.11
        self.train_img_px = 8000  # в реальности - 1280, но это ужатые 8000 с ld = 0.0923

        self.psnt_connection = LoadPercentConnection()

        self.is_openvino = is_openvino
        self.nc = nc

    def run(self):

        torch.cuda.empty_cache()

        if self.image_list:
            self.run_yolo8_image_list(self.image_list)
            return

        if self.img_path == None:
            print("img_path doesn't set. Stop detection")
            return

        img_path_full = os.path.join(self.img_path, self.img_name)

        self.run_yolo8(img_path_full, self.scanning)

    def run_yolo8_image_list(self, image_list):
        self.psnt_connection.percent.emit(0)
        self.image_list_results = []
        id_tek = 0

        for im_num, img_path_full in enumerate(image_list):
            img = cv2.imread(img_path_full)
            image_height, image_width = img.shape[1], img.shape[0]
            image_data = {'filename': os.path.basename(img_path_full), 'shapes': []}
            if self.is_openvino:
                results = predict_and_return_mask_openvino(self.model, img, conf=self.conf_thres,
                                                           iou=self.iou_thres, save_txt=False, nc=self.nc)

                boxes = results["det"]
                masks = results.get("segment")

                shapes = []

                for idx, (*xyxy, conf, lbl) in enumerate(boxes):
                    shapes.append({'cls_num': int(lbl), 'points': masks[idx].astype(int), 'conf': conf})

                image_data['shapes'] = shapes
                self.image_list_results.append(image_data)
            else:

                results = predict_and_return_masks(self.model, img, conf=self.conf_thres,
                                                   iou=self.iou_thres, save_txt=False)

                shapes = []
                for res in results:
                    for i, mask in enumerate(res['masks']):
                        points = yolo8masks2points(mask, simplify_factor=self.simplify_factor, width=image_width,
                                                   height=image_height)
                        if not points:
                            continue
                        cls_num = res['classes'][i]

                        shape = {'id': id_tek, 'cls_num': cls_num, 'points': points}
                        id_tek += 1
                        shapes.append(shape)

                image_data['shapes'] = shapes
                self.image_list_results.append(image_data)

            self.psnt_connection.percent.emit(int(im_num * 100.0 / len(image_list)))

        self.psnt_connection.percent.emit(100)

    def run_yolo8(self, img_path_full, is_scanning, is_progress_show=True):
        img = cv2.imread(img_path_full)
        shape = img.shape

        if is_scanning:

            if is_progress_show:
                self.psnt_connection.percent.emit(0)

            self.run_yolo8(img_path_full, is_scanning=False, is_progress_show=False)

            if self.img_ld:
                frag_size = int(self.train_img_px * self.train_ld / self.img_ld)
            else:
                frag_size = 1280

            if math.fabs(frag_size - 1280) < 300:
                if is_progress_show:
                    self.psnt_connection.percent.emit(100)
                return

            scanning_results = [res for res in self.mask_results]

            parts = hf.split_into_fragments(img, frag_size)
            crop_x_y_sizes, x_parts_num, y_parts_num = hf.calc_parts(shape[1], shape[0], frag_size)

            print(f'Crop image into {x_parts_num}x{y_parts_num}')
            # print(crop_x_y_sizes)

            part_tek = 0
            for part, part_size in zip(parts, crop_x_y_sizes):

                if self.is_openvino:
                    part_mask_results = predict_and_return_mask_openvino(self.model, part, conf=self.conf_thres,
                                                                         iou=self.iou_thres, save_txt=False, nc=self.nc)

                    x_min, x_max = part_size[0]
                    y_min, y_max = part_size[1]

                    boxes = part_mask_results["det"]
                    masks = part_mask_results.get("segment")

                    for idx, (*xyxy, conf, lbl) in enumerate(boxes):
                        points = masks[idx].astype(int)
                        points_shifted = []
                        for x, y in points:
                            points_shifted.append([x + x_min, y + y_min])
                        scanning_results.append({'cls_num': int(lbl), 'points': points_shifted, 'conf': conf})


                else:

                    part_mask_results = predict_and_return_masks(self.model, part, conf=self.conf_thres,
                                                                 iou=self.iou_thres, save_txt=False)
                    x_min, x_max = part_size[0]
                    y_min, y_max = part_size[1]

                    for res in part_mask_results:
                        for i, mask in enumerate(res['masks']):
                            points = yolo8masks2points(mask, simplify_factor=self.simplify_factor, width=x_max - x_min,
                                                       height=y_max - y_min)
                            if not points:
                                continue
                            points_shifted = []
                            for x, y in points:
                                points_shifted.append([x + x_min, y + y_min])
                            cls_num = res['classes'][i]
                            conf = res['confs'][i]
                            scanning_results.append({'cls_num': cls_num, 'points': points_shifted, 'conf': conf})

                part_tek += 1
                if is_progress_show:
                    self.psnt_connection.percent.emit(90.0 * part_tek / len(parts))

            self.mask_results = hf.filter_masks(scanning_results, conf_thres=self.conf_thres, iou_filter=0.05)

            if is_progress_show:
                self.psnt_connection.percent.emit(100)

        else:
            if is_progress_show:
                self.psnt_connection.percent.emit(0)

            if self.is_openvino:
                results = predict_and_return_mask_openvino(self.model, img, conf=self.conf_thres,
                                                           iou=self.iou_thres, save_txt=False, nc=self.nc)

                boxes = results["det"]
                masks = results.get("segment")

                mask_results = []

                for idx, (*xyxy, conf, lbl) in enumerate(boxes):
                    mask_results.append({'cls_num': int(lbl), 'points': masks[idx].astype(int), 'conf': conf})

                self.mask_results = mask_results

            else:

                results = predict_and_return_masks(self.model, img, conf=self.conf_thres,
                                                   iou=self.iou_thres, save_txt=False)

                if is_progress_show:
                    self.psnt_connection.percent.emit(50)

                mask_results = []
                for res in results:
                    for i, mask in enumerate(res['masks']):
                        points = yolo8masks2points(mask, simplify_factor=self.simplify_factor, width=shape[1],
                                                   height=shape[0])
                        if not points:
                            continue
                        cls_num = res['classes'][i]
                        conf = res['confs'][i]
                        mask_results.append({'cls_num': cls_num, 'points': points, 'conf': conf})

                self.mask_results = mask_results

            if is_progress_show:
                self.psnt_connection.percent.emit(100)
