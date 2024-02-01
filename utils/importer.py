import os
import shutil

import cv2
import numpy as np
from PIL import Image
from PySide2 import QtCore

from ui.signals_and_slots import LoadPercentConnection, ErrorConnection, InfoConnection
from utils import help_functions as hf
from utils.sam_predictor import mask_to_seg, predict_by_box


class Importer(QtCore.QThread):

    def __init__(self, coco_data=None, alpha=120, yaml_data=None, is_seg=False, copy_images_path=None,
                 is_coco=True, dataset="train", coco_name=None, convert_to_masks=False,
                 sam_predictor=None):
        super(Importer, self).__init__()

        # SIGNALS
        self.load_percent_conn = LoadPercentConnection()
        self.info_conn = InfoConnection()
        self.err_conn = ErrorConnection()

        self.coco_data = coco_data
        self.project = None
        self.alpha = alpha
        self.dataset = dataset
        self.yaml_data = yaml_data
        self.is_coco = is_coco
        self.is_seg = is_seg
        self.copy_images_path = copy_images_path
        self.coco_name = coco_name
        self.convert_to_masks = convert_to_masks
        self.sam_predictor = sam_predictor

    def get_project(self):
        return self.project

    def set_is_seg(self, is_seg):
        self.is_seg = is_seg

    def set_copy_images_path(self, path):
        self.copy_images_path = path

    def set_yaml_path(self, path):
        self.yaml_path = path

    def set_coco_data(self, data):
        self.coco_data = data

    def set_alpha(self, alpha):
        self.alpha = alpha

    def set_dataset_type(self, dataset_type):
        self.dataset = dataset_type

    def filter_data_annotations_by_cls_size(self, data_annotations, cls_size):
        """
        Delete shapes from data if cls_num >= cls_size
        """
        data_annotations_new = []
        """
        seg = {"segmentation": all_points, "area": int(area), "bbox": bbox, "iscrowd": 0, "id": seg_id,
                           "image_id": id_map[filename], "category_id": cls_num + 1}
                    export_json["annotations"].append(seg)
        """
        for seg in data_annotations:
            if seg["category_id"] < cls_size:
                data_annotations_new.append(seg)
            else:
                self.info_conn.info_message.emit(f'Filtered seg with category_id {seg["category_id"]}')

        return data_annotations_new

    def import_from_coco(self):

        self.info_conn.info_message.emit(f"Start import data from {self.coco_name}")
        data = self.coco_data
        alpha = self.alpha

        label_names = [d["name"] for d in data["categories"]]

        data["annotations"] = self.filter_data_annotations_by_cls_size(data["annotations"], len(label_names))

        label_colors = hf.get_label_colors(label_names, alpha=alpha)

        if self.copy_images_path:
            # change paths
            images = {}

            for i, im in enumerate(data["images"]):
                im_copy = im
                # make sense copy from real folder, not from flickr_url
                save_images_folder = self.copy_images_path
                if os.path.exists(im['flickr_url']):
                    shutil.copy(im['flickr_url'], os.path.join(save_images_folder, im["file_name"]))

                elif os.path.exists(os.path.join(os.path.dirname(self.coco_name), im["file_name"])):
                    shutil.copy(os.path.join(os.path.dirname(self.coco_name), im["file_name"]),
                                os.path.join(save_images_folder, im["file_name"]))
                else:
                    continue

                im_copy['flickr_url'] = os.path.join(save_images_folder, im["file_name"])
                im_copy['coco_url'] = os.path.join(save_images_folder, im["file_name"])
                images[im["file_name"]] = im_copy

                self.load_percent_conn.percent.emit(int(i * 100.0 / len(data["images"])))

            data["images"] = images

        project_path = os.path.dirname(self.coco_name)

        project = {'path_to_images': project_path,
                   "images":  {}, 'labels': label_names, 'labels_color': label_colors}

        id_num = 0
        for i, im in enumerate(data["images"]):
            im_id = im["id"]
            filename = im["file_name"]
            proj_im = {'shapes': []}
            for seg in data["annotations"]:
                if seg["image_id"] == im_id:
                    cls = seg["category_id"] - 1
                    points = [[seg["segmentation"][0][i], seg["segmentation"][0][i + 1]] for i in
                              range(0, len(seg["segmentation"][0]), 2)]
                    shape = {"id": id_num, "cls_num": cls, 'points': points}
                    id_num += 1
                    proj_im["shapes"].append(shape)

            project['images'][filename] = proj_im

            self.load_percent_conn.percent.emit(int(i * 100.0 / len(data["images"])))

        self.project = project

    def import_from_yolo_yaml(self):
        yaml_data = self.yaml_data

        alpha = self.alpha
        is_seg = self.is_seg
        copy_images_path = self.copy_images_path

        if self.dataset == 'all':
            datasets = ['train', 'val', 'test']
        else:
            datasets = [self.dataset]

        for ds in datasets:

            path_to_labels = os.path.join(yaml_data["path"], "labels", ds)

            if copy_images_path:
                path_to_images = os.path.join(yaml_data["path"], "images", ds)
                images = [im for im in os.listdir(path_to_images) if hf.is_im_path(im)]

                # copy images
                for i, im in enumerate(images):
                    shutil.copy(os.path.join(path_to_images, im), os.path.join(copy_images_path, im))
                    self.load_percent_conn.percent.emit(int(i * 100.0 / len(images)))

                path_to_images = copy_images_path

            else:
                path_to_images = os.path.join(yaml_data["path"], "images", ds)

            labels_names = yaml_data["names"]
            label_colors = hf.get_label_colors(labels_names, alpha=alpha)

            if is_seg:
                self.import_from_yolo_seg(path_to_labels, path_to_images, labels_names, label_colors)
            else:
                self.import_from_yolo_box(path_to_labels, path_to_images, labels_names, label_colors,
                                          convert_to_masks=self.convert_to_masks, sam_predictor=self.sam_predictor)

    def import_from_yolo_box(self, path_to_labels, path_to_images, labels_names, labels_color, convert_to_masks=False,
                             sam_predictor=None):
        project = {'path_to_images': path_to_images, 'images': {}, "labels": labels_names, "labels_color": labels_color}
        images = [im for im in os.listdir(path_to_images) if hf.is_im_path(im)]
        id_num = 0
        for i, filename in enumerate(images):

            if convert_to_masks and sam_predictor:
                sam_predictor.set_image(cv2.imread(os.path.join(path_to_images, filename)))

            width, height = Image.open(os.path.join(path_to_images, filename)).size
            im_shape = [height, width]

            txt_name = hf.convert_image_name_to_txt_name(filename)
            image_data = {'shapes': []}

            if not os.path.exists(os.path.join(path_to_labels, txt_name)):
                self.load_percent_conn.percent.emit(int(i * 100.0 / len(images)))
                continue

            with open(os.path.join(path_to_labels, txt_name), 'r') as f:
                for line in f:
                    shape = {}
                    cls_data = line.strip().split(' ')

                    shape["cls_num"] = int(cls_data[0])
                    shape["id"] = int(id_num)
                    id_num += 1

                    x = int(float(cls_data[1]) * im_shape[1])
                    y = int(float(cls_data[2]) * im_shape[0])
                    w = int(float(cls_data[3]) * im_shape[1])
                    h = int(float(cls_data[4]) * im_shape[0])

                    shape["points"] = [[x - w / 2, y - h / 2], [x + w / 2, y - h / 2], [x + w / 2, y + h / 2],
                                       [x - w / 2, y + h / 2]]

                    if convert_to_masks and sam_predictor:
                        input_box = np.array([shape["points"][0][0], shape["points"][0][1], shape["points"][2][0],
                                              shape["points"][2][1]])
                        masks = predict_by_box(sam_predictor, input_box)
                        shape["points"] = mask_to_seg(masks)[0]

                    image_data["shapes"].append(shape)

            project['images'][filename] = image_data
            self.load_percent_conn.percent.emit(int(i * 100.0 / len(images)))

        self.project = project

    def import_from_yolo_seg(self, path_to_labels, path_to_images, labels_names, labels_color):
        project = {'path_to_images': path_to_images, 'images': {}, "labels": labels_names, "labels_color": labels_color}
        images = [im for im in os.listdir(path_to_images) if hf.is_im_path(im)]
        id_num = 0
        for i, filename in enumerate(images):
            # im_shape = cv2.imread(os.path.join(path_to_images, im)).shape
            width, height = Image.open(os.path.join(path_to_images, filename)).size
            im_shape = [height, width]

            txt_name = hf.convert_image_name_to_txt_name(filename)
            image_data = {'shapes': []}

            if not os.path.exists(os.path.join(path_to_labels, txt_name)):
                self.load_percent_conn.percent.emit(int(i * 100.0 / len(images)))
                self.info_conn.info_message.emit(f"Can't find labels for {filename}")
                continue

            with open(os.path.join(path_to_labels, txt_name), 'r') as f:
                for line in f:
                    shape = {}
                    cls_data = line.strip().split(' ')

                    shape["cls_num"] = int(cls_data[0])
                    shape["id"] = int(id_num)
                    id_num += 1

                    shape["points"] = [
                        [int(float(cls_data[i]) * im_shape[1]), int(float(cls_data[i + 1]) * im_shape[0])] for i in
                        range(1, len(cls_data), 2)]
                    image_data["shapes"].append(shape)

            project['images'][filename] = image_data
            self.load_percent_conn.percent.emit(int(i * 100.0 / len(images)))

        self.project = project

    def run(self):
        if self.is_coco:
            self.import_from_coco()
        else:
            self.import_from_yolo_yaml()
