import datetime
import os

import cv2
import numpy as np
import ujson
from PIL import Image
from PyQt5.QtWidgets import QWidget
from shapely import Polygon
from utils.exporter import Exporter

import utils.config as config
import utils.help_functions as hf
from ui.signals_and_slots import LoadPercentConnection, InfoConnection, ProjectSaveLoadConn
from utils.blur_image import blur_image_by_mask, get_mask_from_yolo_txt


class ProjectHandler(QWidget):
    """
    Класс для работы с данными проекта
    Хранит данные разметки в виде словаря
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.export_percent_conn = LoadPercentConnection()
        self.info_conn = InfoConnection()
        self.export_finished = ProjectSaveLoadConn()
        self.init()

    def check_json(self, json_project_data):
        for field in ["path_to_images", "images", "labels", "labels_color"]:
            if field not in json_project_data:
                return False
        return True

    def clear(self):
        self.init()

    def init(self):
        self.data = dict()
        self.data["path_to_images"] = ""
        self.data["images"] = []
        self.data["labels"] = []
        self.data["labels_color"] = {}
        self.is_loaded = False

    def calc_dataset_balance(self):
        labels = self.get_labels()
        labels_nums = {}
        for im in self.data['images']:
            for shape in im['shapes']:
                cls_num = shape['cls_num']
                label_name = labels[cls_num]
                if label_name not in labels_nums:
                    labels_nums[label_name] = 1
                else:
                    labels_nums[label_name] += 1
        return labels_nums

    def load(self, json_path):
        with open(json_path, 'r', encoding='utf8') as f:
            self.data = ujson.load(f)
            self.update_ids()
            self.is_loaded = True

    def save(self, json_path):
        with open(json_path, 'w', encoding='utf8') as f:
            ujson.dump(self.data, f)

    def update_ids(self):
        id_num = 0
        for im in self.data['images']:
            for shape in im['shapes']:
                shape['id'] = id_num
                id_num += 1

    def set_data(self, data):
        self.data = data
        self.is_loaded = True

    def set_image_lrm(self, image_name, lrm):
        im = self.get_image_data(image_name)
        if im:
            im["lrm"] = round(lrm, 6)
        else:
            im_names_in_folder = os.listdir(self.get_image_path())
            if image_name in im_names_in_folder:
                self.data["images"].append({"filename": image_name, "shapes": [], "lrm": round(lrm, 6)})

    def set_lrm_for_all_images(self, lrms_data):
        set_names = []
        unset_names = []
        im_names_in_folder = os.listdir(self.get_image_path())
        for image_name in lrms_data:
            if image_name not in im_names_in_folder:
                unset_names.append(image_name)
            else:

                lrm = lrms_data[image_name]
                im = self.get_image_data(image_name)
                if im:
                    im["lrm"] = round(lrm, 6)
                else:
                    self.set_image_data({"filename": image_name, "shapes": [], "lrm": round(lrm, 6)})

                set_names.append(image_name)

        return set_names, unset_names

    def set_labels(self, labels):
        self.data["labels"] = labels

    def set_path_to_images(self, path):
        self.data["path_to_images"] = path

    def set_blank_data_for_images_names(self, images_names):
        im_names_in_folder = os.listdir(self.get_image_path())
        for im_name in images_names:
            if im_name in im_names_in_folder:
                im = self.get_image_data(im_name)
                if not im:
                    self.set_image_data({"filename": im_name, "shapes": [], "lrm": None})

    def set_label_color(self, cls_name, color=None, alpha=None):

        if not color:
            if not alpha:
                alpha = 255

            cls_color = self.get_label_color(cls_name)
            if not cls_color:
                proj_colors = self.get_colors()

                selected_color = config.COLORS[0]
                tek_color_num = 0
                is_break = False
                while selected_color in proj_colors:
                    tek_color_num += 1
                    if tek_color_num == len(config.COLORS) - 1:
                        is_break = True
                        break
                    selected_color = config.COLORS[tek_color_num]

                if is_break:
                    selected_color = hf.create_random_color(alpha)

                self.data["labels_color"][cls_name] = selected_color

        else:
            if alpha:
                color = [color[0], color[1], color[2], alpha]
            self.data["labels_color"][cls_name] = color

    def set_labels_colors(self, labels_names, rewrite=False):
        if rewrite:
            self.data["labels_color"] = {}

        for label_name in labels_names:
            if label_name not in self.data["labels_color"]:
                self.set_label_color(label_name)

    def set_labels_names(self, labels):
        self.data["labels"] = labels

    def set_all_images(self, images_new):
        self.data["images"] = images_new

    def set_image_data(self, image_data):
        im_names_in_folder = os.listdir(self.get_image_path())
        image_name = image_data["filename"]

        if image_name not in im_names_in_folder:
            return

        is_found = False
        images_new = []
        for im in self.data["images"]:
            if image_name == im["filename"]:
                is_found = True
                images_new.append(image_data)
            else:
                images_new.append(im)

        if not is_found:
            images_new.append(image_data)

        self.data["images"] = images_new

    def get_data(self):
        return self.data

    def get_label_color(self, cls_name):
        if cls_name in self.data["labels_color"]:
            return self.data["labels_color"][cls_name]

        return None

    def get_label_num(self, label_name):
        for i, label in enumerate(self.data["labels"]):
            if label == label_name:
                return i
        return -1

    def get_images_num(self):
        return len(self.data["images"])

    def get_colors(self):
        return [tuple(self.data["labels_color"][key]) for key in
                self.data["labels_color"]]

    def get_image_data(self, image_name):
        for im in self.data["images"]:
            if image_name == im["filename"]:
                return im
        return None

    def get_labels(self):
        return self.data["labels"]

    def get_label_name(self, cls_num):
        if cls_num < len(self.data["labels"]):
            return self.data["labels"][cls_num]

    def get_image_lrm(self, image_name):
        for im in self.data["images"]:
            if image_name == im["filename"]:
                if "lrm" in im:
                    return im["lrm"]
                else:
                    return None
        return None

    def get_image_path(self):
        return self.data["path_to_images"]

    def get_export_map(self, export_label_names):
        label_names = self.get_labels()
        export_map = {}
        export_cls_num = 0
        for i, name in enumerate(label_names):
            if name in export_label_names:
                export_map[name] = export_cls_num
                export_cls_num += 1
            else:
                export_map[name] = 'del'

        return export_map

    def change_cls_num_by_id(self, lbl_id, new_cls_num):
        new_images = []
        for im in self.data['images']:
            new_im = im
            new_shapes = []
            for shape in im['shapes']:
                if lbl_id == shape['id']:
                    new_shape = shape
                    new_shape["cls_num"] = new_cls_num
                    new_im['shapes']
                    new_shapes.append(new_shape)
                else:
                    new_shapes.append(shape)
            new_im['shapes'] = new_shapes
            new_images.append(new_im)

        self.data['images'] = new_images

    def rename_color(self, old_name, new_name):
        if old_name in self.data["labels_color"]:
            color = self.data["labels_color"][old_name]
            self.data["labels_color"][new_name] = color
            del self.data["labels_color"][old_name]

    def change_name(self, old_name, new_name):
        self.rename_color(old_name, new_name)
        labels = []
        for i, label in enumerate(self.data["labels"]):
            if label == old_name:
                labels.append(new_name)
            else:
                labels.append(label)
        self.data["labels"] = labels

    def delete_label_color(self, label_name):
        if label_name in self.data["labels_color"]:
            del self.data["labels_color"][label_name]

    def delete_label(self, label_name):
        labels = []
        for label in self.data['labels']:
            if label != label_name:
                labels.append(label)
        self.set_labels(labels)

    def delete_image(self, image_name):

        images = []
        for image in self.data["images"]:
            if image_name != image["filename"]:
                images.append(image)

        self.data["images"] = images

    def delete_data_by_class_name(self, cls_name):
        for i, label in enumerate(self.data["labels"]):
            if label == cls_name:
                self.delete_data_by_class_number(i)
                self.delete_label_color(label)
                self.delete_label(label)
                return

    def delete_data_by_class_number(self, cls_num):

        images = []
        for image in self.data["images"]:
            new_shapes = []
            for shape in image["shapes"]:
                if shape["cls_num"] < cls_num:
                    new_shapes.append(shape)
                elif shape["cls_num"] > cls_num:
                    shape_new = {}
                    shape_new["cls_num"] = shape["cls_num"] - 1

                    shape_new["points"] = shape["points"]
                    shape_new["id"] = shape["id"]
                    new_shapes.append(shape_new)

            new_image = image
            new_image["shapes"] = new_shapes

            images.append(new_image)

        self.data["images"] = images

    def change_data_class_from_to(self, from_cls_name, to_cls_name):
        # Two stage:
        # 1) change
        from_cls_num = self.get_label_num(from_cls_name)
        to_cls_num = self.get_label_num(to_cls_name)

        images = []
        for image in self.data["images"]:
            new_shapes = []
            for shape in image["shapes"]:

                if shape["cls_num"] < from_cls_num:
                    # save without change
                    new_shapes.append(shape)

                else:

                    shape_new = {}

                    if shape["cls_num"] == from_cls_num:
                        shape_new["cls_num"] = to_cls_num
                    else:
                        # Все номера от from старше должны уменьшиться на 1
                        shape_new["cls_num"] = shape["cls_num"] - 1

                    shape_new["points"] = shape["points"]
                    shape_new["id"] = shape["id"]
                    new_shapes.append(shape_new)

            new_image = image
            new_image["shapes"] = new_shapes

            images.append(new_image)

        self.data["images"] = images

        # labels
        labels = []

        for label in self.data["labels"]:
            if label != from_cls_name:
                labels.append(label)
        self.set_labels(labels)

        self.delete_label_color(from_cls_name)

    def is_blurred_classes(self, export_map):
        for label in export_map:
            if export_map[label] == 'blur':
                return True

        return False

    def create_images_labels_subdirs(self, export_dir):
        images_dir = os.path.join(export_dir, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        labels_dir = os.path.join(export_dir, 'labels')
        if not os.path.exists(labels_dir):
            os.makedirs(labels_dir)

        return images_dir, labels_dir

    def export(self, export_dir, export_map=None, format='yolo_seg'):

        self.exporter = Exporter(self.data, export_dir=export_dir, format=format, export_map=export_map)

        self.exporter.export_percent_conn.percent.connect(self.on_exporter_percent_change)
        self.exporter.info_conn.info_message.connect(self.on_exporter_message)
        self.exporter.err_conn.error_message.connect(self.on_exporter_message)

        self.exporter.finished.connect(self.on_export_finished)

        if not self.exporter.isRunning():
            self.exporter.start()

    def on_exporter_percent_change(self, percent):
        self.export_percent_conn.percent.emit(percent)

    def on_exporter_message(self, message):
        self.info_conn.info_message.emit(message)

    def on_export_finished(self):
        print('Finished')
        self.export_finished.on_finished.emit(True)

    def clear_not_existing_images(self):
        images = []
        im_path = self.get_image_path()
        for im in self.data['images']:
            if os.path.exists(os.path.join(im_path, im['filename'])):
                images.append(im)
            else:
                print(f"Checking files: image {im['filename']} doesn't exist")

        self.data['images'] = images


if __name__ == '__main__':
    proj_path = "D:\python\\ai_annotator\projects\\test.json"

    proj = ProjectHandler()
    proj.load(proj_path)
    # print(proj.exportToYOLOBox("D:\python\\ai_annotator\labels"))
