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


def create_blank_image():
    return {"shapes": [], "lrm": None, 'status': 'empty'}


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
        self.data["images"] = {}
        self.data["labels"] = []
        self.data["labels_color"] = {}
        self.is_loaded = False

    def calc_dataset_balance(self):
        labels = self.get_labels()
        labels_nums = {}
        for im in self.data['images'].values():
            for shape in im['shapes']:
                cls_num = shape['cls_num']
                label_name = labels[cls_num]
                if label_name not in labels_nums:
                    labels_nums[label_name] = 1
                else:
                    labels_nums[label_name] += 1
        return labels_nums

    def check_and_convert_old_data_to_new(self):
        if isinstance(self.data["images"], list):
            print("Old version of project. Convert to new version")
            images = {}
            for im in self.data["images"]:
                filename = im["filename"]
                im_dict = {k: v for k, v in im.items() if k != 'filename'}
                images[filename] = im_dict
            self.data["images"] = images

    def load(self, json_path):
        with open(json_path, 'r', encoding='utf8') as f:
            self.data = ujson.load(f)
            self.check_and_convert_old_data_to_new()
            self.update_ids()
            self.is_loaded = True

    def save(self, json_path):
        with open(json_path, 'w', encoding='utf8') as f:
            ujson.dump(self.data, f)

    def update_ids(self):

        if not self.data["images"]:
            return
        id_num = 0
        for im in self.data['images'].values():
            for shape in im['shapes']:
                shape['id'] = id_num
                id_num += 1

    def set_data(self, data):
        self.data = data
        self.is_loaded = True

    def set_image_lrm(self, image_name, lrm):
        im = self.get_image_data(image_name)  # im = {shapes:[], lrm:float, status:str}
        if im:
            im["lrm"] = round(lrm, 6)
        else:
            im = create_blank_image()
            im['lrm'] = round(lrm, 6)
            self.data["images"][image_name] = im

    def get_image_status(self, image_name):
        im = self.get_image_data(image_name)  # im = {shapes:[], lrm:float, status:str}
        if im:
            status = im.get("status", None)
            if status:
                return status
            self.set_image_status(image_name, 'empty')
            return 'empty'

    def get_all_images_info(self):
        res = {}
        for image_name, im_data in self.data["images"].items():
            status = im_data.get("status", "empty")
            last_user = im_data.get("last_user", "unknown")

            res[image_name] = {"status": status, "last_user": last_user}
        return res

    def set_image_status(self, image_name, status):
        if status not in ['empty', 'in_work', 'approve']:
            return
        im = self.get_image_data(image_name)  # im = {shapes:[], lrm:float, status:str}
        if im:
            im["status"] = status
            self.data["images"][image_name] = im
        else:
            im = create_blank_image()
            im["status"] = status
            self.data["images"][image_name] = im

    def get_image_last_user(self, image_name):
        im = self.get_image_data(image_name)
        if im:
            return im.get("last_user", None)

    def set_image_last_user(self, image_name, last_user):
        im = self.get_image_data(image_name)
        if im:
            im["last_user"] = last_user
            self.data["images"][image_name] = im
        else:
            im = create_blank_image()
            im["last_user"] = last_user
            self.data["images"][image_name] = im

    def set_lrm_for_all_images(self, lrms_data):
        set_names = []
        unset_names = []
        im_names_in_folder = os.listdir(self.get_image_path())
        for image_name in lrms_data:
            if image_name not in im_names_in_folder:
                unset_names.append(image_name)
            else:

                lrm = lrms_data[image_name]
                im = self.get_image_data(image_name)  # im = {shapes:[], lrm:float, status:str}
                if im:
                    im["lrm"] = round(lrm, 6)
                    self.data["images"][image_name] = im
                else:
                    im = create_blank_image()
                    im['lrm'] = round(lrm, 6)
                    self.data["images"][image_name] = im

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
                    im = create_blank_image()
                    self.data["images"][im_name] = im

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

    def add_shapes_to_images(self, dicts_with_shapes):
        """
        Добавить данные по полигонам. Не затираются предыдущие полигоны, а добавляются поверх имеющихся
        dicts_with_shapes - { image_name1: {'shapes':[...]}, image_name2: {'shapes':[...]}, ...}
        """
        for im_name, im_data in dicts_with_shapes.items():
            if im_name in self.data['images']:
                for shape in im_data['shapes']:  # im_data в формате {"shapes": [...]}
                    self.data["images"][im_name]['shapes'].append(shape)
            else:
                im_blank = create_blank_image()
                for shape in im_data['shapes']:  # im_data в формате {"shapes": [...]}
                    im_blank['shapes'].append(shape)
                self.data["images"][im_name] = im_blank

    def set_image_data(self, image_name, image_data):
        self.data["images"][image_name] = image_data

    def get_data(self):
        return self.data

    def get_label_color(self, cls_name):
        return self.data["labels_color"].get(cls_name, None)

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
        return self.data["images"].get(image_name, None)

    def get_labels(self):
        return self.data["labels"]

    def get_label_name(self, cls_num):
        if cls_num < len(self.data["labels"]):
            return self.data["labels"][cls_num]

    def get_image_lrm(self, image_name):
        im = self.get_image_data(image_name)  # im = {shapes:[], lrm:float, status:str}
        if im:
            return im.get("lrm", None)

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

    def change_cls_num_by_ids(self, image_name, lbl_ids, new_cls_num):
        im = self.get_image_data(image_name)  # im = {shapes:[], lrm:float, status:str}
        new_shapes = []

        for shape in im['shapes']:
            if shape['id'] in lbl_ids:
                new_shape = shape
                new_shape["cls_num"] = new_cls_num
                new_shapes.append(new_shape)
            else:
                new_shapes.append(shape)
        im['shapes'] = new_shapes
        self.data["images"][image_name] = im

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
        if image_name in self.data["images"]:
            del self.data["images"][image_name]

    def delete_data_by_class_name(self, cls_name):
        name_to_name_map = {}  # Конвертер старого имени в новое
        old_name_to_num = {}
        new_labels = []

        for i, label in enumerate(self.data["labels"]):

            if label != cls_name:
                name_to_name_map[label] = label
                new_labels.append(label)
            else:
                name_to_name_map[label] = None

            old_name_to_num[label] = i

        new_name_to_num = {}
        for i, label in enumerate(new_labels):
            new_name_to_num[label] = i

        num_to_num = {}
        for label, old_num in old_name_to_num.items():
            new_name = name_to_name_map[label]
            if new_name:
                new_num = new_name_to_num[new_name]
                num_to_num[old_num] = new_num
            else:
                num_to_num[old_num] = -1

        for im_name, image in self.data["images"].items():  # image = {shapes:[], lrm:float, status:str}
            new_shapes = []
            for shape in image["shapes"]:

                new_num = num_to_num[shape["cls_num"]]
                if new_num != -1:
                    shape_new = {}
                    shape_new["cls_num"] = new_num
                    shape_new["points"] = shape["points"]
                    shape_new["id"] = shape["id"]
                    new_shapes.append(shape_new)

            image["shapes"] = new_shapes
            self.data["images"][im_name] = image

        self.set_labels(new_labels)

        self.delete_label_color(cls_name)

    def delete_data_by_class_number(self, cls_num):

        for im_name, image in self.data["images"].items():  # image = {shapes:[], lrm:float, status:str}
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

            image["shapes"] = new_shapes
            self.data["images"][im_name] = image

    def change_data_class_from_to(self, from_cls_name, to_cls_name):

        name_to_name_map = {}  # Конвертер старого имени в новое
        old_name_to_num = {}
        new_labels = []

        for i, label in enumerate(self.data["labels"]):

            if label != from_cls_name:
                name_to_name_map[label] = label
                new_labels.append(label)
            else:
                name_to_name_map[label] = to_cls_name

            old_name_to_num[label] = i

        new_name_to_num = {}
        for i, label in enumerate(new_labels):
            new_name_to_num[label] = i

        num_to_num = {}
        for label, old_num in old_name_to_num.items():
            new_name = name_to_name_map[label]
            new_num = new_name_to_num[new_name]
            num_to_num[old_num] = new_num


        for im_name, image in self.data["images"].items():  # image = {shapes:[], lrm:float, status:str}
            new_shapes = []
            for shape in image["shapes"]:
                shape_new = {}
                shape_new["cls_num"] = num_to_num[shape["cls_num"]]
                shape_new["points"] = shape["points"]
                shape_new["id"] = shape["id"]
                new_shapes.append(shape_new)

            image["shapes"] = new_shapes
            self.data["images"][im_name] = image


        self.set_labels(new_labels)

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
        images = {}
        im_path = self.get_image_path()
        for filename, im in self.data['images'].items():
            if os.path.exists(os.path.join(im_path, filename)):
                images[filename] = im
            else:
                print(f"Checking files: image {filename} doesn't exist")

        self.data['images'] = images


if __name__ == '__main__':
    proj_path = "D:\python\\ai_annotator\projects\\test.json"

    proj = ProjectHandler()
    proj.load(proj_path)
    # print(proj.exportToYOLOBox("D:\python\\ai_annotator\labels"))
