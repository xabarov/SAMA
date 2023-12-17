import datetime
import os

import cv2
import numpy as np
import ujson
from PIL import Image
from PySide2 import QtCore
from shapely import Polygon
import shutil
from ui.signals_and_slots import LoadPercentConnection, ErrorConnection, InfoConnection
from utils import help_functions as hf
from utils.blur_image import blur_image_by_mask, get_mask_from_yolo_txt
from utils.datasets_converter.yolo_converter import create_yaml
from utils.dataset_preprocessing.splitter import train_test_val_splitter


class Exporter(QtCore.QThread):

    def __init__(self, project_data, export_dir, format='yolo_seg', export_map=None, dataset_name='dataset',
                 variant_idx=0, splits=None, split_method='names', sim=0):
        """
        variant_idx = 0 Train/Val/Test
        1 - Train/Val
        2 - Only Train
        3 - Only Val
        4 - Only Test
        """
        super(Exporter, self).__init__()

        # SIGNALS
        self.export_percent_conn = LoadPercentConnection()
        self.info_conn = InfoConnection()
        self.err_conn = ErrorConnection()

        self.export_dir = export_dir
        self.format = format
        self.export_map = export_map
        self.dataset_name = dataset_name

        self.variant_idx = variant_idx
        self.splits = splits
        self.split_method = split_method
        self.sim = sim

        self.data = project_data

    def run(self):
        if self.format == 'yolo_seg':
            self.exportToYOLOSeg()
        elif self.format == 'yolo_box':
            self.exportToYOLOBox()
        else:
            self.exportToCOCO()

    def get_labels(self):
        return self.data["labels"]

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

    def is_blurred_classes(self, export_map):
        for label in export_map:
            if export_map[label] == 'blur':
                return True

        return False

    def create_images_labels_subdirs(self, export_dir, is_labels_need=True):
        images_dir = os.path.join(export_dir, 'images')
        if not os.path.exists(images_dir):
            os.makedirs(images_dir)

        for folder in ["train", "val", "test"]:
            folder_name = os.path.join(images_dir, folder)
            if not os.path.exists(folder_name):
                os.makedirs(folder_name)

        if is_labels_need:
            labels_dir = os.path.join(export_dir, 'labels')
            if not os.path.exists(labels_dir):
                os.makedirs(labels_dir)

            for folder in ["train", "val", "test"]:
                folder_name = os.path.join(labels_dir, folder)
                if not os.path.exists(folder_name):
                    os.makedirs(folder_name)

            return images_dir, labels_dir

        return images_dir

    def create_blur_dir(self, export_dir):
        blur_dir = os.path.join(export_dir, 'blur')
        if not os.path.exists(blur_dir):
            os.makedirs(blur_dir)
        return blur_dir

    def write_yolo_seg_line(self, shape, im_shape, f, cls_num):
        points = shape["points"]
        line = f"{cls_num}"
        for point in points:
            line += f" {point[0] / im_shape[1]} {point[1] / im_shape[0]}"

        f.write(f"{line}\n")

    def get_split_image_names(self):
        """
        Возвращает списки изображений, разбитых по категориям
        if idx == 0: Train/Val/Test
        1: Train/Val
        2: Train
        3: Val
        4: Test
        """
        if self.sim == 0:
            sim_method = "random"
        elif self.sim == 1:
            sim_method = "names"
        else:
            sim_method = "images"

        if self.variant_idx == 0:
            train_names, val_names, test_names = train_test_val_splitter(self.data["path_to_images"], self.splits[0],
                                                                         self.splits[1],
                                                                         self.splits[2], sim_method=sim_method,
                                                                         percent_hook=self.emit_percent)
            return {"train": train_names, "val": val_names, "test": test_names}
        if self.variant_idx == 1:
            train_names, val_names = train_test_val_splitter(self.data["path_to_images"], self.splits[0],
                                                             self.splits[1], sim_method=sim_method,
                                                             percent_hook=self.emit_percent)

            return {"train": train_names, "val": val_names}

        if self.variant_idx == 2:
            train_names = [im for im in os.listdir(self.data["path_to_images"]) if hf.is_im_path(im)]

            return {"train": train_names}

        if self.variant_idx == 3:
            val_names = [im for im in os.listdir(self.data["path_to_images"]) if hf.is_im_path(im)]

            return {"val": val_names}

        if self.variant_idx == 4:
            test_names = [im for im in os.listdir(self.data["path_to_images"]) if hf.is_im_path(im)]

            return {"test": test_names}

    def write_yolo_box_line(self, shape, im_shape, f, cls_num):
        points = shape["points"]
        xs = []
        ys = []
        for point in points:
            xs.append(point[0])
            ys.append(point[1])
        min_x = min(xs)
        max_x = max(xs)
        min_y = min(ys)
        max_y = max(ys)
        w = abs(max_x - min_x)
        h = abs(max_y - min_y)

        x_center = min_x + w / 2
        y_center = min_y + h / 2

        f.write(
            f"{cls_num} {x_center / im_shape[1]} {y_center / im_shape[0]} {w / im_shape[1]} {h / im_shape[0]}\n")

    def exportToYOLOSeg(self):
        """
        export_map - {label_name: cls_num или 'del' или 'blur' , ... } Экспортируемых меток может быть меньше
        """
        export_dir = self.export_dir
        export_map = self.export_map

        if not os.path.isdir(export_dir):
            return

        self.clear_not_existing_images()
        labels_names = self.get_labels()

        if not export_map:
            export_map = self.get_export_map(labels_names)

        images_dir, labels_dir = self.create_images_labels_subdirs(export_dir)
        is_blur = self.is_blurred_classes(export_map)

        if is_blur:
            blur_dir = self.create_blur_dir(export_dir)

        export_label_names = {}
        unique_values = []
        for k, v in export_map.items():
            if v != 'del' and v != 'blur' and v not in unique_values:
                export_label_names[k] = v
                unique_values.append(v)

        use_test = True if self.variant_idx == 0 else False
        create_yaml(f"{self.dataset_name}.yaml", export_dir, list(export_label_names.keys()),
                    dataset_name=self.dataset_name, use_test=use_test)

        split_names = self.get_split_image_names()
        im_num = 0

        for split_folder, image_names in split_names.items():

            for filename, image in self.data["images"].items():

                if filename not in image_names:
                    continue

                if not len(image["shapes"]):  # чтобы не создавать пустых файлов
                    continue

                fullname = os.path.join(self.data["path_to_images"], filename)

                if not os.path.exists(fullname):
                    continue

                txt_yolo_name = hf.convert_image_name_to_txt_name(filename)

                width, height = Image.open(fullname).size
                im_shape = [height, width]

                if is_blur:
                    blur_txt_name = os.path.join(blur_dir, txt_yolo_name)
                    blur_f = open(blur_txt_name, 'w')

                with open(os.path.join(labels_dir, split_folder, txt_yolo_name), 'w') as f:
                    for shape in image["shapes"]:
                        cls_num = shape["cls_num"]

                        if cls_num == -1 or cls_num > len(labels_names) - 1:
                            continue

                        label_name = labels_names[cls_num]
                        export_cls_num = export_map[label_name]

                        if export_cls_num == 'del':
                            continue

                        elif export_cls_num == 'blur':
                            self.write_yolo_seg_line(shape, im_shape, blur_f, 0)

                        else:
                            self.write_yolo_seg_line(shape, im_shape, f, export_cls_num)

                if is_blur:
                    blur_f.close()
                    mask = get_mask_from_yolo_txt(fullname, blur_txt_name, [0])
                    blurred_image_cv2 = blur_image_by_mask(fullname, mask)
                    cv2.imwrite(os.path.join(images_dir, split_folder, filename), blurred_image_cv2)
                else:
                    shutil.copy(fullname, os.path.join(images_dir, split_folder, filename))

                im_num += 1
                self.export_percent_conn.percent.emit(int(100 * im_num / (len(self.data['images']))))

    def emit_percent(self, value):
        self.export_percent_conn.percent.emit(value)

    def exportToYOLOBox(self):

        export_dir = self.export_dir
        export_map = self.export_map

        if not os.path.isdir(export_dir):
            return

        self.clear_not_existing_images()
        labels_names = self.get_labels()

        if not export_map:
            export_map = self.get_export_map(labels_names)

        images_dir, labels_dir = self.create_images_labels_subdirs(export_dir)
        is_blur = self.is_blurred_classes(export_map)

        if is_blur:
            blur_dir = self.create_blur_dir(export_dir)

        export_label_names = {}
        unique_values = []
        for k, v in export_map.items():
            if v != 'del' and v != 'blur' and v not in unique_values:
                export_label_names[k] = v
                unique_values.append(v)

        use_test = True if self.variant_idx == 0 else False
        create_yaml(f"{self.dataset_name}.yaml", export_dir, list(export_label_names.keys()),
                    dataset_name=self.dataset_name, use_test=use_test)

        split_names = self.get_split_image_names()
        im_num = 0

        for split_folder, image_names in split_names.items():

            for filename, image in self.data["images"].items():

                if filename not in image_names:
                    continue

                if not len(image["shapes"]):  # чтобы не создавать пустых файлов
                    continue

                fullname = os.path.join(self.data["path_to_images"], filename)
                txt_yolo_name = hf.convert_image_name_to_txt_name(filename)

                width, height = Image.open(fullname).size
                im_shape = [height, width]

                if is_blur:
                    blur_txt_name = os.path.join(blur_dir, txt_yolo_name)
                    blur_f = open(blur_txt_name, 'w')

                with open(os.path.join(labels_dir, split_folder, txt_yolo_name), 'w') as f:
                    for shape in image["shapes"]:
                        cls_num = shape["cls_num"]

                        if cls_num == -1 or cls_num > len(labels_names) - 1:
                            continue

                        label_name = labels_names[cls_num]

                        export_cls_num = export_map[label_name]

                        if export_cls_num == 'del':

                            continue

                        elif export_cls_num == 'blur':
                            self.write_yolo_box_line(shape, im_shape, blur_f, 0)

                        else:
                            self.write_yolo_box_line(shape, im_shape, f, export_cls_num)
                if is_blur:
                    blur_f.close()
                    mask = get_mask_from_yolo_txt(fullname, blur_txt_name, [0])
                    blurred_image_cv2 = blur_image_by_mask(fullname, mask)
                    cv2.imwrite(os.path.join(images_dir, split_folder, filename), blurred_image_cv2)

                else:
                    shutil.copy(fullname, os.path.join(images_dir, split_folder, filename))

                im_num += 1
                self.export_percent_conn.percent.emit(int(100 * im_num / (len(self.data['images']))))

    def exportToCOCO(self):

        export_dir = self.export_dir
        export_map = self.export_map

        self.clear_not_existing_images()
        labels_names = self.get_labels()

        if not export_map:
            export_map = self.get_export_map(labels_names)

        images_dir = self.create_images_labels_subdirs(export_dir, is_labels_need=False)
        is_blur = self.is_blurred_classes(export_map)

        if is_blur:
            blur_dir = self.create_blur_dir(export_dir)

        split_names = self.get_split_image_names()

        for split_folder, image_names in split_names.items():

            # split folder - one of [train, val, test]

            export_json = {}
            export_json["info"] = {"year": datetime.date.today().year, "version": "1.0",
                                   "description": "exported to COCO format using AI Annotator", "contributor": "",
                                   "url": "", "date_created": datetime.date.today().strftime("%c")}

            export_json["images"] = []

            id_tek = 1
            id_map = {}

            for filename, image in self.data["images"].items():

                if filename not in image_names:
                    continue

                id_map[filename] = id_tek
                im_full_path = os.path.join(self.data["path_to_images"], filename)
                im_save_path = os.path.join(images_dir, split_folder, filename)

                if not os.path.exists(im_full_path):
                    continue

                width, height = Image.open(im_full_path).size
                im_shape = [height, width]

                width = im_shape[1]
                height = im_shape[0]
                im_dict = {"id": id_tek, "width": width, "height": height, "file_name": filename, "license": 0,
                           "flickr_url": im_save_path, "coco_url": im_save_path, "date_captured": ""}
                export_json["images"].append(im_dict)

                id_tek += 1

            export_json["annotations"] = []

            seg_id = 1
            im_num = 0
            for filename, image in self.data["images"].items():

                if filename not in image_names:
                    continue

                fullname = os.path.join(self.data["path_to_images"], filename)

                txt_yolo_name = hf.convert_image_name_to_txt_name(filename)
                if not os.path.exists(fullname):
                    continue

                if is_blur:
                    blur_txt_name = os.path.join(blur_dir, txt_yolo_name)
                    blur_f = open(blur_txt_name, 'w')

                for shape in image["shapes"]:

                    cls_num = shape["cls_num"]

                    if cls_num == -1 or cls_num > len(labels_names) - 1:
                        continue

                    label_name = labels_names[cls_num]
                    export_cls_num = export_map[label_name]

                    if export_cls_num == 'del':
                        continue

                    elif export_cls_num == 'blur':
                        self.write_yolo_seg_line(shape, im_shape, blur_f, 0)

                    else:
                        points = shape["points"]
                        xs = []
                        ys = []
                        all_points = [[]]
                        for point in points:
                            xs.append(point[0])
                            ys.append(point[1])
                            all_points[0].append(int(point[0]))
                            all_points[0].append(int(point[1]))

                        seg = np.array(all_points[0])

                        poly = np.reshape(seg, (seg.size // 2, 2))
                        poly = Polygon(poly)
                        area = poly.area

                        min_x = min(xs)
                        max_x = max(xs)
                        min_y = min(ys)
                        max_y = max(ys)
                        w = abs(max_x - min_x)
                        h = abs(max_y - min_y)

                        x_center = min_x + w / 2
                        y_center = min_y + h / 2

                        bbox = [int(x_center), int(y_center), int(width), int(height)]

                        seg = {"segmentation": all_points, "area": int(area), "bbox": bbox, "iscrowd": 0, "id": seg_id,
                               "image_id": id_map[filename], "category_id": export_cls_num + 1}
                        export_json["annotations"].append(seg)
                        seg_id += 1

                if is_blur:
                    blur_f.close()
                    mask = get_mask_from_yolo_txt(fullname, blur_txt_name, [0])
                    blurred_image_cv2 = blur_image_by_mask(fullname, mask)
                    cv2.imwrite(os.path.join(images_dir, filename), blurred_image_cv2)

                else:
                    shutil.copy(fullname, os.path.join(images_dir, split_folder, filename))

                im_num += 1
                self.export_percent_conn.percent.emit(int(100 * im_num / (len(self.data['images']))))

            export_json["licenses"] = [{"id": 0, "name": "Unknown License", "url": ""}]
            export_json["categories"] = []

            for label in export_map:
                if export_map[label] != 'del' and export_map[label] != 'blur':
                    category = {"supercategory": "type", "id": export_map[label] + 1, "name": label}
                    export_json["categories"].append(category)

            with open(os.path.join(images_dir, split_folder, f"{split_folder}.json"), 'w') as f:
                ujson.dump(export_json, f)

    def get_image_path(self):
        return self.data["path_to_images"]

    def clear_not_existing_images(self):
        images = {}
        im_path = self.get_image_path()
        for filename, im in self.data['images'].items():
            if os.path.exists(os.path.join(im_path, filename)):
                images[filename] = im
            else:
                print(f"Checking files: image {filename} doesn't exist")

        self.data['images'] = images
