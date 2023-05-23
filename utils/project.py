import utils.config as config
import cv2
import datetime
import utils.help_functions as hf
import json
import numpy as np
import os
import shutil
import yaml

from shapely import Polygon
from ui.signals_and_slots import LoadPercentConnection


class ProjectHandler:
    """
    Класс для работы с данными проекта
    Хранит данные разметки в виде словаря
    """

    def __init__(self):
        self.data = dict()
        self.data["path_to_images"] = ""
        self.data["images"] = []
        self.data["labels"] = []
        self.data["labels_color"] = {}

        self.load_percent_conn = LoadPercentConnection()

    def check_json(self, json_project_data):
        for field in ["path_to_images", "images", "labels", "labels_color"]:
            if field not in json_project_data:
                return False
        return True

    def load(self, json_path):
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                data = json.load(f)

                if not self.check_json(data):
                    return False

                self.data = data
                return True

        return False

    def save(self, json_path):

        with open(json_path, 'w') as f:
            json.dump(self.data, f)

    def get_data(self):
        return self.data

    def get_label_color(self, cls_name):
        if cls_name in self.data["labels_color"]:
            return self.data["labels_color"][cls_name]

        return None

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

    def get_image_path(self):
        return self.data["path_to_images"]

    def rename_color(self, old_name, new_name):
        if old_name in self.data["labels_color"]:
            color = self.data["labels_color"][old_name]
            self.data["labels_color"][new_name] = color
            del self.data["labels_color"][old_name]

    def set_labels(self, labels):
        self.data["labels"] = labels

    def set_path_to_images(self, path):
        self.data["path_to_images"] = path

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

    def set_labels_colors(self, labels_names):
        for label_name in labels_names:
            if label_name not in self.data["labels_color"]:
                self.set_label_color(label_name)

    def set_labels_names(self, labels):
        self.data["labels"] = labels

    def set_image_data(self, image_data):
        image_name = image_data["filename"]
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

    def delete_label_color(self, label_name):
        if label_name in self.data["labels_color"]:
            del self.data["labels_color"][label_name]

    def del_image(self, image_name):

        images = []
        for image in self.data["images"]:
            if image_name != image["filename"]:
                images.append(image)

        self.data["images"] = images

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

            new_image = {}
            new_image["filename"] = image["filename"]
            new_image["shapes"] = new_shapes

            images.append(new_image)

        self.data["images"] = images

    def change_images_class_from_to(self, from_cls_num, to_cls_num):
        images = []
        for image in self.data["images"]:
            new_shapes = []
            for shape in image["shapes"]:

                if shape["cls_num"] == from_cls_num:
                    shape_new = {}
                    shape_new["cls_num"] = to_cls_num
                    shape_new["points"] = shape["points"]
                    shape_new["id"] = shape["id"]
                    new_shapes.append(shape_new)
                else:
                    new_shapes.append(shape)

            new_image = {}
            new_image["filename"] = image["filename"]
            new_image["shapes"] = new_shapes

            images.append(new_image)

        self.data["images"] = images

    def exportToYOLOSeg(self, export_dir):
        if os.path.isdir(export_dir):
            for image in self.data["images"]:
                if len(image["shapes"]):  # чтобы не создавать пустых файлов
                    filename = image["filename"]
                    fullname = os.path.join(self.data["path_to_images"], filename)
                    im = cv2.imread(fullname)  # height, width
                    txt_yolo_name = hf.convert_image_name_to_txt_name(filename)
                    im_shape = im.shape
                    with open(os.path.join(export_dir, txt_yolo_name), 'w') as f:
                        for shape in image["shapes"]:
                            cls_num = shape["cls_num"]
                            points = shape["points"]
                            line = f"{cls_num}"
                            for point in points:
                                line += f" {point[0] / im_shape[1]} {point[1] / im_shape[0]}"

                            f.write(f"{line}\n")
            return True
        return False

    def exportToCOCO(self, export_сoco_name):

        if os.path.isdir(os.path.dirname(export_сoco_name)):
            export_json = {}
            export_json["info"] = {"year": datetime.date.today().year, "version": "1.0",
                                   "description": "exported to COCO format using AI Annotator", "contributor": "",
                                   "url": "", "date_created": datetime.date.today().strftime("%c")}

            export_json["images"] = []

            id_tek = 1
            id_map = {}

            for image in self.data["images"]:
                filename = image["filename"]
                id_map[filename] = id_tek
                im_full_path = os.path.join(self.data["path_to_images"], filename)
                im = cv2.imread(im_full_path)
                im_shape = im.shape
                width = im_shape[1]
                height = im_shape[0]
                im_dict = {"id": id_tek, "width": width, "height": height, "file_name": filename, "license": 0,
                           "flickr_url": im_full_path, "coco_url": im_full_path, "date_captured": ""}
                export_json["images"].append(im_dict)

                id_tek += 1

            export_json["annotations"] = []

            seg_id = 1
            for image in self.data["images"]:
                filename = image["filename"]
                for shape in image["shapes"]:

                    cls_num = shape["cls_num"]
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
                           "image_id": id_map[filename], "category_id": cls_num + 1}
                    export_json["annotations"].append(seg)
                    seg_id += 1

            export_json["licenses"] = [{"id": 0, "name": "Unknown License", "url": ""}]
            export_json["categories"] = []

            for i, label in enumerate(self.data["labels"]):
                category = {"supercategory": "type", "id": i + 1, "name": label}
                export_json["categories"].append(category)

            with open(export_сoco_name, 'w') as f:
                json.dump(export_json, f)

            return True

        return False

    def import_from_coco(self, data, alpha=120, label_names=None):
        if not label_names:
            label_names = [d["name"] for d in data["categories"]]

        label_colors = hf.get_label_colors(label_names, alpha=alpha)
        
        project = {'path_to_images': os.path.dirname(data["images"][0]["coco_url"]),
                   "images": [], 'labels': label_names, 'labels_color': label_colors}
        id_num = 0
        for i, im in enumerate(data["images"]):
            im_id = im["id"]
            proj_im = {'filename': im["file_name"], 'shapes': []}
            for seg in data["annotations"]:
                if seg["image_id"] == im_id:
                    cls = seg["category_id"]
                    points = [[seg["segmentation"][0][i], seg["segmentation"][0][i + 1]] for i in
                              range(0, len(seg["segmentation"][0]), 2)]
                    shape = {"id": id_num, "cls_num": cls, 'points': points}
                    id_num += 1
                    proj_im["shapes"].append(shape)

            project['images'].append(proj_im)

            self.load_percent_conn.percent.emit(int(i * 100.0 / len(data["images"])))

        self.data = project

    def import_from_yolo_yaml(self, yaml_path, dataset="train", alpha=120, is_seg=False, copy_images_path=None):
        yaml_data = hf.read_yolo_yaml(yaml_path)
        path_to_labels = os.path.join(yaml_data["path"], "labels", dataset)

        if copy_images_path:
            path_to_images = os.path.join(yaml_data["path"], "images", dataset)
            images = [im for im in os.listdir(path_to_images) if hf.is_im_path(im)]

            # copy images
            for i, im in enumerate(images):
                shutil.copy(os.path.join(path_to_images, im), os.path.join(copy_images_path, im))
                self.load_percent_conn.percent.emit(int(i * 100.0 / len(images)))

            path_to_images = copy_images_path

        else:
            path_to_images = os.path.join(yaml_data["path"], "images", dataset)

        labels_names = yaml_data["names"]
        label_colors = hf.get_label_colors(labels_names, alpha=alpha)

        if is_seg:
            self.import_from_yolo_seg(path_to_labels, path_to_images, labels_names, label_colors)
        else:
            self.import_from_yolo_box(path_to_labels, path_to_images, labels_names, label_colors)

    def import_from_yolo_box(self, path_to_labels, path_to_images, labels_names, labels_color):
        project = {'path_to_images': path_to_images, 'images': [], "labels": labels_names, "labels_color": labels_color}
        images = [im for im in os.listdir(path_to_images) if hf.is_im_path(im)]
        id_num = 0
        for i, im in enumerate(images):
            im_shape = cv2.imread(os.path.join(path_to_images, im)).shape
            txt_name = hf.convert_image_name_to_txt_name(im)
            image_data = {"filename": im, 'shapes': []}
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
                    image_data["shapes"].append(shape)

            project['images'].append(image_data)
            self.load_percent_conn.percent.emit(int(i * 100.0 / len(images)))

        self.data = project

    def import_from_yolo_seg(self, path_to_labels, path_to_images, labels_names, labels_color):
        project = {'path_to_images': path_to_images, 'images': [], "labels": labels_names, "labels_color": labels_color}
        images = [im for im in os.listdir(path_to_images) if hf.is_im_path(im)]
        id_num = 0
        for i, im in enumerate(images):
            im_shape = cv2.imread(os.path.join(path_to_images, im)).shape
            txt_name = hf.convert_image_name_to_txt_name(im)
            image_data = {"filename": im, 'shapes': []}
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

            project['images'].append(image_data)
            self.load_percent_conn.percent.emit(int(i * 100.0 / len(images)))

        self.data = project

    def exportToYOLOBox(self, export_dir):

        if os.path.isdir(export_dir):
            for image in self.data["images"]:
                if len(image["shapes"]):  # чтобы не создавать пустых файлов
                    filename = image["filename"]
                    fullname = os.path.join(self.data["path_to_images"], filename)
                    im = cv2.imread(fullname)  # height, width
                    txt_yolo_name = hf.convert_image_name_to_txt_name(filename)
                    im_shape = im.shape
                    with open(os.path.join(export_dir, txt_yolo_name), 'w') as f:
                        for shape in image["shapes"]:
                            cls_num = shape["cls_num"]
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

            return True
        return False


if __name__ == '__main__':
    proj_path = "D:\python\\ai_annotator\projects\\test.json"

    proj = ProjectHandler()
    proj.load(proj_path)
    # print(proj.exportToYOLOBox("D:\python\\ai_annotator\labels"))
    print(proj.exportToCOCO("D:\python\\ai_annotator\labels\\coco.json"))
