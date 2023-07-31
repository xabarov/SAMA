import utils.config as config
import datetime
import utils.help_functions as hf
import ujson
import numpy as np
import os

from PIL import Image

from shapely import Polygon


class ProjectHandler:
    """
    Класс для работы с данными проекта
    Хранит данные разметки в виде словаря
    """

    def __init__(self):
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
        cls_nums = {}
        for im in self.data['images']:
            for shape in im['shapes']:
                cls_num = shape['cls_num']
                if cls_num not in cls_nums:
                    cls_nums[cls_num] = 1
                else:
                    cls_nums[cls_num] += 1
        return cls_nums

    def load(self, json_path, on_load_callback=None):
        with open(json_path, 'r', encoding='utf8') as f:
            self.data = ujson.load(f)
            self.update_ids()
            self.is_loaded = True

            if on_load_callback:
                on_load_callback()

    def save(self, json_path, on_save_callback=None):
        with open(json_path, 'w', encoding='utf8') as f:
            ujson.dump(self.data, f)

            if on_save_callback:
                on_save_callback()

    def update_ids(self):
        id_num = 0
        for im in self.data['images']:
            for shape in im['shapes']:
                shape['id'] = id_num
                id_num += 1

    def set_data(self, data):
        self.data = data
        self.is_loaded = True

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

    def get_image_path(self):
        return self.data["path_to_images"]

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

    def delete_label(self, label_name):
        labels = []
        for label in self.data['labels']:
            if label != label_name:
                labels.append(label)
        self.set_labels(labels)

    def del_image(self, image_name):

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

            new_image = {}
            new_image["filename"] = image["filename"]
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

            new_image = {}
            new_image["filename"] = image["filename"]
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

    def get_export_map(self, export_label_names):
        label_names = self.get_labels()
        export_map = {}
        export_cls_num = 0
        for i, name in enumerate(label_names):
            if name in export_label_names:
                export_map[i] = export_cls_num
                export_cls_num += 1
            else:
                export_map[i] = -1

        return export_map

    def exportToYOLOSeg(self, export_dir, export_label_names=None):

        self.clear_not_existing_images()

        if not export_label_names:
            export_label_names = self.get_labels()
        export_map = self.get_export_map(export_label_names)

        if os.path.isdir(export_dir):
            for image in self.data["images"]:
                if len(image["shapes"]):  # чтобы не создавать пустых файлов
                    filename = image["filename"]
                    fullname = os.path.join(self.data["path_to_images"], filename)
                    # im = cv2.imread(fullname)  # height, width
                    txt_yolo_name = hf.convert_image_name_to_txt_name(filename)
                    # im_shape = im.shape
                    if not os.path.exists(fullname):
                        continue

                    width, height = Image.open(fullname).size
                    im_shape = [height, width]

                    with open(os.path.join(export_dir, txt_yolo_name), 'w') as f:
                        for shape in image["shapes"]:
                            cls_num = shape["cls_num"]
                            export_cls_num = export_map[cls_num]
                            if export_cls_num != -1:
                                points = shape["points"]
                                line = f"{export_cls_num}"
                                for point in points:
                                    line += f" {point[0] / im_shape[1]} {point[1] / im_shape[0]}"

                                f.write(f"{line}\n")
            return True
        return False

    def clear_not_existing_images(self):
        images = []
        im_path = self.get_image_path()
        for im in self.data['images']:
            if os.path.exists(os.path.join(im_path, im['filename'])):
                images.append(im)
            else:
                print(f"Checking files: image {im['filename']} doesn't exist")

        self.data['images'] = images

    def exportToCOCO(self, export_сoco_name, export_label_names=None):

        self.clear_not_existing_images()

        if not export_label_names:
            export_label_names = self.get_labels()
        export_map = self.get_export_map(export_label_names)

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
                # im = cv2.imread(im_full_path)
                # im_shape = im.shape

                if not os.path.exists(im_full_path):
                    continue

                width, height = Image.open(im_full_path).size
                im_shape = [height, width]

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
                    export_cls_num = export_map[cls_num]

                    if export_cls_num != -1:
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

            export_json["licenses"] = [{"id": 0, "name": "Unknown License", "url": ""}]
            export_json["categories"] = []

            for i, label in enumerate(export_label_names):
                category = {"supercategory": "type", "id": i + 1, "name": label}
                export_json["categories"].append(category)

            with open(export_сoco_name, 'w') as f:
                ujson.dump(export_json, f)

            return True

        return False

    def exportToYOLOBox(self, export_dir, export_label_names=None):

        self.clear_not_existing_images()

        if not export_label_names:
            export_label_names = self.get_labels()
        export_map = self.get_export_map(export_label_names)

        if os.path.isdir(export_dir):
            for image in self.data["images"]:
                if len(image["shapes"]):  # чтобы не создавать пустых файлов
                    filename = image["filename"]
                    fullname = os.path.join(self.data["path_to_images"], filename)
                    # im = cv2.imread(fullname)  # height, width
                    txt_yolo_name = hf.convert_image_name_to_txt_name(filename)
                    # im_shape = im.shape

                    width, height = Image.open(fullname).size
                    im_shape = [height, width]

                    with open(os.path.join(export_dir, txt_yolo_name), 'w') as f:
                        for shape in image["shapes"]:
                            cls_num = shape["cls_num"]

                            export_cls_num = export_map[cls_num]
                            if export_cls_num != -1:

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
                                    f"{export_cls_num} {x_center / im_shape[1]} {y_center / im_shape[0]} {w / im_shape[1]} {h / im_shape[0]}\n")

            return True
        return False


if __name__ == '__main__':
    proj_path = "D:\python\\ai_annotator\projects\\test.json"

    proj = ProjectHandler()
    proj.load(proj_path)
    # print(proj.exportToYOLOBox("D:\python\\ai_annotator\labels"))
    print(proj.exportToCOCO("D:\python\\ai_annotator\labels\\coco.json"))
