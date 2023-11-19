from xml.dom import minidom
import cv2
from collections import namedtuple
import os
import shutil
import numpy as np
from PIL import Image
from tqdm import tqdm
from yolo_converter import create_yaml

Label = namedtuple('Label', ['name', 'points'])

classes = {
    "Passenger Ship": {"id": 0, "category": "Ship"},
    "Motorboat": {"id": 1, "category": "Ship"},
    "Fishing Boat": {"id": 2, "category": "Ship"},
    "Tugboat": {"id": 3, "category": "Ship"},
    "other-ship": {"id": 4, "category": "Ship"},
    "Engineering Ship": {"id": 5, "category": "Ship"},
    "Liquid Cargo Ship": {"id": 6, "category": "Ship"},
    "Dry Cargo Ship": {"id": 7, "category": "Ship"},
    "Warship": {"id": 8, "category": "Ship"},
    "Small Car": {"id": 9, "category": "Vehicle"},
    "Bus": {"id": 10, "category": "Vehicle"},
    "Cargo Truck": {"id": 11, "category": "Vehicle"},
    "Dump Truck": {"id": 12, "category": "Vehicle"},
    "other-vehicle": {"id": 13, "category": "Vehicle"},
    "Van": {"id": 14, "category": "Vehicle"},
    "Trailer": {"id": 15, "category": "Vehicle"},
    "Tractor": {"id": 16, "category": "Vehicle"},
    "Excavator": {"id": 17, "category": "Vehicle"},
    "Truck Tractor": {"id": 18, "category": "Vehicle"},
    "Boeing737": {"id": 19, "category": "Airplane"},
    "Boeing747": {"id": 20, "category": "Airplane"},
    "Boeing777": {"id": 21, "category": "Airplane"},
    "Boeing787": {"id": 22, "category": "Airplane"},
    "ARJ21": {"id": 23, "category": "Airplane"},
    "C919": {"id": 24, "category": "Airplane"},
    "A220": {"id": 25, "category": "Airplane"},
    "A321": {"id": 26, "category": "Airplane"},
    "A330": {"id": 27, "category": "Airplane"},
    "A350": {"id": 28, "category": "Airplane"},
    "other-airplane": {"id": 29, "category": "Airplane"},
    "Baseball Field": {"id": 30, "category": "Court"},
    "Basketball Court": {"id": 31, "category": "Court"},
    "Football Field": {"id": 32, "category": "Court"},
    "Tennis Court": {"id": 33, "category": "Court"},
    "Roundabout": {"id": 34, "category": "Road"},
    "Intersection": {"id": 35, "category": "Road"},
    "Bridge": {"id": 36, "category": "Road"},
}


def convert_box_to_xywh(points, width, height):
    left_corner = points[0]
    right_corner = points[1]
    # absolute
    w = float(right_corner[0] - left_corner[0])
    h = float(right_corner[1] - left_corner[1])
    # center
    x = left_corner[0] + w / 2.0
    y = left_corner[1] + h / 2.0
    # relative

    w /= width
    h /= height
    x /= width
    y /= height

    return x, y, w, h


def convert_oriented_to_box(points):
    x_min = 1e12
    x_max = 0
    y_min = 1e12
    y_max = 0
    for p in points:
        if p[0] < x_min:
            x_min = p[0]
        if p[0] > x_max:
            x_max = p[0]
        if p[1] < y_min:
            y_min = p[1]
        if p[1] > y_max:
            y_max = p[1]

    return [[x_min, y_min], [x_max, y_max]]


def get_label_name(object_tag):
    return object_tag.getElementsByTagName('possibleresult')[0].getElementsByTagName('name')[0].firstChild.data


def get_points(object_tag):
    points = object_tag.getElementsByTagName('points')[0].getElementsByTagName('point')
    res = []
    for p in points:
        xy = p.firstChild.data.split(',')
        res.append([int(float(xy[0])), int(float(xy[1]))])

    return res


def get_labels_and_points(file_name):
    res = []
    mydoc = minidom.parse(file_name)

    tif_name = mydoc.getElementsByTagName('filename')[0].firstChild.data

    objects = mydoc.getElementsByTagName('object')

    for i, s in enumerate(objects):
        name = get_label_name(s)
        points = get_points(s)
        res.append(Label(name, points))

    return res


def create_yolo_dataset(fair1m_folder, yolo_folder, dataset_name='FAIR1M', fair1m_label_names=None):
    if not fair1m_label_names:
        fair1m_label_names = list(classes.keys())
    if not os.path.exists(yolo_folder):
        os.makedirs(yolo_folder)
    create_yaml(f'{dataset_name}.yaml', yolo_folder, fair1m_label_names, dataset_name=dataset_name)

    # create labels folder
    labels_folder = os.path.join(yolo_folder, 'labels')
    if not os.path.exists(labels_folder):
        os.makedirs(labels_folder)

    # create images folder
    images_folder = os.path.join(yolo_folder, 'images')
    if not os.path.exists(images_folder):
        os.makedirs(images_folder)

    # copy images

    for folder in ['train', 'val']:
        print(f'Start {folder} converting...')

        fair_sub_folder = os.path.join(fair1m_folder, folder)
        fair_labels_folder = os.path.join(fair_sub_folder, 'labels')
        fair_images_folder = os.path.join(fair_sub_folder, 'images')

        labels = [lbl for lbl in os.listdir(fair_labels_folder) if '.xml' in lbl]
        for i in tqdm(range(len(labels))):
            f_name = labels[i]
            full_path = os.path.join(fair_labels_folder, f_name)
            labels_and_points = get_labels_and_points(full_path)
            airplanes_labels = []
            for lbl in labels_and_points:
                if lbl.name in fair1m_label_names:
                    airplanes_labels.append(lbl)

            if len(airplanes_labels) > 0:
                image_full_path = os.path.join(fair_images_folder, f_name.split('.xml')[0] + '.tif')
                img = Image.open(image_full_path)
                img_width, img_height = img.size

                shutil.copy(image_full_path, os.path.join(images_folder, folder, f_name.split('.xml')[0] + '.tif'))

                with open(os.path.join(labels_folder, folder, f_name.split('.xml')[0] + '.txt'), 'w') as txt_file:
                    for lbl in airplanes_labels:
                        points = convert_oriented_to_box(lbl.points)
                        x, y, w, h = convert_box_to_xywh(points, img_width, img_height)
                        txt_file.write(f"{fair1m_label_names.index(lbl.name)} {x:0.6f} {y:0.6f} {w:0.6f} {h:0.6f}\n")


def create_airplanes_dataset(fair1m_folder, airplanes_folder, dataset_name='AirplanesFAIR1M'):
    airplane_names = ["Boeing737", "Boeing747", "Boeing777", "Boeing787", "ARJ21",
                      "C919", "A220", "A321", "A330", "A350", "other-airplane"]
    create_yolo_dataset(fair1m_folder, airplanes_folder, dataset_name=dataset_name, fair1m_label_names=airplane_names)


def test_lable(image_path, points):
    image = cv2.imread(image_path)

    start_point = points[0]

    # Ending coordinate, here (220, 220)
    # represents the bottom right corner of rectangle
    end_point = points[2]

    # Blue color in BGR
    color = (255, 0, 0)

    # Line thickness of 2 px
    thickness = 2

    # Using cv2.rectangle() method
    # Draw a rectangle with blue line borders of thickness of 2 px
    image = cv2.rectangle(image, start_point, end_point, color, thickness)

    # Display the image
    cv2.imshow("Image", image)

    # Wait for the user to press a key
    cv2.waitKey(0)

    # Close all windows
    cv2.destroyAllWindows()


if __name__ == '__main__':
    # label_name = 'D:\python\datasets\FAIR1M\\train\labels\\0.xml'
    # labels_and_points = get_labels_and_points(label_name)
    #
    # tif_name = "D:\python\datasets\FAIR1M\\train\images\\0.tif"
    #
    # points = convert_oriented_to_box(labels_and_points[1].points)
    #
    # test_lable(tif_name, points)

    # create_airplanes_dataset("D:\python\datasets\FAIR1M", "D:\python\datasets\FAIR1M\\airplanes")
    # create_yaml('yaml_test.yaml', os.getcwd(), ["Boeing737", "Boeing747", "Boeing777", "Boeing787", "ARJ21",
    #                                             "C919", "A220", "A321", "A330", "A350", "other-airplane"],
    #             dataset_name='AirplanesFAIR1M')
    create_yolo_dataset("D:\python\datasets\FAIR1M", "D:\python\datasets\FAIR1M\\FAIR1M_yolo_box",
                        dataset_name='FAIR1M_yolo_box')
