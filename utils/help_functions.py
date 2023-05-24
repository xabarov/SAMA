from PyQt5 import QtCore
from utils import config

import math
import matplotlib as mpl
import numpy as np
import yaml


def generate_set_of_label_colors():
    colors = []
    color_sec = ['tab10']
    for sec in color_sec:
        cmap = mpl.color_sequences[sec]
        for color in cmap:
            color_rgba = [int(255 * c) for c in color]
            color_rgba.append(255)
            colors.append(tuple(color_rgba))
    return colors


def get_label_colors(names, alpha=120):
    colors = {}
    if not alpha:
        alpha = 255

    for name in names:
        selected_color = config.COLORS[0]
        tek_color_num = 0
        is_break = False
        while selected_color in colors.values():
            tek_color_num += 1
            if tek_color_num == len(config.COLORS) - 1:
                is_break = True
                break
            selected_color = config.COLORS[tek_color_num]

        if is_break:
            selected_color = create_random_color(alpha)

        colors[name] = selected_color

    return colors

def read_yolo_yaml(yolo_yaml):
    with open(yolo_yaml, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        return yaml_data


def convert_percent_to_alpha(percent, alpha_min=15, alpha_max=200):
    return alpha_min + int(percent * (alpha_max - alpha_min) / 100.0)


def set_alpha_to_max(rgba):
    return (rgba[0], rgba[1], rgba[2], 255)


def distance(p1, p2):
    return math.sqrt(pow(p1.x() - p2.x(), 2) + pow(p1.y() - p2.y(), 2))


def calc_abc(p1, p2):
    """
    Вычисление параметров A, B, C из уравнения прямой по двум заданным точкам p1 и p2
    """
    a = p2.y() - p1.y()
    b = -(p2.x() - p1.x())
    c = p2.x() * p1.y() - p2.y() * p1.x()
    return a, b, c


def distance_from_point_to_line(p, line_p1, line_p2):
    a, b, c = calc_abc(line_p1, line_p2)
    chisl = abs(a * p.x() + b * p.y() + c)
    znam = math.sqrt(a * a + b * b)
    if znam > 1e-8:
        return chisl / znam
    return 0


def find_nearest_edge_of_polygon(polygon, point):
    d_min = 1e12
    edge = None
    size = len(polygon)
    for i in range(size):
        p1 = polygon[i]
        if i == size - 1:
            p2 = polygon[0]
        else:
            p2 = polygon[i + 1]
        if p1 != p2:
            d = distance_from_point_to_line(point, p1, p2)
            if d < d_min:
                d_min = d
                edge = p1, p2

    return edge


def density_slider_to_value(value, min_value=config.MIN_DENSITY_VALUE, max_value=config.MAX_DENSITY_VALUE):
    b = 0.01 * math.log(max_value / min_value)
    return min_value * math.exp(b * value)


def get_closest_to_line_point(point, p1_line, p2_line):
    a, b, c = calc_abc(p1_line, p2_line)
    znam = a * a + b * b

    if znam != 0:
        x = (b * (b * point.x() - a * point.y()) - a * c) / znam

        y = (a * (-b * point.x() + a * point.y()) - b * c) / znam

        return QtCore.QPointF(x, y)

    return None


def create_random_color(alpha):
    rgba = [0, 0, 0, alpha]
    for i in range(3):
        rgba[i] = np.random.randint(0, 256)

    return rgba


def is_im_path(im_path, suffixes=['jpg', 'tiff', 'png', 'jpeg']):
    for s in suffixes:
        if im_path.endswith(s):
            return True
    return False


def calc_ellips_point_coords(ellipse_rect, angle):
    tl = ellipse_rect.topLeft()
    br = ellipse_rect.bottomRight()
    width = abs(br.x() - tl.x())
    height = abs(br.y() - tl.y())
    a = width / 2.0
    b = height / 2.0
    sn = math.sin(angle)
    cs = math.cos(angle)
    t = math.atan2(a * sn, b * cs)
    sn = math.sin(t)
    cs = math.cos(t)
    x_center = tl.x() + a
    y_center = tl.y() + b
    return QtCore.QPointF(x_center + a * cs, y_center + b * sn)


def convert_image_name_to_txt_name(image_name):
    splitted_name = image_name.split('.')
    txt_name = ""
    for i in range(len(splitted_name) - 1):
        txt_name += splitted_name[i]

    return txt_name + ".txt"


def convert_text_name_to_image_name(text_name):
    splitted_name = text_name.split('.')
    img_name = ""
    for i in range(len(splitted_name) - 1):
        img_name += splitted_name[i]

    return img_name + ".txt"


if __name__ == '__main__':
    print(density_slider_to_value(10))