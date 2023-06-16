from collections import namedtuple
from PIL import Image
from utils.gdal_translate import get_extent, get_data

import cv2
import os

Coords = namedtuple('Coords', ['latitude', 'longitude'])
CNN_coords = namedtuple('CNN_coords', ['class_name', 'prob', 'upper_left_coords', 'bottom_right_coords'])


def load_coords(file_name):
    ext = get_ext(file_name)
    if ext == 'map':
        return load_coords_map(file_name)

    elif ext == 'dat':

        with open(file_name, 'r', encoding='UTF-8') as file:
            coords_net = []
            for i, line in enumerate(file):
                if i != 0 and i < 5:
                    str_ = line.rstrip()
                    lon, lat = str_.split(',')
                    coords_net.append(Coords(float(lat), float(lon)))

            return coords_net


def get_geo_extent(image_filename):
    coords_file = find_coords_file(image_filename)
    if coords_file:
        coords_net = load_coords(coords_file)
        if coords_net:
            return get_lat_lon_min_max_coords(coords_net)

    if get_ext(image_filename) == 'tif':

        extent = get_extent(image_filename)  # [(left_poit lat, lon), (right_point]
        coords_net = []
        coords_net.append(Coords(float(extent[0][1]), float(extent[0][0])))
        coords_net.append(Coords(float(extent[1][1]), float(extent[1][0])))

        return get_lat_lon_min_max_coords(coords_net)


def lrm_from_gdal_data(image_name, gdal_data):
    extent = get_extent(image_name) # [left top, bottom right]
    max_y = extent[0][1]
    min_y = extent[1][1]
    delta_lat = abs(max_y - min_y)  # в градусах
    meters = delta_lat * 111.32 * 1000

    return meters / gdal_data.RasterYSize


def find_coords_file(file_name):
    basename_list = os.path.basename(file_name).split('.')
    base_name = ""
    for s in range(len(basename_list) - 1):
        base_name += basename_list[s]

    for ext in ['.kml', '.dat', '.map']:
        name = base_name + ext
        name = os.path.join(os.path.dirname(file_name), name)

        if os.path.exists(name):
            return name


def get_ext(file_name):
    return file_name.split('.')[-1]


# def load_coords_kml(file_name):
#     mydoc = minidom.parse(file_name)
#     coords_net = []
#     north = mydoc.getElementsByTagName('north')[0].firstChild.data
#     east = mydoc.getElementsByTagName('east')[0].firstChild.data
#     west = mydoc.getElementsByTagName('west')[0].firstChild.data
#     south = mydoc.getElementsByTagName('south')[0].firstChild.data
#
#     coords_net.append(Coords(float(north), float(east)))
#     coords_net.append(Coords(float(south), float(west)))
#
#     return coords_net


def load_coords_map(file_name):
    with open(file_name, 'r', encoding='UTF-8') as file:
        coords_net = []
        for i, line in enumerate(file):
            if 'MMPLL' in line:
                lon, lat = [float(coord) for coord in line.strip().split(',')[-2:]]
                coords_net.append(Coords(float(lat), float(lon)))
        return coords_net


def get_lat_lon_min_max_coords(coords_net):
    lat_max = max([x.latitude for x in coords_net])
    lat_min = min([x.latitude for x in coords_net])
    lon_max = max([x.longitude for x in coords_net])
    lon_min = min([x.longitude for x in coords_net])

    return lat_min, lat_max, lon_min, lon_max


def get_n_s_w_e(coords_net):
    res = []
    if coords_net[0].latitude > 0:
        res.append('N')
    else:
        res.append('S')

    if coords_net[0].longitude > 0:
        res.append('E')
    else:
        res.append('W')

    return res


def conver_grad_min_sec_to_text(coords, lat=True, n_s="N", e_w="E"):
    if lat:
        text = "{0:02d} {1:02d}'{2:02d}''{3:s}".format(coords[0], coords[1], int(coords[2]), n_s)
    else:
        text = "{0:02d} {1:02d}'{2:02d}''{3:s}".format(coords[0], coords[1], int(coords[2]), e_w)

    return text


def add_grid(image_file_name, save_name, num_of_grid_cells=None, dat_file_name=None,
             font=cv2.FONT_HERSHEY_COMPLEX,
             fontScale=None,
             fontColor=(0, 255, 255),
             thickness=None,
             lineType=2,
             grid_color=(0, 0, 0)):
    img = cv2.imread(image_file_name)

    shape = img.shape
    if not fontScale:
        fontScale = min(shape[0], shape[1]) / 1900

    if not thickness:
        thickness = int(min(shape[0], shape[1]) / 1900) + 1

    if not num_of_grid_cells:
        num_of_grid_cells = 8

    delta = int(max(shape) / num_of_grid_cells)

    dx, dy = delta, delta

    img[:, ::dy, :] = grid_color
    img[::dx, :, :] = grid_color

    if shape[0] > 3000:
        img[:, ::dy + 1, :] = grid_color
        img[::dx + 1, :, :] = grid_color

    if shape[0] > 8000:
        img[:, ::dy + 2, :] = grid_color
        img[::dx + 2, :, :] = grid_color

    if dat_file_name:

        coords = load_coords(dat_file_name)

        min_max = get_lat_lon_min_max_coords(coords)
        lon_start = min_max[2]
        lon_finish = min_max[3]
        delta_lon = lon_finish - lon_start
        ns = get_n_s_w_e(coords)
        grad_in_pix = delta_lon / shape[1]

        num_of_w_steps = int(shape[1] / dx)

        for st in range(num_of_w_steps + 1):
            # направление зависит от знака
            if ns[1] == 'E':
                next_lon = lon_start + grad_in_pix * delta * (st + 1)
                bottomLeftCornerOfText = [st * delta + 6, int(shape[0] / 2)]
            else:
                tail = int(shape[1] - num_of_w_steps * delta)
                start_w = shape[1] - tail

                if st == 0:
                    next_lon = lon_start + grad_in_pix * tail
                else:
                    next_lon = lon_start + grad_in_pix * delta * st
                bottomLeftCornerOfText = [start_w - st * delta + 6, int(shape[0] / 2)]

            lon_text = conver_grad_min_sec_to_text(convert_coords_to_grad_min_sec(next_lon), lat=False, e_w=ns[1])
            cv2.putText(img, lon_text,
                        bottomLeftCornerOfText,
                        font,
                        fontScale,
                        fontColor,
                        thickness,
                        lineType)

            lat_start = min_max[0]
            lat_finish = min_max[1]
            delta_lat = lat_finish - lat_start
            grad_in_pix = delta_lat / shape[0]

            num_of_h_steps = int(shape[0] / dy)
            tail = int(shape[0] - num_of_h_steps * delta)
            start_h = shape[0] - tail

            for st in range(num_of_h_steps + 1):
                if st == int(num_of_h_steps / 2):
                    continue

                if ns[0] == 'N':
                    if st == 0:
                        next_lat = lat_start + grad_in_pix * tail
                    else:
                        next_lat = lat_start + grad_in_pix * delta * st
                    bottomLeftCornerOfText = [int(shape[1] / 2) - 34, start_h - st * delta - 12]
                else:
                    next_lat = lat_start + grad_in_pix * delta * (st + 1)
                    bottomLeftCornerOfText = [int(shape[1] / 2) - 34, st * delta + 24]

                lat_text = conver_grad_min_sec_to_text(convert_coords_to_grad_min_sec(next_lat), lat=True, n_s=ns[0])
                cv2.putText(img, lat_text,
                            bottomLeftCornerOfText,
                            font,
                            fontScale,
                            fontColor,
                            thickness,
                            lineType)

    cv2.imwrite(save_name, img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def convert_coords_to_grad_min_sec(lat_or_lon):
    grads = int(lat_or_lon)
    mins_float = (lat_or_lon - grads) * 60
    mins = int(mins_float)
    secs_float = (mins_float - mins) * 60

    return (grads, mins, secs_float)


def convert_coco_to_coords(coco_label_file, dat_file, img_file_name):
    coords_net = load_coords(dat_file)
    img = Image.open(img_file_name)

    img_width, img_height = img.size

    n_s_w_e = get_n_s_w_e(coords_net)
    min_max = get_lat_lon_min_max_coords(coords_net)

    lat_start = min_max[0]
    lat_finish = min_max[1]
    delta_lat = lat_finish - lat_start
    grad_lat_in_pix = delta_lat / im_height

    lon_start = min_max[2]
    lon_finish = min_max[3]
    delta_lon = lon_finish - lon_start
    grad_lon_in_pix = delta_lon / img_width

    res = []

    if not os.path.exists(coco_label_file):
        print("Coco label file doesn't exist")
        return res

    with open(coco_label_file, "r") as f:
        for i, line in enumerate(f):
            str_y = line.rstrip().split(' ')
            class_name = int(str_y[1])
            prob = float(str_y[3])

            upper_left_x = float(str_y[5].rstrip())
            upper_left_y = float(str_y[6].rstrip())

            bottom_right_x = float(str_y[7].rstrip())
            bottom_right_y = float(str_y[8].rstrip())

            if n_s_w_e[0] == "N":
                ul_lat_yolo = lat_start + (img_height - upper_left_y) * grad_lat_in_pix
                br_lat_yolo = lat_start + (img_height - bottom_right_y) * grad_lat_in_pix

            else:
                ul_lat_yolo = -(lat_start + upper_left_y * grad_lat_in_pix)
                br_lat_yolo = -(lat_start + bottom_right_y * grad_lat_in_pix)

            if n_s_w_e[1] == "E":
                ul_lon_yolo = lon_start + upper_left_x * grad_lon_in_pix
                br_lon_yolo = lon_start + bottom_right_x * grad_lon_in_pix

            else:
                ul_lon_yolo = -(lon_start + (img_width - upper_left_x) * grad_lon_in_pix)
                br_lon_yolo = -(lon_start + (img_width - bottom_right_x) * grad_lon_in_pix)

            res.append(
                CNN_coords(class_name, prob, Coords(ul_lat_yolo, ul_lon_yolo), Coords(br_lat_yolo, br_lon_yolo)))

    return res


def crop_from_coords(dat_file, img_file_name, x_min, x_max, y_min, y_max, save_dat_name):
    coords_net = load_coords(dat_file)
    img = Image.open(img_file_name)

    img_width, img_height = img.size

    n_s_w_e = get_n_s_w_e(coords_net)
    min_max = get_lat_lon_min_max_coords(coords_net)

    lat_start = min_max[0]
    lat_finish = min_max[1]
    delta_lat = lat_finish - lat_start
    grad_lat_in_pix = delta_lat / im_height

    lon_start = min_max[2]
    lon_finish = min_max[3]
    delta_lon = lon_finish - lon_start
    grad_lon_in_pix = delta_lon / img_width

    res = []

    if n_s_w_e[0] == "N":
        ul_lat_yolo = lat_start + (img_height - y_min) * grad_lat_in_pix
        br_lat_yolo = lat_start + (img_height - y_max) * grad_lat_in_pix

    else:
        ul_lat_yolo = -(lat_start + y_min * grad_lat_in_pix)
        br_lat_yolo = -(lat_start + y_max * grad_lat_in_pix)

    if n_s_w_e[1] == "E":
        ul_lon_yolo = lon_start + x_min * grad_lon_in_pix
        br_lon_yolo = lon_start + x_max * grad_lon_in_pix

    else:
        ul_lon_yolo = -(lon_start + (img_width - x_min) * grad_lon_in_pix)
        br_lon_yolo = -(lon_start + (img_width - x_max) * grad_lon_in_pix)

    with open(save_dat_name, "w") as safe_f:
        safe_f.write('2\n')
        for i in range(2):
            safe_f.write('{0:f},{1:f}\n'.format(ul_lon_yolo, ul_lat_yolo))
            safe_f.write('{0:f},{1:f}\n'.format(br_lon_yolo, br_lat_yolo))

    return res


def convert_yolo_to_coords(yolo_label_file, dat_file, img_file_name):
    coords_net = load_coords(dat_file)
    img = Image.open(img_file_name)

    img_width, img_height = img.size

    n_s_w_e = get_n_s_w_e(coords_net)
    min_max = get_lat_lon_min_max_coords(coords_net)

    lat_start = min_max[0]
    lat_finish = min_max[1]
    delta_lat = lat_finish - lat_start
    grad_lat_in_pix = delta_lat / im_height

    lon_start = min_max[2]
    lon_finish = min_max[3]
    delta_lon = lon_finish - lon_start
    grad_lon_in_pix = delta_lon / img_width

    res = []

    if not os.path.exists(yolo_label_file):
        print("Yolo label file doesn't exist")
        return res

    with open(yolo_label_file, "r") as f:
        for i, line in enumerate(f):
            str_y = line.rstrip().split(' ')
            class_name = int(str_y[0])
            prob = float(str_y[-1])

            box_center_x, box_center_y = float(str_y[1]), float(str_y[2])
            box_center_w, box_center_h = float(str_y[3]), float(str_y[4])

            upper_left_x = box_center_x - box_center_w / 2
            upper_left_y = box_center_y - box_center_h / 2

            bottom_right_x = box_center_x + box_center_w / 2
            bottom_right_y = box_center_y + box_center_h / 2

            if n_s_w_e[0] == "N":
                ul_lat_yolo = lat_start + (1.0 - upper_left_y) * img_height * grad_lat_in_pix
                br_lat_yolo = lat_start + (1.0 - bottom_right_y) * img_height * grad_lat_in_pix

            else:
                ul_lat_yolo = -(lat_start + upper_left_y * img_height * grad_lat_in_pix)
                br_lat_yolo = -(lat_start + bottom_right_y * img_height * grad_lat_in_pix)

            if n_s_w_e[1] == "E":
                ul_lon_yolo = lon_start + upper_left_x * img_width * grad_lon_in_pix
                br_lon_yolo = lon_start + bottom_right_x * img_width * grad_lon_in_pix

            else:
                ul_lon_yolo = -(lon_start + (1.0 - upper_left_x) * img_width * grad_lon_in_pix)
                br_lon_yolo = -(lon_start + (1.0 - bottom_right_x) * img_width * grad_lon_in_pix)

            res.append(
                CNN_coords(class_name, prob, Coords(ul_lat_yolo, ul_lon_yolo), Coords(br_lat_yolo, br_lon_yolo)))

    return res


def get_lrm(coords_net, img_height):
    """
    Расчет разрешения на местности в метрах (сколько метров в одном пикселе)
    Длина в метрах 1° широты = всегда 111,32 км. Длина в метрах 1° долготы = 40075 км * cos(широта) / 360.
    """
    lat_lon_min_max = get_lat_lon_min_max_coords(coords_net)
    delta_lat = abs(lat_lon_min_max[1] - lat_lon_min_max[0])  # в градусах
    meters = delta_lat * 111.32 * 1000

    return meters / img_height


if __name__ == "__main__":
    # Получаем список географических координат углов изображения.
    # Поддерживаемые форматы .map, .kml, .dat
    img_name ='F:\python\\ai_annotator\projects\geotiff\\byron.tif'
    print(get_geo_extent(img_name))

