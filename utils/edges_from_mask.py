import cv2
import numpy as np
from collections import namedtuple
import os
import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon
from PIL import Image
from utils import help_functions as hf
from skimage.measure._find_contours import find_contours

# from skimage.draw import line, polygon, ellipse

SegResult = namedtuple('SegResult', ['seg', 'cls', 'img_filename', 'prob'])

CLASS_NAME = {'реактор': 1, 'реактор кв': 2, 'градирня': 3, 'градирня кв': 4,
              'градирня вент': 5, 'РУ': 6, 'ВНС': 7, 'турбина': 8, 'БСС': 9, 'машинный зал': 10, 'парковка': 11}

CATEGORIES_CONVERTER = {
    1: 2,  # "ro_pf",
    2: 1,  # "ro_sf",
    3: 13,  # "ro_cil_p", == 1
    4: 15,  # "mz_v",
    5: 14,  # "mz_nv", == 2
    6: 9,  # "tr_",
    7: 11,  # "tr_op", DELETE слишком мало
    8: 7,  # "mz_ot",
    9: 11,  # "ru_ot",
    10: 4,  # "ru_zk", DELETE слишком мало
    11: -1,  # "bns_ot",
    12: -1,  # "bns_zk",
    13: -1,  # "gr_b",
    14: -1,  # "gr_vent_kr",
    15: -1,  # "gr_vent_pr",
    16: -1,  # "bass",
    17: -1,  # "ro_cil_ss", == 1
    18: -1,  # "ro_cil_sp", == 1
    19: -1,  # "gr_b_act", == 13
    20: -1,  # "gr_vent_kr_act" == 15
}

CATEGORIES_ORIGINAL = {
    1: "ro_pf",
    2: "ro_sf",
    3: "ro_cil_p",
    4: "mz_v",
    5: "mz_nv",
    6: "tr_",
    7: "tr_op",
    8: "mz_ot",
    9: "ru_ot",
    10: "ru_zk",
    11: "bns_ot",
    12: "bns_zk",
    13: "gr_b",
    14: "gr_vent_kr",
    15: "gr_vent_pr",
    16: "bass",
    17: "ro_cil_ss",
    18: "ro_cil_sp",
    19: "gr_b_act",
    20: "gr_vent_kr_akt"
}


def get_mask_name(mask_list, frag_name):
    frag_name = frag_name.split('.')[0]
    for m in mask_list:
        if frag_name in m:
            return m


def seg_res_to_masks(seg_res, mask_new_folder, img_width, img_height):
    created_mask_dict = {}

    for seg in seg_res:
        cls = seg.cls
        prob = seg.prob
        x_mass = seg.seg['x']
        y_mass = seg.seg['y']

        pol = []
        for x, y in zip(x_mass, y_mass):
            pol.append((x, y))

        pol = Polygon(pol)

        mask_image = features.rasterize([pol], out_shape=(img_height, img_width), fill=255, default_value=cls)

        if cls not in created_mask_dict:
            created_mask_dict[cls] = 0
        else:
            created_mask_dict[cls] += 1

        mask_tek = created_mask_dict[cls]

        mask_name = f"aggregated class {cls} mask {mask_tek} with score {prob:0.4f}.png"

        new_path = os.path.join(mask_new_folder, mask_name)

        cv2.imwrite(new_path, mask_image)


def seg_res_from_yolo_label(yolo_label_path, image_width, image_height):
    seg_results = []
    if os.path.exists(yolo_label_path):
        with open(yolo_label_path, 'r') as f:
            for line in f:
                cls_and_xy = line.split()
                cls = int(cls_and_xy[0])
                x_points = []
                y_points = []
                for i in range(1, len(cls_and_xy), 2):
                    x_points.append(float(cls_and_xy[i]) * image_width)
                    y_points.append(float(cls_and_xy[i + 1]) * image_height)
                seg = {'cls': cls, 'seg': {'x': x_points, 'y': y_points}}

                seg_results.append(seg)

    return seg_results


def create_png_from_yolo(yolo_label_path, image_path, save_path, background_cls=0):
    img = Image.open(image_path)

    img_width, img_height = img.size

    seg_reults = seg_res_from_yolo_label(yolo_label_path, img_width, img_height)

    final_mask = np.zeros((img_height, img_width))
    final_mask[:, :] = 0

    for seg in seg_reults:
        cls = seg['cls']
        if background_cls == 0:
            cls += 1

        x_mass = seg['seg']['x']
        y_mass = seg['seg']['y']

        pol = []
        for x, y in zip(x_mass, y_mass):
            pol.append((x, y))

        pol = Polygon(pol)

        mask_image = features.rasterize([pol], out_shape=(img_height, img_width),
                                        fill=background_cls,
                                        default_value=cls)

        final_mask[mask_image == cls] = cls
    cv2.imwrite(save_path, final_mask)


def paint_shape_to_mask(mask, shape, cls_num=None):
    """
    Добавляет на mask - np.zeros((img_height, img_width))
    shape = {cls_num: номер класса, points:[ [x1,y1], ...] - точки граней маски }
    cls_num - номер класса. Если не задан - берется из shape
    """
    img_width, img_height = mask.size

    if not cls_num:
        cls_num = shape['cls_num']

    points = shape["points"]
    pol = Polygon(points)

    mask_image = features.rasterize([pol], out_shape=(img_height, img_width),
                                    fill=255,
                                    default_value=cls_num)

    mask[mask_image == cls_num] = cls_num


def create_seg_annotations_from_yolo_seg(images_folder, yolo_seg_folder, png_save_folder, img_suffix='jpg'):
    if not os.path.exists(png_save_folder):
        os.makedirs(png_save_folder)

    print(f'Start annotation creation')

    for im in os.listdir(images_folder):
        if im.endswith(img_suffix):

            base_img_name = im.split('.' + img_suffix)[0]
            txt_name = base_img_name + '.txt'
            txt_path = os.path.join(yolo_seg_folder, txt_name)
            if not os.path.exists(txt_path):
                print(f"    label {txt_path} doesn't exists")
                continue
            png_name = base_img_name + '.png'
            png_path = os.path.join(png_save_folder, png_name)
            create_png_from_yolo(txt_path, os.path.join(images_folder, im), png_path)
            print(f'    annotation for {im} done')

    print(f'Annotation creation done')


def calc_areas(seg_results, lrm, verbose=False, cls_names=None, scale=1):
    if verbose:
        print(f"Старт вычисления площадей с lrm={lrm:0.3f}, всего {len(seg_results)} объектов:")

    areas = []
    for seg in seg_results:
        x_mass = seg.seg['x']
        y_mass = seg.seg['y']
        cls = seg.cls

        pol = []
        for x, y in zip(x_mass, y_mass):
            pol.append((x, y))

        pol = Polygon(pol)

        area = pol.area * lrm * lrm * scale * scale
        areas.append(area)

        if verbose:
            if cls_names:
                cls = cls_names[cls]
            print(f"\tплощадь {cls}: {area:0.3f} кв.м")

    return areas


def segmentation_to_polygons(mm_seg_predicitons, cls_num):
    points_mass = []
    shape = mm_seg_predicitons.shape
    mask = np.zeros((shape[0], shape[1]))
    mask[mm_seg_predicitons == cls_num] = 255
    polygons = mask_to_polygons_layer(mask)
    for pol in polygons:
        try:
            points = []
            xy = np.asarray(pol.boundary.xy, dtype="float")
            x_mass = xy[0].tolist()
            y_mass = xy[1].tolist()
            for x, y in zip(x_mass, y_mass):
                points.append([x, y])
            points_mass.append(points)
        except:
            pass
    return points_mass


def segmentation_to_points(mm_seg_predicitons, cls_num):
    mask = mm_seg_predicitons == cls_num
    points_xx_yy = np.asarray(np.where(mask == True))
    yy = points_xx_yy[0]
    xx = points_xx_yy[1]
    points = []
    for x, y in zip(xx, yy):
        points.append([x, y])
    return points


def mask_results_to_yolo_txt(mask_results, image_path, yolo_txt_file_name, with_conf=False):
    img = Image.open(image_path)

    img_width, img_height = img.size

    with open(yolo_txt_file_name, 'w') as f:
        for res in mask_results:
            cls_num = res['cls_num']
            points = res['points']
            line = f"{cls_num} "
            for p in points:
                x = p[0]
                y = p[1]
                line += f"{x / img_width} {y / img_height} "
            if with_conf:
                line += res["conf"]
            f.write(f"{line}\n")


def yolo8masks2points(yolo_mask, simplify_factor=0.4, width=1280, height=1280):
    img_data = yolo_mask[0] > 128
    shape = img_data.shape

    mask_width = shape[1]
    mask_height = shape[0]

    img_data = np.asarray(img_data[:, :], dtype=np.double)

    polygons = mask_to_polygons_layer(img_data)

    results = []
    for pol in polygons:
        pol_simplified = pol.simplify(simplify_factor, preserve_topology=True)
        points = []
        try:
            xy = np.asarray(pol_simplified.boundary.xy, dtype="float")
            x_mass = xy[0].tolist()
            y_mass = xy[1].tolist()
            for x, y in zip(x_mass, y_mass):
                points.append([x * width / mask_width, y * height / mask_height])
            results.append(points)
        except:
            pass

    return results


def mask2shapes(mask_filename, nc=254, simplify_factor=0.3):
    img_data = cv2.imread(mask_filename, 0)
    shapes = []

    for cls_num in range(nc):

        img_data_cls = img_data == cls_num
        img_data_cls = np.asarray(img_data_cls[:, :], dtype=np.double)
        polygons = mask_to_polygons_layer(img_data_cls)

        for pol in polygons:
            shape = {'cls_num': cls_num}  # pairs of x,y pixels
            pol_simplified = pol.simplify(simplify_factor, preserve_topology=False)

            try:
                xy = np.asarray(pol_simplified.boundary.xy, dtype="int32")
                points = [[x, y] for x, y in zip(xy[0], xy[1])]
                shape['points'] = points
                shapes.append(shape)
            except:
                pass

    return shapes


def mask2seg(mask_filename, simplify_factor=3, cls_num=None):
    img_data = cv2.imread(mask_filename)

    if not cls_num:
        img_data = img_data > 128
    else:
        img_data = img_data == cls_num

    size = img_data.shape[0] * img_data.shape[1]

    img_data = np.asarray(img_data[:, :, 0], dtype=np.double)

    polygons = mask_to_polygons_layer(img_data)

    results = []
    for pol in polygons:
        seg = {}  # pairs of x,y pixels
        pol_simplified = pol.simplify(simplify_factor, preserve_topology=False)

        try:
            xy = np.asarray(pol_simplified.boundary.xy, dtype="int32")
            seg['x'] = xy[0].tolist()
            seg['y'] = xy[1].tolist()
            seg['size'] = size
            results.append(seg)

        except:
            pass

    return results


def mask_folder2seg_results(folder, simpify_factor=3):
    folder_files = os.listdir(folder)

    seg_results = []

    for i in range(len(folder_files)):
        file = folder_files[i]
        if file.endswith('.png'):

            spl = file.split(' ')
            cls = int(spl[2])
            end = spl[-1].split('.')
            prob = float(end[0] + "." + end[1])

            mask_name = os.path.join(folder, file)

            if os.path.exists(mask_name):

                seg = mask2seg(mask_name, simpify_factor=simpify_factor)

                if seg:
                    seg_res = SegResult(seg, cls, file, prob)
                    seg_results.append(seg_res)

    return seg_results


def seg_results2metadata(seg_results):
    metadata = {}

    for seg_result in seg_results:
        meta_id = seg_result.img_filename
        if meta_id not in metadata:
            metadata[meta_id] = {}
            metadata[meta_id]["filename"] = seg_result.img_filename
            metadata[meta_id]["size"] = seg_result.seg['size']
            metadata[meta_id]["regions"] = []

        attr = {}
        attr["shape_attributes"] = {}
        attr["shape_attributes"]["name"] = "polygon"
        attr["shape_attributes"]["all_points_x"] = seg_result.seg['x']
        attr["shape_attributes"]["all_points_y"] = seg_result.seg['y']
        attr["region_attributes"] = {}
        attr["region_attributes"]["type"] = CATEGORIES_ORIGINAL[seg_result.cls]
        metadata[meta_id]["regions"].append(attr)
        metadata[meta_id]["file_attributes"] = {}

    return metadata


def make_edge(mask_filename, save_name='mask_edge.png', is_save=True):
    img_data = cv2.imread(mask_filename)
    img_data = img_data > 128

    img_data = np.asarray(img_data[:, :, 0], dtype=np.double)
    gx, gy = np.gradient(img_data)

    temp_edge = gy * gy + gx * gx

    temp_edge[temp_edge != 0.0] = 255.0

    temp_edge = np.asarray(temp_edge, dtype=np.uint8)

    if is_save:
        cv2.imwrite(save_name, temp_edge)

    return temp_edge


def get_images_not_match(train_folder, val_folder, add_folder):
    all_images = os.listdir(add_folder)
    train_images = os.listdir(train_folder)
    val_images = os.listdir(val_folder)

    all_old = train_images + val_images

    new_images = [im for im in all_images if im not in all_old]
    return new_images


def create_png(dataset_path):
    for folder in ['train', 'val']:
        images_path = f'{dataset_path}\images\\{folder}'
        labels_path = f'{dataset_path}\labels\\{folder}'
        mask_path = f'{dataset_path}\\annotations\\{folder}'
        create_seg_annotations_from_yolo_seg(images_path, labels_path, mask_path)


def test_cv2():
    image_path = "F:\\python\\datasets\\aes_buildings\\argentina_atucha_1.jpg"
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    thresh, BW = cv2.threshold(img, 50, 255, cv2.THRESH_BINARY)
    print(BW.shape)
    contours, _ = cv2.findContours(BW, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    polygons = []
    shapes = []
    for obj in contours:
        coords = []
        for point in obj:
            coords.append([int(point[0][0]), int(point[0][1])])
        polygons.append(coords)
        if len(coords) > 2:
            shapes.append(Polygon(coords))


def mask_to_polygons_layer(mask, method='cv2', is_clear=True, min_size=80):
    """
    Преобразует маску в набор полигонов Shapely Polygon
    mask - бинарное изображение- маска, numpy array
    method - метод для преобразования:
            'rasterio' - библиотекой rasterio
            'cv2' - библиотекой cv2
            'skimage' - библиотекой skimage
    is_clear - нужно ли очищать маски меньше, чем min_size пикелей
    """
    shapes = []
    if is_clear:
        mask = hf.clean_mask(mask, type='remove', min_size=min_size, connectivity=1)
    if method == 'rasterio':
        for shape, value in features.shapes(mask.astype(np.int16), mask=(mask > 0),
                                            transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
            shapes.append(shapely.geometry.shape(shape))
    elif method == 'cv2':
        polygons = []
        mask = mask * 255
        mask = mask.astype(np.uint8)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        for obj in contours:
            coords = []
            for point in obj:
                coords.append([int(point[0][0]), int(point[0][1])])
            polygons.append(coords)
            if len(coords) > 2:
                shapes.append(Polygon(coords))

    elif method == 'skimage':
        contours = find_contours(mask)
        for contour in contours:
            contour = np.flip(contour, axis=1)

            if len(contour) > 2:
                shapes.append(Polygon(contour))
    else:
        print('Unknown method for create masks.')

    return shapes


def show_mask(mask_path, pallete):
    img = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    img_colored = cv2.imread(mask_path)
    for cls_num in pallete:
        img_colored[img == cls_num] = np.array(pallete[cls_num])

    img_colored = cv2.resize(img_colored, (640, 640))

    cv2.imshow('image', img_colored)
    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    mask_path = "F:\\git_test\\33dep_dataset\\nuclear_power_stations\\mmseg\\ann_dir\\train\\france_belleville_1.png"

    labels = ["reactor_sq", "reactor", "engine_room", "pipe", "turbine", "switchgear", "pump", "cooltower",
              "ct_vent_circle", "cl_vent_sq", "ct_active", "discharge", "ISFSI", "tank", "sea", "parking", 'splash_border']
    labels_colors = {"discharge": [255, 0, 164, 120], "ISFSI": [255, 170, 127, 58], "reactor_sq": [170, 0, 255, 58],
                     "reactor": [100, 10, 127, 120], "engine_room": [155, 0, 0, 58], "pipe": [155, 255, 220, 58],
                     "turbine": [255, 170, 0, 120], "switchgear": [255, 255, 0, 58], "pump": [85, 255, 0, 58],
                     "cooltower": [85, 0, 0, 120], "ct_vent_circle": [85, 85, 127, 58], "cl_vent_sq": [13, 45, 85, 120],
                     "ct_active": [178, 155, 191, 120], "tank": [110, 85, 155, 120], "diesel": [255, 85, 0, 120],
                     "sea": [100, 54, 0, 58], "parking": [255, 85, 255, 58], "waste_water_cil": [170, 0, 3, 50],
                     'splash_border':[200,200,0,15]}

    pallete = {}
    for i, lbl in enumerate(labels):
        pallete[i+1] = labels_colors[lbl][:-1]

    show_mask(mask_path, pallete)
