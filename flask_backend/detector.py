import cv2
from detect_yolo8 import predict_and_return_masks
import numpy as np
from rasterio import features, Affine
import shapely
from shapely.geometry import Polygon

train_ld = 0.11
train_img_px = 8000  # в реальности - 1280, но это ужатые 8000 с ld = 0.0923


def filter_masks(masks_results, conf_thres=0.2, iou_filter=0.3):
    """
    Фильтрация боксов
    conf_tresh - убираем все боксы с вероятностями ниже заданной
    iou_filter - убираем дубликаты боксов, дубликатами считаются те, значение IoU которых выше этого порога
    """
    unique_results = []
    skip_nums = []
    for i in range(len(masks_results)):
        if float(masks_results[i]['conf']) < conf_thres:
            continue

        biggest_mask = None

        if i in skip_nums:
            continue

        for j in range(i + 1, len(masks_results)):

            if j in skip_nums:
                continue

            if masks_results[i]['cls_num'] != masks_results[j]['cls_num']:
                continue

            if biggest_mask:
                pol1 = Polygon(biggest_mask['points'])
            else:
                pol1 = Polygon(masks_results[i]['points'])

            pol2 = Polygon(masks_results[j]['points'])

            un = pol1.union(pol2)
            inter = pol1.intersection(pol2)

            if inter and un:
                iou = inter.area / un.area

                if iou > iou_filter:

                    if pol1.area < pol2.area:
                        biggest_mask = masks_results[j]

                    skip_nums.append(j)

        if biggest_mask:
            unique_results.append(biggest_mask)
        else:
            unique_results.append(masks_results[i])

    return unique_results


def yolo8masks2points(yolo_mask, simplify_factor=3, width=1280, height=1280):
    points = []
    img_data = yolo_mask[0] > 128
    shape = img_data.shape

    mask_width = shape[1]
    mask_height = shape[0]

    img_data = np.asarray(img_data[:, :], dtype=np.double)

    polygons = mask_to_polygons_layer(img_data)

    polygon = polygons[0].simplify(simplify_factor, preserve_topology=False)

    try:
        xy = np.asarray(polygon.boundary.xy, dtype="float")
        x_mass = xy[0].tolist()
        y_mass = xy[1].tolist()
        for x, y in zip(x_mass, y_mass):
            points.append([x * width / mask_width, y * height / mask_height])
        return points

    except:

        return None


def mask_to_polygons_layer(mask):
    shapes = []
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask > 0),
                                        transform=Affine(1.0, 0, 0, 0, 1.0, 0)):
        shapes.append(shapely.geometry.shape(shape))

    return shapes


def split_into_fragments(img, frag_size):
    fragments = []

    shape = img.shape

    img_width = shape[1]
    img_height = shape[0]

    crop_x_y_sizes, x_parts_num, y_parts_num = calc_parts(img_width, img_height, frag_size)

    for x_y_crops in crop_x_y_sizes:
        x_min, x_max = x_y_crops[0]
        y_min, y_max = x_y_crops[1]
        fragments.append(img[int(y_min):int(y_max), int(x_min):int(x_max), :])

    return fragments


def calc_width_parts(img_width, frag_size):
    if frag_size / 2 > img_width:
        return [[0, img_width]]
    crop_start_end_coords = []
    tek_pos = 0
    while tek_pos <= img_width:
        if tek_pos == 0:
            if img_width > frag_size:
                crop_start_end_coords.append([tek_pos, frag_size])
            else:
                crop_start_end_coords.append([tek_pos, img_width])
                break

        elif tek_pos + frag_size >= img_width:
            crop_start_end_coords.append([tek_pos, img_width])
            break

        else:
            crop_start_end_coords.append([tek_pos, tek_pos + frag_size])
        tek_pos += int(frag_size / 2)

    return crop_start_end_coords


def calc_parts(img_width, img_height, frag_size):
    crop_x_y_sizes = []
    crop_x_sizes = calc_width_parts(img_width, int(frag_size))
    crop_y_sizes = calc_width_parts(img_height, int(frag_size))
    for y in crop_y_sizes:
        for x in crop_x_sizes:
            crop_x_y_sizes.append([x, y])
    return crop_x_y_sizes, len(crop_x_sizes), len(crop_y_sizes)


def run_yolo8(model, img_path_full, lrm=None, conf_thres=0.5, iou_thres=0.5):
    img = cv2.imread(img_path_full)
    shape = img.shape

    if lrm:

        mask_results = run_yolo8(model, img_path_full, lrm=None)

        frag_size = int(train_img_px * train_ld / lrm)

        scanning_results = [res for res in mask_results]

        parts = split_into_fragments(img, frag_size)
        crop_x_y_sizes, x_parts_num, y_parts_num = calc_parts(shape[1], shape[0], frag_size)

        print(f'Crop image into {x_parts_num}x{y_parts_num}')
        print(crop_x_y_sizes)

        part_tek = 0
        for part, part_size in zip(parts, crop_x_y_sizes):

            part_mask_results = predict_and_return_masks(model, part, conf=conf_thres,
                                                         iou=iou_thres, save_txt=False)
            x_min, x_max = part_size[0]
            y_min, y_max = part_size[1]

            for res in part_mask_results:
                for i, mask in enumerate(res['masks']):
                    points = yolo8masks2points(mask, simplify_factor=3, width=x_max - x_min, height=y_max - y_min)
                    if not points:
                        continue
                    points_shifted = []
                    for x, y in points:
                        points_shifted.append([x + x_min, y + y_min])
                    cls_num = res['classes'][i]
                    conf = res['confs'][i]
                    scanning_results.append({'cls_num': cls_num, 'points': points_shifted, 'conf': conf})

            part_tek += 1

        return filter_masks(scanning_results, conf_thres=conf_thres, iou_filter=0.05)



    else:

        results = predict_and_return_masks(model, img, conf=conf_thres,
                                           iou=iou_thres, save_txt=False)

        mask_results = []
        for res in results:
            for i, mask in enumerate(res['masks']):
                points = yolo8masks2points(mask, simplify_factor=3, width=shape[1], height=shape[0])
                if not points:
                    continue
                cls_num = res['classes'][i]
                conf = res['confs'][i]
                mask_results.append({'cls_num': cls_num, 'points': points, 'conf': conf})

        return mask_results
