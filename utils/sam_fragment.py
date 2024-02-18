import json
import math
import os
import pickle
from typing import Any, Dict, List

import cv2  # type: ignore
import numpy as np
from PIL import Image
from segment_anything import SamAutomaticMaskGenerator

import utils.config as config


def create_random_color():
    rgb = [0, 0, 0]
    for i in range(3):
        rgb[i] = np.random.randint(0, 256)

    return rgb


def create_one_image_from_masks(masks, img_path, pickle_name=None):
    if len(masks) > 0:
        height, width = masks[0]["segmentation"].shape
        image = np.zeros([height, width, 3])

        if pickle_name:
            masks_pickle = []

        for i, mask_data in enumerate(masks):
            mask = mask_data["segmentation"]  # (width * height)

            if pickle_name:
                masks_pickle.append(mask)

            color = create_random_color()
            image[mask > 0, :] = color

        cv2.imwrite(img_path, image)
        if pickle_name:
            with open(pickle_name, 'wb') as f:
                pickle.dump(masks_pickle, f)


def get_amg_kwargs():
    amg_kwargs = {
        "points_per_side": None,
        "points_per_batch": None,
        "pred_iou_thresh": None,
        "stability_score_thresh": None,
        "stability_score_offset": None,
        "box_nms_thresh": None,
        "crop_n_layers": None,
        "crop_nms_thresh": None,
        "crop_overlap_ratio": None,
        "crop_n_points_downscale_factor": None,
        "min_mask_region_area": None,
    }
    amg_kwargs = {k: v for k, v in amg_kwargs.items() if v is not None}
    return amg_kwargs


def write_masks_to_folder(masks: List[Dict[str, Any]], path: str) -> None:
    header = "id,area,bbox_x0,bbox_y0,bbox_w,bbox_h,point_input_x,point_input_y,predicted_iou,stability_score,crop_box_x0,crop_box_y0,crop_box_w,crop_box_h"  # noqa
    metadata = [header]

    for i, mask_data in enumerate(masks):
        mask = mask_data["segmentation"]

        filename = f"{i}.png"
        cv2.imwrite(os.path.join(path, filename), mask * 255)
        mask_metadata = [
            str(i),
            str(mask_data["area"]),
            *[str(x) for x in mask_data["bbox"]],
            *[str(x) for x in mask_data["point_coords"][0]],
            str(mask_data["predicted_iou"]),
            str(mask_data["stability_score"]),
            *[str(x) for x in mask_data["crop_box"]],
        ]
        row = ",".join(mask_metadata)
        metadata.append(row)
    metadata_path = os.path.join(path, "metadata.csv")
    with open(metadata_path, "w") as f:
        f.write("\n".join(metadata))

    return


def create_generator(sam, pred_iou_thresh=0.88, box_nms_thresh=0.7,
                     points_per_side=32, crop_n_points_downscale_factor=1, crop_nms_thresh=0.7,
                     output_mode="binary_mask"):
    # amg_kwargs = get_amg_kwargs()

    generator = SamAutomaticMaskGenerator(sam, pred_iou_thresh=pred_iou_thresh,  # 0.88
                                          stability_score_thresh=0.95,
                                          stability_score_offset=1.0,
                                          points_per_side=points_per_side,
                                          box_nms_thresh=box_nms_thresh,  # 0.7
                                          crop_nms_thresh=crop_nms_thresh,
                                          crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                                          output_mode=output_mode)

    return generator


def create_masks(generator, input_path, output_path=None, one_image_name=None, pickle_name=None,
                 output_mode="binary_mask"):
    if output_path:
        os.makedirs(output_path, exist_ok=True)

    print(f"Processing '{input_path}'...")
    image = cv2.imread(input_path)

    if image is None:
        print(f"Could not load '{input_path}' as an image, skipping...")
        return

    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    masks = generator.generate(image)
    if one_image_name:
        create_one_image_from_masks(masks, one_image_name, pickle_name=pickle_name)

    if output_path:
        base = os.path.basename(input_path)
        base = os.path.splitext(base)[0]
        save_base = os.path.join(output_path, base)
        if output_mode == "binary_mask":
            os.makedirs(save_base, exist_ok=False)
            write_masks_to_folder(masks, save_base)
        else:
            save_file = save_base + ".json"
            with open(save_file, "w") as f:
                json.dump(masks, f)

    print("Done!")

    return [mask['segmentation'] for mask in masks]


def create_segments_for_folder(generator, folder_name, is_pickle=False, is_resize=True, scale_width=1200):
    images = [im for im in os.listdir(folder_name) if
              os.path.isfile(os.path.join(folder_name, im)) and im.split('.')[-1] in ['jpg', 'png']]

    res_path = os.path.join(folder_name, 'results')
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    for im in images:
        ext = im.split('.')[-1]
        im_path = os.path.join(folder_name, im)
        if is_resize:
            im_cv2 = cv2.imread(im_path)
            shape = im_cv2.shape
            scale = scale_width / shape[1]
            width = int(shape[1] * scale)
            height = int(shape[0] * scale)
            dim = (width, height)
            resized = cv2.resize(im_cv2, dim, interpolation=cv2.INTER_AREA)

            name = im.split('.' + ext)[0] + '_resized.' + ext

            im_path = os.path.join(res_path, name)
            cv2.imwrite(im_path, resized)

        seg_im_path = os.path.join(res_path, im)
        if is_pickle:
            pickle_name = seg_im_path.split('.' + ext)[0]
        else:
            pickle_name = None
        create_masks(generator, im_path, output_path=None, one_image_name=seg_im_path, pickle_name=pickle_name)

def calc_points_per_side(self, min_obj_width_meters):
    image = Image.open(self.tek_image_path)
    image_width = image.width

    min_obj_width_px = min_obj_width_meters / self.lrm
    step_px = min_obj_width_px / 2.0
    return math.floor(image_width / step_px)


if __name__ == '__main__':
    pass
