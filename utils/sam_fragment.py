import json
import os
import pickle
from typing import Any, Dict, List

import cv2  # type: ignore
import numpy as np
from segment_anything import SamAutomaticMaskGenerator
from segment_anything import build_sam, build_sam_hq
import gc
import torch


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


def create_masks(input_path, output_path=None, checkpoint='sam_hq_vit_l.pth', device='cuda',
                 one_image_name=None, pickle_name=None, use_sam_hq=True, pred_iou_thresh=0.88, box_nms_thresh=0.7,
                 points_per_side=32, crop_n_points_downscale_factor=1, crop_nms_thresh=0.7):
    print("Loading model...")

    if use_sam_hq:
        sam = build_sam_hq(checkpoint=checkpoint).to(device)
    else:
        sam = build_sam(checkpoint=checkpoint).to(device)

    # sam = sam_model_registry[model_type](checkpoint=checkpoint)
    # _ = sam.to(device=device)
    output_mode = "binary_mask"
    # amg_kwargs = get_amg_kwargs()
    generator = SamAutomaticMaskGenerator(sam, pred_iou_thresh=pred_iou_thresh,  # 0.88
                                          stability_score_thresh=0.95,
                                          stability_score_offset=1.0,
                                          points_per_side=points_per_side,
                                          box_nms_thresh=box_nms_thresh,  # 0.7
                                          crop_nms_thresh=crop_nms_thresh,
                                          crop_n_points_downscale_factor=crop_n_points_downscale_factor,
                                          output_mode=output_mode)

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

    sam.to('cpu')
    del sam
    gc.collect()
    torch.cuda.empty_cache()

    return [mask['segmentation'] for mask in masks]


def create_segments_for_folder(folder_name, is_pickle=False, is_resize=True):
    images = [im for im in os.listdir(folder_name) if
              os.path.isfile(os.path.join(folder_name, im)) and im.split('.')[-1] in ['jpg', 'png']]

    res_path = os.path.join(folder_name, 'resuts')
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    for im in images:
        ext = im.split('.')[-1]
        im_path = os.path.join(folder_name, im)
        if is_resize:
            im_cv2 = cv2.imread(im_path)
            shape = im_cv2.shape
            scale = 1200 / shape[1]
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
        create_masks(im_path, output_path=None, one_image_name=seg_im_path, pickle_name=pickle_name)


if __name__ == '__main__':
    import numpy as np

    input_path = 'F:\python\\aia_git\\ai_annotator\\nuclear_power\crop0.jpg'
    output_path = 'res_cropped'
    pickle_name = 'palo_verde'

    # for pred in np.linspace(0.1, 0.88, 10):
    #     for nms in np.linspace(0.1, 0.7, 5):
    for crop_nms_thresh in np.linspace(0.1, 0.7, 5):
        masks = create_masks(input_path, output_path=None,
                             checkpoint="F:\python\\aia_git\\ai_annotator\sam_models\sam_vit_h_4b8939.pth",
                             one_image_name=f'F:\python\\aia_git\\ai_annotator\\nuclear_power\\crop_nms_thresh {crop_nms_thresh:0.3f}.jpg',
                             pickle_name=None,
                             # pred_iou_thresh=pred, box_nms_thresh=nms,
                             # points_per_side=points_per_side,
                             crop_nms_thresh=crop_nms_thresh,
                             use_sam_hq=False)
