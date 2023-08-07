import cv2
import numpy as np
import os
from utils.edges_from_mask import seg_res_from_yolo_label
from PIL import Image
from shapely import Polygon
from rasterio import features


def blur_image_by_mask(image_path, mask, save=False):
    img_cv2 = cv2.imread(image_path)
    blurred_img = cv2.GaussianBlur(img_cv2, (21, 21), 0)
    out = np.where(mask != (255, 255, 255), img_cv2, blurred_img)
    if save:
        im_suffix = os.path.basename(image_path).split('.')[-1]
        save_path = image_path.split('.' + im_suffix)[0] + "_blurred.jpg"
        cv2.imwrite(save_path, out)
    return out



def get_mask_from_yolo_txt(image_path, yolo_label_path, cls_nums, mask_save=False):
    img = Image.open(image_path)

    img_width, img_height = img.size

    seg_results = seg_res_from_yolo_label(yolo_label_path, img_width, img_height)

    final_mask = np.zeros((img_height, img_width, 3))
    final_mask[:, :] = (0, 0, 0)

    for seg in seg_results:
        cls = seg['cls']
        if cls in cls_nums:

            x_mass = seg['seg']['x']
            y_mass = seg['seg']['y']

            pol = []
            for x, y in zip(x_mass, y_mass):
                pol.append((x, y))

            pol = Polygon(pol)

            mask_image = features.rasterize([pol], out_shape=(img_height, img_width),
                                            fill=0,
                                            default_value=cls + 1)

            final_mask[mask_image == cls + 1] = (255, 255, 255)

    if mask_save:
        im_suffix = os.path.basename(image_path).split('.')[-1]
        mask_save_path = image_path.split('.'+im_suffix)[0] + f"_mask.png"
        cv2.imwrite(mask_save_path, final_mask)

    return final_mask


if __name__ == '__main__':
    image_path = 'argentina_atucha_1.jpg'
    yolo_label = 'argentina_atucha_1.txt'
    mask = get_mask_from_yolo_txt(image_path, yolo_label, [14, 0, 10, 12], mask_save=True)
    blur_image_by_mask(image_path, mask, save=True)
