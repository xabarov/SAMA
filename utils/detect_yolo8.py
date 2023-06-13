from ultralytics import YOLO
from ultralytics.yolo.utils.plotting import Annotator, Colors
import cv2
from PIL import Image
import numpy as np
# from plots import Annotator, Colors
import os

names = ["ro_pf", "ro_sf", "mz_v", "mz_ot", "ru_ot", "bns_ot", "gr_b", "gr_vent_kr", "gr_vent_pr", "gr_b_act",
         "discharge", "diesel"]


# names = ["РО кв", "РО", "МЗ", "Турбина", "РУ", "БНС", "Градирня пасс", "Град.вент.кр", "Град.вент.пр", "Градирня акт",
#          "БСС", "ДГС"]


def create_image_yolo(res, save_path=None, save_name=None, box_thickness=2, pallete=None, txt_color=None, names=None,
                      is_seg=True,
                      mask_path=None):
    path = res.path
    image = cv2.imread(path)

    if not names:
        names = res.names
    else:
        names_dict = {}
        for i, name in enumerate(names):
            names_dict[i] = name
        res.names = names_dict
    colors = Colors()

    if is_seg:
        image = res.plot()
        if res.masks:
            masks_mass = res.masks.cpu().numpy()

            boxes_mass = res.boxes
            cls_nums = []
            confs = []
            for box in boxes_mass:
                cls = int(box.cls[0])
                cls_nums.append(cls)
                confs.append(box.conf[0])

            masks_cls = {}
            for i, mask in enumerate(masks_mass):
                mask = mask.masks
                mask[mask == 1] = 255
                cls = cls_nums[i]
                if cls not in masks_cls:
                    masks_cls[cls] = 0
                else:
                    masks_cls[cls] += 1

                mask_name = f"{os.path.basename(path).split('.')[0]} class {cls} mask {masks_cls[cls]} with score {confs[i]:0.3f}.png"
                if mask_path:
                    if not os.path.exists(mask_path):
                        os.mkdir(mask_path)

                    mask_name = os.path.join(mask_path, mask_name)

                cv2.imwrite(mask_name, mask)

    else:
        boxes_mass = res.boxes.cpu().numpy()
        for box in boxes_mass:
            cls = int(box.cls[0])
            text = f"{names[cls]} {box.conf[0]:0.3f}"
            if pallete:
                color = pallete[cls]
            else:
                color = colors(cls)

            if not txt_color:
                txt_color = (255, 255, 255)

            ann = Annotator(image, box_thickness, example=text)
            ann.box_label(box.xyxy[0], label=text, color=color, txt_color=txt_color)

            image = ann.result()

    im_name = os.path.basename(path)
    name, suffix = im_name.split('.')
    if not save_path:
        save_path = os.getcwd()
    if not save_name:
        save_name = os.path.join(save_path, name + "_detected." + suffix)

    cv2.imwrite(save_name, image)


def predict(source, model_path, save_folder, save_name=None, box_thickness=2, pallete=None, txt_color=None, names=None,
            is_seg=True, mask_path=None, conf=0.25, iou=0.7, device=None, save_txt=False):

    model = YOLO(model_path)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)

    results = model.predict(source=source, save_conf=True, conf=conf, iou=iou, save_txt=save_txt)
    for res in results:
        create_image_yolo(res, save_folder, save_name=save_name, box_thickness=box_thickness, pallete=pallete,
                          txt_color=txt_color, names=names, is_seg=is_seg, mask_path=mask_path)

def predict_and_return_masks(model, source, conf=0.25, iou=0.7, save_txt=False):
    # Test for multiple images:
    # images_dir = 'F:\python\OD_on_graph_view\images'
    # source = [os.path.join(images_dir, im) for im in os.listdir(images_dir)]

    results = model.predict(source=source, save_conf=True, conf=float(conf), iou=float(iou), save_txt=save_txt)

    mask_results = []
    for res in results:  # res for each image
        if res.masks:
            masks_mass = res.masks.cpu().numpy()

            boxes_mass = res.boxes.cpu().numpy()
            cls_nums = []
            confs = []
            for box in boxes_mass:
                cls = int(box.cls[0])
                cls_nums.append(cls)
                confs.append(box.conf[0])

            masks = []
            for i, mask in enumerate(masks_mass):
                mask = mask.data
                mask[mask == 1] = 255
                masks.append(mask)

            mask_results.append({'masks': masks, 'confs': confs, 'classes': cls_nums})

    return mask_results


if __name__ == '__main__':
    import os

    model_path = "F:\python\drones_yolo\\runs\segment\\train2\weights\\best.pt"

    dir_name = 'C:\\Users\\roman\AppData\Roaming\QGIS\QGIS3\profiles\\default\\python\\plugins\\OD plugin\\detected\\temp'
    # folders = os.listdir(dir_name)
    # for folder in folders:
    #     folder_path = os.path.join(dir_name, folder)
    save_folder = os.path.join(os.getcwd(), "segment_results")
    predict(dir_name, model_path, save_folder, names=names, is_seg=True)
