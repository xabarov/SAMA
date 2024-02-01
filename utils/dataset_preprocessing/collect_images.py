import shutil
import os
import ujson
from utils.help_functions import is_im_path


def get_images_contains_cls(project_path, cls_names, save_path=None):
    with open(project_path, 'r') as f:
        data = ujson.load(f)
        labels = list(data["labels"])
        print(f"Labels: {labels}")
        images = list(data["images"])

        export_im_names = set()
        for cls_name in cls_names:
            if cls_name not in labels:
                continue

            cls_idx = labels.index(cls_name)
            for im in images:
                filename = im["filename"]
                shapes = list(im["shapes"])
                for s in shapes:
                    if s["cls_num"] == cls_idx:
                        if filename not in export_im_names:
                            export_im_names.add(filename)
                        break

        if save_path:
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            path_to_images = str(data["path_to_images"])
            for im_name in export_im_names:
                fullpath = os.path.join(path_to_images, im_name)
                if not os.path.exists(fullpath):
                    print(f"Image {fullpath} not exists")
                    continue
                shutil.copy(fullpath, os.path.join(save_path, im_name))

        return list(export_im_names)


def copy_only_one_img_from_group(img_path, save_path):
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    images = [im for im in os.listdir(img_path) if is_im_path(im)]
    looked = set()
    for im_name in images:
        head = im_name.split('_')[0]
        if head not in looked:
            looked.add(head)
            im_full_path = os.path.join(img_path, im_name)
            shutil.copy(im_full_path, os.path.join(save_path, im_name))




if __name__ == '__main__':
    # project_path = "project.json"
    # names = get_images_contains_cls(project_path, cls_names=['cooltower', 'ct_vent_circle', 'cl_vent_sq', 'ct_active'],
    #                                 save_path='export')
    #
    # print(names)
    img_path = "F:\\python\\!ai_practice\\ПЗ6. Object Detection и Instance Segmentation\\light version\\data\\train"
    save_path = "F:\\python\\!ai_practice\\ПЗ6. Object Detection и Instance Segmentation\\light version\\data\\export"
    copy_only_one_img_from_group(img_path, save_path)
