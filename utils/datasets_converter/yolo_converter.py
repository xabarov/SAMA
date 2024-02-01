import os
import shutil
from utils.help_functions import convert_image_name_to_png_name, is_im_path


def create_yaml(yaml_short_name, save_folder, label_names, dataset_name='Dataset', use_test=None):
    yaml_full_name = os.path.join(save_folder, yaml_short_name)
    with open(yaml_full_name, 'w') as f:
        f.write(f"# {dataset_name}\n")
        # Paths:
        path_str = f"path: {save_folder}\n"
        path_str += "train: images/train  # train images (relative to 'path') \n"
        path_str += "val: images/val  # val images (relative to 'path')\n"
        if not use_test:
            path_str += "test:  # test images (optional)\n"
        else:
            path_str += "test:  images/test # test images\n"
        f.write(path_str)
        # Classes:
        f.write("#Classes\n")
        f.write(f"nc: {len(label_names)} # number of classes\n")
        f.write(f"names: {label_names}\n")


def create_mmseg_like_yolo_names(yolo_path: str, mmseg_path: str, save_path='dataset'):
    """
    Перемещает изображения jpg и маски png согласно именам train val yolo
    """
    yolo_names = get_yolo_train_val_names(yolo_path)  # Dict {'train': List(), 'val':List() }
    mm_names = get_mmseg_names(mmseg_path)

    if not os.path.exists(save_path):
        os.makedirs(save_path)

    copy_yaml_if_exist(mmseg_path, save_path)

    for ann_or_img_folder in ['ann_dir', 'img_dir']:

        if not os.path.exists(os.path.join(save_path, ann_or_img_folder)):
            os.makedirs(os.path.join(save_path, ann_or_img_folder))

        for folder in ['train', 'val']:
            cur_save_path = os.path.join(save_path, ann_or_img_folder, folder)

            if not os.path.exists(cur_save_path):
                os.makedirs(cur_save_path)

            for name in yolo_names[folder]:
                if ann_or_img_folder == 'img_dir':
                    source_full_name = os.path.join(yolo_path, "images", folder, name)
                    dest_name = os.path.join(cur_save_path, name)
                else:
                    png_name = convert_image_name_to_png_name(name)
                    if png_name in mm_names['ann_dir']['train']:
                        source_full_name = os.path.join(mmseg_path, "ann_dir", 'train', png_name)
                    elif png_name in mm_names['ann_dir']['val']:
                        source_full_name = os.path.join(mmseg_path, "ann_dir", 'val', png_name)
                    else:
                        print(f"No annotation with name {png_name}")
                        continue
                    dest_name = os.path.join(cur_save_path, png_name)

                shutil.copy(source_full_name, dest_name)


def copy_yaml_if_exist(source, dest):
    dirs = os.listdir(source)
    for d in dirs:
        if d.endswith("yaml"):
            yaml_path = os.path.join(source, d)
            shutil.copy(yaml_path, os.path.join(dest, d))
            break


def get_yolo_train_val_names(path):
    images_path = os.path.join(path, "images")
    res = {}
    for folder in ['train', 'val']:
        folder_path = os.path.join(images_path, folder)
        res[folder] = [im_name for im_name in os.listdir(folder_path) if is_im_path(im_name)]

    return res


def get_mmseg_names(path):
    res = {}
    for ann_or_img_folder in ['ann_dir', 'img_dir']:
        names = {}
        for folder in ['train', 'val']:
            images = os.listdir(os.path.join(path, ann_or_img_folder, folder))
            names[folder] = [im for im in images if is_im_path(im)]

        res[ann_or_img_folder] = names

    return res


if __name__ == '__main__':
    y_path = "D:\\python\\mmseg_last\\mmsegmentation\\data\\dataset_11_01_2024"
    m_path = "D:\\python\\mmseg_last\\mmsegmentation\\data\\aes_mmseg"
    save_path = ''
    create_mmseg_like_yolo_names(y_path, m_path)
