import os
from utils.help_functions import is_im_path
import cv2
import numpy as np
from tqdm import tqdm
import yaml


def update_stat(stat, new_rgb, iter_num):
    for chanel in range(3):
        stat[chanel] = stat[chanel] * (1.0 - (1.0 / iter_num)) + new_rgb[chanel] / iter_num


def get_mm_dataset_mean_std(path_to_train):
    img_names = [im for im in os.listdir(path_to_train) if is_im_path(im)]
    mean = np.zeros(3)
    std = np.zeros(3)
    for i in tqdm(range(len(img_names))):
        im_name = img_names[i]
        im = cv2.imread(os.path.join(path_to_train, im_name))
        m, s = cv2.meanStdDev(im)
        update_stat(mean, m, i + 1)
        update_stat(std, s, i + 1)

    return mean, std


def get_classes_pallete_from_yaml(yaml_path):
    with open(yaml_path, 'r') as f:
        yaml_data = yaml.load(f, Loader=yaml.FullLoader)
        names = yaml_data['names']
        palette = yaml_data['palette']
        classes = list(names.keys())
        palette['background'] = [255, 255, 255]
        palette_new = [palette[cls] for cls in classes]

        return classes, palette_new


if __name__ == '__main__':
    # train_path = 'D:/python/mmseg_last/mmsegmentation/data/aes/img_dir/train'
    # mean, std = get_mm_dataset_mean_std(train_path)
    # print(mean, std)

    classes, palette = get_classes_pallete_from_yaml('D:/python/mmseg_last/mmsegmentation/data/aes/aes.yaml')

    print(classes)
    print(palette)
    # For AES: [103.68238594 115.12355758 111.36336727] [45.7925291  46.47562911 51.79538171]
