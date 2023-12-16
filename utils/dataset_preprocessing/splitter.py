import os
import shutil
import cv2
from PIL import Image
from utils.help_functions import split_into_fragments, calc_parts
from utils.dataset_preprocessing.sentence_sim import get_images_names_similarity
from utils.help_functions import is_im_path
import math

Image.MAX_IMAGE_PIXELS = 933120000


def handle_out_of_image_border(x_px, y_px, w_px, h_px, im_width, im_height):
    x_min = x_px - w_px / 2.0
    x_min = max(0, x_min)

    x_max = x_px + w_px / 2.0
    x_max = min(im_width, x_max)

    y_min = y_px - h_px / 2.0
    y_min = max(0, y_min)

    y_max = y_px + h_px / 2.0
    y_max = min(im_height, y_max)

    w_px_new = x_max - x_min
    h_px_new = y_max - y_min
    x_px_new = x_min + w_px_new / 2.0
    y_px_new = y_min + h_px_new / 2.0

    return x_px_new, y_px_new, w_px_new, h_px_new


def create_labels_from_crop(crop, image_label_path, crop_label_txt_path, image_width, image_height):
    # crop = [[x_min, x_max], [y_min, y_max]]
    labels_in_crop = []
    with open(image_label_path, 'r') as image_label_file:
        for lbl in image_label_file:
            line = lbl.split(' ')
            cls_num = int(line[0])
            x = float(line[1])
            y = float(line[2])
            w = float(line[3])
            h = float(line[4])

            x_px = int(x * image_width)
            y_px = int(y * image_height)
            w_px = int(w * image_width)
            h_px = int(h * image_height)

            crop_w = crop[0][1] - crop[0][0]
            crop_h = crop[1][1] - crop[1][0]

            if abs(1.0 - crop_w / crop_h) > 0.2:
                return False

            if x_px > crop[0][0] and x_px < crop[0][1] and y_px > crop[1][0] and y_px < crop[1][1]:
                # label in crop
                x_inside_crop_px = x_px - crop[0][0]  # minus x_min
                y_inside_crop_px = y_px - crop[1][0]  # minus y_min
                crop_w = crop[0][1] - crop[0][0]
                crop_h = crop[1][1] - crop[1][0]

                x_inside_crop_px, y_inside_crop_px, w_px, h_px = handle_out_of_image_border(x_inside_crop_px,
                                                                                            y_inside_crop_px, w_px,
                                                                                            h_px, crop_w, crop_h)

                labels_in_crop.append([cls_num, float(x_inside_crop_px) / crop_w, float(y_inside_crop_px) / crop_h,
                                       float(w_px) / crop_w, float(h_px) / crop_h])

    if len(labels_in_crop) > 0:
        with open(crop_label_txt_path, 'w') as lbl_crop_txt:
            for lbl in labels_in_crop:
                lbl_crop_txt.write(f"{lbl[0]} {lbl[1]:0.6f} {lbl[2]:0.6f} {lbl[3]:0.6f} {lbl[4]:0.6f}\n")

        return True

    return False


def split_image_and_label(image_path, label_path, save_images_path, save_labels_path, part_size=1280,
                          convert_tif_to_jpg=True):
    img = Image.open(image_path)

    if not os.path.exists(save_labels_path):
        os.makedirs(save_labels_path)

    if not os.path.exists(save_images_path):
        os.makedirs(save_images_path)

    img_width, img_height = img.size

    label_name = os.path.basename(label_path).split('.txt')[0]

    if img_width < part_size * 1.5:

        if convert_tif_to_jpg:
            cv2_im = cv2.imread(image_path)
            jpg_name = os.path.basename(image_path).split('.tif')[0] + '.jpg'
            cv2.imwrite(os.path.join(save_images_path, jpg_name), cv2_im)

        else:
            image_name = os.path.basename(image_path)
            shutil.copy(image_path, os.path.join(save_images_path, image_name))

        shutil.copy(label_path, os.path.join(save_labels_path, os.path.basename(label_path)))

        return

    crop_x_y_sizes, x_parts_num, y_parts_num = calc_parts(img_width, img_height, part_size)

    cv2_im = cv2.imread(image_path)
    fragments = split_into_fragments(cv2_im, part_size)

    for i in range(len(crop_x_y_sizes)):
        crop = crop_x_y_sizes[i]  # [[x_min, x_max], [y_min, y_max]]
        is_label_created = create_labels_from_crop(crop, label_path,
                                                   os.path.join(save_labels_path, f"{label_name}_crop_{i}.txt"),
                                                   img_width, img_height)
        if is_label_created:
            frag = fragments[i]
            cv2.imwrite(os.path.join(save_images_path, f"{label_name}_crop_{i}.jpg"), frag)


def split_yolo_dataset(dataset_folder, splitted_dataset_folder, part_size=1280):
    # in dataset folder
    images_folder_from = os.path.join(dataset_folder, 'images')
    labels_folder_from = os.path.join(dataset_folder, 'labels')

    images_folder_to = os.path.join(splitted_dataset_folder, 'images')
    if not os.path.exists(images_folder_to):
        os.makedirs(images_folder_to)

    labels_folder_to = os.path.join(splitted_dataset_folder, 'labels')
    if not os.path.exists(labels_folder_to):
        os.makedirs(labels_folder_to)

    for folder in ['train', 'val']:
        print(f"Start processing {folder}...")
        images_from = os.listdir(os.path.join(images_folder_from, folder))
        labels_from = os.listdir(os.path.join(labels_folder_from, folder))

        save_im_path = os.path.join(images_folder_to, folder)
        if not os.path.exists(save_im_path):
            os.makedirs(save_im_path)

        save_lbl_path = os.path.join(labels_folder_to, folder)
        if not os.path.exists(save_lbl_path):
            os.makedirs(save_lbl_path)

        for im_name, lbl_name in zip(images_from, labels_from):
            print(f"\t>{im_name}...")
            im_full_path_from = os.path.join(images_folder_from, folder, im_name)
            lbl_full_path_from = os.path.join(labels_folder_from, folder, lbl_name)

            split_image_and_label(im_full_path_from, lbl_full_path_from, save_im_path, save_lbl_path,
                                  part_size=part_size)


def train_test_val_splitter(images_path, train=80.0, val=20.0, test=None, sim_method='names', sim_thresh=0.8):
    """
    Сортирует имена изображений на 3 (2) части - Train/Val/Test. Если Test = None, то Train/Val
    sim_method - способ определения близости
        names - по именам
        clip - по эмбеддингам
    """
    if not test:
        "Split in 2 groups - train/val"
        assert math.fabs(train + val - 100) < 1

        if sim_method == 'names':
            images = [im for im in os.listdir(images_path) if is_im_path(im)]

            sim = get_images_names_similarity(images_path)
            train_names = []
            val_names = []
            train_indices = []
            val_indices = []

            val_percent = 0.0

            N = len(images)

            for i in range(N - 1):
                if val_percent >= val:
                    break
                if i in val_indices:
                    continue

                val_indices.append(i)
                val_names.append(images[i])
                val_percent += 100.0 / N
                n = i + 1
                while sim[i, n] > sim_thresh:
                    if n == N - 1:
                        break
                    if n not in val_indices:
                        val_indices.append(n)
                        val_names.append(images[n])
                        val_percent += 100.0 / N
                        if val_percent > val:
                            break
                    n += 1

            # all other - in val
            for i in range(N - 1):
                if i not in val_indices:
                    train_indices.append(i)
                    train_names.append(images[i])

            return train_names, val_names


    else:
        "Split in 3 groups - train/val/test"
        assert math.fabs(train + val + test - 100) < 1

        if sim_method == 'names':
            images = [im for im in os.listdir(images_path) if is_im_path(im)]

            sim = get_images_names_similarity(images_path)
            train_names = []
            val_names = []
            test_names = []
            train_indices = []
            val_indices = []
            test_indices = []

            test_percent = 0.0
            val_percent = 0.0

            N = len(images)

            for i in range(N - 1):
                if test_percent >= test:
                    break
                if i in test_indices:
                    continue

                test_indices.append(i)
                test_names.append(images[i])
                test_percent += 100.0 / N
                n = i + 1
                while sim[i, n] > sim_thresh:
                    if n == N - 1:
                        break
                    if n not in test_indices:
                        test_indices.append(n)
                        test_names.append(images[n])
                        test_percent += 100.0 / N
                        if test_percent > test:
                            break
                    n += 1

            for i in range(N - 1):
                if val_percent >= val:
                    break
                if i in test_indices or i in val_indices:
                    continue

                val_indices.append(i)
                val_names.append(images[i])
                val_percent += 100.0 / N
                n = i + 1
                while sim[i, n] > sim_thresh:
                    if n == N - 1:
                        break
                    if n not in test_indices and n not in val_indices:
                        val_indices.append(n)
                        val_names.append(images[n])
                        val_percent += 100.0 / N
                        if val_percent > val:
                            break
                    n += 1

            # all other - in val
            for i in range(N - 1):
                if i not in test_indices and i not in val_indices:
                    train_indices.append(i)
                    train_names.append(images[i])

            return train_names, val_names, test_names


if __name__ == '__main__':
    # image_path = "F:\python\datasets\\airplanes_FAIR1M\images\\train\\4640.tif"
    # label_path = "F:\python\datasets\\airplanes_FAIR1M\labels\\train\\4640.txt"
    # save_labels_path = "labels"
    # save_images_path = "images"
    #
    # split_image_and_label(image_path, label_path, save_images_path, save_labels_path, part_size=1280)
    path = "D:\python\\aia_git\\ai_annotator\\projects\\aes_test"

    train, val, test = train_test_val_splitter(path, 80, 10, 10)

    print(len(train))
    print(len(val))
    print(len(test))
