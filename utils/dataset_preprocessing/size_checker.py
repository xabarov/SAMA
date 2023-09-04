import os

from PIL import Image


def dataset_size_checker(dataset_folder, size=1280, img_ext='jpg'):
    print("Start dataset analyse...")

    images = [im for im in os.listdir(dataset_folder) if
              os.path.isfile(os.path.join(dataset_folder, im)) and '.' + img_ext in im]

    success_num = 0
    failure_num = 0

    for im in images:

        im_full_path = os.path.join(dataset_folder, im)
        img = Image.open(im_full_path)
        img_width, img_height = img.size

        if img_width != size or img_height != size:
            print(f"File {im} has wrong sizes {img_width}*{img_height}")
            failure_num += 1
        else:
            success_num += 1

    print(
        f"End dataset analyse. Total images in dataset {len(images)}. Correct size have {success_num} images. Wrong {failure_num}")


if __name__ == '__main__':
    dataset_dir = "D:\python\datasets\\airplanes_hq\\train"
    dataset_size_checker(dataset_dir, size=1280, img_ext='jpg')
