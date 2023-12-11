import os


def rename(dataset_path):
    for folder in ['train', 'test', 'valid']:
        path = os.path.join(dataset_path, folder)
        im_path = os.path.join(path, 'images')
        lbl_path = os.path.join(path, 'labels')

        for i, im in enumerate(os.listdir(im_path)):
            # убираем кракозябры
            basename = f"{folder}_{i}.jpg"
            new_name = os.path.join(im_path, basename)
            os.rename(os.path.join(im_path, im), new_name)

        for i, txt in enumerate(os.listdir(lbl_path)):
            # убираем кракозябры
            basename = f"{folder}_{i}.txt"
            new_name = os.path.join(lbl_path, basename)
            os.rename(os.path.join(lbl_path, txt), new_name)


if __name__ == '__main__':
    dataset_path = "F:\python\datasets\Buildings Instance Segmentation.v4-resize640_2x-v2.yolov8"
    rename(dataset_path)
