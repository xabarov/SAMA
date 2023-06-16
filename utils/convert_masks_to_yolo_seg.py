from utils.edges_from_mask import mask2seg
from PIL import Image
import os

import numpy as np


def convert(train_folder, val_folder, cls_names, verbose=True):
    for folder in [train_folder, val_folder]:
        if verbose:
            print(f'Folder {folder}:')
        pngs = [name for name in os.listdir(folder) if name.endswith('.png')]

        for png_name in pngs:

            if verbose:
                print(f'>>>{png_name}:')

            txt_name = os.path.join(folder, png_name.split('.png')[0] + '.txt')
            with open(txt_name, 'w') as f:
                all_lines = []
                for i in range(1, len(cls_names)):

                    png_full_path = os.path.join(folder, png_name)
                    segments = mask2seg(png_full_path, cls_num=i)

                    if len(segments):

                        img = Image.open(png_full_path)
                        img_width, img_height = img.size

                        for seg in segments:
                            new_line = f'{i - 1} '
                            for x, y in zip(seg['x'], seg['y']):
                                new_line += f'{x/img_width} {y/img_height} '
                            if new_line not in all_lines:
                                all_lines.append(new_line)
                                if verbose:
                                    print(f'      class: {cls_names[i]}: found {len(segments)} segments')

                                f.write(new_line+'\n')


if __name__ == '__main__':
    train_folder = 'D:\\python\\ultralytics\\data\\aes\\train'
    val_folder = 'D:\\python\\ultralytics\\data\\aes\\val'

    cls_names = {0: 'background', 1: 'ro_kv', 2: 'ro_cil', 3: 'mz', 4: 'ru', 5: 'gr_b', 6: 'gr_vent_kr',
                 7: 'gr_vent_pr', 8: 'water'}

    convert(train_folder, val_folder, cls_names)
