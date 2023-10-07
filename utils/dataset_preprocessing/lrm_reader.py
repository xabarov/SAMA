from utils.help_functions import try_read_lrm
import ujson
import os
import numpy as np
from PIL import Image


def write_lrms_to_json(dataset_folder, json_name, normalize_size=1280, img_ext='jpg'):
    images = [im for im in os.listdir(dataset_folder) if
              os.path.isfile(os.path.join(dataset_folder, im)) and '.' + img_ext in im]
    lrms_info = {}
    last_lrm = 0
    for im in images:

        im_full_path = os.path.join(dataset_folder, im)
        lrm = try_read_lrm(im_full_path)

        if lrm:

            img = Image.open(im_full_path)
            img_width, img_height = img.size
            lrm *= min(img_width, img_height) / float(normalize_size)
            last_lrm = lrm
            lrms_info[im] = lrm

            # print(f"Write lrm = {lrm:0.3f} for {im}")

        else:
            print(f"Can't read lrm for {im}. Fill with previous {last_lrm}")
            lrms_info[im] = last_lrm

    with open(json_name, 'w', encoding='utf-8') as f:
        ujson.dump(lrms_info, f, ensure_ascii=False)

    print(f'Writing lrm to {json_name} completed')


def analyze_lrm_json(lrm_json_path):
    with open(lrm_json_path, 'r') as f:
        lrms = ujson.load(f).values()
        print(f"Max lrm = {max(lrms):0.3f}, min lrm = {min(lrms):0.3f}")
        mean = sum(lrms) / len(lrms)
        b2 = sum([lrm * lrm for lrm in lrms]) / len(lrms)
        variance = b2 - mean * mean
        standard_deviation = np.sqrt(variance) / mean
        print(f"Mean = {mean:0.3f}, standard_deviation = {standard_deviation:0.3f}")


if __name__ == '__main__':
    dataset_dir = "D:\python\datasets\\airplanes_hq"
    json_name = "lrms.json"
    write_lrms_to_json(dataset_dir, json_name, img_ext='jpg')
    # analyze_lrm_json(json_name)
