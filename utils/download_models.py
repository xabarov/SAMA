import os
import pathlib
import urllib.request
import urllib.request
from ultralytics import YOLO

from tqdm import tqdm

from utils.config import PATH_TO_SAM_CHECKPOINT, PATH_TO_SAM_HQ_CHECKPOINT, PATH_TO_GROUNDING_DINO_CHECKPOINT


class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)

def download_yolo():
    model = YOLO()


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)


def get_sam_path(is_hq):
    if is_hq:
        path_to_sam = os.path.join(pathlib.Path(__file__).parent.resolve(), "..", PATH_TO_SAM_HQ_CHECKPOINT)
    else:
        path_to_sam = os.path.join(pathlib.Path(__file__).parent.resolve(), "..", PATH_TO_SAM_CHECKPOINT)
    return path_to_sam


def get_gd_path():
    path_to_gd = os.path.join(pathlib.Path(__file__).parent.resolve(), "..", PATH_TO_GROUNDING_DINO_CHECKPOINT)
    return path_to_gd


def check_sam(is_hq):
    path_to_sam = get_sam_path(is_hq)

    return os.path.exists(path_to_sam)


def make_dirs(path):
    dirname = os.path.dirname(path)
    if not os.path.exists(dirname):
        os.makedirs(dirname)


def download_sam(is_hq):
    path_to_sam = get_sam_path(is_hq)

    if not check_sam(is_hq):

        make_dirs(path_to_sam)
        if is_hq:
            url = "https://drive.google.com/file/d/1qobFYrI4eyIANfBSmYcGuWRaSIXfMOQ8/view?usp=sharing"
            download_url(url, path_to_sam)
        else:
            url = "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth"
            download_url(url, path_to_sam)
    else:
        print(f"SAM model checkpoint found in {path_to_sam}")


def download_gd():
    url = "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth"
    path_to_gd = get_gd_path()
    if not os.path.exists(path_to_gd):
        make_dirs(path_to_gd)
        download_url(url, path_to_gd)
    else:
        print(f"GroundingDINO model checkpoint found in {path_to_gd}")


if __name__ == '__main__':
    # download_sam(False)
    download_gd()
