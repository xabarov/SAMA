import numpy as np
import cv2


def find_exteriour_contours(img):
    ret = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(ret) == 2:
        return ret[0]
    elif len(ret) == 3:
        return ret[1]


class MagicWand:

    def __init__(self, np_img, x, y, connectivity=4, tolerance=32):
        self.img = np_img
        h, w = np_img.shape[:2]
        tolerance = (tolerance,) * 3
        self.mask = np.zeros((h, w), dtype=np.uint8)
        self._flood_mask = np.zeros((h + 2, w + 2), dtype=np.uint8)
        self._flood_fill_flags = (connectivity | cv2.FLOODFILL_FIXED_RANGE | cv2.FLOODFILL_MASK_ONLY | 255 << 8)

        self._flood_mask[:] = 0
        cv2.floodFill(self.img,
                      self._flood_mask,
                      (x, y),
                      0,
                      tolerance,
                      tolerance,
                      self._flood_fill_flags)

        flood_mask = self._flood_mask[1:-1, 1:-1].copy()

        self.mask = flood_mask

        self.show()

    def show(self):
        viz = self.img.copy()
        contours = find_exteriour_contours(self.mask)
        viz = cv2.drawContours(viz, contours, -1, color=(255,) * 3, thickness=-1)
        viz = cv2.addWeighted(self.img, 0.75, viz, 0.25, 0)
        viz = cv2.drawContours(viz, contours, -1, color=(255,) * 3, thickness=1)
        cv2.imwrite("test_data/img.jpg", viz)


if __name__ == '__main__':
    img = cv2.imread('test_data/altus_usa_bing_1.jpg')
    mw = MagicWand(img, 500, 100)
