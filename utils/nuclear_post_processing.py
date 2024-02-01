import math
import os
import pickle

from PIL import Image
from PySide2 import QtCore

from ui.signals_and_slots import LoadPercentConnection, InfoConnection
from utils.calc_methods import DNToQGis, DNTheamProc
from utils.sam_fragment import create_masks, create_generator
from utils.settings_handler import AppSettings


class PostProcessingWorker(QtCore.QThread):

    def __init__(self, sam_model, yolo_txt_name: str, tek_image_path: str, edges_stats: str, lrm: float,
                 save_folder: str):
        """
        yolo_txt_name - результаты классификации CNN_Worker
        tek_image_path - путь к изображению
        edges_stats - путь к файлу со статистикой
        lrm - ЛРМ
        save_folder - папка для сохранения
        """
        super(PostProcessingWorker, self).__init__()
        self.yolo_txt_name = yolo_txt_name
        self.tek_image_path = tek_image_path
        self.edges_stats = edges_stats
        self.lrm = lrm
        self.save_folder = save_folder
        self.settings = AppSettings()
        self.sam_model = sam_model

        self.polygons = []

        self.psnt_connection = LoadPercentConnection()
        self.info_connection = InfoConnection()

    def calc_points_per_side(self, min_obj_width_meters):
        image = Image.open(self.tek_image_path)
        image_width = image.width

        min_obj_width_px = min_obj_width_meters / self.lrm
        step_px = min_obj_width_px / 2.0
        return math.floor(image_width / step_px)

    def bns_detection(self):
        self.psnt_connection.percent.emit(0)

        ToQGisObj = DNToQGis(self.tek_image_path,
                             self.yolo_txt_name,
                             self.edges_stats)

        self.psnt_connection.percent.emit(5)
        info_message = f"Начинаю поиск зоны расположения БНС..." if self.settings.read_lang() == 'RU' else f"Start finding BNS local zone..."
        self.info_connection.info_message.emit(info_message)

        bns_zones = ToQGisObj.LocalZoneBNS(self.lrm, 180, 500)

        self.psnt_connection.percent.emit(30)

        crop_names = []
        if not bns_zones:
            self.psnt_connection.percent.emit(100)
            info_message = f"Зоны для поиска БНС не найдены" if self.settings.read_lang() == 'RU' else f"Can't find BNS local zone"
            self.info_connection.info_message.emit(info_message)
            return

        for i, im in enumerate(bns_zones['Imgs']):
            crop_name = os.path.join(self.save_folder, f'crop{i}.jpg')
            crop_names.append(crop_name)
            im.save(crop_name)

        self.psnt_connection.percent.emit(40)

        info_message = f"Зона расположения БНС найдена. Начинаю кластеризацию методом SAM" if self.settings.read_lang() == 'RU' else f"Found BNS local zone. Start clustering with SAM.."
        self.info_connection.info_message.emit(info_message)

        step = 0
        steps = len(crop_names) * 2
        points_per_side = 16  # self.calc_points_per_side(min_obj_width_meters=80)
        print(f"Points per side = {points_per_side}")

        generator = create_generator(self.sam_model, pred_iou_thresh=0.88, box_nms_thresh=0.7,
                                     points_per_side=points_per_side, crop_n_points_downscale_factor=1,
                                     crop_nms_thresh=0.7,
                                     output_mode="binary_mask")

        for i, crop_name in enumerate(crop_names):
            pkl_name = os.path.join(self.save_folder, f'crop{i}.pkl')

            create_masks(generator, crop_name, output_path=None,
                         one_image_name=os.path.join(self.save_folder, f'crop{i}_sam.jpg'),
                         pickle_name=pkl_name)

            step += 1
            self.psnt_connection.percent.emit(40 + 60 * float(step) / steps)

            with open(pkl_name, 'rb') as f:
                Mass = pickle.load(f)
                ContBNS = ToQGisObj.FinedBNS(Mass, bns_zones['Coords'][i], self.lrm)

                info_message = f"Кластеризация методом SAM завершена. Создаю контуры БНС..." if self.settings.read_lang() == 'RU' else f"SAM finished. Start building contours..."
                self.info_connection.info_message.emit(info_message)

                for points in ContBNS:
                    cls_num = 5  # bns
                    self.polygons.append({'cls_num': cls_num, 'points': points})

            step += 1
            self.psnt_connection.percent.emit(40 + 60 * float(step) / steps)

    def mz_detection(self):
        post_proc = DNTheamProc(self.tek_image_path, self.yolo_txt_name, self.edges_stats, self.lrm,
                                self.sam_model)
        post_proc.info_connection.info_message.connect(self.info_connection.info_message)
        post_proc.psnt_connection.percent.connect(self.psnt_connection.percent)

        self.polygons = post_proc.AESProc(self.save_folder)

    def run(self):
        # self.bns_detection()

        self.mz_detection()


if __name__ == '__main__':
    def calc_points_per_side(min_obj_width_meters, tek_image_path, lrm):
        image = Image.open(tek_image_path)
        image_width = image.width

        min_obj_width_px = min_obj_width_meters / lrm
        step_px = min_obj_width_px / 2.0
        return math.floor(image_width / step_px)


    tek_image_path = "../nuclear_power/crop0.jpg"
    lrm = 0.9
    print(calc_points_per_side(100, tek_image_path, lrm))
