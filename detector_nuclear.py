import os
import sys

import torch
from PyQt5 import QtWidgets
from qt_material import apply_stylesheet

import utils.help_functions as hf
from detector import Detector
from utils import cls_settings
from utils import config
from utils.edges_from_mask import mask_results_to_yolo_txt
from utils.nuclear_post_processing import PostProcessingWorker
import gc
from ui.dialogs.ask_next_step_dialog import AskNextStepDialog
from utils.settings_handler import AppSettings

basedir = os.path.dirname(__file__)


class DetectorNuclear(Detector):
    def __init__(self, parent=None):
        self.settings = AppSettings()
        self.settings.write_sam_hq(0)
        super().__init__(parent)

        self.mask_res = []

        self.setWindowTitle("Nuclear power station detector")

    def hide(self) -> bool:
        self.settings.write_sam_hq(1)
        super().hide()


    def on_post_finished(self):

        # Back models
        self.handle_detection_model()
        self.gd_model = self.load_gd_model()

        polygons = self.post_worker.polygons
        for pol in polygons:
            cls_num = pol['cls_num']  # bns

            color = None
            label = self.project_data.get_label_name(cls_num)
            if label:
                color = self.project_data.get_label_color(label)
            if not color:
                color = cls_settings.PALETTE[cls_num]

            cls_name = self.cls_combo.itemText(cls_num)

            label_text_params = self.settings.read_label_text_params()
            if label_text_params['hide']:
                text = None
            else:
                text = f"{cls_name} {1.00}"

            shape_id = self.view.add_polygon_to_scene(cls_num, pol['points'], color=color, text=text)

            shape = {'id': shape_id, 'cls_num': cls_num, 'points': pol['points'], 'conf': 1.0}
            self.detected_shapes.append(shape)

        self.block_geo_coords_message = False

        self.progress_toolbar.hide_progressbar()
        self.save_view_to_project()

        self.statusBar().showMessage(
            f"Найдено {len(self.detected_shapes)} объектов" if self.settings.read_lang() == 'RU' else f"{len(self.detected_shapes)} objects has been detected",
            3000)

    def on_post_signal(self, message):
        self.statusBar().showMessage(
            message,
            3000)

    def cuda_model_clear(self, model):
        model.to('cpu')
        del model
        gc.collect()
        torch.cuda.empty_cache()

    def start_post_processing(self):
        """
        Старт пост-обработки классическими методами
        """
        self.ask_next_step.hide()

        if len(self.mask_res) > 0:
            self.statusBar().showMessage(
                f"Начинаю пост-обработку..." if self.settings.read_lang() == 'RU' else f"Start post-processing...",
                3000)

            self.progress_toolbar.show_progressbar()
            self.block_geo_coords_message = True

            yolo_txt_name = os.path.join(hf.handle_temp_folder(os.getcwd()), f'{self.tek_image_name.split(".jpg")[0]}.txt')

            if self.tek_image_name in self.map_geotiff_names:
                tek_image_path = self.map_geotiff_names[self.tek_image_name]
            else:
                tek_image_path = self.tek_image_path

            mask_results_to_yolo_txt(self.mask_res, tek_image_path, yolo_txt_name)
            edges_stats = os.path.join(basedir, 'nuclear_power', 'out_rebra.csv')

            # before SAM - move other models to CPU and del
            for m in [self.yolo, self.gd_model]:
                self.cuda_model_clear(m)

            self.post_worker = PostProcessingWorker(self.sam.model, yolo_txt_name=yolo_txt_name,
                                                    tek_image_path=tek_image_path,
                                                    edges_stats=edges_stats, lrm=self.lrm,
                                                    save_folder=os.path.join(basedir, 'nuclear_power'))

            self.progress_toolbar.set_signal(self.post_worker.psnt_connection.percent)

            self.post_worker.info_connection.info_message.connect(self.on_post_signal)
            self.post_worker.finished.connect(self.on_post_finished)

            if not self.post_worker.isRunning():
                self.post_worker.start()

    def on_cnn_finished(self):
        """
        При завершении классификации.
        """
        self.mask_res = self.CNN_worker.mask_results

        super(DetectorNuclear, self).on_cnn_finished()

        message = f"Найдено {len(self.detected_shapes)} объектов" if self.settings.read_lang() == 'RU' else f"{len(self.detected_shapes)} objects has been detected"

        self.ask_next_step = AskNextStepDialog(None, "Поиск объектов нейросетью", "Поиск БНС", message,
                                               width_percent=0.2, height_percent=0.2)
        self.ask_next_step.nextBtn.clicked.connect(self.start_post_processing)
        self.ask_next_step.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             'primaryTextColor': '#ffffff'}

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = DetectorNuclear()
    w.show()
    sys.exit(app.exec_())
