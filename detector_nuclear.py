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
    def __init__(self, parent=None, hide_classes=None):

        """
        hide_classes - №№ не отображаемых классов (меток). По умолчанию отображаются все

        Текущий вариант для АЭС
        self.yolo.names = {0: 'reactor_sq', 1: 'reactor', 2: 'engine_room', 3: 'pipe', 4: 'turbine',
                            5: 'switchgear', 6: 'pump', 7: 'cooltower',
                            8: 'ct_vent_circle', 9: 'ct_vent_sq', 10: 'ct_active',
                            11: 'ISFSI', 12: 'tank', 13: 'parking', 14: 'waste_water_cil'}


        """
        self.settings = AppSettings()
        super().__init__(parent)

        self.mask_res = []

        if self.settings.read_platform() == 'cuda':
            self.cuda_model_clear(self.sam.model)
        self.use_hq_before = self.settings.read_sam_hq()
        self.settings.write_sam_hq(1)

        self.sam = self.load_sam('b')
        self.setWindowTitle("Nuclear power station detector")

        self.names = self.yolo.names

        if hide_classes:
            self.names = {k: v for k, v in self.names.items() if k not in hide_classes}

    def hide(self) -> bool:
        self.settings.write_sam_hq(self.use_hq_before)
        super().hide()
        return True

    def on_post_finished(self):

        polygons = self.post_worker.polygons

        alpha_tek = self.settings.read_alpha()
        alpha_edge = self.settings.read_edges_alpha()

        self.view.remove_all_polygons()
        self.detected_shapes.clear()

        for pol in polygons:

            cls_num = pol['cls_num']

            if cls_num not in self.names.keys():
                # Фильтрация не отображаемых классов
                continue

            color = None
            label = self.project_data.get_label_name(cls_num)
            if pol['cnn_found']:
                if label:
                    color = self.project_data.get_label_color(label)
                if not color:
                    color = cls_settings.PALETTE[cls_num]
                alpha_edge_tek = alpha_edge
            else:
                alpha_edge_tek = 1.0
                color = (255, 10, 10, 120)

            cls_name = self.cls_combo.itemText(cls_num)

            label_text_params = self.settings.read_label_text_params()
            if label_text_params['hide']:
                text = None
            else:
                text = f"{cls_name}"

            shape_id = self.view.add_polygon_to_scene(cls_num, pol['points'], color=color, text=text, alpha=alpha_tek,
                                                      alpha_edge=alpha_edge_tek)

            shape = {'id': shape_id, 'cls_num': cls_num, 'points': pol['points'], 'conf': 1.0}
            self.detected_shapes.append(shape)

        self.block_geo_coords_message = False

        self.progress_toolbar.hide_progressbar()
        self.save_view_to_project()

        self.statusBar().showMessage(
            f"Найдено {len(self.detected_shapes)} объектов" if self.settings.read_lang() == 'RU' else f"{len(self.detected_shapes)} objects has been detected",
            3000)

        # Back models
        self.handle_detection_model()
        self.gd_model = self.load_gd_model()

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

            yolo_txt_name = os.path.join(hf.handle_temp_folder(os.getcwd()),
                                         f'{self.tek_image_name.split(".jpg")[0]}.txt')

            if self.tek_image_name in self.map_geotiff_names:
                tek_image_path = self.map_geotiff_names[self.tek_image_name]
            else:
                tek_image_path = self.tek_image_path

            mask_results_to_yolo_txt(self.mask_res, tek_image_path, yolo_txt_name)
            edges_stats = os.path.join(basedir, 'nuclear_power', 'out_rebra.csv')

            # before SAM - move other models to CPU and del
            if self.settings.read_platform() == 'cuda':
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

        if self.scanning_mode:
            self.scanning_mode = False

        alpha_tek = self.settings.read_alpha()
        alpha_edge = self.settings.read_edges_alpha()

        if self.detected_image != self.tek_image_name:
            # куда-то переключили во время детекции
            self.save_view_to_project()  # сохраняем что успели наделать на том изображении
            self.tek_image_name = self.detected_image
            self.tek_image_path = os.path.join(self.dataset_dir, self.detected_image)
            self.reload_image(is_tek_image_changed=True)
            self.images_list_widget.move_to_image_name(self.detected_image)

        self.detected_shapes.clear()

        for res in self.mask_res:
            cls_num = res['cls_num']

            if cls_num not in self.names.keys():
                # Фильтрация не отображаемых классов
                continue

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
                text = f"{cls_name} {res['conf']:0.3f}"

            shape_id = self.view.add_polygon_to_scene(cls_num, res['points'], color=color, text=text, alpha=alpha_tek,
                                                      alpha_edge=alpha_edge)

            shape = {'id': shape_id, 'cls_num': cls_num, 'points': res['points'], 'conf': res['conf']}
            self.detected_shapes.append(shape)

        self.progress_toolbar.hide_progressbar()
        self.save_view_to_project()
        if self.settings.read_lang() == 'RU':
            message = f"Найдено {len(self.detected_shapes)} объектов"
        else:
            message = f"{len(self.detected_shapes)} objects has been detected"

        self.statusBar().showMessage(
            message,
            3000)

        self.ask_next_step = AskNextStepDialog(None, "Поиск объектов нейросетью",
                                               "Поиск объектов на основе геометрических признаков", message,
                                               width_percent=0.2, height_percent=0.2)
        self.ask_next_step.nextBtn.clicked.connect(self.start_post_processing)
        self.ask_next_step.show()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             'primaryTextColor': '#ffffff'}

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = DetectorNuclear(hide_classes=[3, 11, 12, 13])
    w.show()
    sys.exit(app.exec_())
