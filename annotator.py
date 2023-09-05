import os

import cv2
import matplotlib.pyplot as plt
import torch
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon, QCursor
from PyQt5.QtWidgets import QAction, QMessageBox, QMenu
from shapely import Polygon
from ultralytics import YOLO

import utils.help_functions as hf
from gd.gd_sam import load_model as gd_load_model
from gd.gd_worker import GroundingSAMWorker
from ui.base_window import MainWindow
from ui.import_dialogs import ImportFromYOLODialog
from ui.input_dialog import PromptInputDialog
from ui.settings_window import SettingsWindow
from ui.show_image_widget import ShowImgWindow
from utils import cls_settings
from utils import config
from utils.cnn_worker import CNN_worker
from utils.importer import Importer
from utils.predictor import SAMImageSetter
from utils.sam_predictor import load_model as sam_load_model
from utils.sam_predictor import mask_to_seg, predict_by_points, predict_by_box


class Annotator(MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Annotator")

        # Current CUDA model
        self.try_to_set_platform('cuda')
        self.last_sam_use_hq = self.settings.read_sam_hq()

        # Detector
        self.started_cnn = None

        # GroundingDINO
        self.gd_worker = None
        self.prompts = []

        # SAM
        self.image_set = False
        self.image_setter = None
        self.queue_to_image_setter = []

        self.view.mask_end_drawing.on_mask_end_drawing.connect(self.ai_mask_end_drawing)

        self.handle_cuda_models()

        self.scanning_mode = False
        self.lrm = None

        self.detected_shapes = []

    def try_to_set_platform(self, platform):
        lang = self.settings.read_lang()

        if platform in ['cuda', 'Auto'] and torch.cuda.is_available():
            self.last_platform = 'cuda'
            print("CUDA is available")
            if lang == 'RU':
                self.statusBar().showMessage(
                    "Найдено устройство NVIDIA CUDA. Нейросеть будет использовать ее для ускорения", 3000)
            else:
                self.statusBar().showMessage(
                    "NVIDIA CUDA is found. Neural networks will use it for acceleration", 3000)
        else:
            self.last_platform = 'cpu'
            if lang == 'RU':
                self.statusBar().showMessage(
                    "Нейросеть будет использовать ресурсы процессора", 3000)
            else:
                self.statusBar().showMessage(
                    "Neural networks will use CPU", 3000)

    def createActions(self):
        super(Annotator, self).createActions()

        self.balanceAct = QAction("Информация о датасете" if self.settings.read_lang() == 'RU' else "Dataset info",
                                  self,
                                  enabled=False, triggered=self.on_dataset_balance_clicked)

        self.syncLabelsAct = QAction(
            "Синхронизировать имена меток" if self.settings.read_lang() == 'RU' else "Fill label names from AI model",
            self, enabled=False,
            triggered=self.sync_labels)

        # Object detector
        self.detectAct = QAction(
            "Обнаружить объекты за один проход" if self.settings.read_lang() == 'RU' else "Detect objects", self,
            shortcut="Ctrl+Y", enabled=False,
            triggered=self.detect)

        self.detectAllImagesAct = QAction(
            "Обнаружить объекты на всех изображениях" if self.settings.read_lang() == 'RU' else "Detect objects at all image",
            self, enabled=False,
            triggered=self.detect_all_images)

        # AI Annotators
        self.aiAnnotatorPointsAct = QAction(
            "Сегментация по точкам" if self.settings.read_lang() == 'RU' else "SAM by points",
            self, enabled=False, shortcut="Ctrl+A",
            triggered=self.ai_points_pressed,
            checkable=True)
        self.aiAnnotatorMaskAct = QAction(
            "Сегментация внутри бокса" if self.settings.read_lang() == 'RU' else "SAM by box", self,
            enabled=False, shortcut="Ctrl+M",
            triggered=self.ai_mask_pressed,
            checkable=True)

        self.GroundingDINOSamAct = QAction(
            "GroundingDINO + SAM" if self.settings.read_lang() == 'RU' else "GroundingDINO + SAM", self,
            enabled=False, shortcut="Ctrl+G",
            triggered=self.grounding_sam_pressed,
            checkable=True)

    def sync_labels(self):
        if self.yolo:
            names = self.yolo.names  # dict like {0:name1, 1:name2...}
            self.cls_combo.clear()
            labels = []
            for key in names:
                label = names[key]
                self.cls_combo.addItem(label)
                labels.append(label)

            self.project_data.set_labels(labels)
            self.project_data.set_labels_colors(labels, rewrite=True)

    def toggle_act(self, is_active):
        super(Annotator, self).toggle_act(is_active)
        self.aiAnnotatorMethodMenu.setEnabled(is_active)
        self.aiAnnotatorPointsAct.setEnabled(is_active)
        self.aiAnnotatorMaskAct.setEnabled(is_active)
        self.aiAnnotatorMethodMenu.setEnabled(is_active)

        self.syncLabelsAct.setEnabled(is_active)

        self.GroundingDINOSamAct.setEnabled(is_active)
        self.balanceAct.setEnabled(is_active)
        self.detectAllImagesAct.setEnabled(is_active)
        self.detectAct.setEnabled(is_active)

    def createMenus(self):
        super(Annotator, self).createMenus()

        self.aiAnnotatorMethodMenu = QMenu("С помощью ИИ" if self.settings.read_lang() == 'RU' else "AI", self)

        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorPointsAct)
        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorMaskAct)
        self.aiAnnotatorMethodMenu.addAction(self.GroundingDINOSamAct)

        self.AnnotatorMethodMenu.addMenu(self.aiAnnotatorMethodMenu)

        self.classifierMenu = QMenu("Классификатор" if self.settings.read_lang() == 'RU' else "Classifier", self)
        self.classifierMenu.addAction(self.detectAct)
        self.classifierMenu.addAction(self.detectAllImagesAct)
        self.classifierMenu.addAction(self.syncLabelsAct)
        self.annotatorMenu.addAction(self.balanceAct)

        self.menuBar().clear()
        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.classifierMenu)
        self.menuBar().addMenu(self.annotatorMenu)
        self.menuBar().addMenu(self.settingsMenu)
        self.menuBar().addMenu(self.helpMenu)

    def set_icons(self):
        super(Annotator, self).set_icons()
        # AI
        self.aiAnnotatorMethodMenu.setIcon(QIcon(self.icon_folder + "/ai.png"))
        self.aiAnnotatorPointsAct.setIcon(QIcon(self.icon_folder + "/mouse.png"))
        self.aiAnnotatorMaskAct.setIcon(QIcon(self.icon_folder + "/ai_select.png"))
        self.detectAllImagesAct.setIcon(QIcon(self.icon_folder + "/detect_all.png"))

        self.syncLabelsAct.setIcon(QIcon(self.icon_folder + "/sync.png"))

        self.GroundingDINOSamAct.setIcon(QIcon(self.icon_folder + "/dino.png"))

        self.balanceAct.setIcon(QIcon(self.icon_folder + "/bar-chart.png"))

        # classifier
        self.detectAct.setIcon(QIcon(self.icon_folder + "/detect.png"))

    def open_image(self, image_name):
        super(Annotator, self).open_image(image_name)

        self.image_set = False
        self.queue_image_to_sam(image_name)

    def reload_image(self):
        super(Annotator, self).reload_image()
        self.view.clear_ai_points()

    def set_image(self, image_name):
        self.cv2_image = cv2.imread(image_name)
        self.image_set = False
        self.image_setter.set_image(self.cv2_image)
        self.queue_to_image_setter = []
        self.statusBar().showMessage(
            "Нейросеть SAM еще не готова. Подождите секунду..." if self.settings.read_lang() == 'RU' else "SAM is loading. Please wait...",
            3000)
        self.image_setter.start()

    def get_jpg_path(self, image_name):
        # для поддержки с детектором. У него данный метод переписан и поддерживает конвертацию в tif
        return image_name

    def on_image_set(self):

        if len(self.queue_to_image_setter) != 0:
            image_name = self.queue_to_image_setter[-1]  # geo_tif_names
            jpg_path = self.get_jpg_path(image_name)
            self.set_image(jpg_path)

        else:
            self.statusBar().showMessage(
                "Нейросеть SAM готова к сегментации" if self.settings.read_lang() == 'RU' else "SAM ready to work",
                3000)
            self.image_set = True

    def showSettings(self):
        """
        Показать окно с настройками приложения
        """

        self.settings_window = SettingsWindow(self)

        self.settings_window.okBtn.clicked.connect(self.on_settings_closed)
        self.settings_window.cancelBtn.clicked.connect(self.on_settings_closed)

        self.settings_window.show()

    def about(self):
        """
        Окно о приложении
        """
        QMessageBox.about(self, "AI Annotator",
                          "<p><b>AI Annotator</b></p>"
                          "<p>Программа для разметки изображений с поддержкой автоматической сегментации</p>" if
                          self.settings.read_lang() == 'RU' else "<p>Labeling Data for Object Detection and Instance Segmentation "
                                                                 "with Segment Anything Model (SAM) and GroundingDINO.</p>")

    def handle_sam_model(self):
        self.sam = self.load_sam()
        self.image_setter = SAMImageSetter()
        self.image_setter.set_predictor(self.sam)
        self.image_setter.finished.connect(self.on_image_set)
        if self.tek_image_path:
            self.queue_image_to_sam(self.tek_image_path)

    def handle_cuda_models(self):

        # Start on Loading Animation
        self.view.start_circle_progress()

        self.handle_sam_model()

        cfg_path, weights_path = cls_settings.get_cfg_and_weights_by_cnn_name('YOLOv8')
        config_path = os.path.join(os.getcwd(), cfg_path)
        model_path = os.path.join(os.getcwd(), weights_path)

        self.yolo = YOLO(model_path)
        self.yolo.data = config_path

        dev_set = 'cpu'
        # if self.settings.read_platform() == "cuda":
        #     dev_set = 0

        self.yolo.to(dev_set)
        self.yolo.overrides['data'] = config_path

        self.gd_model = self.load_gd_model()

        self.view.stop_circle_progress()

    def on_settings_closed(self):
        super(Annotator, self).on_settings_closed()

        sam_hq = self.settings.read_sam_hq()
        if sam_hq != self.last_sam_use_hq:
            self.handle_sam_model()

        platform = self.settings.read_platform()

        if platform != self.last_platform:
            self.try_to_set_platform(platform)

    def ai_points_pressed(self):

        self.ann_type = "AiPoints"
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings.read_alpha()

        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def ai_mask_pressed(self):

        self.ann_type = "AiMask"
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings.read_alpha()
        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def add_sam_polygon_to_scene(self, sam_mask, cls_num=None):
        simplify_factor = float(self.settings.read_simplify_factor())
        points_mass = mask_to_seg(sam_mask, simplify_factor=simplify_factor)

        if len(points_mass) > 0:
            filtered_points_mass = []
            for points in points_mass:
                shapely_pol = Polygon(points)
                area = shapely_pol.area

                if area > config.POLYGON_AREA_THRESHOLD:

                    filtered_points_mass.append(points)

                else:
                    if self.settings.read_lang() == 'RU':
                        self.statusBar().showMessage(
                            f"Метку сделать не удалось. Площадь маски слишком мала {area:0.3f}. Попробуйте еще раз",
                            3000)
                    else:
                        self.statusBar().showMessage(
                            f"Can't create label. Area of label is too small {area:0.3f}. Try again", 3000)

            if not cls_num:
                cls_num = self.cls_combo.currentIndex()
            cls_name = self.cls_combo.itemText(cls_num)
            alpha_tek = self.settings.read_alpha()
            color = self.project_data.get_label_color(cls_name)

            self.view.add_polygons_group_to_scene(cls_num, filtered_points_mass, color, alpha_tek)

            self.update_labels()

    def ai_mask_end_drawing(self):

        self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))
        input_box = self.view.get_sam_mask_input()

        self.view.remove_active()

        if len(input_box):
            if self.image_set and not self.image_setter.isRunning():
                mask = predict_by_box(self.sam, input_box)
                self.add_sam_polygon_to_scene(mask)

        self.view.end_drawing()
        self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

    def start_drawing(self):
        super(Annotator, self).start_drawing()
        self.view.clear_ai_points()

    def break_drawing(self):
        super(Annotator, self).break_drawing()
        if self.ann_type == "AiPoints":
            self.view.clear_ai_points()
            self.view.remove_active()

    def end_drawing(self):
        super(Annotator, self).end_drawing()

        if self.ann_type == "AiPoints":

            self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))

            input_point, input_label = self.view.get_sam_input_points_and_labels()

            if len(input_label):
                if self.image_set and not self.image_setter.isRunning():
                    masks = predict_by_points(self.sam, input_point, input_label, multi=False)
                    for mask in masks:
                        self.add_sam_polygon_to_scene(mask)

            else:
                self.view.remove_active()

            self.view.clear_ai_points()
            self.view.end_drawing()

            self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

            self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def on_quit(self):
        self.exit_box.hide()

        self.write_size_pos()

        self.hide()  # Скрываем окно

        if self.image_setter:
            self.image_setter.running = False  # Изменяем флаг выполнения
            self.image_setter.wait(5000)  # Даем время, чтобы закончить
        if self.gd_worker:
            self.gd_worker.running = False
            self.gd_worker.wait(5000)

        self.is_asked_before_close = True
        self.close()

    def load_gd_model(self):
        config_file = os.path.join(os.getcwd(),
                                   config.PATH_TO_GROUNDING_DINO_CONFIG)
        grounded_checkpoint = os.path.join(os.getcwd(),
                                           config.PATH_TO_GROUNDING_DINO_CHECKPOINT)

        return gd_load_model(config_file, grounded_checkpoint, device=self.settings.read_platform())

    def load_sam(self):
        use_hq = self.settings.read_sam_hq()
        if use_hq:
            sam_model_path = os.path.join(os.getcwd(), config.PATH_TO_SAM_HQ_CHECKPOINT)
        else:
            sam_model_path = os.path.join(os.getcwd(), config.PATH_TO_SAM_CHECKPOINT)

        return sam_load_model(sam_model_path, device=self.settings.read_platform(), use_sam_hq=use_hq)

    def queue_image_to_sam(self, image_name):

        if not self.image_setter.isRunning():
            self.image_setter.set_image(self.cv2_image)
            self.statusBar().showMessage(
                "Начинаю загружать изображение в нейросеть SAM..." if self.settings.read_lang() == 'RU' else "Start loading image to SAM...",
                3000)
            self.image_setter.start()
        else:
            self.queue_to_image_setter.append(image_name)

            self.statusBar().showMessage(
                f"Изображение {os.path.split(image_name)[-1]} добавлено в очередь на обработку." if self.settings.read_lang() == 'RU' else f"Image {os.path.split(image_name)[-1]} is added to queue...",
                3000)

    def detect(self):
        # на вход воркера - исходное изображение

        jpg_path = self.get_jpg_path(self.tek_image_path)

        img_name = os.path.basename(jpg_path)
        img_path = os.path.dirname(jpg_path)

        self.run_detection(img_name=img_name, img_path=img_path)

    def run_detection(self, img_name, img_path):
        """
        Запуск классификации
        img_name - имя изображения
        img_path - путь к изображению
        """

        self.started_cnn = self.settings.read_cnn_model()

        conf_thres_set = self.settings.read_conf_thres()
        iou_thres_set = self.settings.read_iou_thres()
        simplify_factor = self.settings.read_simplify_factor()

        if self.scanning_mode:
            str_text = "Начинаю классифкацию СНС {0:s} сканирующим окном".format(self.started_cnn)
        else:
            str_text = "Начинаю классифкацию СНС {0:s}".format(self.started_cnn)

        self.statusBar().showMessage(str_text, 3000)

        self.CNN_worker = CNN_worker(model=self.yolo, conf_thres=conf_thres_set, iou_thres=iou_thres_set,
                                     img_name=img_name, img_path=img_path,
                                     scanning=self.scanning_mode,
                                     linear_dim=self.lrm, simplify_factor=simplify_factor)

        self.CNN_worker.started.connect(self.on_cnn_started)

        self.progress_toolbar.set_signal(self.CNN_worker.psnt_connection.percent)

        self.CNN_worker.finished.connect(self.on_cnn_finished)

        if not self.CNN_worker.isRunning():
            self.CNN_worker.start()

    def on_cnn_started(self):
        """
        При начале классификации
        """
        self.progress_toolbar.show_progressbar()
        self.statusBar().showMessage(
            f"Начинаю поиск объектов на изображении..." if self.settings.read_lang() == 'RU' else f"Start searching object on image...",
            3000)

    def on_cnn_finished(self):
        """
        При завершении классификации
        """

        if self.scanning_mode:
            self.scanning_mode = False

        self.detected_shapes = []
        for res in self.CNN_worker.mask_results:

            shape_id = self.view.get_unique_label_id()

            cls_num = res['cls_num']
            points = res['points']

            color = None
            label = self.project_data.get_label_name(cls_num)
            if label:
                color = self.project_data.get_label_color(label)
            if not color:
                color = cls_settings.PALETTE[cls_num]

            self.view.add_polygon_to_scene(cls_num, points, color=color, id=shape_id)

            shape = {'id': shape_id, 'cls_num': cls_num, 'points': points, 'conf': res['conf']}
            self.detected_shapes.append(shape)

        self.progress_toolbar.hide_progressbar()
        self.update_labels()

        self.statusBar().showMessage(
            f"Найдено {len(self.CNN_worker.mask_results)} объектов" if self.settings.read_lang() == 'RU' else f"{len(self.CNN_worker.mask_results)} objects has been detected",
            3000)

    def grounding_sam_pressed(self):

        theme = self.settings.read_theme()

        self.prompt_input_dialog = PromptInputDialog(self, theme=theme,
                                                     class_names=self.project_data.get_labels(),
                                                     on_ok_clicked=self.start_grounddino, prompts_variants=self.prompts)
        self.prompt_input_dialog.show()

    def run_gd(self, image_name, prompt):
        self.progress_toolbar.show_progressbar()

        self.prompt_cls_name = self.prompt_input_dialog.getClsName()
        self.prompt_cls_num = self.prompt_input_dialog.getClsNumber()

        if prompt not in self.prompts:
            self.prompts.append(prompt)

        config_file = os.path.join(os.getcwd(),
                                   config.PATH_TO_GROUNDING_DINO_CONFIG)
        grounded_checkpoint = os.path.join(os.getcwd(),
                                           config.PATH_TO_GROUNDING_DINO_CHECKPOINT)

        self.gd_worker = GroundingSAMWorker(config_file=config_file, grounded_checkpoint=grounded_checkpoint,
                                            sam_predictor=self.sam, tek_image_path=image_name,
                                            grounding_dino_model=self.gd_model,
                                            box_threshold=self.prompt_input_dialog.get_box_threshold(),
                                            text_threshold=self.prompt_input_dialog.get_text_threshold(),
                                            prompt=prompt)

        self.progress_toolbar.set_percent(10)

        self.gd_worker.finished.connect(self.on_gd_worker_finished)

        if not self.gd_worker.isRunning():
            self.statusBar().showMessage(
                f"Начинаю поиск {self.prompt_cls_name} на изображении..." if self.settings.read_lang() == 'RU' else f"Start searching {self.prompt_cls_name} on image...",
                3000)
            self.gd_worker.start()

    def start_grounddino(self):
        prompt = self.prompt_input_dialog.getPrompt()
        self.prompt_input_dialog.close()

        jpg_path = self.get_jpg_path(self.tek_image_path)

        if prompt:
            self.run_gd(jpg_path, prompt)

    def on_gd_worker_finished(self):
        masks = self.gd_worker.getMasks()
        self.progress_toolbar.set_percent(50)

        for i, mask in enumerate(masks):
            self.add_sam_polygon_to_scene(mask, cls_num=self.prompt_cls_num)
            self.progress_toolbar.set_percent(50 + int(i + 1) * 100.0 / len(masks))

        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())
        self.progress_toolbar.hide_progressbar()

    def on_dataset_balance_clicked(self):
        balance_data = self.project_data.calc_dataset_balance()

        labels = list(balance_data.keys())
        values = list(balance_data.values())

        dataset_dir = self.project_data.get_image_path()
        balance_txt_name = os.path.join(os.path.dirname(dataset_dir), 'label_balance.txt')
        with open(balance_txt_name, 'w') as f:
            for label, size in balance_data.items():
                f.write(f"{label}: {size}\n")

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.bar(labels, values,
               # color=config.THEMES_COLORS[self.theme_str],
               width=0.8)

        ax.set_xlabel("Label names")
        ax.set_ylabel("No. of labels")
        ax.tick_params(axis='x', rotation=70)
        plt.title('Баланс меток')

        temp_folder = self.handle_temp_folder()
        fileName = os.path.join(temp_folder, 'balance.jpg')
        plt.savefig(fileName)

        ShowImgWindow(self, title='Баланс меток', img_file=fileName, icon_folder=self.icon_folder)

    def detect_all_images(self):
        """
        Запуск классификации
        img_name - имя изображения
        img_path - путь к изображению
        """

        self.started_cnn = self.settings.read_cnn_model()

        conf_thres_set = self.settings.read_conf_thres()
        iou_thres_set = self.settings.read_iou_thres()

        str_text = "Начинаю классифкацию СНС {0:s}".format(self.started_cnn)

        self.statusBar().showMessage(str_text, 3000)

        images_list = [os.path.join(self.dataset_dir, im_name) for im_name in self.dataset_images]

        self.CNN_worker = CNN_worker(model=self.yolo, conf_thres=conf_thres_set, iou_thres=iou_thres_set,
                                     img_name=None, img_path=None,
                                     images_list=images_list,
                                     scanning=None)

        self.CNN_worker.started.connect(self.on_cnn_started)

        self.progress_toolbar.set_signal(self.CNN_worker.psnt_connection.percent)

        self.CNN_worker.finished.connect(self.on_all_images_finished)

        if not self.CNN_worker.isRunning():
            self.CNN_worker.start()

    def on_all_images_finished(self):
        """
        При завершении классификации всех изображений
        """

        self.progress_toolbar.hide_progressbar()

        self.project_data.set_all_images(self.CNN_worker.image_list_results)

        self.load_image_data(self.tek_image_name)
        self.fill_labels_on_tek_image_list_widget()
        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def importFromYOLOBox(self):

        self.close_project()

        self.import_dialog = ImportFromYOLODialog(self, on_ok_clicked=self.on_import_yolo_clicked, convert_to_mask=True)
        self.is_seg_import = False
        self.import_dialog.show()

    def on_import_yolo_clicked(self):
        yaml_data = self.import_dialog.getData()

        if self.sam:
            convert_to_masks = self.import_dialog.convert_to_mask_checkbox.isChecked()
        else:
            convert_to_masks = False

        copy_images_path = None
        if yaml_data['is_copy_images']:
            copy_images_path = yaml_data['save_images_dir']

        self.importer = Importer(yaml_data=yaml_data, alpha=self.settings.read_alpha(), is_seg=self.is_seg_import,
                                 copy_images_path=copy_images_path, dataset=yaml_data["selected_dataset"],
                                 is_coco=False, convert_to_masks=convert_to_masks, sam_predictor=self.sam)

        self.importer.load_percent_conn.percent.connect(self.on_import_percent_change)
        self.importer.info_conn.info_message.connect(self.on_importer_message)
        self.importer.err_conn.error_message.connect(self.on_importer_message)

        self.importer.finished.connect(self.on_import_finished)

        if not self.importer.isRunning():
            self.importer.start()


if __name__ == '__main__':
    import sys
    from qt_material import apply_stylesheet

    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             # 'primaryTextColor': '#ffffff',
             }

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra, invert_secondary=False)

    w = Annotator()
    w.show()
    sys.exit(app.exec_())
