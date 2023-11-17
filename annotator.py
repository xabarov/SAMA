import ast
import os

import cv2
import matplotlib.pyplot as plt
import torch
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtGui import QIcon, QCursor, QKeySequence
from PyQt5.QtWidgets import QAction, QMessageBox, QMenu
from bs4 import BeautifulSoup
from openvino.runtime import Core
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
    """
    Класс для создания разметки с поддержкой ИИ (модели SAM, GroundingDINO, YOLOv8 и т.д.)
    """

    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Annotator")

        # Current CUDA model
        self.try_to_set_platform('cuda')
        self.last_sam_use_hq = self.settings.read_sam_hq()
        self.last_cnn = self.settings.read_cnn_model()

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
        """
        Ищет CUDA устройство. Если не находит - используем CPU
        """
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

    def reset_shortcuts(self):

        super(Annotator, self).reset_shortcuts()

        shortcuts = self.settings.read_shortcuts()

        for sc, act in zip(
                ['detect_single', 'gd', 'sam_box', 'sam_points'],
                [self.detectAct, self.GroundingDINOSamAct, self.aiAnnotatorMaskAct, self.aiAnnotatorPointsAct]):
            shortcut = shortcuts[sc]
            appearance = shortcut['appearance']
            act.setShortcut(QKeySequence(appearance))

    def createActions(self):
        """
        Добавляем новые действия к базовой модели
        """


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
            enabled=False,
            triggered=self.detect)

        self.detectAllImagesAct = QAction(
            "Обнаружить объекты на всех изображениях" if self.settings.read_lang() == 'RU' else "Detect objects at all image",
            self, enabled=False,
            triggered=self.detect_all_images)

        # AI Annotators
        self.aiAnnotatorPointsAct = QAction(
            "Сегментация по точкам" if self.settings.read_lang() == 'RU' else "SAM by points",
            self, enabled=False,
            triggered=self.ai_points_pressed,
            checkable=True)
        self.aiAnnotatorMaskAct = QAction(
            "Сегментация внутри бокса" if self.settings.read_lang() == 'RU' else "SAM by box", self,
            enabled=False,
            triggered=self.ai_mask_pressed,
            checkable=True)

        self.GroundingDINOSamAct = QAction(
            "GroundingDINO + SAM" if self.settings.read_lang() == 'RU' else "GroundingDINO + SAM", self,
            enabled=False,
            triggered=self.grounding_sam_pressed,
            checkable=True)

        super(Annotator, self).createActions()

    def read_detection_model_names(self):
        """
        Чтение имен классов, записанных в модели нейросети для обнаружения
        Поддерживаются:
            - YOLOv8 pt
            - YOLOv8 OpenVino
        """
        detection_model = self.settings.read_cnn_model()

        if detection_model == 'YOLOv8_openvino':
            config, weights = cls_settings.get_cfg_and_weights_by_cnn_name('YOLOv8_openvino')
            with open(config, 'r') as f:
                data = f.read()
                Bs_data = BeautifulSoup(data, "xml")
                names = Bs_data.find('names')  # .find('names')
                return ast.literal_eval(names.get('value'))
        elif detection_model == 'YOLOv8':
            return self.yolo.names  # dict like {0:name1, 1:name2...}

    def sync_labels(self):
        """
        Записать имена модели для обнаружения в чекбокс
        Поддерживаются:
            - YOLOv8 pt
            - YOLOv8 OpenVino
        """
        if self.yolo:
            names = self.read_detection_model_names()
            if not names:
                if self.settings.read_lang() == 'RU':
                    self.statusBar().showMessage(
                        f"Неизвестное имя модели {self.settings.read_cnn_model()}",
                        3000)
                else:
                    self.statusBar().showMessage(
                        f"Unknown detection model {self.settings.read_cnn_model()}", 3000)
                return

            self.cls_combo.clear()
            labels = []
            for key in names:
                label = names[key]
                self.cls_combo.addItem(label)
                labels.append(label)

            self.project_data.set_labels(labels)
            self.project_data.set_labels_colors(labels, rewrite=True)
            self.reload_image(is_tek_image_changed=False)
            self.save_view_to_project()

    def toggle_act(self, is_active):
        """
        Переключение действий, завясящих от состояния is_active
        """
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
        """
        К базовой модели добавляется SAM. Поскольку ему нужен прогрев, создаем очередь загружаемых изображений
        """
        super(Annotator, self).open_image(image_name)

        self.image_set = False
        self.queue_image_to_sam(image_name)  # создаем очередь загружаемых изображений

    def reload_image(self, is_tek_image_changed=False):
        """
        Заново загружает текущее изображение с разметкой
        """
        super(Annotator, self).reload_image(is_tek_image_changed=is_tek_image_changed)
        self.view.clear_ai_points()  # очищаем точки-prompts SAM

    def set_image(self, image_name):
        """
        Старт загрузки изображения в модель SAM
        """
        self.cv2_image = cv2.imread(image_name)
        self.image_set = False
        self.image_setter.set_image(self.cv2_image)
        self.queue_to_image_setter = []
        self.statusBar().showMessage(
            "Нейросеть SAM еще не готова. Подождите секунду..." if self.settings.read_lang() == 'RU' else "SAM is loading. Please wait...",
            3000)
        self.image_setter.start()

    def get_jpg_path(self, image_name):
        """
        Для поддержки детектора. У него данный метод переписан и поддерживает конвертацию в tif
        """
        return image_name

    def on_image_set(self):
        """
        Завершение прогрева модели SAM. Если остались изображения в очереди - берем последнее, а очередь очищаем
        """

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
        """
        Загрузка модели SAM
        """
        self.sam = self.load_sam()
        self.image_setter = SAMImageSetter()
        self.image_setter.set_predictor(self.sam)
        self.image_setter.finished.connect(self.on_image_set)
        if self.tek_image_path:
            self.queue_image_to_sam(self.tek_image_path)

    def handle_detection_model(self):
        """
        Загрузка модели для обнаружения объектов
        """
        cnn_model = self.settings.read_cnn_model()

        if cnn_model == "YOLOv8_openvino":
            core = Core()

            cfg_path, weights_path = cls_settings.get_cfg_and_weights_by_cnn_name('YOLOv8_openvino')
            seg_ov_model = core.read_model(cfg_path)

            if self.settings.read_platform() == "cuda":
                device = "GPU"
                seg_ov_model.reshape({0: [1, 3, 1280, 1280]})
            else:
                device = 'CPU'

            self.yolo = core.compile_model(seg_ov_model, device)

        elif cnn_model == "YOLOv8":
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

        else:
            print("Unknown detection model")
            if self.settings.read_lang() == 'RU':
                self.statusBar().showMessage(
                    f"Неизвестное имя модели {cnn_model}",
                    3000)
            else:
                self.statusBar().showMessage(
                    f"Unknown detection model {cnn_model}", 3000)

    def handle_cuda_models(self):
        """
        Загрузка всех моделей нейросетей
        """
        # Start on Loading Animation
        self.view.start_circle_progress()

        self.handle_sam_model()

        self.handle_detection_model()

        self.gd_model = self.load_gd_model()

        self.view.stop_circle_progress()

    def on_settings_closed(self):
        """
        После закрытия окна настроек проверяем не изменились ли модели нейросетей и платформа вычислений
        """
        super(Annotator, self).on_settings_closed()

        sam_hq = self.settings.read_sam_hq()
        if sam_hq != self.last_sam_use_hq:
            self.handle_sam_model()

        platform = self.settings.read_platform()

        if platform != self.last_platform:
            self.try_to_set_platform(platform)

        cnn_model = self.settings.read_cnn_model()

        if cnn_model != self.last_cnn:
            self.last_cnn = cnn_model
            self.handle_detection_model()

    def ai_points_pressed(self):
        """
        Нажатие левой или правой кнопки мыши в режиме точек-prompts SAM
        """

        self.ann_type = "AiPoints"
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings.read_alpha()

        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def ai_mask_pressed(self):
        """
        Старт рисования прямоугольной области в режиме SAM
        """
        self.ann_type = "AiMask"
        self.set_labels_color()
        cls_txt = self.cls_combo.currentText()
        cls_num = self.cls_combo.currentIndex()

        label_color = self.project_data.get_label_color(cls_txt)

        alpha_tek = self.settings.read_alpha()
        self.view.start_drawing(self.ann_type, color=label_color, cls_num=cls_num, alpha=alpha_tek)

    def add_sam_polygon_to_scene(self, sam_mask, cls_num=None, is_save_to_temp_folder=True):
        """
        Добавление полигонов SAM на сцену
        """
        simplify_factor = float(self.settings.read_simplify_factor())
        sam_mask = hf.clean_mask(sam_mask, type='remove', min_size=80, connectivity=1)
        if is_save_to_temp_folder:
            mask_name = hf.create_unique_image_name(self.tek_image_name)
            mask_name = os.path.join(self.handle_temp_folder(), mask_name)
            hf.save_mask_as_image(sam_mask, mask_name)
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
                            f"Метку сделать не удалось. Площадь маски слишком мала {area:0.3f}",
                            3000)
                    else:
                        self.statusBar().showMessage(
                            f"Can't create label. Area of label is too small {area:0.3f}", 3000)
                    continue

            if not cls_num:
                cls_num = self.cls_combo.currentIndex()
            cls_name = self.cls_combo.itemText(cls_num)
            alpha_tek = self.settings.read_alpha()
            color = self.project_data.get_label_color(cls_name)

            label_text_params = self.settings.read_label_text_params()
            if label_text_params['hide']:
                text = None
            else:
                text = cls_name

            self.view.add_polygons_group_to_scene(cls_num, filtered_points_mass, color, alpha_tek, text=text)

            self.save_view_to_project()

    def ai_mask_end_drawing(self):
        """
        Завершение рисования прямоугольной области SAM
        """
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
        """
        Старт рисования метки
        """
        super(Annotator, self).start_drawing()
        self.view.clear_ai_points()  # очищение точек SAM

    def break_drawing(self):
        """
        Прерывание рисования метки
        """
        super(Annotator, self).break_drawing()
        if self.ann_type == "AiPoints":
            self.view.clear_ai_points()
            self.view.remove_active()

    def end_drawing(self):
        """
        Завершение рисования метки
        """
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

            self.view.end_drawing()  # clear points inside view

            self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

            self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def on_quit(self):
        """
        Выход из приложения
        """
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
        """
        Загрузка модели groundingDINO
        """
        config_file = os.path.join(os.getcwd(),
                                   config.PATH_TO_GROUNDING_DINO_CONFIG)
        grounded_checkpoint = os.path.join(os.getcwd(),
                                           config.PATH_TO_GROUNDING_DINO_CHECKPOINT)

        return gd_load_model(config_file, grounded_checkpoint, device=self.settings.read_platform())

    def load_sam(self):
        """
        Загрузка модели SAM
        """
        use_hq = self.settings.read_sam_hq()
        if use_hq:
            sam_model_path = os.path.join(os.getcwd(), config.PATH_TO_SAM_HQ_CHECKPOINT)
        else:
            sam_model_path = os.path.join(os.getcwd(), config.PATH_TO_SAM_CHECKPOINT)

        return sam_load_model(sam_model_path, device=self.settings.read_platform(), use_sam_hq=use_hq)

    def queue_image_to_sam(self, image_name):
        """
        Постановка в очередь изображения для загрузки в модель SAM
        """
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
        """
        Старт обнаружения объектов на текущем изображении
        """
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

        names = self.read_detection_model_names()

        self.CNN_worker = CNN_worker(model=self.yolo, conf_thres=conf_thres_set, iou_thres=iou_thres_set,
                                     img_name=img_name, img_path=img_path,
                                     scanning=self.scanning_mode, model_name=self.settings.read_cnn_model(),
                                     linear_dim=self.lrm, simplify_factor=simplify_factor, nc=len(names.keys()))

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

            cls_name = self.cls_combo.itemText(cls_num)

            label_text_params = self.settings.read_label_text_params()
            if label_text_params['hide']:
                text = None
            else:
                text = cls_name

            self.view.add_polygon_to_scene(cls_num, points, color=color, id=shape_id, text=text)

            shape = {'id': shape_id, 'cls_num': cls_num, 'points': points, 'conf': res['conf']}
            self.detected_shapes.append(shape)

        self.progress_toolbar.hide_progressbar()
        self.save_view_to_project()

        self.statusBar().showMessage(
            f"Найдено {len(self.CNN_worker.mask_results)} объектов" if self.settings.read_lang() == 'RU' else f"{len(self.CNN_worker.mask_results)} objects has been detected",
            3000)

    def grounding_sam_pressed(self):
        """
        Открытие диалогового окна GroundingDINO
        """

        theme = self.settings.read_theme()

        self.prompt_input_dialog = PromptInputDialog(self, theme=theme,
                                                     class_names=self.project_data.get_labels(),
                                                     on_ok_clicked=self.start_grounddino, prompts_variants=self.prompts)
        self.prompt_input_dialog.show()

    def run_gd(self, image_name, prompt):
        """
        Нажатие кнопки ОК в диалоговом окне GroundingDINO
        """
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
        """
        Запуск обнаружения объектов по текстовому prompt GroundingDINO
        """
        prompt = self.prompt_input_dialog.getPrompt()
        self.prompt_input_dialog.close()

        jpg_path = self.get_jpg_path(self.tek_image_path)

        if prompt:
            self.run_gd(jpg_path, prompt)

    def on_gd_worker_finished(self):
        """
        Завершение обнаружения объектов по текстовому prompt GroundingDINO
        """
        masks = self.gd_worker.getMasks()
        self.progress_toolbar.set_percent(50)

        for i, mask in enumerate(masks):
            self.add_sam_polygon_to_scene(mask, cls_num=self.prompt_cls_num)
            self.progress_toolbar.set_percent(50 + int(i + 1) * 100.0 / len(masks))

        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())
        self.progress_toolbar.hide_progressbar()

    def on_dataset_balance_clicked(self):
        """
        Вывод инфо о балансе датасета
        """
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
        Запуск классификации на всех изображениях датасета
        """

        self.started_cnn = self.settings.read_cnn_model()

        conf_thres_set = self.settings.read_conf_thres()
        iou_thres_set = self.settings.read_iou_thres()

        str_text = "Начинаю классифкацию СНС {0:s}".format(self.started_cnn)

        self.statusBar().showMessage(str_text, 3000)

        images_list = [os.path.join(self.dataset_dir, im_name) for im_name in self.dataset_images]

        names = self.read_detection_model_names()

        self.CNN_worker = CNN_worker(model=self.yolo, conf_thres=conf_thres_set, iou_thres=iou_thres_set,
                                     img_name=None, img_path=None,
                                     images_list=images_list, model_name=self.settings.read_cnn_model(),
                                     scanning=None, nc=len(names.keys()))

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

        self.project_data.add_shapes_to_images(self.CNN_worker.all_images_results)

        self.reload_image(is_tek_image_changed=False)

    def importFromYOLOBox(self):
        """
        По нажатии иконки Импорт модели из YOLO Box - открытие диалогового окна и закрытие проекта
        """

        self.close_project()

        self.import_dialog = ImportFromYOLODialog(self, on_ok_clicked=self.on_import_yolo_clicked, convert_to_mask=True)
        self.is_seg_import = False
        self.import_dialog.show()

    def on_import_yolo_clicked(self):
        """
        При нажатии импорта из проекта YOLO
        """
        yaml_data = self.import_dialog.getData()

        if self.sam and not self.is_seg_import:
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
