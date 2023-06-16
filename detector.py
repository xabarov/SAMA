from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QAction, QMessageBox, QMenu, QToolBar, QToolButton
from PyQt5.QtGui import QIcon, QCursor

from ui.base_window import MainWindow
from utils import config

from ultralytics import YOLO

from ui.input_dialog import PromptInputDialog
from ui.show_image_widget import ShowImgWindow
from ui.settings_window import SettingsWindow
from ui.progress import ProgressWindow
from ui.edit_with_button import EditWithButton

from utils.predictor import SAMImageSetter
from utils.cnn_worker import CNN_worker
from utils.sam_predictor import load_model as sam_load_model
from utils import cls_settings
from utils.edges_from_mask import yolo8masks2points
from utils.sam_predictor import mask_to_seg, predict_by_points, predict_by_box
from gd.gd_worker import GroundingSAMWorker

from gd.gd_sam import load_model as gd_load_model
from shapely import Polygon

import matplotlib.pyplot as plt
import utils.help_functions as hf
import cv2
import os


class Detector(MainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)

        self.setWindowTitle("AI Detector")

        # CUDA platform
        self.last_platform = self.settings.read_platform()
        self.message_cuda_available()

        # Detector
        self.started_cnn = None
        self.scanning_mode = None

        # GroundingDINO
        self.gd_worker = None
        self.prompts = []

        # SAM
        self.image_set = False
        self.image_setter = None
        self.queue_to_image_setter = []

        self.view.mask_end_drawing.on_mask_end_drawing.connect(self.ai_mask_end_drawing)

        self.lrm = None  # name of file with geo coords
        self.detected_shapes = []

        self.handle_cuda_models()



    def createActions(self):
        super(Detector, self).createActions()

        self.balanceAct = QAction("Информация о датасете" if self.settings.read_lang() == 'RU' else "Dataset info",
                                  self,
                                  enabled=False, triggered=self.on_dataset_balance_clicked)

        # Object detector
        self.detectAct = QAction(
            "Обнаружить объекты за один проход" if self.settings.read_lang() == 'RU' else "Detect objects", self,
            shortcut="Ctrl+Y", enabled=False,
            triggered=self.detect)
        self.detectScanAct = QAction(
            "Обнаружить объекты сканированием" if self.settings.read_lang() == 'RU' else "Detect objects with scanning",
            self, enabled=False,
            triggered=self.detect_scan)
        self.syncLabelsAct = QAction(
            "Синхронизировать имена меток" if self.settings.read_lang() == 'RU' else "Fill label names from AI model",
            self, enabled=False,
            triggered=self.sync_labels)

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

        self.exportToESRIAct = QAction(
            "Экспорт в ESRI shapefile" if self.settings.read_lang() == 'RU' else "Export to ESRI shapefile", self,
            enabled=False,
            triggered=self.export_to_esri)

    def createMenus(self):
        super(Detector, self).createMenus()

        self.aiAnnotatorMethodMenu = QMenu("С помощью ИИ" if self.settings.read_lang() == 'RU' else "AI", self)

        self.aiAnnotatorMethodMenu.setEnabled(False)

        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorPointsAct)
        self.aiAnnotatorMethodMenu.addAction(self.aiAnnotatorMaskAct)
        self.aiAnnotatorMethodMenu.addAction(self.GroundingDINOSamAct)

        self.AnnotatorMethodMenu.addMenu(self.aiAnnotatorMethodMenu)

        self.classifierMenu = QMenu("Классификатор" if self.settings.read_lang() == 'RU' else "Classifier", self)
        self.classifierMenu.addAction(self.detectAct)
        self.classifierMenu.addAction(self.detectScanAct)
        self.classifierMenu.addAction(self.syncLabelsAct)

        self.annotatorMenu.addAction(self.balanceAct)
        self.annotatorExportMenu.addAction(self.exportToESRIAct)

        self.menuBar().clear()
        self.menuBar().addMenu(self.fileMenu)
        self.menuBar().addMenu(self.viewMenu)
        self.menuBar().addMenu(self.classifierMenu)
        self.menuBar().addMenu(self.annotatorMenu)
        self.menuBar().addMenu(self.settingsMenu)
        self.menuBar().addMenu(self.helpMenu)

    def createToolbar(self):

        self.create_right_toolbar()
        self.create_top_toolbar()

        toolBar = QToolBar("Панель инструментов" if self.settings.read_lang() == 'RU' else "ToolBar", self)
        toolBar.addAction(self.createNewProjectAct)
        toolBar.addSeparator()
        toolBar.addAction(self.zoomInAct)
        toolBar.addAction(self.zoomOutAct)
        toolBar.addAction(self.fitToWindowAct)
        toolBar.addSeparator()

        self.annotatorToolButton = QToolButton(self)
        self.annotatorToolButton.setDefaultAction(self.polygonAct)
        self.annotatorToolButton.setPopupMode(QToolButton.MenuButtonPopup)
        self.annotatorToolButton.triggered.connect(self.ann_triggered)

        self.annotatorToolButton.setMenu(self.AnnotatorMethodMenu)

        toolBar.addWidget(self.annotatorToolButton)

        toolBar.addSeparator()
        toolBar.addAction(self.detectAct)
        toolBar.addAction(self.detectScanAct)

        toolBar.addSeparator()
        toolBar.addAction(self.settingsAct)
        toolBar.addSeparator()

        self.toolBarLeft = toolBar
        self.addToolBar(QtCore.Qt.LeftToolBarArea, self.toolBarLeft)

    def set_icons(self):
        super(Detector, self).set_icons()
        # AI
        self.aiAnnotatorMethodMenu.setIcon(QIcon(self.icon_folder + "/ai.png"))
        self.aiAnnotatorPointsAct.setIcon(QIcon(self.icon_folder + "/mouse.png"))
        self.aiAnnotatorMaskAct.setIcon(QIcon(self.icon_folder + "/ai_select.png"))
        self.GroundingDINOSamAct.setIcon(QIcon(self.icon_folder + "/dino.png"))

        self.balanceAct.setIcon(QIcon(self.icon_folder + "/bar-chart.png"))

        # classifier
        self.detectAct.setIcon(QIcon(self.icon_folder + "/detect_all.png"))
        self.detectScanAct.setIcon(QIcon(self.icon_folder + "/slide.png"))

    def open_image(self, image_name):

        super(Detector, self).open_image(image_name)

        self.aiAnnotatorPointsAct.setEnabled(True)
        self.aiAnnotatorMaskAct.setEnabled(True)
        self.aiAnnotatorMethodMenu.setEnabled(True)
        self.GroundingDINOSamAct.setEnabled(True)
        self.exportToESRIAct.setEnabled(True)
        self.image_set = False
        self.queue_image_to_sam(image_name)
        lrm = hf.try_read_lrm(image_name)
        if lrm:
            self.lrm = lrm

    def reload_image(self):
        super(Detector, self).reload_image()
        self.view.clear_ai_points()

    def on_image_set(self):

        if len(self.queue_to_image_setter) != 0:
            image_name = self.queue_to_image_setter[-1]
            image = cv2.imread(image_name)
            self.cv2_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            self.image_set = False
            self.image_setter.set_image(self.cv2_image)
            self.queue_to_image_setter = []
            self.statusBar().showMessage(
                "Нейросеть SAM еще не готова. Подождите секунду..." if self.settings.read_lang() == 'RU' else "SAM is loading. Please wait...",
                3000)
            self.image_setter.start()

        else:
            self.statusBar().showMessage(
                "Нейросеть SAM готова к сегментации" if self.settings.read_lang() == 'RU' else "SAM ready to work",
                3000)
            self.image_set = True

    def export_to_esri(self):

        self.esri_path_window = EditWithButton(None, in_separate_window=True,
                                               theme=self.settings.read_theme(),
                                               on_button_clicked_callback=self.on_input_esri_path,
                                               is_dir=False, dialog_text='ESRI shapefile name',
                                               title=f"Choose ESRI shapefile name", file_type='shp',
                                               placeholder='ESRI shapefile name', is_existing_file_only=False)
        self.esri_path_window.show()

    def on_input_esri_path(self):

        esri_filename = self.esri_path_window.getEditText()

        self.esri_path_window.hide()

        if hf.get_extension(esri_filename) != 'shp':

            if self.settings.read_lang() == 'RU':
                message = f"Имя ESRI shapefile файла должно иметь расширение .shp"
            else:
                message = f"ESRI shapefile has to have '.shp' extension."

            self.statusBar().showMessage(
                message, 3000)

            return

        if self.lrm:
            esri_shapes = []
            view_shapes = self.view.get_all_shapes()
            for shape in view_shapes:
                shape_id = shape['id']
                is_found = False
                for det_shape in self.detected_shapes:
                    if det_shape['id'] == shape_id:
                        esri_shapes.append(det_shape)
                        is_found = True
                        break
                if not is_found:
                    esri_shapes.append(shape)

            hf.convert_shapes_to_esri(esri_shapes, self.tek_image_path, out_shapefile=esri_filename)

            if self.settings.read_lang() == 'RU':
                message = f"ESRI shapefile файл создан. Добавлено {len(esri_shapes)} объектов"
            else:
                message = f"ESRI shapefile has been created with {len(esri_shapes)} objects."

            self.statusBar().showMessage(
                message, 3000)

        else:
            if self.settings.read_lang() == 'RU':
                message = f"ESRI shapefile файл не создан. Не найден файл геоданных"
            else:
                message = f"Can't create ESRI shapefile. No geo data"

            self.statusBar().showMessage(
                message, 3000)

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
        QMessageBox.about(self, "AI Detector",
                          "<p><b>AI Detector</b></p>"
                          "<p>Программа для обнаружения объектов</p>" if
                          self.settings.read_lang() == 'RU' else "<p>Object detection and instance segmentation program</p>")

    def on_checking_project_success(self, dataset_dir):
        super(Detector, self).on_checking_project_success(dataset_dir)
        self.balanceAct.setEnabled(True)
        self.detectAct.setEnabled(True)
        self.detectScanAct.setEnabled(True)
        self.syncLabelsAct.setEnabled(True)

    def handle_cuda_models(self):

        # Start on Loading Animation
        self.start_gif(is_prog_load=True)

        self.sam = self.load_sam()
        self.image_setter = SAMImageSetter()
        self.image_setter.set_predictor(self.sam)
        self.image_setter.finished.connect(self.on_image_set)
        if self.tek_image_path:
            self.queue_image_to_sam(self.tek_image_path)

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

        self.splash.finish(self)

    def on_settings_closed(self):
        super(Detector, self).on_settings_closed()
        platform = self.settings.read_platform()

        if platform != self.last_platform:
            self.sam = self.load_sam()
            self.last_platform = platform
            self.message_cuda_available()

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

    def add_sam_polygon_to_scene(self, sam_mask, id):
        points_mass = mask_to_seg(sam_mask)

        if points_mass:
            shapely_pol = Polygon(points_mass)
            area = shapely_pol.area
            if area > config.POLYGON_AREA_THRESHOLD:

                cls_num = self.cls_combo.currentIndex()
                cls_name = self.cls_combo.itemText(cls_num)
                alpha_tek = self.settings.read_alpha()
                color = self.project_data.get_label_color(cls_name)

                self.view.add_polygon_to_scene(cls_num, points_mass, color, alpha_tek, id=id)
                self.write_scene_to_project_data()
                self.fill_labels_on_tek_image_list_widget()
            else:
                if self.settings.read_lang() == 'RU':
                    self.statusBar().showMessage(
                        f"Метку сделать не удалось. Площадь маски слишком мала {area:0.3f}. Попробуйте еще раз", 3000)
                else:
                    self.statusBar().showMessage(
                        f"Can't create label. Area of label is too small {area:0.3f}. Try again", 3000)

                self.view.remove_label_id(id)
                self.write_scene_to_project_data()
                self.fill_labels_on_tek_image_list_widget()
        else:
            self.view.remove_label_id(id)
            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()

        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

    def ai_mask_end_drawing(self):

        self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))
        input_box = self.view.get_sam_mask_input()

        self.view.remove_active()

        if len(input_box):
            if self.image_set and not self.image_setter.isRunning():
                mask = predict_by_box(self.sam, input_box)
                self.add_sam_polygon_to_scene(mask, self.view.get_unique_label_id())

        self.view.end_drawing()
        self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

    def start_drawing(self):
        super(Detector, self).start_drawing()
        self.view.clear_ai_points()

    def break_drawing(self):
        super(Detector, self).break_drawing()
        if self.ann_type == "AiPoints":
            self.view.clear_ai_points()
            self.view.remove_active()

    def end_drawing(self):
        super(Detector, self).end_drawing()

        if self.ann_type == "AiPoints":

            self.view.setCursor(QCursor(QtCore.Qt.BusyCursor))

            input_point, input_label = self.view.get_sam_input_points_and_labels()

            if len(input_label):
                if self.image_set and not self.image_setter.isRunning():
                    masks = predict_by_points(self.sam, input_point, input_label, multi=False)
                    for mask in masks:
                        self.add_sam_polygon_to_scene(mask, id=self.view.get_unique_label_id())

            else:
                self.view.remove_active()

            self.view.clear_ai_points()
            self.view.end_drawing()

            self.view.setCursor(QCursor(QtCore.Qt.ArrowCursor))

            self.write_scene_to_project_data()
            self.fill_labels_on_tek_image_list_widget()

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

    def message_cuda_available(self):
        """
        Инициализация настроек приложения
        """
        lang = self.settings.read_lang()
        platform = self.settings.read_platform()

        if platform == 'cuda':
            print("CUDA is available")
            if lang == 'RU':
                self.statusBar().showMessage(
                    "Найдено устройство NVIDIA CUDA. Нейросеть будет использовать ее для ускорения", 3000)
            else:
                self.statusBar().showMessage(
                    "NVIDIA CUDA is found. SAM will use it for acceleration", 3000)

        else:
            if lang == 'RU':
                self.statusBar().showMessage(
                    "Не найдено устройство NVIDIA CUDA. Нейросеть будет использовать ресурсы процессора", 3000)
            else:
                self.statusBar().showMessage(
                    "Cant't find NVIDIA CUDA. SAM will use CPU", 3000)

    def load_gd_model(self):
        config_file = os.path.join(os.getcwd(),
                                   config.PATH_TO_GROUNDING_DINO_CONFIG)
        grounded_checkpoint = os.path.join(os.getcwd(),
                                           config.PATH_TO_GROUNDING_DINO_CHECKPOINT)

        return gd_load_model(config_file, grounded_checkpoint, device=self.settings.read_platform())

    def load_sam(self):
        sam_model_path = os.path.join(os.getcwd(), config.PATH_TO_SAM_CHECKPOINT)
        return sam_load_model(sam_model_path, model_type="vit_h", device=self.settings.read_platform())

    def queue_image_to_sam(self, image_name):

        if self.image_setter and not self.image_setter.isRunning():
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

        img_path = self.dataset_dir
        img_name = os.path.basename(self.tek_image_name)

        self.run_detection(img_name=img_name, img_path=img_path)

    def detect_scan(self):
        self.scanning_mode = True

        self.detect()

    def run_detection(self, img_name, img_path):
        """
        Запуск классификации
        img_name - имя изображения
        img_path - путь к изображению
        """

        self.started_cnn = self.settings.read_cnn_model()

        conf_thres_set = self.settings.read_conf_thres()
        iou_thres_set = self.settings.read_iou_thres()

        if self.scanning_mode:
            str_text = "Начинаю классифкацию СНС {0:s} сканирующим окном".format(self.started_cnn)
        else:
            str_text = "Начинаю классифкацию СНС {0:s}".format(self.started_cnn)
        print(str_text)

        self.CNN_worker = CNN_worker(model=self.yolo, conf_thres=conf_thres_set, iou_thres=iou_thres_set,
                                     img_name=img_name, img_path=img_path,
                                     scanning=self.scanning_mode,
                                     linear_dim=self.lrm)

        self.CNN_worker.started.connect(self.on_cnn_started)

        self.progress_bar = ProgressWindow(self,
                                           'Обнаружение объектов...' if self.settings.read_lang() == 'RU' else 'Object detection...')

        self.CNN_worker.psnt_connection.percent.connect(self.progress_bar_changed)
        self.CNN_worker.finished.connect(self.on_cnn_finished)

        if not self.CNN_worker.isRunning():
            self.CNN_worker.start()

    def on_cnn_started(self):
        """
        При начале классификации
        """
        self.progress_bar.show()
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

        self.write_scene_to_project_data()
        self.fill_labels_on_tek_image_list_widget()
        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())

        self.progress_bar.hide()

        self.statusBar().showMessage(
            f"Найдено {len(self.CNN_worker.mask_results)} объектов" if self.settings.read_lang() == 'RU' else f"{len(self.CNN_worker.mask_results)} objects has been detected",
            3000)

    def grounding_sam_pressed(self):

        self.prompt_input_dialog = PromptInputDialog(self,
                                                     class_names=self.project_data.get_labels(),
                                                     on_ok_clicked=self.start_grounddino, prompts_variants=self.prompts)
        self.prompt_input_dialog.show()

    def start_grounddino(self):
        prompt = self.prompt_input_dialog.getPrompt()

        if prompt:

            self.prompt_cls_name = self.prompt_input_dialog.getClsName()
            self.prompt_cls_num = self.prompt_input_dialog.getClsNumber()

            if prompt not in self.prompts:
                self.prompts.append(prompt)

            config_file = os.path.join(os.getcwd(),
                                       config.PATH_TO_GROUNDING_DINO_CONFIG)
            grounded_checkpoint = os.path.join(os.getcwd(),
                                               config.PATH_TO_GROUNDING_DINO_CHECKPOINT)

            self.gd_worker = GroundingSAMWorker(config_file=config_file, grounded_checkpoint=grounded_checkpoint,
                                                sam_predictor=self.sam, tek_image_path=self.tek_image_path,
                                                grounding_dino_model=self.gd_model,
                                                prompt=prompt)

            self.prompt_input_dialog.set_progress(10)

            self.gd_worker.finished.connect(self.on_gd_worker_finished)

            if not self.gd_worker.isRunning():
                self.statusBar().showMessage(
                    f"Начинаю поиск {self.prompt_cls_name} на изображении..." if self.settings.read_lang() == 'RU' else f"Start searching {self.prompt_cls_name} on image...",
                    3000)
                self.gd_worker.start()

    def on_gd_worker_finished(self):
        masks = self.gd_worker.getMasks()

        shape = self.cv2_image.shape
        for i, mask in enumerate(masks):
            self.prompt_input_dialog.set_progress(10 + int(90.0 * i / len(masks)))
            points = yolo8masks2points(mask * 255, simplify_factor=3, width=shape[1], height=shape[0])
            if points:
                self.view.add_polygon_to_scene(self.prompt_cls_num, points,
                                               color=self.project_data.get_label_color(self.prompt_cls_name))

        self.write_scene_to_project_data()
        self.fill_labels_on_tek_image_list_widget()
        self.labels_count_conn.on_labels_count_change.emit(self.labels_on_tek_image.count())
        self.prompt_input_dialog.close()

    def on_dataset_balance_clicked(self):
        balance_data = self.project_data.calc_dataset_balance()

        label_names = self.project_data.get_data()['labels']
        labels = list(balance_data.keys())
        labels = [label_names[int(i)] for i in labels]
        values = list(balance_data.values())

        fig, ax = plt.subplots(figsize=(10, 8))

        ax.bar(labels, values,
               # color=config.THEMES_COLORS[self.theme_str],
               width=0.8)

        ax.set_xlabel("Label names")
        ax.set_ylabel("No. of labels")
        ax.tick_params(axis='x', rotation=70)
        plt.title('Баланс меток')

        plt.savefig('temp.jpg')
        fileName = 'temp.jpg'
        ShowImgWindow(self, title='Баланс меток', img_file=fileName, icon_folder=self.icon_folder)


if __name__ == '__main__':
    import sys
    from qt_material import apply_stylesheet

    app = QtWidgets.QApplication(sys.argv)
    extra = {'density_scale': hf.density_slider_to_value(config.DENSITY_SCALE),
             # 'font_size': '14px',
             # 'primaryTextColor': '#ffffff',
             }

    apply_stylesheet(app, theme='dark_blue.xml', extra=extra)

    w = Detector()
    w.show()
    sys.exit(app.exec_())
