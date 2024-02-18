from PyQt5.QtWidgets import QLabel, QGroupBox, QFormLayout, QCheckBox

from utils import ml_config
from ui.settings_window_base import SettingsWindowBase
from ui.custom_widgets.styled_widgets import StyledComboBox, StyledDoubleSpinBox, StyledSpinBox
import numpy as np
from PyQt5 import QtWidgets
import sys

from qt_material import apply_stylesheet


class SettingsWindow(SettingsWindowBase):
    def __init__(self, parent, test_mode=False):
        super().__init__(parent, test_mode=test_mode)
        self.create_cnn_layout()
        self.create_sam_layout()
        self.create_segmentation_layout()

        self.tabs.addTab(self.classifierGroupBox, "Обнаружение" if self.lang == 'RU' else 'Detection')
        self.tabs.addTab(self.samGroupBox, "SAM" if self.lang == 'RU' else 'SAM')
        self.tabs.addTab(self.segmentationGroupBox, "Сегментация" if self.lang == 'RU' else 'Segmentation')

    def create_sam_layout(self):
        theme = self.settings.read_theme()

        self.where_sam_calc_combo = StyledComboBox(self, theme=theme)
        self.where_sam_vars = np.array(["cpu", "cuda", 'Auto'])
        self.where_sam_calc_combo.addItems(self.where_sam_vars)
        where_label = QLabel("Платформа:" if self.lang == 'RU' else 'Platform:')

        self.where_sam_calc_combo.setCurrentIndex(0)
        idx = np.where(self.where_sam_vars == self.settings.read_sam_platform())[0][0]
        self.where_sam_calc_combo.setCurrentIndex(idx)

        # Настройки обнаружения
        self.samGroupBox = QGroupBox()

        sam_layout = QFormLayout()

        self.sam_combo = StyledComboBox(self, theme=theme)
        sam_list = list(ml_config.SAM_DICT.keys())
        self.sams = np.array(sam_list)
        self.sam_combo.addItems(self.sams)

        sam_label = QLabel("Модель:" if self.lang == 'RU' else "Model:")

        sam_layout.addRow(sam_label, self.sam_combo)
        sam_layout.addRow(where_label, self.where_sam_calc_combo)

        self.sam_combo.setCurrentIndex(0)
        idx = np.where(self.sams == self.settings.read_sam_model())[0][0]
        self.sam_combo.setCurrentIndex(idx)

        self.clear_sam_spin = StyledSpinBox(self, theme=theme)
        clear_sam_size = self.settings.read_clear_sam_size()
        self.clear_sam_spin.setValue(int(clear_sam_size))

        self.clear_sam_spin.setMinimum(1)
        self.clear_sam_spin.setMaximum(500)
        sam_layout.addRow(QLabel(
            "Размер удаляемых мелких областей, px:" if self.lang == 'RU' else "Remove small objects size, px"),
                                 self.clear_sam_spin)

        self.samGroupBox.setLayout(sam_layout)


    def create_segmentation_layout(self):
        theme = self.settings.read_theme()

        self.where_segmentation_calc_combo = StyledComboBox(self, theme=theme)
        self.where_segmentation_vars = np.array(["cpu", "cuda", 'Auto'])
        self.where_segmentation_calc_combo.addItems(self.where_segmentation_vars)
        where_label = QLabel("Платформа:" if self.lang == 'RU' else 'Platform:')

        self.where_segmentation_calc_combo.setCurrentIndex(0)
        idx = np.where(self.where_segmentation_vars == self.settings.read_segmentation_platform())[0][0]
        self.where_segmentation_calc_combo.setCurrentIndex(idx)

        # Настройки обнаружения
        self.segmentationGroupBox = QGroupBox()

        segmentation_layout = QFormLayout()

        self.segmentation_combo = StyledComboBox(self, theme=theme)
        segmentation_list = list(ml_config.SEG_DICT.keys())
        self.segmentations = np.array(segmentation_list)
        self.segmentation_combo.addItems(self.segmentations)

        segmentation_label = QLabel("Модель:" if self.lang == 'RU' else "Model:")

        segmentation_layout.addRow(segmentation_label, self.segmentation_combo)
        segmentation_layout.addRow(where_label, self.where_segmentation_calc_combo)

        self.segmentation_combo.setCurrentIndex(0)
        idx = np.where(self.segmentations == self.settings.read_seg_model())[0][0]
        self.segmentation_combo.setCurrentIndex(idx)

        self.segmentationGroupBox.setLayout(segmentation_layout)

    def create_cnn_layout(self):
        theme = self.settings.read_theme()

        self.where_detector_calc_combo = StyledComboBox(self, theme=theme)
        self.where_detector_vars = np.array(["cpu", "cuda", 'Auto'])
        self.where_detector_calc_combo.addItems(self.where_detector_vars)
        where_label = QLabel("Платформа:" if self.lang == 'RU' else 'Platform:')

        self.where_detector_calc_combo.setCurrentIndex(0)
        idx = np.where(self.where_detector_vars == self.settings.read_detector_platform())[0][0]
        self.where_detector_calc_combo.setCurrentIndex(idx)

        # Настройки обнаружения
        self.classifierGroupBox = QGroupBox()

        classifier_layout = QFormLayout()

        self.cnn_combo = StyledComboBox(self, theme=theme)
        cnn_list = list(ml_config.CNN_DICT.keys())
        self.cnns = np.array(cnn_list)
        self.cnn_combo.addItems(self.cnns)

        cnn_label = QLabel("Модель:" if self.lang == 'RU' else "Model:")

        classifier_layout.addRow(cnn_label, self.cnn_combo)
        classifier_layout.addRow(where_label, self.where_detector_calc_combo)

        self.cnn_combo.setCurrentIndex(0)
        idx = np.where(self.cnns == self.settings.read_detector_model())[0][0]
        self.cnn_combo.setCurrentIndex(idx)

        self.conf_thres_spin = StyledDoubleSpinBox(self, theme=theme)
        self.conf_thres_spin.setDecimals(3)
        conf_thres = self.settings.read_conf_thres()
        self.conf_thres_spin.setValue(float(conf_thres))

        self.conf_thres_spin.setMinimum(0.01)
        self.conf_thres_spin.setMaximum(1.00)
        self.conf_thres_spin.setSingleStep(0.01)
        classifier_layout.addRow(QLabel("Доверительный порог:" if self.lang == 'RU' else "Conf threshold"),
                                 self.conf_thres_spin)

        self.IOU_spin = StyledDoubleSpinBox(self, theme=theme)
        self.IOU_spin.setDecimals(3)
        iou_thres = self.settings.read_iou_thres()
        self.IOU_spin.setValue(float(iou_thres))

        self.IOU_spin.setMinimum(0.01)
        self.IOU_spin.setMaximum(1.00)
        self.IOU_spin.setSingleStep(0.01)
        classifier_layout.addRow(QLabel("IOU порог:" if self.lang == 'RU' else "IoU threshold"), self.IOU_spin)

        self.simplify_spin = StyledDoubleSpinBox(self, theme=theme)
        self.simplify_spin.setDecimals(3)
        simplify_factor = self.settings.read_simplify_factor()
        self.simplify_spin.setValue(float(simplify_factor))

        self.simplify_spin.setMinimum(0.01)
        self.simplify_spin.setMaximum(10.00)
        self.simplify_spin.setSingleStep(0.01)
        classifier_layout.addRow(QLabel("Коэффициент упрощения полигонов:" if self.lang == 'RU' else "Simplify factor"),
                                 self.simplify_spin)

        self.classifierGroupBox.setLayout(classifier_layout)

    def on_ok_clicked(self):
        super(SettingsWindow, self).on_ok_clicked()

        self.settings.write_detector_platform(self.where_detector_vars[self.where_detector_calc_combo.currentIndex()])
        self.settings.write_sam_platform(self.where_sam_vars[self.where_sam_calc_combo.currentIndex()])
        self.settings.write_segmentation_platform(self.where_segmentation_vars[self.where_segmentation_calc_combo.currentIndex()])

        self.settings.write_detector_model(self.cnns[self.cnn_combo.currentIndex()])
        self.settings.write_sam_model(self.sams[self.sam_combo.currentIndex()])
        self.settings.write_seg_model(self.segmentations[self.segmentation_combo.currentIndex()])

        self.settings.write_iou_thres(self.IOU_spin.value())
        self.settings.write_conf_thres(self.conf_thres_spin.value())
        self.settings.write_simplify_factor(self.simplify_spin.value())

        self.settings.write_clear_sam_size(self.clear_sam_spin.value())




if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)

    apply_stylesheet(app, theme='dark_blue.xml', invert_secondary=False)

    w = SettingsWindow(None)
    w.show()
    sys.exit(app.exec_())
