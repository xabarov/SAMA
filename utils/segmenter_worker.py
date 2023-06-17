from PySide2 import QtCore

from utils.run_mm_seg import segment


class SegmenterWorker(QtCore.QThread):

    def __init__(self, image_path=None, config_path=None, checkpoint_path=None, palette=None, classes=None,
                 device='cuda'):
        super(SegmenterWorker, self).__init__()
        self.image_path = image_path
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        self.palette = palette
        self.classes = classes
        self.device = device

        self.results = None

        # Example
        # print(result['visualization'].shape)
        # (512, 683, 3)

        # 'predictions' includes segmentation mask with label indices
        # print(result['predictions'].shape)
        # (512, 683)

    def set_image_path(self, image_path):
        self.image_path = image_path

    def set_checkpoint_path(self, checkpoint_path):
        self.checkpoint_path = checkpoint_path

    def set_config_path(self, config_path):
        self.config_path = config_path

    def set_palette(self, palette):
        self.palette = palette

    def set_classes(self, classes):
        self.classes = classes

    def run(self):
        if self.image_path:
            self.results = segment(image_path=self.image_path, config_path=self.config_path,
                                   checkpoint_path=self.checkpoint_path, palette=self.palette, classes=self.classes,
                                   device=self.device)
