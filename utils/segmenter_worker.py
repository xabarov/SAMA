from PySide2 import QtCore

from utils.run_mm_seg import segment
from utils.edges_from_mask import segmentation_to_points
from ui.signals_and_slots import LoadPercentConnection

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
        self.psnt_connection = LoadPercentConnection()


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
        self.psnt_connection.percent.emit(0)
        self.results = segment(image_path=self.image_path, config_path=self.config_path,
                               checkpoint_path=self.checkpoint_path, palette=self.palette, classes=self.classes,
                               device=self.device)
        self.psnt_connection.percent.emit(50)



if __name__ == '__main__':

    def on_start():
        print('Started')


    def on_finished():
        print('Finished')
        print(worker.results)


    config = "../mm_segmentation/configs/psp_aes.py"
    checkpoint = "../mm_segmentation/checkpoints/iter_52000_83_59.pth"
    im = 'D:\python\datasets\\aes_2200_copy\\argentina_atucha_1.jpg'
    palette = [[0, 0, 0], [255, 0, 0], [0, 255, 0]]
    classes = [
        'background', 'water', 'vapor'
    ]
    worker = SegmenterWorker(image_path=im, config_path=config, checkpoint_path=checkpoint, palette=palette,
                             classes=classes, device='cuda')

    worker.started.connect(on_start)
    worker.finished.connect(on_finished)

    if not worker.isRunning():
        worker.start()
