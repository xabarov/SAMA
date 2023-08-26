from PySide2 import QtCore

from utils.sam_predictor import predictor_set_image


class SAMImageSetter(QtCore.QThread):

    def __init__(self):
        super(SAMImageSetter, self).__init__()

    def set_image(self, image):
        self.image = image

    def set_predictor(self, predictor):
        self.predictor = predictor

    def run(self):
        predictor_set_image(self.predictor, self.image)


