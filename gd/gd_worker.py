from PySide2 import QtCore
from .gd_sam import predict


class GroundingSAMWorker(QtCore.QThread):

    def __init__(self, config_file=None, grounded_checkpoint=None,
                 sam_predictor=None,
                 tek_image_path=None,
                 prompt=None):
        super(GroundingSAMWorker, self).__init__()
        self.config_file = config_file
        self.grounded_checkpoint = grounded_checkpoint
        self.sam_predictor = sam_predictor
        self.tek_image_path = tek_image_path
        self.prompt = prompt
        self.masks = []

    def set_config(self, config):
        self.config_file = config

    def set_grounded_checkpoint(self, grounded_checkpoint):
        self.grounded_checkpoint = grounded_checkpoint

    def set_sam_predictor(self, sam_predictor):
        self.sam_predictor = sam_predictor

    def set_prompt(self, prompt):
        self.prompt = prompt

    def set_tek_image_path(self, tek_image_path):
        self.tek_image_path = tek_image_path

    def run(self):
        self.masks = predict(self.tek_image_path, self.prompt, config_file=self.config_file,
                             grounded_checkpoint=self.grounded_checkpoint,
                             sam_predictor=self.sam_predictor
                             )

    def getMasks(self):
        return self.masks
