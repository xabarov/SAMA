from PySide2 import QtCore
from .gd_sam2 import predict


class GroundingSAMWorker(QtCore.QThread):

    def __init__(self, config_file=None, grounded_checkpoint=None,
                 sam_predictor=None, grounding_dino_model=None,
                 tek_image_path=None, box_threshold=0.4, text_threshold=0.55,
                 prompt=None, device='cuda', output_dir=None):
        super(GroundingSAMWorker, self).__init__()
        self.config_file = config_file
        self.grounded_checkpoint = grounded_checkpoint
        self.grounding_dino_model = grounding_dino_model
        self.sam_predictor = sam_predictor
        self.tek_image_path = tek_image_path
        self.prompt = prompt
        self.masks = []
        self.device = device
        self.output_dir = output_dir
        self.box_threshold = box_threshold
        self.text_threshold = text_threshold

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
        self.masks = predict(self.sam_predictor, self.tek_image_path, self.prompt, self.output_dir,
                             grounded_checkpoint=self.grounded_checkpoint,
                             box_threshold=self.box_threshold, text_threshold=self.text_threshold, device=self.device,
                             config_file=self.config_file)

    def getMasks(self):
        return self.masks
