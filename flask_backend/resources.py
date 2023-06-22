from flask_restful import Resource, reqparse
from ultralytics import YOLO
from werkzeug.datastructures import FileStorage
from detector import run_yolo8
import os
from mask_encoder import encode
from sam import load_sam, set_image, predict_by_points, mask_to_seg, predict_by_box
import json
import numpy as np

UPLOAD_FOLDER = 'uploads'

config_path = os.path.join(os.getcwd(), 'yolov8/aes_yolo_seg.yaml')
model_path = os.path.join(os.getcwd(), 'yolov8/weights/best.pt')

yolo = YOLO(model_path)
yolo.data = config_path

dev_set = 'cpu'
# if self.settings.read_platform() == "cuda":
#     dev_set = 0

yolo.to(dev_set)
yolo.overrides['data'] = config_path
sam = load_sam()


class Detect(Resource):
    def post(self):
        target = os.path.join(UPLOAD_FOLDER, 'images')
        if not os.path.isdir(target):
            os.mkdir(target)

        upload_parser = reqparse.RequestParser()
        upload_parser.add_argument('file', type=FileStorage, location='files')
        upload_parser.add_argument('conf', location='headers')
        upload_parser.add_argument('iou', location='headers')
        upload_parser.add_argument('lrm', location='headers')

        data = upload_parser.parse_args()
        file = data['file']
        conf = float(data['conf'])
        iou = float(data['iou'])
        lrm = float(data['lrm'])
        destination = "/".join([target, 'image.jpg'])
        file.save(destination)
        if lrm > 0.0:
            print('lrm gt 0')
            mask_results = run_yolo8(yolo, destination, conf_thres=conf, iou_thres=iou, lrm=lrm)
        else:
            mask_results = run_yolo8(yolo, destination, conf_thres=conf, iou_thres=iou, lrm=None)

        return encode(mask_results)


class SamPoints(Resource):
    def post(self):
        upload_parser = reqparse.RequestParser()
        upload_parser.add_argument('input_points', location='json', type=list)
        upload_parser.add_argument('input_labels', location='json', type=list)

        data = upload_parser.parse_args()

        points = np.asarray(data['input_points'])
        labels = np.asarray(data['input_labels'])

        masks = predict_by_points(sam, points, labels, multi=False)
        results = []
        print(len(masks))
        for mask in masks:
            res = mask_to_seg(mask)
            print(res)
            results.append(res)

        return results


class SamBox(Resource):
    def post(self):
        upload_parser = reqparse.RequestParser()
        upload_parser.add_argument('input_box', location='json', type=list)

        data = upload_parser.parse_args()

        box = np.asarray(data['input_box'])

        masks = predict_by_box(sam, box)
        res = mask_to_seg(masks)

        return res


class SamSetImage(Resource):
    def post(self):
        target = os.path.join(UPLOAD_FOLDER, 'images')
        if not os.path.isdir(target):
            os.mkdir(target)

        upload_parser = reqparse.RequestParser()
        upload_parser.add_argument('file', type=FileStorage, location='files')
        data = upload_parser.parse_args()

        file = data['file']
        destination = "/".join([target, 'image_sam.jpg'])
        file.save(destination)

        set_image(sam, destination)

        return {'msg': 'Image is set to SAM'}


class SyncNames(Resource):
    def get(self):
        names = yolo.names  # dict like {0:name1, 1:name2...}
        return names
