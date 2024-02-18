import os
from pathlib import Path

PALETTE = [
    (252, 66, 123, 120),
    (192, 57, 43, 120),
    (58, 53, 117, 120),
    (58, 53, 117, 120),
    (39, 174, 96, 120),
    (244, 164, 96, 120),
    (25, 25, 112, 120),
    (0, 191, 255, 120),
    (0, 206, 209, 120),
    (0, 191, 255, 120),
    (255, 20, 147, 120),
    (44, 160, 44, 255),
    (255, 127, 14, 255), (214, 39, 40, 255),
    (148, 103, 189, 255), (140, 86, 75, 255), (227, 119, 194, 255), (188, 189, 34, 255),
    (23, 190, 207, 255), (31, 119, 180, 255),  # tab10 without gray
]

CLASSES_RU = ["РО кв", "РО", "МЗ", "Турбина", "РУ", "БНС", "Градирня пасс", "Град.вент.кр", "Град.вент.пр",
              "Градирня акт", "БСС", "ДГС"]

CLASSES_ENG = ["ro_pf", "ro_sf", "mz_v", "mz_ot", "ru_ot", "bns_ot", "gr_b", "gr_vent_kr", "gr_vent_pr",
               "gr_b_act", "discharge", "diesel"]

CLASS_CONFIDENCE_MULTIPLIER = [0.7, 1.0, 0.7, 0.7, 0.9, 0.6, 1.0, 0.7, 0.7,
                               1.0, 0.2, 0.3]

PATH_TO_DOVERIT_STAT = "nuclear_power/doverit_stat_aes2.csv"

CNN_LIST = [
    # 'YOLOv3',
    # 'YOLOv5l6',
    # 'YOLOv5x6',
    'YOLOv8',
    # 'YOLOR',
    # Detectron2 models:
    # 'Mask-R-CNN-R50',
    # 'Mask-R-CNN-R101',
    # 'Mask-R-CNN-X101',
    # 'Cascade R-CNN',
    # 'Retina-R50',
    # 'Retina-R101',
    # 'R-CNN-Faster-R50',
    # 'R-CNN-Faster-R101',
    # 'R-CNN-Faster-X101',
    # 'GN',
    # 'Deformable-Conv',
    # MMdetection models:
    # 'YOLACT-R101',
    # 'SSD',
    # 'DDOD-R50',
    # 'PAA-R101',
    # 'TOOD-R101-Dconv',
    # 'Sparce-RCNN-R50-FPN-300prop-3x',
    # 'Sparce-RCNN-R101-FPN-3x',
    # 'Dynamic R-CNN',
    # 'VerifocalNet-R101',
    # 'SOLOv2-R50',
    # 'SOLOv2-R101',
    # 'HTC-R50-1',
    # 'MS-R-CNN-R50',
    # 'SCNet-R50-20e',
    # 'TridentNet-R50',
    # 'AutoAssign',
    # 'FCOS-R101',
    # 'NAS-FCOS-R50',
    # 'ATSS-R101',
    # 'Grid R-CNN'
]

CNN_DEFAULT = 'YOLOv8'

CNN_TYPES = ["YOLO8", "YOLO8_openvino", "YOLO5", "YOLOR", "MM_OD", "MM_MASK", "D2_OD", "D2_MASK", "D2_RETINA"]

CNN_DICT = {
    # 'YOLOv3': {'weights': 'mm_detector/checkpoints/yolov3_d53_mstrain-608_273e/epoch_270.pth',
    #            'config': "mm_detector/configs/yolo/yolov3_d53_mstrain-608_273e_coco.py", "type": "MM_OD"},
    # 'YOLOv5l6': {'weights': 'yolo//yolo_weights//v5l6_1280//best.pt', 'config': "yolo//yamls//aes.yaml",
    #              "type": "YOLO5"},
    # 'YOLOv5x6': {'weights': 'yolo//yolo_weights//v5x6_640//best.pt', 'config': "yolo//yamls//aes.yaml",
    #              "type": "YOLO5"},
    'YOLOv8': {'weights': 'yolov8//weights//best.pt', 'config': "yolov8//aes_yolo_seg.yaml",
               "type": "YOLO8"},
    # 'YOLOv8': {'weights': 'yolov8//weights//yolov8x-seg.pt', 'config': "yolov8//yolov8-seg.yaml",
    #            "type": "YOLO8"},
    'YOLOv8_openvino': {'weights': 'yolov8//weights//best_openvino_model//best.bin',
                        'config': "yolov8//weights//best_openvino_model//best.xml",
                        "type": "YOLO8_openvino"
                        }
    # 'YOLOR': {'weights': 'yolor//weights//best.pt', 'config': "yolor//cfg//yolor_p6.cfg", "type": "YOLOR"},
    # Detectron2 models:
    # 'Mask-R-CNN-R50': {'weights': "mask_r_cnn/run_train/model_final.pth",
    #                    'config': "mask_r_cnn/mask_config.yml", "type": "D2_MASK"},
    # 'Mask-R-CNN-R101': {'weights': "mask_r_cnn/run_train_mask_101/model_final.pth",
    #                     'config': "mask_r_cnn/mask_config_101.yml", "type": "D2_MASK"},
    # 'Mask-R-CNN-X101': {'weights': "mask_r_cnn/run_train_mask_x101/model_0063999.pth",
    #                     'config': "mask_r_cnn/mask_x_101_config.yml", "type": "D2_MASK"},
    # 'Cascade R-CNN': {'weights': "mask_r_cnn/run_train_cascade/model_final.pth",
    #                   'config': "mask_r_cnn/config_cascade.yml", "type": "D2_MASK"},
    # 'Retina-R50': {'weights': "mask_r_cnn/run_train_retina/model_final.pth",
    #                'config': "mask_r_cnn/mask_config_retina.yml", "type": "D2_RETINA"},
    # 'Retina-R101': {'weights': "mask_r_cnn/run_train_retina_101/model_final.pth",
    #                 'config': "mask_r_cnn/config_retina_101.yml", "type": "D2_RETINA"},
    # 'R-CNN-Faster-R50': {'weights': "mask_r_cnn/run_train_faster/model_final.pth",
    #                      'config': "mask_r_cnn/faster_config.yml", "type": "D2_OD"},
    # 'R-CNN-Faster-R101': {'weights': "mask_r_cnn/run_train_faster_101/model_final.pth",
    #                       'config': "mask_r_cnn/config_faster_101.yml", "type": "D2_OD"},
    # 'R-CNN-Faster-X101': {'weights': "mask_r_cnn/run_train_faster_x101/model_0054999.pth",
    #                       'config': "mask_r_cnn/faster_x101_config.yml", "type": "D2_OD"},
    # 'GN': {'weights': "mask_r_cnn/run_train_gn/model_final.pth", 'config': "mask_r_cnn/config_gn.yml",
    #        "type": "D2_MASK"},
    # # 'Deformable-Conv',
    # # MMdetection models:
    # 'YOLACT-R101': {'weights': 'mm_detector/checkpoints/yolact_101/latest.pth',
    #                 'config': "mm_detector/configs/yolact/yolact_r101_1x8_coco.py", "type": "MM_MASK"},
    # 'SSD': {'weights': 'mm_detector/checkpoints/ssd_512/latest.pth',
    #         'config': "mm_detector/configs/ssd/ssd512_coco.py", "type": "MM_OD"},
    # 'DDOD-R50': {'weights': 'mm_detector/checkpoints/ddod/latest.pth',
    #              'config': "mm_detector/configs/ddod/ddod_r50_fpn_1x_coco.py", "type": "MM_OD"},
    # 'PAA-R101': {'weights': 'mm_detector/checkpoints/paa_101_3/latest.pth',
    #              'config': "mm_detector/configs/paa/paa_r101_fpn_mstrain_3x_coco.py", "type": "MM_OD"},
    # 'TOOD-R101-Dconv': {'weights': 'mm_detector/checkpoints/tood_r101_fpn_dconv_c3-c5_mstrain_2x/latest.pth',
    #                     'config': "mm_detector/configs/tood/tood_r101_fpn_dconv_c3-c5_mstrain_2x_coco.py",
    #                     "type": "MM_OD"},
    # 'Sparce-RCNN-R50-FPN-300prop-3x': {'weights': 'mm_detector/checkpoints/sparce_50_300prop_3/latest.pth',
    #                                    'config': "mm_detector/configs/sparse_rcnn/sparse_rcnn_r50_fpn_300_proposals_crop_mstrain_480-800_3x_coco.py",
    #                                    "type": "MM_OD"},
    # 'Sparce-RCNN-R101-FPN-3x': {
    #     'weights': 'mm_detector/checkpoints/sparse_rcnn_r101_fpn_mstrain_480-800_3x/latest.pth',
    #     'config': "mm_detector/configs\sparse_rcnn\sparse_rcnn_r101_fpn_mstrain_480-800_3x_coco.py", "type": "MM_OD"},
    # 'Dynamic R-CNN': {'weights': 'mm_detector/checkpoints/dynamic_rcnn/latest.pth',
    #                   'config': "mm_detector/configs/dynamic_rcnn/dynamic_rcnn_r50_fpn_1x_coco.py", "type": "MM_OD"},
    # 'VerifocalNet-R101': {
    #     'weights': 'mm_detector/checkpoints/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x/latest.pth',
    #     'config': "mm_detector/configs/vfnet/vfnet_r101_fpn_mdconv_c3-c5_mstrain_2x_coco.py", "type": "MM_OD"},
    # 'SOLOv2-R50': {'weights': 'mm_detector/checkpoints/solov2_r50_fpn_3x/latest.pth',
    #                'config': "mm_detector/configs/solov2\solov2_r50_fpn_3x_coco.py", "type": "MM_MASK"},
    # 'SOLOv2-R101': {'weights': 'mm_detector/checkpoints/solov2_r101_fpn_3x/latest.pth',
    #                 'config': "mm_detector/configs/solov2\solov2_r101_fpn_3x_coco.py", "type": "MM_MASK"},
    # 'HTC-R50-1': {'weights': 'mm_detector/checkpoints/htc-50-1/latest.pth',
    #               'config': "mm_detector/configs/htc/htc_without_semantic_r50_fpn_1x_coco.py", "type": "MM_OD"},
    # 'MS-R-CNN-R50': {'weights': 'mm_detector/checkpoints/ms_rcnn_50/latest.pth',
    #                  'config': "mm_detector/configs/ms_rcnn/ms_rcnn_r50_fpn_1x_coco.py", "type": "MM_OD"},
    # 'SCNet-R50-20e': {'weights': 'mm_detector/checkpoints/scnet_r50_fpn_20e/latest.pth',
    #                   'config': "mm_detector/configs/scnet\scnet_r50_fpn_20e_coco.py", "type": "MM_OD"},
    # 'TridentNet-R50': {'weights': 'mm_detector/checkpoints/tridentnet_r50_caffe_mstrain_3x/epoch_32.pth',
    #                    'config': "mm_detector/configs/tridentnet/tridentnet_r50_caffe_mstrain_3x_coco.py",
    #                    "type": "MM_OD"},
    # 'AutoAssign': {'weights': 'mm_detector/checkpoints/autoassign_r50_fpn_8x2_1x/latest.pth',
    #                'config': "mm_detector/configs/autoassign/autoassign_r50_fpn_8x2_1x_coco.py", "type": "MM_OD"},
    # 'FCOS-R101': {
    #     'weights': 'mm_detector/checkpoints/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x/latest.pth',
    #     'config': "mm_detector/configs/fcos/fcos_r101_caffe_fpn_gn-head_mstrain_640-800_2x_coco.py", "type": "MM_OD"},
    # 'NAS-FCOS-R50': {'weights': 'mm_detector/checkpoints/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x/latest.pth',
    #                  'config': "mm_detector/configs/nas_fcos/nas_fcos_nashead_r50_caffe_fpn_gn-head_4x4_1x_coco.py",
    #                  "type": "MM_OD"},
    # 'ATSS-R101': {'weights': 'mm_detector/checkpoints/atss_r101_fpn_1x/latest.pth',
    #               'config': "mm_detector/configs/atss/atss_r101_fpn_1x_coco.py", "type": "MM_OD"},
    # 'Grid R-CNN': {'weights': 'mm_detector/checkpoints/grid_rcnn/epoch_33.pth',
    #                'config': "mm_detector/configs/grid_rcnn/grid_rcnn_r101_fpn_gn-head_2x_coco.py", "type": "MM_OD"}
}

SEG_DICT = {
    "PSPNetR50": {"weights": "mm_segmentation\checkpoints\iter_52000_83_59.pth",
                  "config": "mm_segmentation\configs\psp_aes.py",
                  "classes": ['background', 'water', 'vapor'],
                  "palette": [
                      [0, 0, 0],  # фон
                      [69, 170, 242],  # вода, голубой
                      [235, 77, 75]  # пар, красный
                  ]},
    "SegFormerMiTB5": {"weights": "mm_segmentation\checkpoints\segformer_mit_b5_160k_aes_iter_81000.pth",
                       "config": "mm_segmentation\configs\segformer_mit_b5_160k_aes.py",
                       "classes": ['background', 'reactor_sq', 'reactor', 'engine_room', 'pipe', 'switchgear', 'pump',
                                   'cooltower', 'ct_vent_circle', 'ct_vent_sq', 'ct_active', 'discharge', 'ISFSI',
                                   'tank', 'sea', 'parking', 'splash_pool'],
                       "palette": [
                           [0, 0, 0],  # фон
                           [219, 108, 110], [255, 140, 142], [255, 157, 92],
                           [26, 26, 26], [255, 255, 127], [106, 255, 83],
                           [255, 85, 255], [255, 85, 255], [255, 85, 255],
                           [255, 85, 255], [0, 255, 200], [91, 22, 23],
                           [129, 129, 0], [189, 201, 255], [0, 0, 255],
                           [220, 0, 216]

                       ]},

}

SAM_DICT = {
    'SAM_HQ_VIT_B': {'weights': 'sam_models//sam_hq_vit_b.pth', 'config': "",
                     "type": "SAM_HQ"},
    'SAM_HQ_VIT_H': {'weights': 'sam_models//sam_hq_vit_h.pth', 'config': "",
                     "type": "SAM_HQ"},
    'SAM_HQ_VIT_L': {'weights': 'sam_models//sam_hq_vit_l.pth', 'config': "",
                     "type": "SAM_HQ"},
    'SAM_VIT_H': {'weights': 'sam_models//sam_vit_h_4b8939.pth', 'config': "",
                  "type": "SAM"},
    'FastSAM': {'weights': 'sam_models/FastSAM-x.pt', 'config': "",
                "type": "FastSAM"}
}


def get_cfg_and_weights_by_cnn_name(cnn_name):
    dir_path = Path(os.path.curdir)
    dir_path = dir_path.parent
    cnn_dict = CNN_DICT[cnn_name]
    return os.path.join(dir_path.name, cnn_dict["config"]), os.path.join(dir_path.name, cnn_dict["weights"])


def get_cfg_and_weights_by_cnn_seg_name(cnn_name):
    cnn_dict = SEG_DICT[cnn_name]
    return cnn_dict["config"], cnn_dict["weights"]


def get_cnn_type(cnn_name):
    cnn_dict = CNN_DICT[cnn_name]
    return cnn_dict["type"]
