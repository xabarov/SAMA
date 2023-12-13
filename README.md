# SAMA - SAM Annotator

Labeling Data for Object Detection and Instance Segmentation with The Segment Anything Model (SAM) and GroundingDINO.
The project also contains an object detector, a segmenter with SAM, YOLO and GroundingDINO support.

![alt text](assets/main.png?raw=true)
### Install requirements

1. Install pytorch from  `https://pytorch.org/get-started/locally/`
2. Install GroundingDINO and SAM
   - download repo `https://github.com/IDEA-Research/Grounded-Segment-Anything`
   - python -m pip install -e segment_anything
   - python -m pip install -e GroundingDINO
3. Install requirements from `requirements.txt` as `pip install -r requirements.txt`

### SAM model

1. Download SAM models from `https://github.com/facebookresearch/segment-anything#model-checkpoints`
2. Place model to the `/sam_models`

### GroundingDINO model

1. Download GroundingDINO model from `https://github.com/IDEA-Research/GroundingDINO`
2. Place model to the `/gd`
3. Replace string `text_encoder_type` in configs from `gd/GroundingDINO/groundingdino/config/` to "bert-base-uncased"
   for online download BERT model or for local path to BERT-BASE-UNCASED model

### YOLOv8 for detector

1. Replace `weights` and `config` in CNN_DICT for 'YOLOv8' in `utils/cls_settings.py` to your model weight and YAML
   config paths
2. Replace CLASSES_ENG and CLASSES_RU in `utils/cls_settings.py` for your classes names



### Usage

1. Run `annotator_light.py` for Annotator version without SAM, GroundingDINO, YOLO etc.
2. Run `annotator.py` for Annotator version with SAM, GroundingDINO, YOLO etc.
3. Run `detector.py` for Detector version with SAM, GroundingDINO, YOLO etc.
4. Run `segmentator.py` for Segmentator version with SAM, GroundingDINO, YOLO etc.

### Common shortcuts

**S** - draw a new label  
**D** - delete current label    
**Space** - finish drawing the current label

**Ctrl + C** - copy current label  
**Ctrl + V** - paste current label

**Ctrl + A** - SAM by points  
**Ctrl + M** - SAM by box  
**Ctrl + G** - GroundingDINO + SAM

### In segmentation mode using SAM

1. Segmentation by points:

- Left mouse button set point inside segment
- Right mouse button set a point outside the segment (background)
- Space run the SAM neural network to draw the label

2. Segmentation inside the box

1) Draw a rectangular label with an area for segmentation
2) Wait for the SAM label to appear

### About SAM

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it
can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1
billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

@article{kirillov2023segany,
title={Segment Anything},
author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura
and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
journal={arXiv:2304.02643},
year={2023}
}

### GroundingDINO

@article{liu2023grounding,
title={Grounding dino: Marrying dino with grounded pre-training for open-set object detection},
author={Liu, Shilong and Zeng, Zhaoyang and Ren, Tianhe and Li, Feng and Zhang, Hao and Yang, Jie and Li, Chunyuan and
Yang, Jianwei and Su, Hang and Zhu, Jun and others},
journal={arXiv preprint arXiv:2303.05499},
year={2023}
}