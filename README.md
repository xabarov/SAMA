# AIA - Annotator with artificial intellegence support (based on SAM)

 Labeling Data for Object Detection and Instance Segmentation with The Segment Anything Model (SAM).

 ## Usage

 ### Common shortcuts

**S** -  draw a new label  
**D** - delete current label    
**Space** - finish drawing the current label  
**Ctrl + C** - copy current label  
**Ctrl + V** - paste current label  

### In segmentation mode using a neural network

1. Segmentation by points

- Left mouse button set point inside segment
- Right mouse button set a point outside the segment (background)
- Space run the SAM neural network to draw the label

2. Segmentation inside the box

1) Draw a rectangular label with an area for segmentation
2) Wait for the SAM label to appear

 ## About SAM

The Segment Anything Model (SAM) produces high quality object masks from input prompts such as points or boxes, and it can be used to generate masks for all objects in an image. It has been trained on a dataset of 11 million images and 1.1 billion masks, and has strong zero-shot performance on a variety of segmentation tasks.

 ### More about SAM:
    `https://github.com/facebookresearch/segment-anything`
 @article{kirillov2023segany,
  title={Segment Anything},
  author={Kirillov, Alexander and Mintun, Eric and Ravi, Nikhila and Mao, Hanzi and Rolland, Chloe and Gustafson, Laura and Xiao, Tete and Whitehead, Spencer and Berg, Alexander C. and Lo, Wan-Yen and Doll{\'a}r, Piotr and Girshick, Ross},
  journal={arXiv:2304.02643},
  year={2023}
}
