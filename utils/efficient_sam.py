from ultralytics import FastSAM
from utils.fast_sam_promt import FastSAMPrompt
from utils.edges_from_mask import mask_to_polygons_layer
import numpy as np


def mask2points(mask, simplify_factor=0.4):
    polygons = mask_to_polygons_layer(mask)

    results = []
    for pol in polygons:
        pol_simplified = pol.simplify(simplify_factor, preserve_topology=True)
        points = []
        try:
            xy = np.asarray(pol_simplified.boundary.xy, dtype="float")
            x_mass = xy[0].tolist()
            y_mass = xy[1].tolist()
            for x, y in zip(x_mass, y_mass):
                points.append([x, y])
            results.append(points)
        except:
            pass

    return results


def ann_to_shapes(ann, simplify_factor=0.1):
    mask_results = []
    for res in ann:
        mask = res.cpu().numpy()

        mask[mask == 1] = 255

        mask_results.append({'masks': mask})

    shapes = []
    id_tek = 1
    for res in mask_results:
        points_mass = mask2points(res['masks'], simplify_factor=simplify_factor)
        for points in points_mass:
            cls_num = id_tek
            shape = {'id': id_tek, 'cls_num': cls_num, 'points': points}
            id_tek += 1
            shapes.append(shape)

    return shapes


class FastSAMPredictor:

    def __init__(self, checkpoint_path):
        self.model = FastSAM(checkpoint_path)  # or FastSAM-x.pt

    def set_image(self, source, imgsz=1024, conf=0.4, iou=0.9):
        self.everything_results = self.model(source, device='cuda', retina_masks=True, imgsz=imgsz, conf=conf, iou=iou)

    def box_prompt(self, source, bbox):
        """
        Example
            source = 'D:/python/aia_git/ai_annotator/sam_models/bus.jpg'
            bbox = [200, 200, 300, 300]
        """

        # Prepare a Prompt Process object
        prompt_process = FastSAMPrompt(source, self.everything_results, device='cuda')

        ann = prompt_process.box_prompt(bbox=bbox)

        return ann_to_shapes(ann)

    def everything_prompt(self, source):
        # Prepare a Prompt Process object
        prompt_process = FastSAMPrompt(source, self.everything_results, device='cuda')

        # Everything prompt
        ann = prompt_process.everything_prompt()

        return ann_to_shapes(ann)

    def point_prompt(self, source, points, pointlabel):
        """
        Example points=[[200, 200]], pointlabel=[1]
        """

        # Prepare a Prompt Process object
        prompt_process = FastSAMPrompt(source, self.everything_results, device='cuda')

        ann = prompt_process.point_prompt(points=points, pointlabel=pointlabel)

        return ann_to_shapes(ann)


# # Define an inference source
# source = 'D:/python/aia_git/ai_annotator/sam_models/bus.jpg'
#
# # Create a FastSAM model
# sam_path = 'D:/python/aia_git/ai_annotator/sam_models/FastSAM-x.pt'
# model = FastSAM(sam_path)  # or FastSAM-x.pt
#
# # Run inference on an image
# everything_results = model(source, device='cuda', retina_masks=True, imgsz=1024, conf=0.4, iou=0.9)
#
# # Prepare a Prompt Process object
# prompt_process = FastSAMPrompt(source, everything_results, device='cuda')
#
# # Everything prompt
# ann = prompt_process.everything_prompt()
#
# # Bbox default shape [0,0,0,0] -> [x1,y1,x2,y2]
# ann = prompt_process.box_prompt(bbox=[200, 200, 300, 300])
#
# # Text prompt
# # ann = prompt_process.text_prompt(text='a photo of a dog')
#
# # Point prompt
# # points default [[0,0]] [[x1,y1],[x2,y2]]
# # point_label default [0] [1,0] 0:background, 1:foreground
# ann = prompt_process.point_prompt(points=[[200, 200]], pointlabel=[1])
# prompt_process.plot(annotations=ann, output='image_resuls')

if __name__ == '__main__':
    import cv2

    sam_path = 'D:/python/aia_git/ai_annotator/sam_models/FastSAM-x.pt'
    source = 'D:/python/aia_git/ai_annotator/sam_models/bus.jpg'
    points = [[500, 500]]
    pointlabel = [1]

    fast_sam = FastSAMPredictor(sam_path)
    image = cv2.imread(source)
    fast_sam.set_image(image)
    shapes = fast_sam.point_prompt(image, points, pointlabel)
    # shapes = fast_sam.everything_prompt(source)
    print(shapes)
