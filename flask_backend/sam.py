import os
import config
from segment_anything import sam_model_registry, SamPredictor
import cv2
import numpy as np
import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon


def load_model(model_path, model_type="vit_h", device="cuda"):
    sam_checkpoint = model_path

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    return predictor


def load_sam():
    sam_model_path = os.path.join(os.getcwd(), config.PATH_TO_SAM_CHECKPOINT)
    return load_model(sam_model_path, device='cuda')


def set_image(predictor, image_name):
    image = cv2.imread(image_name)
    predictor.set_image(image)


def predict_by_points(predictor, input_point, input_label, multi=True):
    if multi:
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )
        return masks

    else:
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        return [masks]


def mask_to_polygons_layer(mask):
    shapes = []
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask > 0),
                                        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        shapes.append(shapely.geometry.shape(shape))

    return shapes


def mask_to_seg(mask, simplify_factor=2):
    img_data = np.asarray(mask[:, :], dtype=np.double)
    polygons = mask_to_polygons_layer(img_data)

    results = []
    for pol in polygons:
        pol_simplified = pol.simplify(simplify_factor, preserve_topology=False)

        try:
            xy = np.asarray(pol_simplified.boundary.xy, dtype="int32")
            points = []

            xs = xy[0].tolist()
            ys = xy[1].tolist()
            for x, y in zip(xs, ys):
                points.append([x, y])

            results.append(points)

        except:
            pass

    return results


def predict_by_box(predictor, input_box, is_best=True):
    # input_box = np.array([425, 600, 700, 875])

    if is_best:
        masks, scores, _ = predictor.predict(
            point_coords=None,
            point_labels=None,
            box=input_box[None, :],
            multimask_output=True,
        )
        max_score = -1
        best_num = 0
        for i, sc in enumerate(scores):
            if sc > max_score:
                max_score = sc
                best_num = i

        return masks[best_num]

    masks, _, _ = predictor.predict(
        point_coords=None,
        point_labels=None,
        box=input_box[None, :],
        multimask_output=False,
    )
    return masks