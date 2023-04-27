import numpy as np
import torch
import matplotlib.pyplot as plt
import cv2
from segment_anything import sam_model_registry, SamPredictor
import rasterio
from rasterio import features
import shapely
from shapely.geometry import Point, Polygon

def show_mask(mask, ax, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)


def show_points(coords, labels, ax, marker_size=375):
    pos_points = coords[labels == 1]
    neg_points = coords[labels == 0]
    ax.scatter(pos_points[:, 0], pos_points[:, 1], color='green', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)
    ax.scatter(neg_points[:, 0], neg_points[:, 1], color='red', marker='*', s=marker_size, edgecolor='white',
               linewidth=1.25)


def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(plt.Rectangle((x0, y0), w, h, edgecolor='green', facecolor=(0, 0, 0, 0), lw=2))


def load_model(model_path, model_type="vit_h", device="cuda"):
    sam_checkpoint = model_path

    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)

    predictor = SamPredictor(sam)

    return predictor


def predictor_set_image(predictor, image):
    predictor.set_image(image)


def predict_by_points(predictor, input_point, input_label, is_best=True):
    # input_point = np.array([[500, 375]])
    # input_label = np.array([1])

    if is_best:
        masks, scores, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=True,
        )

        max_score = -1
        best_num = 0
        for i, sc in enumerate(scores):
            if sc > max_score:
                max_score = sc
                best_num = i

        return masks[best_num]

    else:
        masks, _, _ = predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            multimask_output=False,
        )

        return masks


def mask_to_polygons_layer(mask):
    for shape, value in features.shapes(mask.astype(np.int16), mask=(mask > 0),
                                        transform=rasterio.Affine(1.0, 0, 0, 0, 1.0, 0)):
        return shapely.geometry.shape(shape)


def mask_to_seg(mask, simplify_factor=2):
    img_data = np.asarray(mask[:, :], dtype=np.double)
    polygon = mask_to_polygons_layer(img_data).simplify(simplify_factor, preserve_topology=False)

    try:
        xy = np.asarray(polygon.boundary.xy, dtype="int32")
        points = []

        xs = xy[0].tolist()
        ys = xy[1].tolist()
        for x, y in zip(xs, ys):
            points.append([x, y])
        return points

    except:

        return None


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


if __name__ == '__main__':
    # mask_name = "F:\python\\ai_annotator\\test_temp\mask.png"
    # mask = cv2.imread(mask_name)
    #
    # print(mask_to_seg(mask))

    predictor = load_model("F:\python\\ai_annotator\seg_models\sam_vit_h_4b8939.pth")
    input_point = np.array([[500, 375]])
    input_label = np.array([1])
    image = cv2.imread("F:\python\\ai_annotator\images\diablo.jpg")
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    mask = predict_by_points(predictor, image, input_point, input_label)
    print(mask.shape)
    seg = mask_to_seg(mask)
    print(seg)
