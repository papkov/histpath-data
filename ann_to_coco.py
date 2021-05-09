import time

import pandas as pd
from matplotlib.patches import Ellipse
import numpy as np


def label_to_category_id(label):
    return {
        "label_color 1": 1,
        "label_color 2": 2,
        "label_color 3": 3,
        "label_color 4": 4,
        "label_color 5": 5,
    }[label]


def get_annotation_label(annotation):
    available_keys = [
        "label_color 1",
        "label_color 2",
        "label_color 3",
        "label_color 4",
        "label_color 5",
    ]

    keys = annotation["user_attrs"].keys()
    for key in keys:
        if key in available_keys:
            return key

    return None


def get_annotation_area(annotation):
    return annotation["stats"]["area_px"]


def calculate_bbox(annotation, top_left):
    ann_points = annotation["geometry"]["points"]
    x1, y1 = ann_points[0][0], ann_points[0][1]
    x2, y2 = ann_points[1][0], ann_points[1][1]
    # Drawing points are relative to the top left
    # as the new image top left is (0,0)
    point1 = (x1 - top_left[0], y1 - top_left[1])
    point2 = (x2 - top_left[0], y2 - top_left[1])

    # Swap if points are flipped. Point1 should be upper left.
    if point1 > point2:
        point1, point2 = point2, point1

    width = point2[0] - point1[0]
    height = point2[1] - point1[1]

    return [point1[0], point1[1], width, height]


def get_annotation_filename(annotation):
    return "{}_{}_{}.png".format(
        annotation["filename"], annotation["top_left"][0], annotation["top_left"][1]
    )


def ellipse_to_polygon(bbox):
    # Top left pixel.
    x = bbox[0]
    y = bbox[1]

    width = bbox[2]
    height = bbox[3]

    center = ((x + width) / 2, (y + height) / 2)

    # TODO: if the last point should be the same as first?
    # Currently they are the same.

    ell = Ellipse((center[0], center[1]), width, height)
    # Polygon
    poly = np.array(ell.get_path().to_polygons())
    poly = poly.reshape(poly.shape[1], poly.shape[2])  # Reduce first dimension

    # Scale points with width and height
    poly = poly * [width, height]

    # Convert coordinates to original image
    poly = poly + [center[0], center[1]]

    # For json serialization.
    poly = poly.tolist()

    return poly


def create_COCO_annotations(tiles_with_annotation):
    df = pd.DataFrame(tiles_with_annotation)
    coco = {}

    coco["info"] = {
        "description": "HistoPathology dataset",
        "url": "http://cocodataset.org",
        "version": "1.0",
        "year": 2021,
        "contributor": "Kaarel R",
        "date_created": time.asctime(),
    }

    coco["licenses"] = {}
    coco["images"] = []
    coco["annotations"] = []

    ann_id = 0
    # Iterating trough only annotated images.
    tiles_with_annotations = df[df.annotations.str.len() > 0]
    for id, row in tiles_with_annotations.iterrows():
        im = {
            "license": 0,
            "file_name": get_annotation_filename(row),
            "coco_url": "",
            "height": row.image_size,
            "width": row.image_size,
            "date_captured": "2021-05-8 17:00:00",
            "flickr_url": "",
            "id": id,
        }

        prev_annotation = row.annotations
        for ann in prev_annotation:
            bbox = calculate_bbox(ann, row.top_left)
            poly = ellipse_to_polygon(bbox)
            coco_ann = {
                "id": ann_id,
                "image_id": id,
                "category_id": label_to_category_id(get_annotation_label(ann)),
                "segmentation": poly,
                "area": get_annotation_area(ann),
                "bbox": bbox,
                "iscrowd": 0,
            }
            coco["annotations"].append(coco_ann)
            ann_id += 1

        coco["images"].append(im)

    coco["categories"] = [
        {"supercategory": "cell", "id": 1, "name": "Spermatogonia"},
        {"supercategory": "cell", "id": 2, "name": "Sertoli"},
        {"supercategory": "cell", "id": 3, "name": "Primary spermatocyte"},
        {"supercategory": "cell", "id": 4, "name": "Spermatid"},
        {"supercategory": "cell", "id": 5, "name": "Spermatozoa"},
    ]

    return coco
