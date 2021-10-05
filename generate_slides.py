import argparse
import glob
import json
import os
import numpy as np
import pandas as pd
from openslide import OpenSlide, open_slide

from references.ann_to_coco import create_COCO_annotations, get_annotation_filename


def load_mrxs_files(data_dir):
    """
    Returns list of mrxs files in data_dir.
    """
    full_paths = glob.glob(data_dir + "*.mrxs")
    return [os.path.basename(file) for file in full_paths]


def load_annotations(data_dir):
    """
    Parse annotation JSON files from data_dir.
    """
    anns = []
    ann_files = [file.replace("mrxs", "json") for file in load_mrxs_files(data_dir)]
    for file in ann_files:
        with open(data_dir + file) as f:
            ann_json = json.load(f)
            anns.append(ann_json)

    return anns


def load_slides(data_dir):
    """
    Load slides in data_dir as OpenSlide objects.
    """
    slides = []
    for file in load_mrxs_files(data_dir):
        slides.append(OpenSlide(data_dir + file))
    return slides


def show_tiles_statistics(annotated_tiles):
    """
    Prints out slide statistics based of generated annotations.
    """
    tot_count = len(annotated_tiles)
    max_annotations_in_image = 0
    min_annotations_in_image = np.inf
    tile_size = annotated_tiles[0]["image_size"]
    lvl = annotated_tiles[0]["lvl"]
    anns_in_image = []

    for tile in annotated_tiles:
        num_annotations = len(tile["annotations"])
        anns_in_image.append(num_annotations)
        if num_annotations > max_annotations_in_image:
            max_annotations_in_image = num_annotations
        if num_annotations < min_annotations_in_image:
            min_annotations_in_image = num_annotations

    print("Zoom lvl: \t\t{}".format(lvl))
    print("Tile size: \t\t{}".format(tile_size))
    print("Total slides: \t\t{}".format(tot_count))
    print("Max annotations: \t{}".format(max_annotations_in_image))
    print("Min annotations: \t{}".format(min_annotations_in_image))
    print("Average annotations: \t{}".format(np.round(np.mean(anns_in_image), 2)))


def ann_in_image(img_top_left, img_bot_right, ann_top_left, ann_bot_right):
    """
    Determines if annotation center is on an image.

    ---
    img_top_left: tuple(x,y)
        Top left point coordinates on WSI image.
    img_bot_right: tuple(x,y)
        Bottom right point coordinates on WSI image.
    ann_top_left: tuple(x,y)
        Top left coordinates of annotation on WSI image.
    ann_bot_right: tuple()
        Bottom right coordinates of annotation on WSI image.
    """
    x1 = img_top_left[0]
    y1 = img_top_left[1]

    x2 = img_bot_right[0]
    y2 = img_bot_right[1]

    ann_x1 = ann_top_left[0]
    ann_y1 = ann_top_left[1]

    ann_x2 = ann_bot_right[0]
    ann_y2 = ann_bot_right[1]

    # Annotation bbox center should be fully on an image.
    ann_center = ((ann_x1 + ann_x2) / 2, (ann_y1 + ann_y2) / 2)

    ann_center_x_on_image = ann_center[0] > x1 and ann_center[0] < x2
    ann_center_y_on_image = ann_center[1] > y1 and ann_center[1] < y2

    if ann_center_x_on_image and ann_center_y_on_image:
        return True

    return False


def create_tiles_with_annotation(annotations, slide, lvl=0, tile_size=1024):
    """
    Generates tiles with annotations. Moves above the WSI with selected tile size
    and finds all annotations for selected region.
    ---
    annotations : array
        Annotations of WSI slide.
    slide : OpenSlide
        OpenSlide object of WSI.
    tile_size : int
        Size of the generated image.
    """

    x_dim = slide.level_dimensions[0][0]
    y_dim = slide.level_dimensions[0][1]
    filename = slide._filename.split("/")[-1].split(".")[0]
    # Scale with lvl downsampling.
    downsample = int(slide.level_downsamples[lvl])

    tile_annotations = []
    # Moving through image with tiles.
    for x in range(0, x_dim, tile_size * downsample):
        for y in range(0, y_dim, tile_size * downsample):
            tile_info = {}

            # Top left pixel of tile.
            x_0 = (x, y)
            # Bottom right pixel.
            x_1 = (x + tile_size * downsample, y + tile_size * downsample)
            # Collect all annotations inside tile.
            tile_anns = []
            # Check if there are annotations inside tile.
            for ann in annotations["annotations"]:
                points = annotations["annotations"][ann]["geometry"]["points"]
                if ann_in_image(
                    img_top_left=x_0,
                    img_bot_right=x_1,
                    ann_top_left=(points[0][0], points[0][1]),
                    ann_bot_right=(points[1][0], points[1][1]),
                ):
                    tile_anns.append(annotations["annotations"][ann])
            if len(tile_anns) > 0:
                tile_info["top_left"] = x_0
                tile_info["lvl"] = lvl
                tile_info["image_size"] = tile_size
                tile_info["annotations"] = tile_anns
                tile_info["filename"] = filename
                tile_info["downsample"] = downsample
                tile_annotations.append(tile_info)

    return tile_annotations


def save_images(annotations, data_folder, output_dir):
    image_dir = output_dir + "data" + os.path.sep

    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    tiles = pd.DataFrame(annotations)
    for _, row in tiles.iterrows():
        slide = open_slide(data_folder + row.filename + ".mrxs")
        im = slide.read_region(
            location=row.top_left, level=row.lvl, size=(row.image_size, row.image_size)
        ).convert("RGB")
        im.save(
            image_dir + get_annotation_filename(row), "PNG",
        )


def main(args):
    root = args.data_folder
    mrx_files = load_mrxs_files(root)
    annotations = load_annotations(root)
    slides = load_slides(root)

    if args.output_dir[-1] != os.path.sep:
        args.output_dir = args.output_dir + os.path.sep

    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    print("Generating annotations!")
    print("Data folder: {}".format(root))
    print("All slides: {}".format(mrx_files))

    total_annotations = []
    for slide, annotation in zip(slides, annotations):
        print("Slide: {}".format(slide._filename))
        for lvl in range(args.max_level + 1):
            annotations = create_tiles_with_annotation(
                annotation, slide, lvl=lvl, tile_size=args.image_size
            )

            total_annotations.extend(annotations)
            show_tiles_statistics(annotations)
            print()

    print("Saving labels!")
    coco = create_COCO_annotations(total_annotations)
    with open(args.output_dir + "labels.json", "w") as f:
        json.dump(coco, f, indent=2, sort_keys=True)

    if args.save_images:
        print("Saving images!")
        save_images(total_annotations, args.data_folder, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits WHI into slides and returns JSON with annotation information"
    )

    parser.add_argument(
        "data_folder", type=str, help="Directory where .mrxs files live"
    )

    parser.add_argument(
        "--max_level", type=int, help="Max openslide level images to use", default=0,
    )

    parser.add_argument(
        "--image_size", type=int, help="Tile size of generated images", default=1024,
    )

    parser.add_argument(
        "--save_images",
        type=bool,
        default=True,
        help="Create images from annotated slides",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="output/",
        help="Directory where to save generated output",
    )

    args = parser.parse_args()
    main(args)
