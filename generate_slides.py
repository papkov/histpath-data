import argparse
import glob
import json
import os

import pandas as pd
from openslide import OpenSlide, open_slide

from ann_to_coco import create_COCO_annotations


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


def save_annotated_image(image_dir, slide, top_left, size):
    """
    Saves annotated section of certain region from the whole slide image with
    certain size starting from the top left corner of WSI.
    Format of saved images {WSI_FILENAME}_{TOP_LEFT_X}_{TOP_LEFT_Y}.png

    ---
    image_dir : str
        Directory where to save images.
    slide : OpenSlide
        OpenSlide object of WSI.
    top_left : tuple
        Top left coordinates of the image. (x,y)
    size : int
        Size of the generated image
    """

    filename = slide._filename.split("/")[-1].split(".")[0]
    im = slide.read_region(location=top_left, level=0, size=(size, size)).convert("RGB")
    im.save(
        "{}_{}_{}.png".format(image_dir + filename, top_left[0], top_left[1]), "PNG",
    )


def show_tiles_statistics(annotated_tiles):
    """
    Prints out slide statistics based of generated annotations.
    """
    tot_count = len(annotated_tiles)
    empty_images = 0
    annotated_images = 0
    max_annotations_in_image = 0

    for tile in annotated_tiles:
        if len(tile["annotations"]) == 0:
            empty_images += 1
        else:
            annotated_images += 1
            num_annotations = len(tile["annotations"])
            if num_annotations > max_annotations_in_image:
                max_annotations_in_image = num_annotations

    print("Total slides: \t\t{}".format(tot_count))
    print("Slides not annotated: \t{}".format(empty_images))
    print("Annotated slides: \t{}".format(annotated_images))
    print("Max annotations: \t{}".format(max_annotations_in_image))


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


def create_tiles_with_annotation(annotations, slide, tile_size=1024):
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

    tile_annotations = []

    # Moving through image with tiles.
    for x in range(0, x_dim, tile_size):
        for y in range(0, y_dim, tile_size):
            tile_info = {}

            # Top left pixel of tile.
            x_0 = (x, y)
            # Bottom right pixel.
            x_1 = (x + tile_size, y + tile_size)

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

            tile_info["top_left"] = x_0
            tile_info["image_size"] = tile_size
            tile_info["annotations"] = tile_anns
            tile_info["filename"] = filename

            tile_annotations.append(tile_info)

    return tile_annotations


def save_images(annotations, data_folder, output_dir):
    image_dir = output_dir + "data" + os.path.sep

    if not os.path.exists(image_dir):
        os.mkdir(image_dir)

    tiles = pd.DataFrame(annotations)
    tiles_with_annotations = tiles[tiles.annotations.str.len() > 0]
    for _, row in tiles_with_annotations.iterrows():
        slide = open_slide(data_folder + row.filename + ".mrxs")
        save_annotated_image(image_dir, slide, row.top_left, row.image_size)


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
        annotations = create_tiles_with_annotation(annotation, slide, args.image_size)

        total_annotations.extend(annotations)
        show_tiles_statistics(annotations)
        print()

    coco = create_COCO_annotations(total_annotations)
    with open(args.output_dir + "labels.json", "w") as f:
        json.dump(coco, f, indent=2, sort_keys=True)

    if args.save_images:
        save_images(total_annotations, args.data_folder, args.output_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Splits WHI into slides and returns JSON with annotation information"
    )

    parser.add_argument(
        "data_folder", type=str, help="Directory where .mrxs files live"
    )

    parser.add_argument(
        "--image_size", type=int, help="Tile size of generated images", default=1024,
    )

    parser.add_argument(
        "--save_images",
        type=bool,
        default=False,
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
