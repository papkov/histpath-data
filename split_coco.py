import json
import argparse
import os
import funcy
from sklearn.model_selection import train_test_split

# Mostly from
# https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py


def save_coco(file_path, info, licences, images, annotations, categories):
    """
    Save COCO annotations to a file.
    """
    with open(file_path, "w") as coco:
        json.dump(
            {
                "info": info,
                "licences": licences,
                "images": images,
                "annotations": annotations,
                "categories": categories,
            },
            coco,
            indent=2,
            sort_keys=True,
        )


def filter_annotations(annotations, images):
    """
    Filter out annotations for available images.
    """
    image_ids = funcy.lmap(lambda i: int(i["id"]), images)
    return funcy.lfilter(lambda a: int(a["image_id"]) in image_ids, annotations)


def main(args):
    # Create dirs if not exist.
    if not os.path.isdir(args.save_dir):
        os.mkdir(args.save_dir)
        os.mkdir(os.path.join(args.save_dir, "train"))
        os.mkdir(os.path.join(args.save_dir, "train", "data"))
        os.mkdir(os.path.join(args.save_dir, "test"))
        os.mkdir(os.path.join(args.save_dir, "test", "data"))

    with open(args.annotations, "rt") as annotations:
        coco = json.load(annotations)
        info = coco["info"]
        licenses = coco["licenses"]
        images = coco["images"]
        annotations = coco["annotations"]
        categories = coco["categories"]

        x, y = train_test_split(images, train_size=args.split)
        train_file = os.path.join(args.save_dir, "train") + os.path.sep + "labels.json"
        test_file = os.path.join(args.save_dir, "test") + os.path.sep + "labels.json"
        save_coco(
            train_file,
            info,
            licenses,
            x,
            filter_annotations(annotations, x),
            categories,
        )
        save_coco(
            test_file,
            info,
            licenses,
            y,
            filter_annotations(annotations, y),
            categories,
        )

        print("Saved {} training entries and {} test entries".format(len(x), len(y)))

    # Move images.
    # Training.
    new_train_img_dir = os.path.join(args.save_dir, "train", "data") + os.path.sep
    for im in x:
        org_path = args.image_dir + im["file_name"]
        new_path = new_train_img_dir + im["file_name"]
        os.rename(org_path, new_path)

    # Testing.
    new_test_img_dir = os.path.join(args.save_dir, "test", "data") + os.path.sep
    for im in y:
        org_path = args.image_dir + im["file_name"]
        new_path = new_test_img_dir + im["file_name"]
        os.rename(org_path, new_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COCO dataset train test split for CenterNet-better"
    )

    parser.add_argument(
        "--annotations",
        type=str,
        help="Path to COCO annotations",
        default="output/labels.json",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        help="Location of annotated image",
        default="output/data/",
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        help="Location where to save train/test datasets",
        default="coco/",
    )
    parser.add_argument(
        "--split", type=float, help="Percentage of training images", default=0.8
    )

    args = parser.parse_args()

    dir = args.image_dir
    if args.image_dir[-1] != os.path.sep:
        args.image_dir = dir + os.path.sep
    main(args)
