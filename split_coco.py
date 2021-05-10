import json
import argparse
import os
import funcy
from sklearn.model_selection import train_test_split

# Mostly from
# https://github.com/akarazniewicz/cocosplit/blob/master/cocosplit.py


def save_coco(file, info, licences, images, annotations, categories):
    """
    Save COCO annotations to a file.
    """
    with open("coco/annotations/{}".format(file), "w") as coco:
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
    if not os.path.isdir("coco"):
        os.mkdir("coco")
        os.mkdir("coco/annotations")
        os.mkdir("coco/train2017")
        os.mkdir("coco/val2017")

    with open(args.annotations, "rt") as annotations:
        coco = json.load(annotations)
        info = coco["info"]
        licenses = coco["licenses"]
        images = coco["images"]
        annotations = coco["annotations"]
        categories = coco["categories"]

        x, y = train_test_split(images, train_size=args.split)
        save_coco(
            "instances_train2017.json",
            info,
            licenses,
            x,
            filter_annotations(annotations, x),
            categories,
        )
        save_coco(
            "instances_val2017.json",
            info,
            licenses,
            y,
            filter_annotations(annotations, y),
            categories,
        )

        print("Saved {} training entries and {} test entries".format(len(x), len(y)))

    # Move images.
    for im in x:
        org_path = args.image_dir + im["file_name"]
        new_path = "coco/train2017/" + im["file_name"]
        os.rename(org_path, new_path)

    for im in y:
        org_path = args.image_dir + im["file_name"]
        new_path = "coco/val2017/" + im["file_name"]
        os.rename(org_path, new_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="COCO dataset train test split for CenterNet-better"
    )

    parser.add_argument("annotations", type=str, help="Path to COCO annotations")
    parser.add_argument(
        "-s",
        type=float,
        dest="split",
        required=True,
        help="Percentage of training images",
    )
    parser.add_argument(
        "-d", type=str, dest="image_dir", required=True, help="Image directory"
    )

    args = parser.parse_args()

    dir = args.image_dir
    if args.image_dir[-1] != os.path.sep:
        args.image_dir = dir + os.path.sep
    main(args)
