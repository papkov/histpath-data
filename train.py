import argparse
import glob
import math
import os
from datetime import datetime

import albumentations
import matplotlib.pyplot as plt
import numpy as np
import torch
import torchvision
from albumentations.augmentations.transforms import (
    Flip,
    HueSaturationValue,
    Normalize,
    RandomBrightnessContrast,
)
from albumentations.pytorch.transforms import ToTensorV2
from numpy.core.fromnumeric import std
from PIL import Image, ImageStat
from pycocotools.coco import COCO
from torch.utils.data import DataLoader, Dataset
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from references import transforms as T
from references import utils
from references.engine import evaluate, train_one_epoch


def get_model(num_classes, pretrained_model=True):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        pretrained=pretrained_model
    )
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class torchDataset(Dataset):
    def __init__(self, root, annotations, transforms=None):
        self.root = root
        self.transforms = transforms
        self.coco = COCO(annotations)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, index):
        coco = self.coco

        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        coco_annotation = coco.loadAnns(ann_ids)

        path = coco.loadImgs(img_id)[0]["file_name"]
        img = Image.open(os.path.join(self.root, path)).convert("RGB")
        num_objects = len(coco_annotation)

        boxes = []
        labels = []
        areas = []
        for i in range(num_objects):
            xmin = coco_annotation[i]["bbox"][0]
            ymin = coco_annotation[i]["bbox"][1]
            xmax = xmin + coco_annotation[i]["bbox"][2]
            ymax = ymin + coco_annotation[i]["bbox"][3]
            boxes.append([xmin, ymin, xmax, ymax])  # pascal_voc
            labels.append(coco_annotation[i]["category_id"])
            areas.append(coco_annotation[i]["area"])

        if self.transforms is not None:
            sample = {"image": np.array(img), "bboxes": boxes, "labels": labels}
            sample = self.transforms(**sample)
            img = sample["image"]
            # plt.imshow(img.permute(1, 2, 0))
            # plt.show()
            boxes = sample["bboxes"]
            labels = sample["labels"]

        # If there are no more bboxes left after augmentation.
        if len(boxes) == 0:
            boxes = np.array([[0.0, 0.0, 1.0, 1.0]])
            areas = [1.0]
            labels = [0]

        # Collect
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(areas, dtype=torch.float32)
        iscrowd = torch.zeros((num_objects,), dtype=torch.int64)
        image_id = torch.tensor([img_id])

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": image_id,
        }

        return img, target

    def __len__(self):
        return len(self.ids)


def get_train_transform(mean, std):
    return albumentations.Compose(
        [
            Flip(),
            RandomBrightnessContrast(brightness_limit=0.5, contrast_limit=0.5),
            HueSaturationValue(
                hue_shift_limit=10, sat_shift_limit=50, val_shift_limit=50
            ),
            Normalize(mean=mean, std=std),
            ToTensorV2(),
        ],
        bbox_params=albumentations.BboxParams(
            format="pascal_voc", label_fields=["labels"]
        ),
    )


def calc_norm_sats(img_dir):
    mean_list = []
    std_list = []
    for img_p in glob.glob(img_dir + "/*.png"):
        img = Image.open(img_p)
        stats = ImageStat.Stat(img)
        mean = np.array(stats.mean) / 255
        std = np.array(stats.stddev) / 255
        mean_list.append(mean)
        std_list.append(std)

    means = np.array(mean_list).mean(axis=0)
    stds = np.array(std_list).mean(axis=0)

    return means, stds


def get_test_transform(mean, std):
    return albumentations.Compose(
        [Normalize(mean=mean, std=std), ToTensorV2()],
        bbox_params=albumentations.BboxParams(
            format="pascal_voc", label_fields=["labels"]
        ),
    )


def do_training(model, torch_dataset, torch_dataset_test, num_epochs):
    data_loader = DataLoader(
        torch_dataset, batch_size=8, shuffle=True, collate_fn=utils.collate_fn
    )
    data_loader_test = DataLoader(
        torch_dataset_test, batch_size=2, shuffle=False, collate_fn=utils.collate_fn
    )

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("Using device {}".format(device))
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    for epoch in range(num_epochs):
        train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10)
        lr_scheduler.step()
        evaluate(model, data_loader_test, device)


def main(args):
    train_dir = "coco/train/data"
    train_coco = "coco/train/labels.json"
    test_dir = "coco/test/data"
    test_coco = "coco/test/labels.json"

    print("Calculating mean and std for training and testing images!")
    train_mean, train_std = calc_norm_sats(train_dir)
    test_mean, tests_std = calc_norm_sats(test_dir)

    dataset = torchDataset(
        root=train_dir,
        annotations=train_coco,
        transforms=get_train_transform(train_mean, train_std),
    )
    test_dataset = torchDataset(
        root=test_dir,
        annotations=test_coco,
        transforms=get_test_transform(test_mean, tests_std),
    )

    num_classes = 6
    model = get_model(num_classes, args.pretrained_model)

    do_training(model, dataset, test_dataset, args.epochs)

    if not os.path.isdir("models/rcnn"):
        os.mkdir("models/rcnn")

    torch.save(
        model.state_dict(),
        "models/rcnn/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S") + ".pth",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a model")

    parser.add_argument(
        "-e", type=int, dest="epochs", default=1, help="Number of epochs to train",
    )
    parser.add_argument(
        "--pretrained_model",
        type=bool,
        help="Use pretrained model weights for training",
        default=True,
    )

    args = parser.parse_args()
    main(args)
