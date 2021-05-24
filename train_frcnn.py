import argparse
import math

from pycocotools.coco import COCO
import os
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from references.engine import train_one_epoch, evaluate
from references import utils
from references import transforms as T
from datetime import datetime


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
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
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(coco_annotation[i]["category_id"])
            areas.append(coco_annotation[i]["area"])

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

        if self.transforms is not None:
            img, target = self.transforms(img, target)

        return img, target

    def __len__(self):
        return len(self.ids)


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


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
    train_dir = "coco/train2017"
    train_coco = "coco/annotations/instances_train2017.json"
    test_dir = "coco/val2017"
    test_coco = "coco/annotations/instances_val2017.json"

    dataset = torchDataset(
        root=train_dir, annotations=train_coco, transforms=get_transform(train=True)
    )
    test_dataset = torchDataset(
        root=test_dir, annotations=test_coco, transforms=get_transform(train=False)
    )

    num_classes = 6
    model = get_model(num_classes)

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

    args = parser.parse_args()
    main(args)
