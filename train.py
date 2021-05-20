import argparse
import math

import fiftyone as fo
import fiftyone.utils.coco as fouc
import torch
import torchvision
from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from references.engine import train_one_epoch, evaluate
from references import utils
from datetime import datetime


def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    return model


class TorchDataset(Dataset):
    def __init__(
        self, fiftyone_dataset, transforms=None, gt_field="ground_truth", classes=None,
    ):
        self.samples = fiftyone_dataset
        self.transforms = transforms
        self.gt_field = gt_field
        self.img_paths = self.samples.values("filepath")

        self.classes = classes
        if not self.classes:
            self.classes = self.samples.distinct("%s.detections.label" % gt_field)
        if self.classes[0] != "background":
            self.classes = ["background"] + self.classes

        self.labels_map_rev = {c: i for i, c in enumerate(self.classes)}

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        sample = self.samples[img_path]
        metadata = sample.metadata

        img = Image.open(img_path)

        boxes = []
        labels = []
        areas = []
        iscrowd = []
        detections = sample[self.gt_field].detections
        for det in detections:
            coco_obj = fouc.COCOObject.from_detection(
                det, metadata, self.labels_map_rev
            )
            x, y, w, h = coco_obj.bbox
            boxes.append([x, y, x + w, y + h])
            labels.append(coco_obj.category_id)
            areas.append(coco_obj.area)
            iscrowd.append(coco_obj.iscrowd)

        # Collect
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)
        area = torch.as_tensor(areas, dtype=torch.float32)
        image_id = torch.as_tensor([index])
        iscrowd = torch.as_tensor(iscrowd, dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "area": area,
            "iscrowd": iscrowd,
            "image_id": image_id,
        }

        if self.transforms is not None:
            img = self.transforms(img)

        return img, target

    def __len__(self):
        return len(self.img_paths)

    def get_classes(self):
        return self.classes


def do_training(model, torch_dataset, torch_dataset_test, num_epochs):
    data_loader = DataLoader(
        torch_dataset, batch_size=2, shuffle=True, collate_fn=utils.collate_fn
    )
    data_loader_test = DataLoader(
        torch_dataset_test, batch_size=1, shuffle=False, collate_fn=utils.collate_fn
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
    dataset = fo.Dataset.from_dir(
        args.dataset_folder, fo.types.COCODetectionDataset, name="histpath-dataset",
    )

    train_transforms = transforms.Compose(
        [transforms.ToTensor(), transforms.RandomHorizontalFlip(0.5)]
    )
    test_transforms = transforms.Compose([transforms.ToTensor()])

    train_len = math.floor(len(dataset) * args.split)

    train_split = dataset.take(train_len)
    test_split = dataset.exclude(train_split)

    torch_dataset = TorchDataset(train_split, train_transforms)
    torch_dataset_test = TorchDataset(test_split, test_transforms)

    num_classes = len(torch_dataset.get_classes())
    model = get_model(num_classes)

    do_training(model, torch_dataset, torch_dataset_test, args.epochs)

    torch.save(
        model.state_dict(),
        "models/rcnn/" + datetime.now().strftime("%d_%m_%Y_%H_%M_%S"),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Trains a model")
    parser.add_argument(
        "dataset_folder", type=str, help="Directory where training dataset live",
    )

    parser.add_argument(
        "-s",
        type=float,
        dest="split",
        required=True,
        help="Percentage of training images",
    )

    parser.add_argument(
        "-e", type=int, dest="epochs", default=1, help="Number of epochs to train",
    )

    args = parser.parse_args()
    main(args)
