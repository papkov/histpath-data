import argparse

import fiftyone as fo
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F


def get_model(device, weights_path, num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=False)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_fo_dataset(dir, name):
    if not fo.dataset_exists(name):
        dataset = fo.Dataset.from_dir(dir, fo.types.COCODetectionDataset, name)
        dataset.persistent = True
    else:
        dataset = fo.load_dataset(name)
    return dataset


def main(args):
    dataset = load_fo_dataset(args.dataset_dir, args.name)
    print("Dataset loaded!")
    print(dataset)

    classes = dataset.default_classes
    num_classes = len(classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(device, args.weights_path, num_classes)

    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            # Load image
            image = Image.open(sample.filepath)
            image = F.to_tensor(image).to(device)
            c, h, w = image.shape

            preds = model([image])[0]
            labels = preds["labels"].cpu().detach().numpy()
            scores = preds["scores"].cpu().detach().numpy()
            boxes = preds["boxes"].cpu().detach().numpy()

            detections = []
            for label, score, box in zip(labels, scores, boxes):
                x1, y1, x2, y2 = box
                rel_box = [
                    x1 / w,
                    y1 / h,
                    (x2 - x1) / w,
                    (y2 - y1) / h,
                ]  # fiftyone format

                detections.append(
                    fo.Detection(
                        label=classes[label], bounding_box=rel_box, confidence=score
                    )
                )

            sample["faster_rcnn"] = fo.Detections(detections=detections)
            sample.save()

    print("Finished adding predictions")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add predictions to dataset with selected model!"
    )

    parser.add_argument("dataset_dir", type=str, help="Directory of fiftyone dataset")

    parser.add_argument("name", type=str, help="Created/existing fiftyone dataset name")

    parser.add_argument(
        "weights_path", type=str, help="Location of Faster-RCNN model weights",
    )

    args = parser.parse_args()
    main(args)
