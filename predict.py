import argparse

import fiftyone as fo
import torch
import torchvision
from PIL import Image
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.transforms import functional as F
from torchvision.ops import nms


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


def add_predictions(dataset, sample_label, device, model):
    classes = dataset.default_classes
    with fo.ProgressBar() as pb:
        for sample in pb(dataset):
            # Load image
            image = Image.open(sample.filepath)
            image = F.to_tensor(image).to(device)
            c, h, w = image.shape

            preds = model([image])[0]

            # Non-Max suppression.
            # Indices to keep.
            idx = nms(boxes=preds["boxes"], scores=preds["scores"], iou_threshold=0.1)
            boxes = preds["boxes"][idx]
            labels = preds["labels"][idx]
            scores = preds["scores"][idx]

            labels = labels.cpu().detach().numpy()
            scores = scores.cpu().detach().numpy()
            boxes = boxes.cpu().detach().numpy()

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

            sample[sample_label] = fo.Detections(detections=detections)
            sample.save()


def main(args):
    dataset = load_fo_dataset(args.dataset_dir, args.name)
    print("Dataset loaded!")
    print(dataset)
    generate_predictions = True

    # Dont overwrite predictions if they exists with same label.
    label_exists = dataset.has_sample_field(args.label)
    if label_exists:
        print("Dataset has allready predictions with label: {}".format(args.label))
        u_input = input("Would you like to generate new predictions? (y/n) : ")
        if str.lower(u_input) == "n":
            print("Skipping adding predictions!")
            generate_predictions = False

    classes = dataset.default_classes
    num_classes = len(classes)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = get_model(device, args.weights_path, num_classes)

    if generate_predictions:
        add_predictions(dataset, args.label, device, model)
        print("Finished adding predictions")

    results = fo.evaluate_detections(dataset, args.label, compute_mAP=True)
    print("mAP: {}".format(results.mAP()))
    results.print_report()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Add predictions to dataset with selected model!"
    )

    parser.add_argument("dataset_dir", type=str, help="Directory of fiftyone dataset")

    parser.add_argument("name", type=str, help="Created/existing fiftyone dataset name")

    parser.add_argument(
        "weights_path", type=str, help="Location of Faster-RCNN model weights",
    )

    parser.add_argument(
        "label", type=str, help="Label for created predictions",
    )

    args = parser.parse_args()
    main(args)
