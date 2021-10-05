# Data handling and backend for the histopath blogpost

## Setup

```
conda env create -f venv.yml
conda activate histpath-project
jupyter notebook
```
## Usage

>Generating slides

datafolder - location where your .mrxs files live.
image_size - default 1024 square image

Default output is `/output` folder for generated images and annotations.

`python3 generate_slides.py <datafolder>`

This creates a folder with a structure that can be viewed with fiftyone - fiftyone dataset.

view:
`fiftyone app view --dataset-dir output/ --type fiftyone.types.COCODetectionDataset`

This enables to look at ground truth and a check if annotations are correctly implemented.

> Splitting training and testing datasets

The default implementation will find the `output` folder and generate a `coco` folder with train/test split with 80% of training images.

`python3 split_coco.py`


> Training classifier

Training loop expects to receive training images from `coco/train` folder and test images from `coco/test` folder. Folder structure is default fiftyone dataset folder structure as was for `output` folder.

Default command will train 1 epoch with FastRCNNPredictor default weights and save a model in `models/` directory with a name generated from the time the training finishes.

`python3 train.py`

> Adding a dataset to predict on.

You could use another split or whole image to predict values on.

To predict on a second image generate dataset with `generate_slides.py <dir_to_another_image_folder> --output_dir test_dataset/`

This will generate images and annotations based of another image and save into `test_dataset/` folder.

You can add dataset to fiftyone dataset lists with command

`fiftyone datasets create -n mydataset -d test_dataset/ -t fiftyone.types.COCODetectionDataset`


> Predicting on a dataset.

To view not only ground truth, but also predictions that model makes you can add predictions into fiftyone dataset.

The weights are the weights that were saved after running `train.py` script and should live inside `models/` directory.

Prediction label should be something clear that predictions are annotated with ex, <model_name_predictions> so you could compare multiple models as the annotations can be switched on and off based on the label.

`python3 predict.py test_dataset/ mydataset <weights.pth> <prediction_label>`

In the end the precision, recall, f1-score and support values are displayed for all classes.

> Viewing predictions


To launch a fiftyone app in a browser:

`fiftyone app launch`

In upper menu you should see a `select a dataset` button and should be able to see a `mydataset`.

Labels have now two options to select from - `ground_truth` and your `<prediction_label>`

# Next steps

This project was done in a neural networks course in Tartu University. The time limit although didn't allow me to improve the model as most of the time went to generate a system that allowed to handle WSI images - split images into datasets, train and visualize results. The next step would be to improve the model as the model currently is just out of the box implementation. This repo however allows to modify model and visualize results in few steps and that makes playing with different solutions much faster.

