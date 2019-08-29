# Detect & Track
## Overview
From traffic video footages from the roundabout obtained through UAV. We determine the direction where the ehicles moving. To count the vehicles, it must be tracked across a line. This means the object must be tracked prior to crossing the line. Lines must be placed in such a way that allows the object to cross.

Also, a unique ID to each tracked object is necessary for counting objects and to monitor their performance. Intersection Over Union (IOU) – which is a metric to find the degree of overlap between two shapes was used. Using IOU we calculated the overlap ratio between an object’s bounding with all the other existing bounding boxes. If the IOU greater than 95% we assumed that it was the same object. Making us possible to assign a unique id to detected objects across multiple numbers of frames.

## Getting Started

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 

### Structure
- **aerial_pedestrain_detection**: Has training scripts 
- **tracking**: Has scripts for tracking 
- **preprocessing**: Has scripts that converts data into the required format (path,x1,y1,x2,y2,label)


### Installing
- `git clone git@github.com:nishontan/vechical-detect-and-track.git`
- `conda create --name detect-and-track tensorflow-gpu`
- `conda activate detect-and-track`
- `cd vechical-detect-and-track/aerial_pedestrian_detection/`
- `pip install .`


## Verifying Installation
Run `python aerial_pedestrian_detection/keras_retinanet/bin/train.py -h`
It should show
```
Using TensorFlow backend.
usage: train.py [-h]
                [--snapshot SNAPSHOT | --imagenet-weights | --weights WEIGHTS | --no-weights]
                [--backbone BACKBONE] [--batch-size BATCH_SIZE] [--gpu GPU]
                [--multi-gpu MULTI_GPU] [--multi-gpu-force] [--epochs EPOCHS]
                [--steps STEPS] [--lr LR] [--snapshot-path SNAPSHOT_PATH]
                [--tensorboard-dir TENSORBOARD_DIR] [--no-snapshots]
                [--no-evaluation] [--freeze-backbone] [--random-transform]
                [--image-min-side IMAGE_MIN_SIDE]
                [--image-max-side IMAGE_MAX_SIDE] [--config CONFIG]
                [--weighted-average] [--workers WORKERS]
                [--max-queue-size MAX_QUEUE_SIZE]
                {coco,pascal,kitti,oid,csv} ...

Simple training script for training a RetinaNet network.

positional arguments:
  {coco,pascal,kitti,oid,csv}
                        Arguments for specific dataset types.

optional arguments:
  -h, --help            show this help message and exit
  --snapshot SNAPSHOT   Resume training from a snapshot.
  --imagenet-weights    Initialize the model with pretrained imagenet weights.
                        This is the default behaviour.
  --weights WEIGHTS     Initialize the model with weights from a file.
  --no-weights          Don't initialize the model with any weights.
  --backbone BACKBONE   Backbone model used by retinanet.
  --batch-size BATCH_SIZE
                        Size of the batches.
  --gpu GPU             Id of the GPU to use (as reported by nvidia-smi).
  --multi-gpu MULTI_GPU
                        Number of GPUs to use for parallel processing.
  --multi-gpu-force     Extra flag needed to enable (experimental) multi-gpu
                        support.
  --epochs EPOCHS       Number of epochs to train.
  --steps STEPS         Number of steps per epoch.
  --lr LR               Learning rate.
  --snapshot-path SNAPSHOT_PATH
                        Path to store snapshots of models during training
                        (defaults to './snapshots')
  --tensorboard-dir TENSORBOARD_DIR
                        Log directory for Tensorboard output
  --no-snapshots        Disable saving snapshots.
  --no-evaluation       Disable per epoch evaluation.
  --freeze-backbone     Freeze training of backbone layers.
  --random-transform    Randomly transform image and annotations.
  --image-min-side IMAGE_MIN_SIDE
                        Rescale the image so the smallest side is min_side.
  --image-max-side IMAGE_MAX_SIDE
                        Rescale the image if the largest side is larger than
                        max_side.
  --config CONFIG       Path to a configuration parameters .ini file.
  --weighted-average    Compute the mAP using the weighted average of
                        precisions among classes.
  --workers WORKERS     Number of multiprocessing workers. To disable
                        multiprocessing, set workers to 0
  --max-queue-size MAX_QUEUE_SIZE
                        Queue length for multiprocessing workers in fit
                        generator.
```


## Debugging Installation Issues
- `ImportError: aerial_pedestrian_detection/keras_retinanet/bin/../../keras_retinanet/utils/compute_overlap.so: undefined symbol: _Py_ZeroStruct`
- **Run these commands**
	- `cd aerial_pedestrian_detection`
	- `python setup.py build_ext --inplace`
	



## Built With

* [RetinaNET](https://github.com/priya-dwivedi/aerial_pedestrian_detection) - Object Detection
* [OpenCV](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_tutorials.html) - Object Tracking






