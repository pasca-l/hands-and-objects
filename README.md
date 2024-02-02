# Hands and Objects

## Project overview
This project holds codes for training models to accomplish certain tasks. Using egocentric video inputs, the motivation of these tasks is to understand the camera wearer's activity in terms of hand and object interaction. [^ego4d]

[^ego4d]: K. Grauman, et al. [Ego4d: Around the world in 3,000 hours of egocentric video](https://arxiv.org/pdf/2110.07058.pdf). arXiv preprint arXiv:2110.07058, 2021.

## Tasks
1. [Objectness classification](https://github.com/pasca-l/hands-and-objects/tree/main/objectness_classification)
> Inspired by the "state change object detection" task, this task handles object detection in the level of semantic segmentation. The objective is to determine the object that is undergoing a state change at pixel level.

2. [Keypoint estimation](https://github.com/pasca-l/hands-and-objects/tree/main/keypoint_estimation)
> This task incorporates the "object state change classification" and the "PNR temporal localization" task, which handles keyframe detection given a video clip. The objective is to detect the keyframe where the object experiences the point of no return (PNR), in other words, the point at which the state of the object cannot be reversed.

## Usage
To use CUDA version of `pytorch` libraries, use the following command. This replaces the poetry registered CPU-only version to CUDA-version (version 11.8).
```shell
$ make gpu_setting
```

## Datasets
### [Ego4D](https://github.com/facebookresearch/Ego4d)
1. Download [Ego4D CLI](https://github.com/facebookresearch/Ego4d/blob/main/ego4d/cli/README.md).
```shell
$ pip install ego4d
```

2. Run command, below is from the basic usage.
```shell
$ ego4d --output_directory="~/ego4d_data" --datasets full_scale annotations --metadata
```

### [EgoHOS](https://github.com/owenzlz/EgoHOS)
1. Clone repository.
```shell
$ git clone https://github.com/owenzlz/EgoHOS
```

2. Download dataset from script.
```shell
$ bash download_datasets.sh
```

### [PASCAL VOC2012](http://host.robots.ox.ac.uk/pascal/VOC/voc2012)
1. Download dataset from [`Development Kit` section](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/#devkit).

### [Oxford-IIIT Pet](https://www.robots.ox.ac.uk/~vgg/data/pets/)
1. Download dataset from `Downloads` section.
