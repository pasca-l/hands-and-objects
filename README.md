# Hands and Objects

## Project overview
This project holds codes for training models to accomplish certain tasks. Using egocentric video inputs, the motivation of these tasks is to understand the camera wearer's activity in terms of hand and object interaction. [^ego4d]

[^ego4d]: K. Grauman, et al. [Ego4d: Around the world in 3,000 hours of egocentric video](https://arxiv.org/pdf/2110.07058.pdf). arXiv preprint arXiv:2110.07058, 2021.

## Tasks
1. PNR temporal localization
> From a video clip, the objective is to find the key frame where the object experiences the point of no return (PNR), in other words, the point at which the state of the object cannot be reversed.

2. State change object detection
> The objective is to determine the object that is undergoing a state change. 3 frames, PNR frame, and frames one before and one after the PNR point are given as input.