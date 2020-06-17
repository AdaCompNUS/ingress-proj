#!/bin/bash

# CUDA Setup
GPU_IDX=0

# docker run --env CUDA_VISIBLE_DEVICES=$GPU_IDX --env ROS_MASTER_URI=$MASTER_URI --env ROS_IP=$IP --runtime=nvidia -it --rm --net host mohito/ingress:v1.0 /bin/bash

## new version of nvidia docker
docker run --env CUDA_VISIBLE_DEVICES=$GPU_IDX --env ROS_MASTER_URI=$MASTER_URI --env ROS_IP=$IP --gpus all -it --net host --privileged adacompnus/ingress:v1.1 /bin/bash
