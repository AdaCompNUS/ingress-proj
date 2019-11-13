#!/bin/bash

## Setup ROS
# MASTER_URI=http://127.0.0.1:11311
# IP=127.0.0.1

## Example
# MASTER_URI=http://192.168.0.32:11311
# IP=192.168.0.32

# CUDA Setup
GPU_IDX=0

# docker run --env CUDA_VISIBLE_DEVICES=$GPU_IDX --env ROS_MASTER_URI=$MASTER_URI --env ROS_IP=$IP --runtime=nvidia -it --rm --net host mohito/ingress:v1.0 /bin/bash

## new version of nvidia docker
docker run --env CUDA_VISIBLE_DEVICES=$GPU_IDX --env ROS_MASTER_URI=$MASTER_URI --env ROS_IP=$IP --gpus all -it --net host --privileged mohito/ingress:v1.0 /bin/bash
