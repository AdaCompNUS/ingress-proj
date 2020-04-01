#!/bin/bash


# SETTINGS ---------------

# Setup ROS 
# export ROS_MASTER_URI=http://localhost:11311
# export ROS_IP=localhost

# Ingress
DISAMBIGUATE=true # If true, generate questions for disambiguation. Takes more time.

# ------------------------

# Start self-referential server 
cd ~/Programs/densecap
qlua localize_server.lua &

# Start relational server
cd ~/Programs/refexp
if [ "$DISAMBIGUATE" = false ] ; then
	python lib/comprehension_test.py --coco_path ./coco --dataset UNC_RefExp --exp_name mil_context_withNegMargin --split_name val --proposal_source gt
else
	python lib/comprehension_test.py --coco_path ./coco --dataset UNC_RefExp --exp_name mil_context_withNegMargin --split_name val --proposal_source gt --disambiguate
fi
