#!/bin/bash

# Requirements:
# 	pytorch: 1.4.0
# 	neural_render: 1.1.3

# preliminary installs
# conda update -n base -c defaults conda
# sudo apt update
# sudo apt upgrade
# conda update anaconda-navigator  
# conda update navigator-updater 

# opencv install
# sudo apt install python3-opencv
pip install opencv-python

# pytorch and neural_render
# https://pytorch.org/get-started/previous-versions/#v140
conda install pytorch==1.4.0 torchvision==0.5.0 cudatoolkit=10.1 -c pytorch
pip install neural-renderer-pytorch==1.1.3
conda install -c conda-forge tqdm
conda install -c anaconda chainer
