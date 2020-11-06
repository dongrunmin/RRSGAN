#!/usr/bin/env bash

# You may need to modify the following paths before compiling.

CUDA_HOME=/mnt/lustrenew/dongrunmin/cuda-9.0 \
CUDNN_INCLUDE_DIR=/mnt/lustrenew/dongrunmin/cuda-9.0/include \
CUDNN_LIB_DIR=/mnt/lustrenew/dongrunmin/cuda-9.0/lib64 \
python setup.py build develop
