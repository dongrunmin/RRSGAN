#!/bin/sh

ROOT=../..
export PYTHONPATH=$ROOT:$PYTHONPATH


python -u train.py -opt options/RRSNet.yml

