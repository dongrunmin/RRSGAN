#!/bin/sh

ROOT=../..
export PYTHONPATH=$ROOT:$PYTHONPATH

partition=Test

TASK_NUM=8
srun -p ${partition} -n${TASK_NUM} --gres=gpu:8 --ntasks-per-node=8 \
    --job-name=RRSGAN --kill-on-bad-exit=1 \
python -u train.py \
  -opt options/RRSGAN.yml \
  --launcher slurm


