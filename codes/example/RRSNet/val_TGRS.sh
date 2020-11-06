ROOT=../..
export PYTHONPATH=$ROOT:$PYTHONPATH

partition=Test
your_model=
dataset=val_1st

srun -p ${partition} --gres=gpu:1 python -u val.py \
  --model_RRSNet_path=${your_model} \
  --exp_name=RRSNet \
  --dataset=${dataset} \
  --save_path=./results


