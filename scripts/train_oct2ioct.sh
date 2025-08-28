#!/bin/bash
# train a model to segment abdominal MRI (T2 fold of CHAOS challenge)
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='AROIDuke_ONE_block_paste'
NWORKER=16
RUNS=1
ALL_EV=(0) # 5-fold cross validation (0, 1, 2, 3, 4)
###### Training configs ######
NSTEP=12000
DECAY=0.98

MAX_ITER=3000 # defines the size of an epoch
SNAPSHOT_INTERVAL=3000 # interval for saving snapshot
SEED=2025

echo ========================================================================

for EVAL_FOLD in "${ALL_EV[@]}"
do
  PREFIX="train_${DATASET}_cv${EVAL_FOLD}"
  echo $PREFIX
  LOGDIR="./exps/train_on_${DATASET}_cdfs"

  if [ ! -d $LOGDIR ]
  then
    mkdir -p $LOGDIR
  fi

  python3 train.py with \
  mode='train' \
  dataset=$DATASET \
  num_workers=$NWORKER \
  n_steps=$NSTEP \
  eval_fold=$EVAL_FOLD \
  max_iters_per_load=$MAX_ITER \
  seed=$SEED \
  save_snapshot_every=$SNAPSHOT_INTERVAL \
  lr_step_gamma=$DECAY \
  path.log_dir=$LOGDIR
done
