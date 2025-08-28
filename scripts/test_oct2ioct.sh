#!/bin/bash
# test a model to segment abdominal/cardiac MRI
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
ALL_DATASET=("horizontal" "vertical" "erm")
SHOTS=1
NWORKER=16
RUNS=1
TEST_LABEL=[0,1,2,3]
ALL_EV=(0) # (0 1 2 3 4)
###### Training configs ######
NSTEP=39001
DECAY=0.98

MAX_ITER=3000 # defines the size of an epoch
SNAPSHOT_INTERVAL=3000 # interval for saving snapshot
SEED=2025

echo ========================================================================
for EVAL_FOLD in "${ALL_EV[@]}"
do
  for DATASET in "${ALL_DATASET[@]}"
  do
    PREFIX="test_${DATASET}_cv${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./results"

    if [ ! -d $LOGDIR ]
    then
      mkdir -p $LOGDIR
    fi

    # RELOAD_PATH='please feed the absolute path to the trained weights here' # path to the reloaded model
    RELOAD_MODEL_PATH="./weights/12000.pth"
    python3 test.py with \
    mode="test" \
    dataset=$DATASET \
    n_shot=$SHOTS\
    num_workers=$NWORKER \
    n_steps=$NSTEP \
    test_label=$TEST_LABEL \
    eval_fold=$EVAL_FOLD \
    max_iters_per_load=$MAX_ITER \
    seed=$SEED \
    reload_model_path=$RELOAD_MODEL_PATH \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR
  done
done
