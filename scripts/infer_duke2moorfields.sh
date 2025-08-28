#!/bin/bash
# test a model to segment abdominal/cardiac MRI
GPUID1=0
export CUDA_VISIBLE_DEVICES=$GPUID1

###### Shared configs ######
DATASET='vertical' # 'horizontal' 'vertical'
SHOTS=5
NWORKER=16
RUNS=1
ALL_EV=(3) # (0 1 2 3 4) # 5-fold cross validation (0, 1, 2, 3, 4)
TEST_LABEL=[0,1,2,3]
###### Training configs ######
NSTEP=39001
DECAY=0.98

MAX_ITER=3000 # defines the size of an epoch
SNAPSHOT_INTERVAL=3000 # interval for saving snapshot
SEED=2025

echo ========================================================================
for EVAL_FOLD in "${ALL_EV[@]}"
do
    PREFIX="infer_${DATASET}_cv${EVAL_FOLD}"
    echo $PREFIX
    LOGDIR="./results"

    if [ ! -d $LOGDIR ]
    then
      mkdir -p $LOGDIR
    fi

    # RELOAD_PATH='please feed the absolute path to the trained weights here' # path to the reloaded model
    RELOAD_MODEL_PATH="./exps/exps_train_on_AROIDuke_ONE_block_paste/train_AROIDuke_ONE_block_paste_cv${EVAL_FOLD}/aroiduke_ob_paste_cm_head_FDloss/snapshots/12000.pth"
    VIDEO_PATH="./data/OCT/video/OS-2025-03-18_145615_test.mp4"
    python3 inference.py with \
    mode="infer" \
    dataset=$DATASET \
    n_shot=$SHOTS\
    num_workers=$NWORKER \
    n_steps=$NSTEP \
    eval_fold=$EVAL_FOLD \
    max_iters_per_load=$MAX_ITER \
    test_label=$TEST_LABEL \
    seed=$SEED \
    reload_model_path=$RELOAD_MODEL_PATH \
    save_snapshot_every=$SNAPSHOT_INTERVAL \
    lr_step_gamma=$DECAY \
    path.log_dir=$LOGDIR \
    video_path=$VIDEO_PATH
done






