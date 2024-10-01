#!/bin/bash

source ./util.sh
root=$(dirname `pwd`)

# add trainerpath to python path
export PYTHONPATH=$root/trainer/trainer:$PYTHONPATH

ts=`date +%Y_%m_%d_%H_%M_%S`

horovodrun -np 1 -H localhost:0 python demo_train.py train_and_validate dsp_ctr_lzs_test_v5.json > log/dsp_ctr_lzs_test_v5_trainer_$ts.log 2>&1