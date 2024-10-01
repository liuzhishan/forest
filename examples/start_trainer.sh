#!/bin/bash

source ./util.sh
root=$(dirname `pwd`)

# add trainerpath to python path
export PYTHONPATH=$root/trainer/trainer:$PYTHONPATH

ts=`date +%Y_%m_%d_%H_%M_%S`

trainer_list=$(echo $TRAIN | sed -e "s/\.[^,]*,/:${MY_GPU},/g")
trainer_list=$(echo $trainer_list | sed -e "s/\.[^,]*$/:${MY_GPU}/g")

trainer=$TRAIN
trainer_names=(${trainer//,/ })
trainer_num=${#trainer_names[@]}
total_gpu=$((trainer_num * MY_GPU))

if [ $trainer_num == 1 ]; then
    trainer_list="localhost:${MY_GPU}"
fi

echo "total_gpu: "${total_gpu}", trainer_list: "${trainer_list}

horovodrun -np ${total_gpu} -H ${trainer_list} python demo_train.py train_and_validate dsp_ctr_lzs_test_v5.json > log/dsp_ctr_lzs_test_v5_trainer_${total_gpu}gpu_${ts}.log 2>&1
