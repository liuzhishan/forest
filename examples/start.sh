#!/bin/bash

export HADOOP_USER_NAME=ad
export JAVA_HOME=/home/hadoop/software/java
export HADOOP_HDFS_HOME=/home/hadoop/software/hadoop
export PATH=$PATH:$HADOOP_HDFS_HOME/bin:$HADOOP_HDFS_HOME/sbin
export CLASSPATH=$($HADOOP_HDFS_HOME/bin/hadoop classpath --glob)

workspace=$(dirname `pwd`)
path=$(dirname $workspace)
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$JAVA_HOME/jre/lib/amd64/server:/usr/local/cuda-10.0/extras/CUPTI/lib64:./
export PYTHONPATH=$workspace:$PYTHONPATH

horovodrun -np 1 -H localhost:0 python demo_train.py train_and_validate dsp_ctr_lzs_test_v5.json
