#!/usr/bin/env sh
set -e

export PATH=/root/miniconda3/bin:/home/hadoop/software/hive/bin:$PATH

export HADOOP_HOME=/home/hadoop/software/hadoop
export HIVE_HOME=/home/hadoop/software/hive
export JAVA_HOME=/home/hadoop/software/java
export PATH=$PATH:$JAVA_HOME/bin:$HADOOP_HOME/bin:$HADOOP_HOME/sbin:$HIVE_HOME/bin:$SPARK_HOME/bin:$SPARK_HOME/sbin:$PATH

source ${HADOOP_HOME}/libexec/hadoop-config.sh
export CLASSPATH=$(${HADOOP_HOME}/bin/hadoop classpath --glob)

export LD_LIBRARY_PATH=/home/hadoop/software/hadoop/lib/native:${LD_LIBRARY_PATH}:${JAVA_HOME}/jre/lib/amd64/server:/usr/local/lib
export LIBRARY_PATH=/home/hadoop/software/hadoop/lib/native/:/usr/local/include/:/usr/lib/x86_64-linux-gnu/:/usr/include:$LIBRARY_PATH:/usr/local/lib

root=$(dirname `pwd`)