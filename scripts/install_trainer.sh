#!/usr/bin/env bash

set -e

script=$(readlink -f "$0")
scriptpath=$(dirname "$script")

trainer_path=$(readlink -f "$scriptpath/../trainer")

cd ${trainer_path}
echo "cd to trainer_path: ${trainer_path}"

# download tensorflow lib
# reference: https://www.tensorflow.org/install/lang_c?hl=zh-cn
#
# Check if libtensorflow exists
if [ ! -d "/usr/local/lib/libtensorflow" ]; then
    echo "libtensorflow not found. Downloading and extracting..."
    
    # Download libtensorflow
    if [ ! -f libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz ]; then
        wget https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz -O libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
    else
        echo "libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz already exists. Skipping download."
    fi
    
    # Extract to /usr/local
    tar -C /usr/local -xzf libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
    
    # Configure the linker
    ldconfig /usr/local/lib
    
    # Clean up the downloaded tar file
    rm libtensorflow-gpu-linux-x86_64-2.6.0.tar.gz
    
    echo "libtensorflow has been installed."
else
    echo "libtensorflow already exists. Skipping download and extraction."
fi

./configure.sh
bazel clean --expunge

{
    bazel build trainer/core/operators:trainer_ops.so
} || {
    dirname=`ls -htrl /root/.cache/bazel/_bazel_root/ | tail -1 | awk '{print $9}'`
    echo "dirname: ${dirname}"
    sed -i 's/@zlib/@zlib_archive/g' /root/.cache/bazel/_bazel_root/${dirname}/external/com_google_protobuf/BUILD

    bazel build trainer/core/operators:trainer_ops.so
}

# --incompatible_no_support_tools_in_action_inputs=false
cp bazel-bin/trainer/core/operators/trainer_ops.so trainer/trainer

# cd trainer
# for x in `ls python`; do
#    if [ ! -L $x ]; then
#        ln -s python/$x .
#    fi
# done
