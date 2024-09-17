#!/usr/bin/env bash

set -e

script=$(readlink -f "$0")
scriptpath=$(dirname "$script")

trainer_path=$(readlink -f "$scriptpath/../trainer")

cd ${trainer_path}

echo "cd to trainer_path: ${trainer_path}"

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
# cp bazel-bin/core/operators/trainer_ops.so trainer

# cd trainer
# for x in `ls python`; do
#    if [ ! -L $x ]; then
#        ln -s python/$x .
#    fi
# done
