# build

`trainer` 需要和 `tensorflow` 一起编译，得到 `so` 供 `python` 使用。使用的版本是 `1.15`。

## mac

`mac` 上编译 `cpu` 版本。

    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow
    
    git checkout v1.15.0

    brew install bazel
