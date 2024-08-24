# build

`trainer` 需要和 `tensorflow` 一起编译，得到 `so` 供 `python` 使用。使用的版本是最新的 `master`, `commit` 是
`a3dbab6eaf0b14892acd35f1b9e8b281a37cd943`, 对应 `v2.17.0`。

## mac

`mac` 上编译 `cpu` 版本。

    git clone https://github.com/tensorflow/tensorflow.git
    cd tensorflow

    brew install bazel

    ./configure
    
    cd "/usr/local/Cellar/bazel/7.3.1/libexec/bin" && curl -fLO https://releases.bazel.build/6.5.0/release/bazel-6.5.0-darwin-x86_64 && chmod +x bazel-6.5.0-darwin-x86_64

    bazel clean --expunge 
    sudo xcode-select -s /Applications/Xcode.app/Contents/Developer
    sudo xcodebuild -license
    bazel clean --expunge 
    bazel build --config=opt //tensorflow/tools/pip_package:build_pip_package.py
