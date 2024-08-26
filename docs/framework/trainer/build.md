# build

`trainer` 需要和 `tensorflow` 一起编译，得到 `so` 供 `python` 使用。使用的版本是 `v1.15.0`。
必须使用 `python 3.7`。

## mac

### build trainer

1. 安装 `pyenv` 依赖: `brew install bzip2 ncurses zlib`。
2. 设置环境变量:

    export LDFLAGS="-L/usr/local/opt/zlib/lib -L/usr/local/opt/bzip2/lib"
    export CPPFLAGS="-I/usr/local/opt/zlib/include -I/usr/local/opt/bzip2/include"

3. 安装 `pyenv`: `brew install pyenv`。
4. 安装 `python 3.6`: `pyenv install 3.6`。
5. 当前目录使用 `python 3.6`: `pyenv local 3.6.15`。
6. 升级 `pip`: `pip install --upgrade pip`。
7. 安装 `tensorflow 1.15`: `pip install tensorflow==1.15`。
8. 删除部分依赖: `brew uninstall protobuf protobuf-c abseil`。
9. 设置 `c++` 版本为 `c++11`: `tools/build/configure.py` 中设置 `--cxxopt="-std=c++11"\n`。


### `mac` 上编译 `tensorflow cpu` 版本。

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
  
### 编译问题

#### 报错 `error: 'CZString' is missing exception specification 'noexcept'`

    external/jsoncpp_git/src/lib_json/json_value.cpp:262:18: error: 'CZString' is missing exception specification 'noexcept'
    Value::CZString::CZString(CZString&& other)
                    ^
                                                noexcept
    /usr/local/include/json/value.h:266:5: note: previous declaration is here
        CZString(CZString&& other) noexcept;
        ^
    external/jsoncpp_git/src/lib_json/json_value.cpp:288:35: error: 'operator=' is missing exception specification 'noexcept'
    Value::CZString& Value::CZString::operator=(CZString&& other) {
                                      ^
                                                                  noexcept
    /usr/local/include/json/value.h:269:15: note: previous declaration is here
        CZString& operator=(CZString&& other) noexcept;
                  ^

##### 解决方案

https://github.com/onnx/onnx/issues/5532

    brew uninstall protobuf
    
#### 报错 `external/com_google_protobuf/src/google/protobuf/stubs/strutil.cc:1275:19: error: expected unqualified-id`

    external/com_google_protobuf/src/google/protobuf/stubs/strutil.cc:1275:19: error: expected unqualified-id
      } else if (std::isnan(value)) {
                      ^
    /usr/local/include/math.h:165:5: note: expanded from macro 'isnan'
        ( sizeof(x) == sizeof(float)  ? __inline_isnanf((float)(x))          \
        ^
    external/com_google_protobuf/src/google/protobuf/stubs/strutil.cc:1393:19: error: expected unqualified-id
      } else if (std::isnan(value)) {
                      ^
    /usr/local/include/math.h:165:5: note: expanded from macro 'isnan'
        ( sizeof(x) == sizeof(float)  ? __inline_isnanf((float)(x))          \
        
      
##### 解决方案

https://github.com/RcppCore/Rcpp/issues/1160

brew 和 xcode 冲突。重装 xcode。


#### 报错 `no such package '@@zlib//': The repository '@@zlib' could not be resolved: '@@zlib' is not a repository rule`


    ERROR: no such package '@@zlib//': The repository '@@zlib' could not be resolved: '@@zlib' is not a repository rule
    ERROR: /xxxxx/80dd2cb23e8ab2858122c509ba8c6f86/external/com_google_protobuf/BUILD:148:11: no such package '@@zlib//': The repository '@@zlib' could not be resolved: '@@zlib' is not a repository rule and referenced by '@@com_google_protobuf//:protobuf'
    ERROR: Analysis of target '//trainer/core/operators:trainer_ops.so' failed; build aborted: Analysis failed
    
##### 解决方案

手动修改 `/xxxxx/80dd2cb23e8ab2858122c509ba8c6f86/external/com_google_protobuf/BUILD` 中的 `@zlib//:zlib` 为 `@zlib_archive//:zlib`。可能需要修改配置来最终解决，待以后修复。
