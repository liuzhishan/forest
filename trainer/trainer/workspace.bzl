# Borrowed from tensorflow
load("//third_party:repo.bzl", "trainer_http_archive")
load("//third_party/systemlibs:syslibs_configure.bzl", "syslibs_configure")
load("//third_party/py:python_configure.bzl", "python_configure")

def initialize_third_party():
    """ Load third party repositories.  See above load() statements. """

# Sanitize a dependency so that it works correctly from code that includes
# Trainer as a submodule.
def clean_dep(dep):
    return str(Label(dep))

def trainer_workspace():
    # Note that we check the minimum bazel version in WORKSPACE.
    syslibs_configure(name = "local_config_syslibs")
    python_configure(name = "local_config_python")

    initialize_third_party()

    trainer_http_archive(
        name = "com_google_absl",
        build_file = clean_dep("//third_party:com_google_absl.BUILD"),
        sha256 = "acd93f6baaedc4414ebd08b33bebca7c7a46888916101d8c0b8083573526d070",
        strip_prefix = "abseil-cpp-43ef2148c0936ebf7cb4be6b19927a9d9d145b8f",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/abseil/abseil-cpp/archive/43ef2148c0936ebf7cb4be6b19927a9d9d145b8f.tar.gz",
            "https://github.com/abseil/abseil-cpp/archive/43ef2148c0936ebf7cb4be6b19927a9d9d145b8f.tar.gz",
        ],
    )

    trainer_http_archive(
        name = "com_google_googletest",
        sha256 = "ff7a82736e158c077e76188232eac77913a15dac0b22508c390ab3f88e6d6d86",
        strip_prefix = "googletest-b6cd405286ed8635ece71c72f118e659f4ade3fb",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
            "https://github.com/google/googletest/archive/b6cd405286ed8635ece71c72f118e659f4ade3fb.zip",
        ],
    )

    trainer_http_archive(
        name = "com_github_gflags_gflags",
        sha256 = "ae27cdbcd6a2f935baa78e4f21f675649271634c092b1be01469440495609d0e",
        strip_prefix = "gflags-2.2.1",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/gflags/gflags/archive/v2.2.1.tar.gz",
            "https://github.com/gflags/gflags/archive/v2.2.1.tar.gz",
        ],
    )

    # https://github.com/bazelbuild/bazel-skylib/releases
    trainer_http_archive(
        name = "bazel_skylib",
        sha256 = "1dde365491125a3db70731e25658dfdd3bc5dbdfd11b840b3e987ecf043c7ca0",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
            "https://github.com/bazelbuild/bazel-skylib/releases/download/0.9.0/bazel_skylib-0.9.0.tar.gz",
        ],
    )

    # 310ba5ee72661c081129eb878c1bbcec936b20f0 is based on 3.8.0 with a fix for protobuf.bzl.
    PROTOBUF_URLS = [
        "https://storage.googleapis.com/mirror.tensorflow.org/github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz",
        "https://github.com/protocolbuffers/protobuf/archive/310ba5ee72661c081129eb878c1bbcec936b20f0.tar.gz",
    ]
    PROTOBUF_SHA256 = "b9e92f9af8819bbbc514e2902aec860415b70209f31dfc8c4fa72515a5df9d59"
    PROTOBUF_STRIP_PREFIX = "protobuf-310ba5ee72661c081129eb878c1bbcec936b20f0"

    # protobuf depends on @zlib, it has to be renamed to @zlib_archive because "zlib" is already
    # defined using bind for grpc.
    PROTOBUF_PATCH = "//third_party/protobuf:protobuf.patch"

    # We need to import the protobuf library under the names com_google_protobuf
    # and com_google_protobuf_cc to enable proto_library support in bazel.
    # Unfortunately there is no way to alias http_archives at the moment.
    trainer_http_archive(
        name = "com_google_protobuf",
        sha256 = PROTOBUF_SHA256,
        strip_prefix = PROTOBUF_STRIP_PREFIX,
        system_build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
        },
        urls = PROTOBUF_URLS,
    )

    trainer_http_archive(
        name = "com_google_protobuf_cc",
        sha256 = PROTOBUF_SHA256,
        strip_prefix = PROTOBUF_STRIP_PREFIX,
        system_build_file = clean_dep("//third_party/systemlibs:protobuf.BUILD"),
        system_link_files = {
            "//third_party/systemlibs:protobuf.bzl": "protobuf.bzl",
        },
        urls = PROTOBUF_URLS,
    )

    # WARNING: make sure ncteisen@ and vpai@ are cc-ed on any CL to change the below rule
    trainer_http_archive(
        name = "grpc",
        sha256 = "67a6c26db56f345f7cee846e681db2c23f919eba46dd639b09462d1b6203d28c",
        strip_prefix = "grpc-4566c2a29ebec0835643b972eb99f4306c4234a3",
        system_build_file = clean_dep("//third_party/systemlibs:grpc.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/grpc/grpc/archive/4566c2a29ebec0835643b972eb99f4306c4234a3.tar.gz",
            "https://github.com/grpc/grpc/archive/4566c2a29ebec0835643b972eb99f4306c4234a3.tar.gz",
        ],
    )

    trainer_http_archive(
        name = "com_github_nanopb_nanopb",
        sha256 = "8bbbb1e78d4ddb0a1919276924ab10d11b631df48b657d960e0c795a25515735",
        build_file = "@grpc//third_party:nanopb.BUILD",
        strip_prefix = "nanopb-f8ac463766281625ad710900479130c7fcb4d63b",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/nanopb/nanopb/archive/f8ac463766281625ad710900479130c7fcb4d63b.tar.gz",
            "https://github.com/nanopb/nanopb/archive/f8ac463766281625ad710900479130c7fcb4d63b.tar.gz",
        ],
    )

    trainer_http_archive(
        name = "boringssl",
        sha256 = "1188e29000013ed6517168600fc35a010d58c5d321846d6a6dfee74e4c788b45",
        strip_prefix = "boringssl-7f634429a04abc48e2eb041c81c5235816c96514",
        system_build_file = clean_dep("//third_party/systemlibs:boringssl.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
            "https://github.com/google/boringssl/archive/7f634429a04abc48e2eb041c81c5235816c96514.tar.gz",
        ],
    )

    trainer_http_archive(
        name = "zlib_archive",
        build_file = clean_dep("//third_party:zlib.BUILD"),
        sha256 = "c3e5e9fdd5004dcb542feda5ee4f0ff0744628baf8ed2dd5d66f8ca1197cb1a1",
        strip_prefix = "zlib-1.2.11",
        system_build_file = clean_dep("//third_party/systemlibs:zlib.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/zlib.net/zlib-1.2.11.tar.gz",
            "https://zlib.net/zlib-1.2.11.tar.gz",
        ],
    )

    trainer_http_archive(
        name = "six_archive",
        build_file = clean_dep("//third_party:six.BUILD"),
        sha256 = "105f8d68616f8248e24bf0e9372ef04d3cc10104f1980f54d57b2ce73a5ad56a",
        strip_prefix = "six-1.10.0",
        system_build_file = clean_dep("//third_party/systemlibs:six.BUILD"),
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
            "https://pypi.python.org/packages/source/s/six/six-1.10.0.tar.gz",
        ],
    )

    trainer_http_archive(
        name = "pybind11",
        urls = [
            "https://mirror.bazel.build/github.com/pybind/pybind11/archive/v2.3.0.tar.gz",
            "https://github.com/pybind/pybind11/archive/v2.3.0.tar.gz",
        ],
        sha256 = "0f34838f2c8024a6765168227ba587b3687729ebf03dc912f88ff75c7aa9cfe8",
        strip_prefix = "pybind11-2.3.0",
        build_file = clean_dep("//third_party:pybind11.BUILD"),
    )

    trainer_http_archive(
        name = "jsoncpp_git",                                                                                                                                                                                                          
        build_file = clean_dep("//third_party:jsoncpp.BUILD"),                       
        sha256 = "8593c1d69e703563d94d8c12244e2e18893eeb9a8a9f8aa3d09a327aa45c8f7d",
        strip_prefix = "jsoncpp-1.9.3",                                              
        system_build_file = clean_dep("//third_party/systemlibs:jsoncpp.BUILD"), 
        urls = [                                                                     
            "http://mirror.tensorflow.org/github.com/open-source-parsers/jsoncpp/archive/1.9.3.tar.gz",
            "https://github.com/open-source-parsers/jsoncpp/archive/1.9.3.tar.gz",
        ],                                                                           
    )

    ##############################################################################
    # BIND DEFINITIONS
    #
    # Please do not add bind() definitions unless we have no other choice.
    # If that ends up being the case, please leave a comment explaining
    # why we can't depend on the canonical build target.

    # gRPC wants a cares dependency but its contents is not actually
    # important since we have set GRPC_ARES=0 in .bazelrc
    native.bind(
        name = "cares",
        actual = "@com_github_nanopb_nanopb//:nanopb",
    )

    # Needed by Protobuf
    native.bind(
        name = "grpc_cpp_plugin",
        actual = "@grpc//:grpc_cpp_plugin",
    )
    native.bind(
        name = "grpc_python_plugin",
        actual = "@grpc//:grpc_python_plugin",
    )

    native.bind(
        name = "grpc_lib",
        actual = "@grpc//:grpc++",
    )

    native.bind(
        name = "grpc_lib_unsecure",
        actual = "@grpc//:grpc++_unsecure",
    )

    # Needed by gRPC
    native.bind(
        name = "libssl",
        actual = "@boringssl//:ssl",
    )

    # Needed by gRPC
    native.bind(
        name = "nanopb",
        actual = "@com_github_nanopb_nanopb//:nanopb",
    )

    # Needed by gRPC
    native.bind(
        name = "protobuf",
        actual = "@com_google_protobuf//:protobuf",
    )

    # gRPC expects //external:protobuf_clib and //external:protobuf_compiler
    # to point to Protobuf's compiler library.
    native.bind(
        name = "protobuf_clib",
        actual = "@com_google_protobuf//:protoc_lib",
    )

    # Needed by gRPC
    native.bind(
        name = "protobuf_headers",
        actual = "@com_google_protobuf//:protobuf_headers",
    )

    # Needed by Protobuf
    native.bind(
        name = "python_headers",
        actual = clean_dep("//third_party/python_runtime:headers"),
    )

    # Needed by Protobuf
    native.bind(
        name = "six",
        actual = "@six_archive//:six",
    )

    # Needed by gRPC
    native.bind(
        name = "zlib",
        actual = "@zlib_archive//:zlib",
    )
