licenses(["notice"])

filegroup(
    name = "LICENSE",
    visibility = ["//visibility:hidden"],
)

cc_library(
    name = "crypto",
    linkopts = ["-lcrypto"],
    visibility = ["//visibility:hidden"],
)

cc_library(
    name = "ssl",
    linkopts = ["-lssl"],
    visibility = ["//visibility:hidden"],
    deps = [
        ":crypto",
    ],
)
