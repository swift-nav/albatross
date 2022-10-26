TAGS = ["manual"]

DEPS = [
    "//:albatross",
    "@gflags",
]

COPTS = ["-std=c++14"]

def example(name, srcs, args):
    native.cc_binary(
        name = name,
        srcs = srcs,
        copts = COPTS,
        tags = TAGS,
        deps = DEPS,
    )

    native.sh_binary(
        name = "run-" + name,
        srcs = ["run_example.sh"],
        args = [
            "$(location :" + name + " )",
            args,
        ],
        data = [":" + name],
        tags = TAGS,
    )
