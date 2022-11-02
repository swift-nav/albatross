load("//:copts.bzl", "COPTS")

TAGS = ["manual"]

DEPS = [
    "//:albatross",
    "@gflags",
]

def example(name, srcs, args):
    native.cc_binary(
        name = name,
        srcs = srcs,
        tags = TAGS,
        deps = DEPS,
        copts = COPTS,
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
