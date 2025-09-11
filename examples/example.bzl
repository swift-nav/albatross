load("@rules_swiftnav//cc:defs.bzl", "swift_cc_binary")

TAGS = ["manual"]

DEPS = [
    "//:albatross",
    "@gflags",
]

def example(name, srcs, args):
    swift_cc_binary(
        name = name,
        srcs = srcs,
        tags = TAGS,
        deps = DEPS,
        rtti = True,
        nocopts = ["-Wfloat-equal"],
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
