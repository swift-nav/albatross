load("@rules_swiftnav//cc:defs2.bzl", "swift_add_internal_cc_binary")

TAGS = ["manual"]

DEPS = [
    "//:albatross",
    "@gflags",
]

def example(name, srcs, args):
    swift_add_internal_cc_binary(
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
