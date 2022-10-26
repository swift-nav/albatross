workspace(name = "albatross")

local_repository(
    name = "gflags",
    path = "third_party/googleflags",
)

local_repository(
    name = "gtest",
    path = "third_party/googletest",
)

new_local_repository(
    name = "eigen",
    build_file = "bazel/eigen.BUILD",
    path = "third_party/eigen",
)

new_local_repository(
    name = "gzip",
    build_file = "bazel/gzip.BUILD",
    path = "third_party/gzip-hpp",
)

new_local_repository(
    name = "variant",
    build_file = "bazel/variant.BUILD",
    path = "third_party/variant",
)

new_local_repository(
    name = "cereal",
    build_file = "bazel/cereal.BUILD",
    path = "third_party/cereal",
)

new_local_repository(
    name = "fast-cpp-csv-parser",
    build_file = "bazel/fast-cpp-csv-parser.BUILD",
    path = "third_party/fast-cpp-csv-parser",
)

new_local_repository(
    name = "nlopt",
    build_file = "bazel/nlopt.BUILD",
    path = "third_party/nlopt",
)

load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Rules for integrating with cmake builds
http_archive(
    name = "rules_foreign_cc",
    strip_prefix = "rules_foreign_cc-c65e8cfbaa002bcd1ce9c26e9fec63b0b866c94b",
    url = "https://github.com/bazelbuild/rules_foreign_cc/archive/c65e8cfbaa002bcd1ce9c26e9fec63b0b866c94b.tar.gz",
)

load("@rules_foreign_cc//foreign_cc:repositories.bzl", "rules_foreign_cc_dependencies")

rules_foreign_cc_dependencies()

# Hedron's Compile Commands Extractor for Bazel
# https://github.com/hedronvision/bazel-compile-commands-extractor
http_archive(
    name = "hedron_compile_commands",
    sha256 = "4b251a482a85de6c5cb0dc34c5671e73190b9ff348e9979fa2c033d81de0f928",
    strip_prefix = "bazel-compile-commands-extractor-5bb5ff2f32d542a986033102af771aa4206387b9",
    url = "https://github.com/hedronvision/bazel-compile-commands-extractor/archive/5bb5ff2f32d542a986033102af771aa4206387b9.tar.gz",
)

load("@hedron_compile_commands//:workspace_setup.bzl", "hedron_compile_commands_setup")

hedron_compile_commands_setup()
