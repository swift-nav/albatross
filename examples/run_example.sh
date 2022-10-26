#! /bin/bash
# Usages:
# run_examples <BINARY> <ARGS>
set -e

BINARY=$(realpath $1)

cd $BUILD_WORKSPACE_DIRECTORY

mkdir -p examples_out/

$BINARY $@
