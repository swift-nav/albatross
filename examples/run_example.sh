#! /bin/bash

BINARY=$(realpath $1)

cd $BUILD_WORKSPACE_DIRECTORY

mkdir -p examples_out/

$BINARY $@
