#!/bin/bash -x
set -e

ensure_clang_format()
{
    mkdir -p build
    cd build
    cmake ../
    make clang-format-all-albatross
    if [[ $(git --no-pager diff --name-only HEAD) ]]; then
        echo "######################################################"
        echo "####### clang-format warning found! Exiting... #######"
        echo "######################################################"
        echo ""
        echo "This should be formatted locally and pushed again..."
        git --no-pager diff
        exit 1
    fi
    cd ..
}

ensure_clang_format
