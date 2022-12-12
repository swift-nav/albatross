#!/bin/bash -x
set -e

run_tidy() {
    mkdir -p build
    cd build
    cmake -DENABLE_AUTOLINT=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -Dalbatross_ENABLE_CLANG_TIDY=ON \
          -DCMAKE_C_COMPILER="$C_COMPILER" \
          -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
          ../
    make clang-tidy-albatross_unit_tests-check
    cd ..
}

run_tidy
