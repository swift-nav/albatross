#!/bin/bash -x
set -e

run_tests() {
    mkdir -p build
    cd build
    cmake -DENABLE_AUTOLINT=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER="$C_COMPILER" \
          -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
          -DSWIFT_SANITIZE_THREAD=ON \
          -DSWIFT_SANITIZE_SUPPRESSION_FILE="$(pwd)/../sanitizers.supp" \
          ../
    TSAN_OPTIONS="force_seq_cst_atomics=1 halt_on_error=1" \
        make \
        -j2 \
        run_albatross_unit_tests \
        run_inspection_example \
        run_sinc_example \
        run_temperature_example \
        run_sampler_example
    cd ..
}

run_tests
