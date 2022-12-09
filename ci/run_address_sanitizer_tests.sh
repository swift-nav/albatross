#!/bin/bash -x
set -e

run_tests() {
    mkdir -p build
    cd build
    cmake -DENABLE_AUTOLINT=ON \
          -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_C_COMPILER="$C_COMPILER" \
          -DCMAKE_CXX_COMPILER="$CXX_COMPILER" \
          -DSWIFT_SANITIZE_ADDRESS=ON \
          -DSWIFT_SANITIZE_UNDEFINED=ON \
          -DSWIFT_SANITIZE_LEAK=ON \
          ../
    ASAN_OPTIONS=check_initialization_order=true:detect_stack_use_after_return=true:strict_string_checks=true:halt_on_error=true \
        UBSAN_OPTIONS=halt_on_error=true:print_stacktrace=true \
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
