#!/bin/bash -x
set -e

run_tests() {
    mkdir -p build
    cd build
    cmake -DENABLE_AUTOLINT=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$C_COMPILER" -DCMAKE_CXX_COMPILER="$CXX_COMPILER" ../
    make -k \
         -j2 \
         do-all-unit-tests \
         run_inspection_example \
         run_sinc_example \
         run_temperature_example \
         run_sampler_example
    cd ..
}

run_tests
