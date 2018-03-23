#!/bin/bash -ex
mkdir build
cd build
# Run the long integration tests in release mode so they're fast.
cmake -DENABLE_AUTOLINT=ON \
  -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$C_COMPILER" -DCMAKE_CXX_COMPILER="$CXX_COMPILER" ../
make run_albatross_unit_tests run_inspection_example run_sinc_example -j4

bash ../ci/ensure_copyright.sh
