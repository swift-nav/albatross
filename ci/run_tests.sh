#!/bin/bash -x
set -e

run_tests() {
    cd build
    # Run the long integration tests in release mode so they're fast.
    cmake -DENABLE_AUTOLINT=ON \
      -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER="$C_COMPILER" -DCMAKE_CXX_COMPILER="$CXX_COMPILER" ../
    make run_albatross_unit_tests run_inspection_example run_sinc_example run_tune_example run_temperature_example -j2
    cd ..
}

ensure_clang_format()
{
    cd build
    make clang-format-all
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

mkdir -p build

run_tests
ensure_clang_format
# Make sure we have the proper header on all files
bash ./ci/ensure_copyright.sh


