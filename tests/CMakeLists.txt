
add_definitions(-DCSV_IO_NO_THREAD)

set(CMAKE_CXX_FLAGS "-Wshadow -Wswitch-default -Wswitch-enum -Wundef -Wuninitialized -Wpointer-arith -Wcast-align -Wformat=2 -Wredundant-decls ${CMAKE_CXX_FLAGS}")

add_executable(albatross_unit_tests
EXCLUDE_FROM_ALL
test_core_distribution.cc
test_core_model.cc
test_csv_utils.cc
test_covariance_functions.cc
test_distance_metrics.cc
test_evaluate.cc
test_map_utils.cc
test_model_adapter.cc
test_models.cc
test_parameter_handling_mixin.cc
test_radial.cc
test_scaling_function.cc
test_serializable_ldlt.cc
test_serialize.cc
test_tune.cc
test_tuning_metrics.cc
test_random_utils.cc
test_outlier.cc
)

add_dependencies(albatross_unit_tests
    albatross
    clang-format-all
)

target_link_libraries(albatross_unit_tests m gtest gtest_main pthread gflags nlopt)

add_custom_target(
run_albatross_unit_tests ALL
COMMAND albatross_unit_tests
COMMENT "Running unit tests"
)

add_dependencies(run_albatross_unit_tests
albatross_unit_tests
)
