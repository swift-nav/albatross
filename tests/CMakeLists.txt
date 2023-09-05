include(SwiftTargets)

# Ensure tests are always built with assertions enabled
string(REPLACE "-DNDEBUG" ""
  CMAKE_CXX_FLAGS_RELEASE
  "${CMAKE_CXX_FLAGS_RELEASE}")
string(REPLACE "-DNDEBUG" ""
  CMAKE_CXX_FLAGS_RELWITHDEBINFO
  "${CMAKE_CXX_FLAGS_RELWITHDEBINFO}")

swift_add_tool(albatross_unit_tests
  SOURCES
  test_apply.cc
  test_async_utils.cc
  test_block_utils.cc
  test_call_trace.cc
  test_callers.cc
  test_compression.cc
  test_conditional_gaussian.cc
  test_concatenate.cc
  test_core_dataset.cc
  test_core_distribution.cc
  test_core_model.cc
  test_covariance_function.cc
  test_covariance_functions.cc
  test_cross_validation.cc
  test_csv_utils.cc
  test_distance_metrics.cc
  test_eigen_utils.cc
  test_error_handling.cc
  test_evaluate.cc
  test_gp.cc
  test_group_by.cc
  test_indexing.cc
  test_linalg_utils.cc
  test_map_utils.cc
  test_minimum_spanning_tree.cc
  test_model_adapter.cc
  test_model_metrics.cc
  test_models.cc
  test_parameter_handling_mixin.cc
  test_prediction.cc
  test_radial.cc
  test_random_utils.cc
  test_ransac.cc
  test_samplers.cc
  test_scaling_function.cc
  test_serializable_ldlt.cc
  test_serialize.cc
  test_sparse_gp.cc
  test_stats.cc
  test_thread_pool.cc
  test_traits_cereal.cc
  test_traits_core.cc
  test_traits_details.cc
  test_traits_covariance_functions.cc
  test_traits_evaluation.cc
  test_traits_indexing.cc
  test_tune.cc
  test_variant_utils.cc
  RTTI
  EXCEPTIONS
  REMOVE_COMPILE_OPTIONS
  -Wfloat-equal
  )
target_include_directories(albatross_unit_tests SYSTEM PRIVATE
  "${gtest_SOURCE_DIR}"
  "${gtest_SOURCE_DIR}/include"
  )
target_compile_definitions(albatross_unit_tests PRIVATE
  CSV_IO_NO_THREAD
  )
target_compile_options(albatross_unit_tests PRIVATE
  ${albatross_COMPILE_OPTIONS}
  )
target_link_libraries(albatross_unit_tests
  albatross
  gtest
  gtest_main
  pthread
  gflags
  nlopt
  z
  zstd::zstd
  suitesparse::spqr
  suitesparse::cholmod
  )
set_target_properties(albatross_unit_tests PROPERTIES CXX_STANDARD 14 CXX_EXTENSIONS OFF)

add_custom_target(run_albatross_unit_tests ALL
  COMMAND albatross_unit_tests
  COMMENT "Running unit tests"
  )
add_dependencies(run_albatross_unit_tests
  albatross_unit_tests
  )
