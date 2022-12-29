include(SwiftTargets)

swift_add_tool(albatross_unit_tests
  SOURCES
  #test_group_by.cc
  #test_block_utils.cc
  #test_sparse_gp.cc
  #test_covariance_functions_block.cc
  test_linalg_structured.cc
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
  )
set_target_properties(albatross_unit_tests PROPERTIES CXX_STANDARD 14 CXX_EXTENSIONS OFF)

add_custom_target(run_albatross_unit_tests ALL
  COMMAND albatross_unit_tests
  COMMENT "Running unit tests"
  )
add_dependencies(run_albatross_unit_tests
  albatross_unit_tests
  )
