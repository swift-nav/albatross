add_executable(albatross_unit_tests 
  test_radial.cc
  test_covariance_functions.cc
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
