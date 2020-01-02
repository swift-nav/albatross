add_executable(sinc_example
  sinc_example.cc
  )
target_link_libraries(sinc_example
  albatross
  gflags
  pthread
  nlopt
  )
add_custom_target(run_sinc_example
  DEPENDS sinc_example
  COMMAND sinc_example -input ${PROJECT_SOURCE_DIR}/examples/sinc_input.csv -output ./sinc_predictions.csv -tune
  COMMENT "Running sinc_example"
  )


add_executable(inspection_example
  inspection.cc
  )
target_link_libraries(inspection_example
  albatross
  gflags
  pthread
  )
add_custom_target(run_inspection_example
  DEPENDS inspection_example
  COMMAND inspection_example -input ${PROJECT_SOURCE_DIR}/examples/inspection_input.csv
  COMMENT "Running inspection_example"
  )


add_executable(temperature_example
  temperature_example/temperature_example.cc
  )
target_link_libraries(temperature_example
  albatross
  gflags
  pthread
  nlopt
  )
add_custom_target(run_temperature_example
  DEPENDS temperature_example
  COMMAND temperature_example -input ${PROJECT_SOURCE_DIR}/examples/temperature_example/gsod.csv -predict ${PROJECT_SOURCE_DIR}/examples/temperature_example/prediction_locations.csv -output ./test_temperature_predictions.csv -thin 5
  COMMENT "Running temperature_example"
  )


add_executable(call_trace_example
  call_trace_example.cc
  )
target_link_libraries(call_trace_example
  albatross
  gflags
  pthread
  )
add_custom_target(run_call_trace_example
  DEPENDS call_trace_example
  COMMAND call_trace_example
  COMMENT "Running call_trace_example"
  )


add_executable(sparse_example
  sparse_example.cc
  )
target_link_libraries(sparse_example
  albatross
  gflags
  pthread
  nlopt
  )
add_custom_target(run_sparse_example
  DEPENDS sparse_example
  COMMAND sparse_example -input ${PROJECT_SOURCE_DIR}/examples/sinc_input.csv -output ./sinc_predictions.csv
  COMMENT "Running sparse_example"
  )


add_executable(patchwork_example
  patchwork_example.cc
  )
target_link_libraries(patchwork_example
  albatross
  gflags
  pthread
  nlopt
  )
add_custom_target(run_patchwork_example
  DEPENDS patchwork_example
  COMMAND patchwork_example -input ${PROJECT_SOURCE_DIR}/examples/sinc_input.csv -output ./sinc_predictions.csv
  COMMENT "Running patchwork_example"
  )

set(albatross_example_BINARIES
  sinc_example
  temperature_example
  call_trace_example
  inspection_example
  sparse_example
  )
foreach(TARGET IN LISTS albatross_example_BINARIES)
  target_compile_options(${TARGET} PRIVATE ${albatross_COMPILE_OPTIONS})
  set_target_properties(${TARGET} PROPERTIES CXX_STANDARD 14 CXX_EXTENSIONS OFF)
endforeach()

