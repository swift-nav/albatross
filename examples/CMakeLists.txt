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
  COMMAND sinc_example -input ${PROJECT_SOURCE_DIR}/examples/sinc_input.csv -output ./sinc_predictions.csv -tune > sinc_example_output.txt
  COMMENT "Running sinc_example"
  )

add_executable(sampler_example
  sampler_example.cc
  )
target_link_libraries(sampler_example
  albatross
  gflags
  pthread
  nlopt
  )
add_custom_target(run_sampler_example
  DEPENDS sampler_example
  COMMAND sampler_example -input ${PROJECT_SOURCE_DIR}/examples/sinc_input.csv -output ./sinc_predictions.csv --maxiter 100 > sampler_example_output.txt
  COMMENT "Running sampler_example"
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


set(albatross_example_BINARIES
  sinc_example
  sampler_example
  inspection_example
  temperature_example
  call_trace_example
  sparse_example
  )
foreach(TARGET IN LISTS albatross_example_BINARIES)
  target_compile_options(${TARGET} PRIVATE ${albatross_COMPILE_OPTIONS})
  set_target_properties(${TARGET} PROPERTIES CXX_STANDARD 14 CXX_EXTENSIONS OFF)
endforeach()
