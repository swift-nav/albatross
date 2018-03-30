
add_executable(temperature_example
    EXCLUDE_FROM_ALL
    temperature.cc
)

target_link_libraries(temperature_example m gflags yaml-cpp pthread)

add_custom_target(
    run_temperature_example ALL
    COMMAND temperature_example
    COMMENT "Running temperature_example"
)

add_dependencies(run_temperature_example
    temperature_example
)

add_executable(linear_example
    EXCLUDE_FROM_ALL
    linear.cc
)

add_dependencies(linear_example
    albatross
)

target_link_libraries(linear_example m gflags yaml-cpp pthread)

add_custom_target(
    run_linear_example ALL
    COMMAND linear_example -input ${PROJECT_SOURCE_DIR}/examples/linear_input.csv -output ./linear_predictions.csv
    COMMENT "Running linear_example"
)

add_dependencies(run_linear_example
    linear_example
)

add_executable(inspection_example
    EXCLUDE_FROM_ALL
    inspection.cc
)

add_dependencies(inspection_example
    albatross
)

target_link_libraries(inspection_example m gflags yaml-cpp pthread)

add_custom_target(
    run_inspection_example ALL
    COMMAND inspection_example -input ${PROJECT_SOURCE_DIR}/examples/inspection_input.csv -output ./inspection_predictions.csv
    COMMENT "Running inspection_example"
)

add_dependencies(run_inspection_example
    inspection_example
)
