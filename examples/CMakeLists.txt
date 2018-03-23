set(example_HEADERS
    example_utils.h
    CACHE INTERNAL ""
)

add_executable(sinc_example
    EXCLUDE_FROM_ALL
    sinc_example.cc
)

add_dependencies(sinc_example
    albatross
)

target_link_libraries(sinc_example m gflags yaml-cpp pthread)

add_custom_target(
    run_sinc_example ALL
    DEPENDS ${example_HEADERS}
    COMMAND sinc_example -input ${PROJECT_SOURCE_DIR}/examples/sinc_input.csv -output ./sinc_predictions.csv
    COMMENT "Running sinc_example"
)

add_dependencies(run_sinc_example
    sinc_example
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
    DEPENDS ${example_HEADERS}
    COMMAND inspection_example -input ${PROJECT_SOURCE_DIR}/examples/inspection_input.csv -output ./inspection_predictions.csv
    COMMENT "Running inspection_example"
)

add_dependencies(run_inspection_example
    inspection_example
)
