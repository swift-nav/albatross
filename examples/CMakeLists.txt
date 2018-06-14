include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/third_party/fast-cpp-csv-parser/")

set(example_HEADERS
    example_utils.h
    sinc_example_utils.h
    CACHE INTERNAL ""
)

add_executable(sinc_example
    EXCLUDE_FROM_ALL
    sinc_example.cc
)

add_dependencies(sinc_example
    albatross
)

target_link_libraries(sinc_example m gflags pthread)

add_custom_target(
    run_sinc_example ALL
    DEPENDS ${example_HEADERS}
    COMMAND sinc_example -input ${PROJECT_SOURCE_DIR}/examples/sinc_input.csv -output ./sinc_predictions.csv
    COMMENT "Running sinc_example"
)

add_dependencies(run_sinc_example
    sinc_example
)


add_executable(tune_example
    EXCLUDE_FROM_ALL
    tune_example.cc
)

add_dependencies(tune_example
    albatross
)

target_link_libraries(tune_example m gflags pthread nlopt)

add_custom_target(
    run_tune_example ALL
    DEPENDS ${example_HEADERS}
    COMMAND tune_example -input ${PROJECT_SOURCE_DIR}/examples/sinc_input.csv -output ./sinc_predictions.csv
    COMMENT "Running tune_example"
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

target_link_libraries(inspection_example m gflags pthread)

add_custom_target(
    run_inspection_example ALL
    DEPENDS ${example_HEADERS}
    COMMAND inspection_example -input ${PROJECT_SOURCE_DIR}/examples/inspection_input.csv -output ./inspection_predictions.csv
    COMMENT "Running inspection_example"
)

add_dependencies(run_inspection_example
    inspection_example
)


add_executable(temperature_example
    EXCLUDE_FROM_ALL
    temperature_example/temperature_example.cc
)

add_dependencies(temperature_example
    albatross
)

target_link_libraries(temperature_example m gflags pthread nlopt)

add_custom_target(
    run_temperature_example ALL
    DEPENDS ${example_HEADERS}
    COMMAND temperature_example -input ${PROJECT_SOURCE_DIR}/examples/temperature_example/gsod.csv -predict ${PROJECT_SOURCE_DIR}/examples/temperature_example/prediction_locations.csv -output ./test_temperature_predictions.csv -thin 5
    COMMENT "Running temperature_example"
)

add_dependencies(run_temperature_example
    temperature_example
)