
set(CMAKE_CXX_FLAGS "-Wshadow -Wswitch-default -Wswitch-enum -Wundef -Wuninitialized -Wpointer-arith -Wcast-align -Wformat=2 -Wredundant-decls ${CMAKE_CXX_FLAGS}")

add_executable(test_albatross
EXCLUDE_FROM_ALL
test_covariance_functions.cc
)

target_link_libraries(test_albatross m gtest gtest_main pthread gflags yaml-cpp)

add_custom_target(
run_test_albatross ALL
COMMAND test_albatross
COMMENT "Running unit tests"
)

add_dependencies(run_test_albatross
test_albatross
)
