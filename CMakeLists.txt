project(albatross)
cmake_minimum_required(VERSION 2.8.7)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(ClangTools)
include(SanitizeTargets)

# Enable Clang auto-formatting, defaults to on. 
option(ENABLE_AUTOFORMAT "Enable automatic formatting and replacement with clang-format." ON)
# Enable Clang auto-linting, defaults to off.
option(ENABLE_AUTOLINT "Enable automatic linting and warning with clang-tidy. " OFF)
# Enable static stack analyzer, defaults to off.
option(ENABLE_STACK_ANALYSIS "Enable stack analysis. Requires gcc." OFF)

################################################################################
# We want this flag to be passed to third-party builds.
################################################################################
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

add_subdirectory(third_party/googletest)
add_subdirectory(third_party/rapidcheck)
add_subdirectory(third_party/rapidcheck/extras/gtest)

include_directories(SYSTEM ${gtest_SOURCE_DIR} ${gtest_SOURCE_DIR}/include)
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/third_party/eigen/")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/third_party/yaml-cpp/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/third_party/fast-cpp-csv-parser/")

add_definitions(-DCSV_IO_NO_THREAD)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fexceptions -frtti")
set("LOG_LEVEL=8")
set(CMAKE_CXX_FLAGS "-std=c++14 -Wshadow -Wswitch-default -Wswitch-enum -Wundef -Wuninitialized -Wpointer-arith -Wcast-align -Wformat=2 -Wredundant-decls ${CMAKE_CXX_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/albatross")
include(${PROJECT_SOURCE_DIR}/cmake/FindGFlags.cmake)
include(${PROJECT_SOURCE_DIR}/cmake/FindYaml-Cpp.cmake)

set(albatros_SRCS
albatross/covariance_functions.cc
CACHE INTERNAL ""
)

add_subdirectory(tests)
add_subdirectory(examples)

