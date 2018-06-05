project(albatross)
cmake_minimum_required(VERSION 2.8.7)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(ClangTools)

# Enable Clang auto-formatting, defaults to on. 
option(ENABLE_AUTOFORMAT "Enable automatic formatting and replacement with clang-format." ON)
# Enable Clang auto-linting, defaults to off.
option(ENABLE_AUTOLINT "Enable automatic linting and warning with clang-tidy. " OFF)
# Enable static stack analyzer, defaults to off.
option(ENABLE_STACK_ANALYSIS "Enable stack analysis. Requires gcc." OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

add_subdirectory(third_party/googletest)

include_directories(SYSTEM ${gtest_SOURCE_DIR} ${gtest_SOURCE_DIR}/include)
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/third_party/eigen/")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/third_party/cereal/include")
include_directories(SYSTEM "${PROJECT_SOURCE_DIR}/third_party/fast-cpp-csv-parser/")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Werror -fexceptions -frtti")
set("LOG_LEVEL=8")
set(CMAKE_CXX_FLAGS "-std=c++14 -Wshadow -Wswitch-default -Wswitch-enum -Wundef -Wuninitialized -Wpointer-arith -Wcast-align -Wformat=2 -Wredundant-decls ${CMAKE_CXX_FLAGS}")

include_directories("${PROJECT_SOURCE_DIR}/albatross")
include(${PROJECT_SOURCE_DIR}/cmake/FindGFlags.cmake)

set(albatross_SRCS
CACHE INTERNAL ""
)

set(albatross_HEADERS
    albatross/tune.h
    albatross/evaluate.h
    albatross/crossvalidation.h
    albatross/tuning_metrics.h
    albatross/map_utils.h
    albatross/core/keys.h
    albatross/core/model.h
    albatross/core/model_adapter.h
    albatross/core/traits.h
    albatross/core/serialize.h
    albatross/core/parameter_handling_mixin.h    
    albatross/cereal/eigen.h
    albatross/eigen/serializable_ldlt.h
    albatross/models/gp.h
    albatross/models/least_squares.h
    albatross/covariance_functions/covariance_term.h
    albatross/covariance_functions/covariance_functions.h
    albatross/covariance_functions/distance_metrics.h
    albatross/covariance_functions/noise.h
    albatross/covariance_functions/polynomials.h
    albatross/covariance_functions/radial.h
    tests/test_utils.h
    CACHE INTERNAL ""
)

add_custom_target(albatross DEPENDS ${albatross_SRCS} ${albatross_HEADERS} COMMENT "albatross dependencies")

add_subdirectory(tests)
add_subdirectory(examples)

