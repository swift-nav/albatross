project(albatross)
cmake_minimum_required(VERSION 2.8.7)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake")
include(ClangTools)

option(ENABLE_AUTOFORMAT "Enable automatic formatting and replacement with clang-format." ON)
option(ENABLE_AUTOLINT "Enable automatic linting and warning with clang-tidy. " OFF)
option(ENABLE_STACK_ANALYSIS "Enable stack analysis. Requires gcc." OFF)

if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
endif()

add_subdirectory(third_party/googletest)

add_library(albatross INTERFACE)
target_include_directories(albatross INTERFACE
  "${PROJECT_SOURCE_DIR}/include"
  )
target_include_directories(albatross SYSTEM INTERFACE
  "${PROJECT_SOURCE_DIR}/third_party/eigen/"
  "${PROJECT_SOURCE_DIR}/third_party/cereal/include"
  "${PROJECT_SOURCE_DIR}/third_party/fast-cpp-csv-parser/"
  )


set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror -fexceptions -frtti")
set(CMAKE_CXX_FLAGS "-std=c++14 -Wshadow -Wswitch-default -Wswitch-enum -Wundef -Wpointer-arith -Wcast-align -Wformat=2 -Wredundant-decls ${CMAKE_CXX_FLAGS}")
set("LOG_LEVEL=8")

include(${PROJECT_SOURCE_DIR}/cmake/FindGFlags.cmake)

add_subdirectory(tests)
add_subdirectory(examples)
