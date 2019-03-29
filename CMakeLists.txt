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

include(${PROJECT_SOURCE_DIR}/cmake/FindGFlags.cmake)

set(albatross_COMPILE_OPTIONS
  -Werror
  -Wall
  -Wno-unused-value
  -Wcast-align
  -Wchar-subscripts
  -Wcomment
  -Wdisabled-optimization
  -Wformat
  -Wformat=2
  -Wformat-nonliteral
  -Wformat-security
  -Wformat-y2k
  -Wimport
  -Winit-self
  -Winvalid-pch
  -Wmissing-braces
  -Wmissing-field-initializers
  -Wmissing-format-attribute
  -Wmissing-include-dirs
  -Wparentheses
  -Wpointer-arith
  -Wredundant-decls
  -Wreturn-type
  -Wsequence-point
  -Wsign-compare
  -Wstack-protector
  -Wswitch
  -Wtrigraphs
  -Wuninitialized
  -Wunknown-pragmas
  -Wunused
  -Wunused-function
  -Wunused-label
  -Wunused-variable
  -Wunused-value
  -Wunused-variable
  -Wvolatile-register-var
  -Wwrite-strings
  )

add_subdirectory(tests)
add_subdirectory(examples)
