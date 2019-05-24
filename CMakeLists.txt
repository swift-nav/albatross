project(albatross)
cmake_minimum_required(VERSION 2.8.7)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" "${CMAKE_CURRENT_LIST_DIR}/cmake/common")

include(SwiftCmakeOptions)
swift_create_project_options(
    HAS_TESTS 
    HAS_EXAMPLES
    TEST_PACKAGES "Googletest" "GFlags"
    )

option(ENABLE_STACK_ANALYSIS "Enable stack analysis. Requires gcc." OFF)

if(albatross_BUILD_EXAMPLES)
  find_package(GFlags REQUIRED)
endif()

set(JUST_INSTALL_CEREAL YES CACHE BOOL "" FORCE) #Don't generate docs, it conflicts with Eigen
find_package(Cereal REQUIRED)
find_package(Eigen REQUIRED)
find_package(FastCSV REQUIRED)
find_package(Gzip-Hpp REQUIRED)

add_library(albatross INTERFACE)
target_include_directories(albatross INTERFACE
  "${PROJECT_SOURCE_DIR}/include"
  "${PROJECT_SOURCE_DIR}/tests"
  )
target_link_libraries(albatross
    INTERFACE
    eigen
    cereal
    fast-csv
    gzip-hpp
    )

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

if(albatross_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(albatross_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

# This is just a temporary hack until we have proper clang tools support again
function(define_if_not_exist target)
  if(NOT TARGET ${target})
    add_custom_target(${target})
  endif()
endfunction()
define_if_not_exist(clang-format-all-albatross)
define_if_not_exist(clang-format-all)
