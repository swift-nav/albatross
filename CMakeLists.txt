cmake_minimum_required(VERSION 3.14)
project(albatross)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" "${CMAKE_CURRENT_LIST_DIR}/cmake/common")

set(SWIFT_CLANG_TIDY_RATCHET_FILE
  "${PROJECT_SOURCE_DIR}/ci/clang_tidy_ratchet.yaml")

include(CCache)
include(ClangTidy)
include(SwiftCmakeOptions)
swift_create_project_options(
    HAS_TESTS 
    HAS_EXAMPLES
    TEST_PACKAGES "Googletest" "GFlags"
    )

include(ClangFormat)
swift_setup_clang_format()

include(SanitizeTargets)

option(ENABLE_STACK_ANALYSIS "Enable stack analysis. Requires gcc." OFF)
option(ENABLE_MKL "Enable Intel MKL Optimizations" OFF)

if(albatross_BUILD_EXAMPLES)
  find_package(GFlags REQUIRED)
endif()

set(JUST_INSTALL_CEREAL YES CACHE BOOL "" FORCE) #Don't generate docs, it conflicts with Eigen
find_package(Cereal REQUIRED)
find_package(Eigen REQUIRED)
find_package(FastCSV REQUIRED)
find_package(Gzip-Hpp REQUIRED)
find_package(Variant REQUIRED)
find_package(Nlopt REQUIRED)
find_package(ThreadPool REQUIRED)
find_package(zstd REQUIRED)
find_package(suitesparse REQUIRED)

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
    zstd::zstd
    variant
    ThreadPool
    suitesparse::cholmod
    suitesparse::spqr
    )
if(ENABLE_MKL)
  find_package(MKL)
  if (MKL_FOUND)
    target_link_libraries(albatross INTERFACE MKL::MKL)
  endif()
endif()

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

swift_create_clang_tidy_targets(WITHOUT_SWIFT_TYPES)
