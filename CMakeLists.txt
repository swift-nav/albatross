cmake_minimum_required(VERSION 3.14)
project(albatross)

set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake" "${CMAKE_CURRENT_LIST_DIR}/cmake/common")

set(SWIFT_CLANG_TIDY_RATCHET_FILE
  "${PROJECT_SOURCE_DIR}/ci/clang_tidy_ratchet.yaml")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

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

function(albatross_add_test target)
  swift_add_test(${target} ${ARGN})
  target_include_directories(${target}
    PUBLIC
      ${PROJECT_SOURCE_DIR}/include
  )
  target_compile_features(${target}
    PUBLIC
      cxx_std_17
  )
  if (MKL_FOUND)
    target_link_libraries(${target}
      PRIVATE
        MKL::MKL
    )
  endif ()
endfunction()

if(albatross_BUILD_TESTS)
  add_subdirectory(tests)
endif()
if(albatross_BUILD_EXAMPLES)
  add_subdirectory(examples)
endif()

swift_create_clang_tidy_targets(WITHOUT_SWIFT_TYPES)
