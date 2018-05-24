cmake_minimum_required(VERSION 2.8)

# This brings in the external project support in cmake
include(ExternalProject)

set(ALBATROSS_GFLAGS_CMAKE_CXX_FLAGS "")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(ALBATROSS_GFLAGS_CMAKE_CXX_FLAGS "-stdlib=libc++")
endif()

# This adds libgflags as an external project with the specified parameters.
ExternalProject_Add(libgflags
  # We use SOURCE_DIR because we use version control to track the
  # version of this library instead of using the build tool
  SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/gflags
  # We don't want to install this globally; we just want to use it in
  # place.
  INSTALL_COMMAND cmake -E echo "Not installing googleflags globally."
  # This determines the subdirectory of the build directory in which
  # gflags gets built.
  PREFIX gflags
  # This simply passes down cmake arguments, which allows us to define
  # gflags-specific cmake flags as arguments to the toplevel cmake
  # invocation.
  CMAKE_ARGS ${CMAKE_ARGS} -DCMAKE_CXX_FLAGS=${ALBATROSS_GFLAGS_CMAKE_CXX_FLAGS})

# This pulls out the variables `source_dir` and `binary_dir` from the
# gflags project, so we can refer to them below.
ExternalProject_Get_Property(libgflags source_dir binary_dir)

# This tells later `target_link_libraries` commands about the gflags
# library.
add_library(gflags STATIC IMPORTED GLOBAL)

# This tells where the static libgflags binary will end up.  I have no
# idea how to control this and just found it with `locate`.
set_property(TARGET gflags
  PROPERTY IMPORTED_LOCATION "${binary_dir}/lib/libgflags.a")

# This makes the gflags library depend on the libgflags external
# project, so that when you ask to link against gflags, the external
# project will get built.
add_dependencies(gflags libgflags)

# This tells where the libgflags headers generated during the build
# process will end up.  I have no idea how to control this and just
# found it with `locate`.  Note that any targets specified after this
# file fragment is included will now include gflags headers as part of
# their compile commands.
include_directories(SYSTEM "${binary_dir}/include")
