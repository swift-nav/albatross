cmake_minimum_required(VERSION 2.8)

# This brings in the external project support in cmake
include(ExternalProject)

set(SWIFT_LIBYAML_CMAKE_CXX_FLAGS "")
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(SWIFT_LIBYAML_CMAKE_CXX_FLAGS "-stdlib=libc++")
endif()

# This adds yaml-cpp as an external project with the specified parameters.
ExternalProject_Add(libyaml-cpp
        # We use SOURCE_DIR because we use version control to track the
        # version of this library instead of using the build tool
        SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/yaml-cpp
        # We don't want to install this globally; we just want to use it in
        # place.
        INSTALL_COMMAND cmake -E echo "Not installing yaml-cpp globally."
        # This determines the subdirectory of the build directory in which
        # yaml-cpp gets built.
        PREFIX yaml-cpp
        # This simply passes down cmake arguments, which allows us to define
        # yaml-cpp-specific cmake flags as arguments to the toplevel cmake
        # invocation.
        CMAKE_ARGS -DCMAKE_CXX_FLAGS=${SWIFT_LIBYAML_CMAKE_CXX_FLAGS} -DYAML_CPP_BUILD_TOOLS=OFF -DYAML_CPP_BUILD_CONTRIB=OFF ${CMAKE_ARGS})

# This pulls out the variables `source_dir` and `binary_dir` from the
# yaml-cpp project, so we can refer to them below.
ExternalProject_Get_Property(libyaml-cpp source_dir binary_dir)

# This tells later `target_link_libraries` commands about the yaml-cpp
# library.
add_library(yaml-cpp STATIC IMPORTED GLOBAL)

# This tells where the static yaml-cpp binary will end up.  I have no
# idea how to control this and just found it with `locate`.
set_property(TARGET yaml-cpp
        PROPERTY IMPORTED_LOCATION "${binary_dir}/libyaml-cpp.a")

# This makes the yaml-cpp library depend on the yaml-cpp external
# project, so that when you ask to link against yaml-cpp, the external
# project will get built.
add_dependencies(yaml-cpp libyaml-cpp)

# This tells where the yaml-cpp headers generated during the build
# process will end up.  I have no idea how to control this and just
# found it with `locate`.  Note that any targets specified after this
# file fragment is included will now include yaml-cpp headers as part of
# their compile commands.
include_directories(SYSTEM "${binary_dir}")
