# CMake script searches for clang-tidy and clang-format and sets the following
# variables:
#
# CLANG_TIDY_PATH    : Fully-qualified path to the clang-tidy executable
# CLANG_FORMAT_PATH  : Fully-qualified path to the clang-format executable
# 
# Additionally defines the following targets:
# 
# clang-tidy-all     : Run clang-tidy over all files.
# clang-tidy-diff    : Run clang-tidy over all files differing from master.
# clang-format-all   : Run clang-format over all files.
# clang-format-diff  : Run clang-format over all files differing from master.
# cppcheck-all       : Run cppcheck over all files.

# Do not use clang tooling when cross compiling.
if(CMAKE_CROSSCOMPILING)
    return()
endif(CMAKE_CROSSCOMPILING)

################################################################################
# Search for tools.
################################################################################

# Check for Clang Tidy
set(CLANG_TIDY_PATH "NOTSET" CACHE STRING "Absolute path to the clang-tidy executable")
if("${CLANG_TIDY_PATH}" STREQUAL "NOTSET")
    find_program(CLANG_TIDY NAMES
        clang-tidy39 clang-tidy-3.9
        clang-tidy38 clang-tidy-3.8
        clang-tidy37 clang-tidy-3.7
        clang-tidy36 clang-tidy-3.6
        clang-tidy35 clang-tidy-3.5
        clang-tidy34 clang-tidy-3.4
        clang-tidy)
    if("${CLANG_TIDY}" STREQUAL "CLANG_TIDY-NOTFOUND")
        message(WARNING "Could not find 'clang-tidy' please set CLANG_TIDY_PATH:STRING")
    else()
        set(CLANG_TIDY_PATH ${CLANG_TIDY})
        message(STATUS "Found: ${CLANG_TIDY_PATH}")
    endif()
else()
    if(NOT EXISTS ${CLANG_TIDY_PATH})
        message(WARNING "Could not find 'clang-tidy': ${CLANG_TIDY_PATH}")
    else()
        message(STATUS "Found: ${CLANG_TIDY_PATH}")
    endif()
endif()

# Check for Clang format
set(CLANG_FORMAT_PATH "NOTSET" CACHE STRING "Absolute path to the clang-format executable")
if("${CLANG_FORMAT_PATH}" STREQUAL "NOTSET")
    find_program(CLANG_FORMAT NAMES
        clang-format39 clang-format-3.9
        clang-format38 clang-format-3.8
        clang-format37 clang-format-3.7
        clang-format36 clang-format-3.6
        clang-format35 clang-format-3.5
        clang-format34 clang-format-3.4
        clang-format)
    if("${CLANG_FORMAT}" STREQUAL "CLANG_FORMAT-NOTFOUND")
        message(WARNING "Could not find 'clang-format' please set CLANG_FORMAT_PATH:STRING")
    else()
        set(CLANG_FORMAT_PATH ${CLANG_FORMAT})
        message(STATUS "Found: ${CLANG_FORMAT_PATH}")
    endif()
else()
    if(NOT EXISTS ${CLANG_FORMAT_PATH})
        message(WARNING "Could not find 'clang-format': ${CLANG_FORMAT_PATH}")
    else()
        message(STATUS "Found: ${CLANG_FORMAT_PATH}")
    endif()
endif()

################################################################################
# Conditionally add targets.
################################################################################

if (EXISTS ${CLANG_TIDY_PATH})
    # Tidy all files .cc files (and their headers) in project
    # Second stage of pipeline makes an absolute path for each file. Note that
    # git ls-files and diff-tree behave differently in prepending the file path.
    add_custom_target(clang-tidy-all
        COMMAND git ls-files -- '../src/*.cc'
        | sed 's/^...//' | sed 's\#\^\#${CMAKE_SOURCE_DIR}/\#'
        | xargs -P 2 -I file "${CLANG_TIDY_PATH}"
            -export-fixes="${CMAKE_SOURCE_DIR}/fixes.yaml" file -- -stdlib=libc++ -std=c++14 "-I${CMAKE_SOURCE_DIR}/include/" "-isystem${CMAKE_SOURCE_DIR}/third_party/eigen/" "-I${CMAKE_SOURCE_DIR}/libfec/include/" "-isystem${CMAKE_SOURCE_DIR}/third_party/Optional" "-isystem${CMAKE_SOURCE_DIR}/third_party/variant/include" "-I${CMAKE_SOURCE_DIR}/refactor/common" "-I${CMAKE_SOURCE_DIR}/include/libswiftnav" "-isystem${CMAKE_SOURCE_DIR}/third_party/json/src"
        )
    # Lint *.cc files that differ from master, and are not listed as being
    # DELETED.
    add_custom_target(clang-tidy-diff
        COMMAND git diff --diff-filter=ACMRTUXB --name-only master -- '../src/*.cc'
        | sed 's\#\^\#${CMAKE_SOURCE_DIR}/\#'
        | xargs -P 2 -I file "${CLANG_TIDY_PATH}" file -- -std=c++14 -stdlib=libc++
            "-I${CMAKE_SOURCE_DIR}/include/" "-isystem${CMAKE_SOURCE_DIR}/third_party/eigen/" "-I${CMAKE_SOURCE_DIR}/libfec/include/" "-isystem${CMAKE_SOURCE_DIR}/third_party/Optional" "-isystem${CMAKE_SOURCE_DIR}/third_party/variant/include" "-I${CMAKE_SOURCE_DIR}/refactor/common" "-I${CMAKE_SOURCE_DIR}/include/libswiftnav" "-isystem${CMAKE_SOURCE_DIR}/third_party/json/src"
        )
endif()
if (EXISTS ${CLANG_FORMAT_PATH})
    # Format all files .cc files (and their headers) in project
    add_custom_target(clang-format-all
        COMMAND git ls-files -- '../*.cc' '../include/libswiftnav/pvt_engine/*.h' '../test_pvt_engine/*.h'
        | sed 's/^...//' | sed 's\#\^\#${CMAKE_SOURCE_DIR}/\#'
        | xargs "${CLANG_FORMAT_PATH}" -i
        )
    # In-place format *.cc files that differ from master, and are not listed as
    # being DELETED.
    add_custom_target(clang-format-diff
        COMMAND git diff --diff-filter=ACMRTUXB --name-only master -- '../*.cc' '../include/libswiftnav/pvt_engine/*.h' '../test_pvt_engine/*.h'
        | sed 's\#\^\#${CMAKE_SOURCE_DIR}/\#'
        | xargs "${CLANG_FORMAT_PATH}" -i
        )
endif()
if (EXISTS ${CPPCHECK_PATH})
    add_custom_target(cppcheck-all
        COMMAND git ls-files -- '../*.cc' '../include/libswiftnav/pvt_engine/*.h' '../test_pvt_engine/*.h'
        | ${CPPCHECK_PATH} --enable=all --std=c++14 -I../include -I../test_pvt_engine/include -isystem../third_party/eigen -isystem../third_party/Optional -isystem../third_party/googletest/googletest/include -isystem../third_party/variant/include -isystem../third_party/json/src -q --file-list=-
        )
endif()

