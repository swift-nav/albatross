# CMake script searches for clang-format and sets the following
# variables:
#
# CLANG_FORMAT_PATH  : Fully-qualified path to the clang-format executable
# 
# Additionally defines the following targets:
# 
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

# Check for Clang format
set(CLANG_FORMAT_PATH "NOTSET" CACHE STRING "Absolute path to the clang-format executable")
if("${CLANG_FORMAT_PATH}" STREQUAL "NOTSET")
  find_program(CLANG_FORMAT NAMES
    clang-format-3.8
    clang-format-6.0
    clang-format-4.0
    )
  message(STATUS "Using clang format: ${CLANG_FORMAT}")
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


if (EXISTS ${CLANG_FORMAT_PATH})
  # Format all files .cc files (and their headers) in project
  add_custom_target(clang-format-all
    COMMAND git ls-files -- '../*.cc' '../*.h'
    | sed 's/^...//' | sed 's\#^\#${CMAKE_SOURCE_DIR}/\#'
    | xargs "${CLANG_FORMAT_PATH}" -i
    )
  # In-place format *.cc files that differ from master, and are not listed as
  # being DELETED.
  add_custom_target(clang-format-diff
    COMMAND git diff --diff-filter=ACMRTUXB --name-only master -- '../*.cc' '../*.h'
    | sed 's\#^\#${CMAKE_SOURCE_DIR}/\#'
    | xargs "${CLANG_FORMAT_PATH}" -i
    )
endif()
if (EXISTS ${CPPCHECK_PATH})
  add_custom_target(cppcheck-all
    COMMAND git ls-files -- '../*.cc' '../*.h'
    | ${CPPCHECK_PATH} --enable=all --std=c++14 -I../include -isystem../third_party/eigen -q --file-list=-
    )
endif()

