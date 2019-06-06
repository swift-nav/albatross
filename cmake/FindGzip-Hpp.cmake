# Non-swift code that doesn't nicely export a library target, can't use the generic module
if(EXISTS "${PROJECT_SOURCE_DIR}/third_party/gzip-hpp/CMakeLists.txt")
  add_library(gzip-hpp INTERFACE)
  target_include_directories(gzip-hpp SYSTEM INTERFACE $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/gzip-hpp/include>)
else()
  if(Gzip-Hpp_FIND_REQUIRED)
    message(FATAL_ERROR "Could not find REQUIRED package Gzip-Hpp")
  else()
    message(WARNING "Could not find package Gzip-Hpp")
  endif()
endif()

