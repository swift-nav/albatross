if(TARGET ThreadPool)
  return()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ThreadPool/")
  add_library(ThreadPool INTERFACE)

  target_include_directories(ThreadPool SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/ThreadPool/")
endif()
