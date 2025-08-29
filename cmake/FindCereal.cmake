# Cereal throws lots of compiler errors if we add_subdirectory it, but we 
# only want the headers anyway
if(TARGET cereal)
  return()
endif()

if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cereal/include")
  add_library(cereal INTERFACE)
  target_include_directories(cereal SYSTEM INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/third_party/cereal/include")
  target_compile_definitions(cereal INTERFACE CEREAL_THREAD_SAFE=1)
  target_link_libraries(cereal INTERFACE pthread)
endif()
