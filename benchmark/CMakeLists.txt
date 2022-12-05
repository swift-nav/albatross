
add_executable(albatross_benchmark 
  covariance.cc
  uninlineable.cc
  )
target_include_directories(albatross_benchmark SYSTEM PRIVATE
  "${benchmark_SOURCE_DIR}"
  "${benchmark_SOURCE_DIR}/include"
  )
target_compile_options(albatross_benchmark PRIVATE
  ${albatross_COMPILE_OPTIONS}
  )
target_link_libraries(albatross_benchmark
  albatross
  pthread
  gflags
  benchmark::benchmark
  z
  )
set_target_properties(albatross_benchmark PROPERTIES CXX_STANDARD 14 CXX_EXTENSIONS OFF)

add_custom_target(run_albatross_benchmark ALL
  COMMAND albatross_benchmark
  COMMENT "Running benchmark"
  )
add_dependencies(run_albatross_benchmark
  albatross_benchmark
)
# target_compile_options(albatross_benchmark PRIVATE "-march=native")
