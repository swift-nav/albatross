# Runtime analysis using Clang sanitization flags.

option(ENABLE_SANITIZERS "Enable sanitizers." OFF)
option(ANALYZE_ADDRESS "Enable address sanitizer." OFF)
option(ANALYZE_LEAK "Enable leak sanitizer." OFF)
option(ANALYZE_MEMORY "Enable memory sanitizer." OFF)
option(ANALYZE_THREAD "Enable thread sanitizer." OFF)
option(ANALYZE_UNDEFINED "Enable undefined behavior sanitizer." OFF)
option(ANALYZE_DATAFLOW "Enable dataflow sanitizer." OFF)

if (ENABLE_SANITIZERS)
  # Some of these options can't be used simultaneously.
  #
  if (ANALYZE_ADDRESS AND ANALYZE_MEMORY )
    message(WARNING "Can't -fsanitize address/memory simultaneously.")
  endif ()
  if (ANALYZE_MEMORY AND ANALYZE_THREAD )
    message(WARNING "Can't -fsanitize memory/thread simultaneously.")
  endif ()
  if (ANALYZE_ADDRESS AND ANALYZE_THREAD )
    message(WARNING "Can't -fsanitize address/thread simultaneously.")
  endif ()
  # Instantiate C/C++ and C++-specific flags.
  #
  set(SANITIZE_FLAGS "")
  # Dispatch sanitizer options based on compiler.
  #
  message(STATUS "Enabling runtime analysis sanitizers!")
  if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    # See http://clang.llvm.org/docs and
    # http://clang.llvm.org/docs/UsersManual.html#controlling-code-generation
    # for more details.
    set(SANITIZE_FLAGS  "-g -O0 -fno-omit-frame-pointer")
    if (ANALYZE_ADDRESS)
      message(STATUS "Enabling address sanitizer.")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize=address")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fno-optimize-sibling-calls")
    elseif (ANALYZE_MEMORY)
      message(STATUS "Enabling memory sanitizer.")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize=memory")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fno-optimize-sibling-calls")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize-memory-track-origins=2")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize-memory-use-after-dtor")
    elseif (ANALYZE_THREAD)
      message(STATUS "Enabling thread sanitizer.")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize=thread")
    endif ()
    if (ANALYZE_LEAK)
      message(STATUS "Enabling leak sanitizer.")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize=leak")
    endif ()
    if (ANALYZE_UNDEFINED)
      message(STATUS "Enabling undefined behavior sanitizer.")
      # The `vptr` sanitizer won't work with `-fno-rtti`.
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize=undefined -fno-sanitize=vptr")
    endif ()
  elseif (CMAKE_CXX_COMPILER_ID MATCHES "GNU" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER 4.8)
    # See: https://gcc.gnu.org/onlinedocs/gcc/Instrumentation-Options.html
    #
    # We seem to need `-fuse-ld=gold` on Travis.
    set(SANITIZE_FLAGS  "-g -O0")
    if (ANALYZE_ADDRESS)
      message(STATUS "Enabling address sanitizer.")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize=address")
    elseif (ANALYZE_MEMORY)
      message(STATUS "Enabling memory sanitizer.")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize=memory")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize-memory-track-origins=2")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize-memory-use-after-dtor")
    elseif (ANALYZE_THREAD)
      message(STATUS "Enabling thread sanitizer.")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize=thread")
    elseif (ANALYZE_LEAK)
      message(STATUS "Enabling leak sanitizer.")
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize=leak")
    endif ()
    if (ANALYZE_UNDEFINED)
      message(STATUS "Enabling undefined behavior sanitizer.")
      # The `vptr` sanitizer won't work with `-fno-rtti`.
      set(SANITIZE_FLAGS  "${SANITIZE_FLAGS} -fsanitize=undefined -fno-sanitize=vptr")
    endif ()
  else ()
    message(FATAL_ERROR "Oh noes! We don't support your compiler.")
  endif ()
  set(CMAKE_C_FLAGS   "${CMAKE_C_FLAGS} ${SANITIZE_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SANITIZE_FLAGS}")
endif ()
