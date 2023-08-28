if (TARGET zstd::zstd)
  return()
endif()

include(FindPackageHandleStandardArgs)

if (APPLE)
  file(GLOB ZSTD_DIRS
    /usr/local/Cellar/zstd/*
    /opt/homebrew/Cellar/zstd/*
  )
endif()

find_path(zstd_INCLUDE_DIR
  NAMES zstd.h
  HINTS ${ZSTD_DIRS}
  PATH_SUFFIXES include
)

find_library(zstd_LIBRARY
  NAMES zstd
  HINTS ${ZSTD_DIRS}
  PATH_SUFFIXES lib
)

find_package_handle_standard_args(zstd REQUIRED_VARS
  zstd_LIBRARY
  zstd_INCLUDE_DIR
)

if (zstd_FOUND)
  mark_as_advanced(zstd_LIBRARY)
  mark_as_advanced(zstd_INCLUDE_DIR)

  add_library(zstd::zstd UNKNOWN IMPORTED)
  set_target_properties(zstd::zstd
    PROPERTIES
      IMPORTED_LOCATION ${zstd_LIBRARY}
      INTERFACE_INCLUDE_DIRECTORIES ${zstd_INCLUDE_DIR}
  )
endif ()
