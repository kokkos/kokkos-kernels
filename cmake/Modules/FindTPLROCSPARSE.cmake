IF (ROCSPARSE_LIBRARY_DIRS AND ROCSPARSE_LIBRARIES)
  KOKKOSKERNELS_FIND_IMPORTED(ROCSPARSE INTERFACE LIBRARIES ${ROCSPARSE_LIBRARIES} LIBRARY_PATHS ${ROCSPARSE_LIBRARY_DIRS})
ELSEIF (ROCSPARSE_LIBRARIES)
  KOKKOSKERNELS_FIND_IMPORTED(ROCSPARSE INTERFACE LIBRARIES ${ROCSPARSE_LIBRARIES})
ELSEIF (ROCSPARSE_LIBRARY_DIRS)
  KOKKOSKERNELS_FIND_IMPORTED(ROCSPARSE INTERFACE LIBRARIES rocsparse LIBRARY_PATHS ${ROCSPARSE_LIBRARY_DIRS})
ELSEIF (DEFINED ENV{ROCM_PATH})
  MESSAGE(STATUS "Detected ROCM_PATH: ENV{ROCM_PATH}")
  SET(ROCSPARSE_ROOT "$ENV{ROCM_PATH}/rocsparse")
  KOKKOSKERNELS_FIND_IMPORTED(ROCSPARSE INTERFACE
    LIBRARIES
      rocsparse
    LIBRARY_PATHS
      ${ROCSPARSE_ROOT}/lib
    HEADERS
      rocsparse.h
    HEADER_PATHS
      ${ROCSPARSE_ROOT}/include
  )
ELSE()
  MESSAGE(ERROR "rocSPARSE was not detected properly, please disable it or provide sufficient information at configure time.")
  # Todo: figure out how to use the target defined during rocsparse installation
  # FIND_PACKAGE(ROCSPARSE REQUIRED)
  # KOKKOSKERNELS_CREATE_IMPORTED_TPL(ROCSPARSE INTERFACE LINK_LIBRARIES ${ROCSPARSE_LIBRARIES})
  # GET_TARGET_PROPERTY(ROCSPARSE_LINK_LIBRARIES ${ROCSPARSE_LIBRARIES} IMPORTED_LINK_INTERFACE_LIBRARIES)
ENDIF()

# Todo figure out how to get the library path added at link time for the TRY_COMPILE example to work
# ## Check that ROCSPARSE is found by CMake and that we can compile against it
# TRY_COMPILE(KOKKOSKERNELS_TRY_COMPILE_ROCSPARSE
#   ${KOKKOSKERNELS_TOP_BUILD_DIR}/tpl_tests
#   ${KOKKOSKERNELS_TOP_SOURCE_DIR}/cmake/compile_tests/rocsparse.cpp
#   LINK_LIBRARIES -lrocsparse
#   OUTPUT_VARIABLE KOKKOSKERNELS_TRY_COMPILE_ROCSPARSE_OUT)
# IF(NOT KOKKOSKERNELS_TRY_COMPILE_ROCSPARSE)
#   MESSAGE(FATAL_ERROR "KOKKOSKERNELS_TRY_COMPILE_ROCSPARSE_OUT=${KOKKOSKERNELS_TRY_COMPILE_ROCSPARSE_OUT}")
# ENDIF()
