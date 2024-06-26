IF (BLAS_LIBRARY_DIRS AND BLAS_LIBRARIES)
  KOKKOSKERNELS_FIND_IMPORTED(BLAS INTERFACE LIBRARIES ${BLAS_LIBRARIES} LIBRARY_PATHS ${BLAS_LIBRARY_DIRS})
ELSEIF (BLAS_LIBRARIES)
  KOKKOSKERNELS_FIND_IMPORTED(BLAS INTERFACE LIBRARIES ${BLAS_LIBRARIES})
ELSEIF (BLAS_LIBRARY_DIRS)
  KOKKOSKERNELS_FIND_IMPORTED(BLAS INTERFACE LIBRARIES blas LIBRARY_PATHS ${BLAS_LIBRARY_DIRS})
ELSE()
  FIND_PACKAGE(BLAS REQUIRED)
  KOKKOSKERNELS_CREATE_IMPORTED_TPL(BLAS INTERFACE LINK_LIBRARIES ${BLAS_LIBRARIES})
ENDIF()
