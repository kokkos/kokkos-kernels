#This assume SuperLU >= 5.0. We don't worry about older versions.
KOKKOSKERNELS_FIND_IMPORTED(SUPERLU LIBRARY superlu HEADER supermatrix.h)
SET(SUPERLU_LIBS KokkosKernels::SUPERLU)
IF (TARGET KokkosKernels::BLAS) #This is an interface library
  #I don't like doing this since it breaks the abstraction of
  #a target is just a thing we link to. CMake doesn't allow
  #us to pass in interface targets to try_compile
  GET_TARGET_PROPERTY(SUPERLU_BLAS_LIBS KokkosKernels::BLAS INTERFACE_LINK_LIBRARIES)
  LIST(APPEND SUPERLU_LIBS ${SUPERLU_BLAS_LIBS})
ENDIF()
IF (TARGET KokkosKernels::LAPACK)
  #I don't like doing this since it breaks the abstraction of
  #a target is just a thing we link to try_compile
  GET_TARGET_PROPERTY(SUPERLU_LAPACK_LIBS KokkosKernels::LAPACK INTERFACE_LINK_LIBRARIES)
  LIST(APPEND SUPERLU_LIBS ${SUPERLU_LAPACK_LIBS})
ENDIF()

TRY_COMPILE(SUPERLU_COMPILE_SUCCEEDS)
  ${KOKKOSKERNELS_TOP_BUILD_DIR}/tpl_tests
  ${KOKKOSKERNELS_TOP_SOURCE_DIR}/cmake/compile_tests/superlu_test.cpp
  LINK_LIBRARIES ${SUPERLU_LIBS}
)

IF (NOT SUPERLU_COMPILE_SUCCEEDS)
  MESSAGE(WARNING "SuperLU failed to correctly compile test."
    " The most likely failure is missing or incorrect BLAS libraries"
    " Please ensure that KokkosKernels is built with same BLAS as SuperLU")
  IF (TARGET KokkosKernels::BLAS)
    MESSAGE(WARNING "KokkosKernels is using BLAS: ${SUPERLU_BLAS_LIBS}")
  ENDIF()
  IF (TARGET KokkosKernels::LAPACK)
    MESSAGE(WARNING "KokkosKernels is using LAPACK: ${SUPERLU_LAPACK_LIBS}")
  ENDIF()
  INCLUDE(FindPackageHandleStandardArgs)
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(TPLSUPERLU DEFAULT_MSG SUPERLU_CORRECT)
ENDIF()
