KOKKOS_CONFIGURE_FILE(${PACKAGE_NAME}_config.h)

KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

SET(KK_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR})
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

#I don't want the complexity of sometimes interface/sometimes not
#Just minimally force an empty dummy file
SET(SOURCES dummy.cpp)

#-----------------------------------------------------------------------------

LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/impl)
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/impl/tpls)
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/blas)
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/blas/impl)

#Include Sparse
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/sparse)
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/sparse/impl)

#Include graph
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/graph)
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/graph/impl)

#Include batched
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/batched)
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/batched/impl)

#Include Common
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/common)
LIST(APPEND KK_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/impl)


FOREACH(DIR ${KK_INCLUDE_DIRS})
  KOKKOS_INCLUDE_DIRECTORIES(${DIR})
  APPEND_GLOB(HEADERS ${DIR}/*.hpp)
ENDFOREACH()

#Blas Cuda wrapper
IF (KOKKOSKERNELS_ENABLE_TPL_CUBLAS OR KOKKOSKERNELS_ENABLE_TPL_MAGMA)
  APPEND_GLOB(HEADERS ${DIR}/impl/tpls/KokkosBlas_Cuda_tpl.hpp)
  APPEND_GLOB(SOURCES ${DIR}/impl/tpls/KokkosBlas_Cuda_tpl.cpp)
ENDIF()

#Include BLAS, Blas host wrapper
IF (KOKKOSKERNELS_ENABLE_TPL_BLAS OR KOKKOSKERNELS_ENABLE_TPL_MKL)
  APPEND_GLOB(HEADERS ${CMAKE_CURRENT_SOURCE_DIR}/impl/tpls/KokkosBlas_Host_tpl.hpp)
  APPEND_GLOB(SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/impl/tpls/KokkosBlas_Host_tpl.cpp)
ENDIF()

SET(IMPL_DIR ${CMAKE_CURRENT_SOURCE_DIR}/impl)
IF (KokkosKernels_INST_EXECSPACE_SERIAL)
  IF (KokkosKernels_INST_DOUBLE)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*inst_double_*_Serial_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_FLOAT)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*inst_float_*_Serial_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_DOUBLE)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*_complex_double_*_Serial_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_FLOAT)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*_complex_float_*_Serial_*.cpp)
  ENDIF()
ENDIF()

IF (KokkosKernels_INST_EXECSPACE_THREADS)
  IF (KokkosKernels_INST_DOUBLE)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*inst_double_*_Threads_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_FLOAT)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*inst_float_*_Threads_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_DOUBLE)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*_complex_double_*_Threads_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_FLOAT)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*_complex_float_*_Threads_*.cpp)
  ENDIF()
ENDIF()

IF (KokkosKernels_INST_EXECSPACE_OPENMP)
  IF (KokkosKernels_INST_DOUBLE)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*inst_double_*_OpenMP_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_FLOAT)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*inst_float_*_OpenMP_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_DOUBLE)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*_complex_double_*_OpenMP_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_FLOAT)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*_complex_float_*_OpenMP_*.cpp)
  ENDIF()
ENDIF()

IF (KokkosKernels_INST_EXECSPACE_CUDA)
  APPEND_GLOB(SOURCES ${IMPL_DIR}/tpls/*_Cuda_*.cpp)
  IF (KokkosKernels_INST_DOUBLE)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*inst_double_*_Cuda_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_FLOAT)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*inst_float_*_Cuda_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_DOUBLE)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*_complex_double_*_Cuda_*.cpp)
  ENDIF()
  IF (KokkosKernels_INST_COMPLEX_FLOAT)
    APPEND_GLOB(SOURCES ${IMPL_DIR}/generated_specializations_cpp/*/*_complex_float_*_Cuda_*.cpp)
  ENDIF()
ENDIF()

LIST(APPEND HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_config.h)
#LIST(APPEND HEADERS ${CMAKE_CURRENT_BINARY_DIR}/${PACKAGE_NAME}_ETIHelperMacros.h)

#-----------------------------------------------------------------------------

KOKKOSKERNELS_ADD_LIBRARY(
  kokkoskernels
  HEADERS ${HEADERS}
  SOURCES ${SOURCES}
)

IF (KOKKOSKERNELS_HAS_TRILINOS)
  #no linking commands required - tribits does this
ELSEIF()
ELSE()
  TARGET_LINK_LIBRARIES(kokkoskernels PUBLIC Kokkos::kokkos)
ENDIF()


KOKKOSKERNELS_LIB_INCLUDE_DIRECTORIES(kokkoskernels ${KK_INCLUDE_DIRS})

KOKKOS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/impl/generated_specializations_hpp)
APPEND_GLOB(HEADERS_GENERATED ${CMAKE_CURRENT_SOURCE_DIR}/impl/generated_specializations_hpp/*.hpp)
INSTALL(FILES ${HEADERS_GENERATED} DESTINATION ${KOKKOSKERNELS_HEADER_DIR}/generated_specializations_hpp/)



#
# Make a trivial change here if you add a file either to this
# directory, or to the impl/ subdirectory.  Keep making changes
# whenever you add or remove a .cpp or .hpp file, and everything will
# be OK.  Here is another such change, and here is yet another.
#

