KOKKOSKERNELS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOSKERNELS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

KOKKOSKERNELS_ADD_EXECUTABLE(
  gmres_ex_real_A
  SOURCES ex_real_A.cpp
  )

# FIXME_SYCL CUDA_ERROR_INVALID_ADDRESS_SPACE
IF(NOT KOKKOS_ENABLE_SYCL)
KOKKOSKERNELS_ADD_EXECUTABLE_AND_TEST(
  gmres_test_prec
  SOURCES test_prec.cpp
  )
ENDIF()

