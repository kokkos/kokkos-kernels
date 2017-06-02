
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR})

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/blas)
INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/blas)

ADD_SUBDIRECTORIES(gtest_unit)


SET(GTEST_SOURCE_DIR ${${PARENT_PACKAGE_NAME}_SOURCE_DIR}/../kokkos/tpls/gtest)
INCLUDE_DIRECTORIES(${GTEST_SOURCE_DIR})

#MD: test is moved to gtest folder.
#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  CrsMatrixUnitTest
#  SOURCES
#    TestCrsMatrix.cpp
#    ${TEUCHOS_STD_UNIT_TEST_MAIN}
#  COMM serial mpi
#  NUM_MPI_PROCS 1
#  FAIL_REGULAR_EXPRESSION "  FAILED  "
#  )

TRIBITS_ADD_EXECUTABLE_AND_TEST(
  ArithTraits
  SOURCES test-arith-traits.cpp
  COMM serial mpi
  NUM_MPI_PROCS 1
  )


IF (Kokkos_ENABLE_Cuda)
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/cuda)
  INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/cuda)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    blas_cuda
    SOURCES
      Test_Main.cpp
      cuda/Test_Cuda_Blas1_abs.cpp
      cuda/Test_Cuda_Blas1_asum.cpp
      cuda/Test_Cuda_Blas1_axpby.cpp
      cuda/Test_Cuda_Blas1_axpy.cpp
      cuda/Test_Cuda_Blas1_dot.cpp
      cuda/Test_Cuda_Blas1_mult.cpp
      cuda/Test_Cuda_Blas1_nrm1.cpp
      cuda/Test_Cuda_Blas1_nrm2.cpp
      cuda/Test_Cuda_Blas1_nrm2_squared.cpp
      cuda/Test_Cuda_Blas1_nrminf.cpp
      cuda/Test_Cuda_Blas1_reciprocal.cpp
      cuda/Test_Cuda_Blas1_scal.cpp
      cuda/Test_Cuda_Blas1_sum.cpp
      cuda/Test_Cuda_Blas1_update.cpp
      cuda/Test_Cuda_Blas2_gemv.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    TESTONLYLIBS kokkoskernels_gtest
    )
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    blas2_MV_Cuda
    SOURCES
      blas2_MV_Cuda
      KokkosKernels_Blas2_MV_GEMV.hpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    )
ENDIF ()

IF (Kokkos_ENABLE_OpenMP)
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/openmp)
  INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/openmp)
  
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    blas_openmp
    SOURCES
      Test_Main.cpp
      openmp/Test_OpenMP_Blas1_abs.cpp
      openmp/Test_OpenMP_Blas1_asum.cpp
      openmp/Test_OpenMP_Blas1_axpby.cpp
      openmp/Test_OpenMP_Blas1_axpy.cpp
      openmp/Test_OpenMP_Blas1_dot.cpp
      openmp/Test_OpenMP_Blas1_mult.cpp
      openmp/Test_OpenMP_Blas1_nrm1.cpp
      openmp/Test_OpenMP_Blas1_nrm2.cpp
      openmp/Test_OpenMP_Blas1_nrm2_squared.cpp
      openmp/Test_OpenMP_Blas1_nrminf.cpp
      openmp/Test_OpenMP_Blas1_reciprocal.cpp
      openmp/Test_OpenMP_Blas1_scal.cpp
      openmp/Test_OpenMP_Blas1_sum.cpp
      openmp/Test_OpenMP_Blas1_update.cpp
      openmp/Test_OpenMP_Blas2_gemv.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    TESTONLYLIBS kokkoskernels_gtest
    )
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    blas2_MV_OpenMP
    SOURCES
      blas2_MV_OpenMP
      KokkosKernels_Blas2_MV_GEMV.hpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    )
ENDIF ()

IF (Kokkos_ENABLE_Serial)
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/serial)
  INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/serial)
  
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    blas_serial
    SOURCES
      Test_Main.cpp
      serial/Test_Serial_Blas1_abs.cpp
      serial/Test_Serial_Blas1_asum.cpp
      serial/Test_Serial_Blas1_axpby.cpp
      serial/Test_Serial_Blas1_axpy.cpp
      serial/Test_Serial_Blas1_dot.cpp
      serial/Test_Serial_Blas1_mult.cpp
      serial/Test_Serial_Blas1_nrm1.cpp
      serial/Test_Serial_Blas1_nrm2.cpp
      serial/Test_Serial_Blas1_nrm2_squared.cpp
      serial/Test_Serial_Blas1_nrminf.cpp
      serial/Test_Serial_Blas1_reciprocal.cpp
      serial/Test_Serial_Blas1_scal.cpp
      serial/Test_Serial_Blas1_sum.cpp
      serial/Test_Serial_Blas1_update.cpp
      serial/Test_Serial_Blas2_gemv.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    TESTONLYLIBS kokkoskernels_gtest
    )
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    blas2_MV_Serial
    SOURCES
      blas2_MV_Serial
      KokkosKernels_Blas2_MV_GEMV.hpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    )
ENDIF ()

IF (Kokkos_ENABLE_Pthread)
  INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR}/threads)
  INCLUDE_DIRECTORIES(REQUIRED_DURING_INSTALLATION_TESTING ${CMAKE_CURRENT_SOURCE_DIR}/threads)

  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    blas_threads
    SOURCES
      Test_Main.cpp
      threads/Test_Threads_Blas1_abs.cpp
      threads/Test_Threads_Blas1_asum.cpp
      threads/Test_Threads_Blas1_axpby.cpp
      threads/Test_Threads_Blas1_axpy.cpp
      threads/Test_Threads_Blas1_dot.cpp
      threads/Test_Threads_Blas1_mult.cpp
      threads/Test_Threads_Blas1_nrm1.cpp
      threads/Test_Threads_Blas1_nrm2.cpp
      threads/Test_Threads_Blas1_nrm2_squared.cpp
      threads/Test_Threads_Blas1_nrminf.cpp
      threads/Test_Threads_Blas1_reciprocal.cpp
      threads/Test_Threads_Blas1_scal.cpp
      threads/Test_Threads_Blas1_sum.cpp
      threads/Test_Threads_Blas1_update.cpp
      threads/Test_Threads_Blas2_gemv.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    TESTONLYLIBS kokkoskernels_gtest
    )
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    blas2_MV_Threads
    SOURCES
      blas2_MV_Threads
      KokkosKernels_Blas2_MV_GEMV.hpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    )
ENDIF ()

ASSERT_DEFINED (${PACKAGE_NAME}_ENABLE_MKL)
IF (${PACKAGE_NAME}_ENABLE_MKL)
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    blas2_MV_unit_MKL
    SOURCES
      blas2_MV_MKL
      KokkosKernels_Blas2_MV_GEMV_MKL.hpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    )
ENDIF ()

ASSERT_DEFINED (${PACKAGE_NAME}_ENABLE_quadmath)
IF (${PACKAGE_NAME}_ENABLE_quadmath)
  TRIBITS_ADD_EXECUTABLE_AND_TEST(
    float128
    SOURCES float128.cpp
    COMM serial mpi
    NUM_MPI_PROCS 1
    )
ENDIF ()

# This test needs to be able to control when Kokkos gets initialized,
# so it can't use the Teuchos unit test "main" .cpp file.
#MD: Test is moved to gtest directory.
#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  findRelOffset
#  SOURCES findRelOffset.cpp
#  COMM serial mpi
#  NUM_MPI_PROCS 1
#  )

# This test needs to be able to control when Kokkos gets initialized,
# so it can't use the Teuchos unit test "main" .cpp file.
#MD:test is moved to gtest
#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  replaceSumInto
#  SOURCES replaceSumInto.cpp
#  COMM serial mpi
#  NUM_MPI_PROCS 1
#  )

# This test needs to be able to control when Kokkos gets initialized,
# so it can't use the Teuchos unit test "main" .cpp file.
#MD: test is moved to gtest
#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  replaceSumIntoLonger
#  SOURCES replaceSumIntoLonger.cpp
#  COMM serial mpi
#  NUM_MPI_PROCS 1
#  )

#This test needs to be able to control when Kokkos gets initialized,
# so it can't use the Teuchos unit test "main" .cpp file.
#MD: test is moved to gtest
#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  copyIntegers
#  SOURCES copyIntegers.cpp
#  COMM serial mpi
#  NUM_MPI_PROCS 1
#  )

# This test needs to be able to control when Kokkos gets initialized,
# so it can't use the Teuchos unit test "main" .cpp file.
#MD: test is moved to gtest
#TRIBITS_ADD_EXECUTABLE_AND_TEST(
#  MKL_sparseMatrixHandle
#  SOURCES MKL_sparseMatrixHandle.cpp
#  COMM serial mpi
#  NUM_MPI_PROCS 1
#  )
 