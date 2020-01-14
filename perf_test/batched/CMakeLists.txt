KOKKOSKERNELS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOSKERNELS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

# KOKKOSKERNELS_ADD_EXECUTABLE(
#   kokkosbatched_test_blockcrs_cuda.x
#   SOURCES KokkosBatched_Test_BlockCrs_Cuda.cpp
#   )

# KOKKOSKERNELS_ADD_EXECUTABLE(
#   kokkosbatched_test_blockcrs_host.x
#   SOURCES KokkosBatched_Test_BlockCrs_Host.cpp
#   )

KOKKOSKERNELS_ADD_EXECUTABLE(
  kokkosbatched_test_blocktridiag_direct.x
  SOURCES KokkosBatched_Test_BlockTridiagDirect.cpp
  )

KOKKOSKERNELS_ADD_EXECUTABLE(
  kokkosbatched_test_blocktridiag_jacobi.x
  SOURCES KokkosBatched_Test_BlockTridiagJacobi.cpp
  )

KOKKOSKERNELS_ADD_EXECUTABLE(
  kokkosbatched_test_block_jacobi.x
  SOURCES KokkosBatched_Test_BlockJacobi.cpp
  )


