KOKKOSKERNELS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
KOKKOSKERNELS_INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

KOKKOSKERNELS_ADD_EXECUTABLE(
  gmres_ex_real_A
  SOURCES ex_real_A.cpp
  )

KOKKOSKERNELS_ADD_EXECUTABLE(
  gmres_ex_cmplx_A
  SOURCES ex_cmplx_A.cpp
  )

KOKKOSKERNELS_ADD_EXECUTABLE(
  gmres_test_real_A
  SOURCES test_real_A.cpp
  )

KOKKOSKERNELS_ADD_EXECUTABLE(
  gmres_test_cmplx_A
  SOURCES test_cmplx_A.cpp
  )

# This is roughly like the Trilinos way to copy tests files,
# but it doesn't work here.

#KOKKOSKERNELS_COPY_FILES_TO_BINARY_DIR(Gmres_Ex_CopyFiles
#  SOURCE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/examples/gmres
#  SOURCE_FILES BentPipe2D100.mtx Laplace3D10.mtx Young1c.mtx
#  EXEDEPS gmres_test_real_A gmres_test_cmplx_A
#  )
