INCLUDE_DIRECTORIES(${CMAKE_CURRENT_BINARY_DIR})
INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR})

TRIBITS_ADD_EXECUTABLE(
  sparse_pcg
  SOURCES KokkosSparse_pcg.cpp
  )

TRIBITS_ADD_EXECUTABLE(
  sparse_spgemm
  SOURCES KokkosSparse_spgemm.cpp
  )

TRIBITS_ADD_EXECUTABLE(
  sparse_spmv_struct
  SOURCES KokkosSparse_spmv_struct.cpp
  )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/spmv)

TRIBITS_ADD_EXECUTABLE(
  sparse_spmv
  SOURCES KokkosSparse_spmv.cpp
  )

TRIBITS_ADD_EXECUTABLE(
  sparse_sptrsv
  SOURCES KokkosSparse_sptrsv.cpp 
  )

TRIBITS_ADD_EXECUTABLE(
  sparse_gs
  SOURCES KokkosSparse_gs.cpp
  )
