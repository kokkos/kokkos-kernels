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
  sparse_spmv_mv
  SOURCES KokkosSparse_spmv_mv.cpp
  )

INCLUDE_DIRECTORIES(${CMAKE_CURRENT_SOURCE_DIR}/spmv)

TRIBITS_ADD_EXECUTABLE(
  sparse_spmv
  SOURCES KokkosSparse_spmv.cpp 
  )
  
  

  

