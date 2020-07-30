FIND_PACKAGE(CUDA)

INCLUDE(FindPackageHandleStandardArgs)
IF (NOT CUDA_FOUND)
  #Important note here: this find Module is named TPLCUBLAS
  #The eventual target is named CUBLAS. To avoid naming conflicts
  #the find module is called TPLCUBLAS. This call will cause
  #the find_package call to fail in a "standard" CMake way
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(TPLCUBLAS REQUIRED_VARS CUDA_FOUND)
ELSE()
  #The libraries might be empty - OR they might explicitly be not found
  IF("${CUDA_cublas_LIBRARY}" MATCHES "NOTFOUND")
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(TPLCUBLAS REQUIRED_VARS CUDA_cublas_LIBRARY)
  ELSE()
    KOKKOSKERNELS_CREATE_IMPORTED_TPL(CUBLAS LIBRARY ${CUDA_cublas_LIBRARY})
  ENDIF()
ENDIF()
