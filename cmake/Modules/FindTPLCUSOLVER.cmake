FIND_PACKAGE(CUDA)

INCLUDE(FindPackageHandleStandardArgs)
IF (NOT CUDA_FOUND)
  #Important note here: this find Module is named TPLCUSOLVER
  #The eventual target is named CUSOLVER. To avoid naming conflicts
  #the find module is called TPLCUSOLVER. This call will cause
  #the find_package call to fail in a "standard" CMake way
  FIND_PACKAGE_HANDLE_STANDARD_ARGS(TPLCUSOLVER REQUIRED_VARS CUDA_FOUND)
ELSE()
  #The libraries might be empty - OR they might explicitly be not found
  IF("${CUDA_CUSOLVER_LIBRARIES}" MATCHES "NOTFOUND")
    FIND_PACKAGE_HANDLE_STANDARD_ARGS(TPLCUSOLVER REQUIRED_VARS CUDA_cusolver_LIBRARY)
  ELSE()
    KOKKOSKERNELS_CREATE_IMPORTED_TPL(CUSOLVER INTERFACE
      LINK_LIBRARIES "${CUDA_cusolver_LIBRARY}")
  ENDIF()
ENDIF()
