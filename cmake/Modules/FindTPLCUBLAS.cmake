if(CUBLAS_LIBRARIES AND CUBLAS_LIBRARY_DIRS AND CUBLAS_INCLUDE_DIRS)
  kokkoskernels_find_imported(CUBLAS INTERFACE
    LIBRARIES ${CUBLAS_LIBRARIES}
    LIBRARY_PATHS ${CUBLAS_LIBRARY_DIRS}
    HEADER_PATHS ${CUBLAS_INCLUDE_DIRS}
  )
elseif(CUBLAS_LIBRARIES AND CUBLAS_LIBRARY_DIRS)
  kokkoskernels_find_imported(CUBLAS INTERFACE
    LIBRARIES ${CUBLAS_LIBRARIES}
    LIBRARY_PATHS ${CUBLAS_LIBRARY_DIRS}
    HEADER cublas.h
  )
elseif(CUBLAS_LIBRARIES)
  kokkoskernels_find_imported(CUBLAS INTERFACE
    LIBRARIES ${CUBLAS_LIBRARIES}
    HEADER cublas.h
  )
elseif(CUBLAS_LIBRARY_DIRS)
  kokkoskernels_find_imported(CUBLAS INTERFACE
    LIBRARIES cublas
    LIBRARY_PATHS ${CUBLAS_LIBRARY_DIRS}
    HEADER cublas.h
  )
elseif(CUBLAS_ROOT OR KokkosKernels_CUBLAS_ROOT) # nothing specific provided, just ROOT
  kokkoskernels_find_imported(CUBLAS INTERFACE
    LIBRARIES cublas
    HEADER cublas.h
  )
elseif(CMAKE_VERSION VERSION_LESS "3.27")
  # backwards compatible way using FIND_PACKAGE(CUDA) (removed in 3.27)
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
    IF("${CUDA_CUBLAS_LIBRARIES}" MATCHES "NOTFOUND")
      FIND_PACKAGE_HANDLE_STANDARD_ARGS(TPLCUBLAS REQUIRED_VARS CUDA_CUBLAS_LIBRARIES)
    ELSE()
      KOKKOSKERNELS_CREATE_IMPORTED_TPL(CUBLAS INTERFACE
        LINK_LIBRARIES "${CUDA_CUBLAS_LIBRARIES}")
    ENDIF()
  ENDIF()
else()
  FIND_PACKAGE(CUDAToolkit REQUIRED)
  KOKKOSKERNELS_CREATE_IMPORTED_TPL(CUBLAS EXISTING_IMPORTED_TARGET CUDA::cublas)
  # X_FOUND_INFO used to print the Kokkos Kernels config summary
  get_target_property(CUBLAS_FOUND_INFO CUDA::cublas IMPORTED_LOCATION)
endif()
