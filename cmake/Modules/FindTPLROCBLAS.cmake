## MPL: v3
## MPL: 12/29/2022: CMake regular way to find a package
#FIND_PACKAGE(ROCBLAS)
#if(TARGET roc::rocsparse)
### MPL: 12/29/2022: Variable TPL_ROCBLAS_IMPORTED_NAME follows the requested convention
### of KokkosKernel (method kokkoskernels_import_tpl of kokkoskernels_tpls.cmake)
  #SET(TPL_ROCBLAS_IMPORTED_NAME roc::rocblas)
  #SET(TPL_IMPORTED_NAME roc::rocblas)
### MPL: 12/29/2022: A target comming from a TPL must follows the requested convention
### of KokkosKernel (method kokkoskernels_link_tpl of kokkoskernels_tpls.cmake)
  #ADD_LIBRARY(KokkosKernels::ROCBLAS ALIAS roc::rocblas)
#ELSE()
#  MESSAGE(FATAL_ERROR "Package ROCBLAS requested but not found")
#ENDIF()

# MPL: v2
# MPL: 12/26/2022: This bloc is not necessary anymore since ROCBLAS installation provide a cmake config file.
# Should we verify for different version of ROCBLAS ?
# GOAL: The code is commented for now and we aim to remove it
IF (ROCBLAS_LIBRARY_DIRS AND ROCBLAS_LIBRARIES)
  KOKKOSKERNELS_FIND_IMPORTED(ROCBLAS INTERFACE LIBRARIES ${ROCBLAS_LIBRARIES} LIBRARY_PATHS ${ROCBLAS_LIBRARY_DIRS})
ELSEIF (ROCBLAS_LIBRARIES)
  KOKKOSKERNELS_FIND_IMPORTED(ROCBLAS INTERFACE LIBRARIES ${ROCBLAS_LIBRARIES})
ELSEIF (ROCBLAS_LIBRARY_DIRS)
ELSE()
    # MPL: 12/26/2022: USE FIND_PACKAGE and check if the requested target is the more modern way to do it
    # MPL: 12/28/2022 : This logical bloc is based on the logical bloc coming from FindTPLCUBLAS. But instead of
    # expecting a ROCBLAS_FOUND variable to be set. We expect the TARGET roc::rocblas to be defined (more modern)
    FIND_PACKAGE(ROCBLAS)
    if(NOT TARGET roc::rocblas)
        MESSAGE( "TARGET roc::rocblas NOT FOUND")
        #Important note here: this find Module is named TPLROCBLAS
        #The eventual target is named roc::rocblas. To avoid naming conflicts
        #the find module is called TPLROCBLAS. This call will cause
        #the find_package call to fail in a "standard" CMake way
        FIND_PACKAGE_HANDLE_STANDARD_ARGS(TPLROCBLAS REQUIRED_VARS ROCBLAS_FOUND)
    ELSE()
        # MPL: 12/26/2022: USING FIND_PACKAGE_HANDLE_STANDARD_ARGS can be ok in modern CMAKE but with a Find module
        # if the package is found, we can verify that some variables are defined using FIND_PACKAGE_HANDLE_STANDARD_ARGS
        MESSAGE( "TARGET roc::rocblas FOUND")
        #The libraries might be empty - OR they might explicitly be not found
        IF("${ROCBLAS_LIBRARIES}" MATCHES "NOTFOUND")
          MESSAGE( "ROCBLAS_LIBRARIES is not found")

          FIND_PACKAGE_HANDLE_STANDARD_ARGS(TPLROCBLAS REQUIRED_VARS ROCBLAS_LIBRARIES)
        ELSE()
            MESSAGE( "ROCBLAS_LIBRARIES is not found")
            # 12/28/2022: ROCBLAS_LIBRARIES is found using find_packge which defines it as a target and not a lib
            message("TPLROCBLAS LIBRARIES VARIABLE IS ${ROCBLAS_LIBRARIES}")
            KOKKOSKERNELS_CREATE_IMPORTED_TPL(ROCBLAS INTERFACE LINK_LIBRARIES ${ROCBLAS_LIBRARIES})
        ENDIF()
    endif()
ENDIF()

## MPL: v1
#IF (ROCBLAS_LIBRARY_DIRS AND ROCBLAS_LIBRARIES)
#  KOKKOSKERNELS_FIND_IMPORTED(ROCBLAS INTERFACE LIBRARIES ${ROCBLAS_LIBRARIES} LIBRARY_PATHS ${ROCBLAS_LIBRARY_DIRS})
#ELSEIF (ROCBLAS_LIBRARIES)
#  KOKKOSKERNELS_FIND_IMPORTED(ROCBLAS INTERFACE LIBRARIES ${ROCBLAS_LIBRARIES})
#ELSEIF (ROCBLAS_LIBRARY_DIRS)
#  KOKKOSKERNELS_FIND_IMPORTED(ROCBLAS INTERFACE LIBRARIES rocblas LIBRARY_PATHS ${ROCBLAS_LIBRARY_DIRS})
#ELSEIF (KokkosKernels_ROCBLAS_ROOT)
#  KOKKOSKERNELS_FIND_IMPORTED(ROCBLAS INTERFACE
#    LIBRARIES
#      rocblas
#    LIBRARY_PATHS
#      ${KokkosKernels_ROCBLAS_ROOT}/lib
#    HEADERS
#      rocblas.h
#    HEADER_PATHS
#      ${KokkosKernels_ROCBLAS_ROOT}/include
#  )
#ELSEIF (DEFINED ENV{ROCM_PATH})
#  MESSAGE(STATUS "Detected ROCM_PATH: ENV{ROCM_PATH}")
#  SET(ROCBLAS_ROOT "$ENV{ROCM_PATH}/rocblas")
#  KOKKOSKERNELS_FIND_IMPORTED(ROCBLAS INTERFACE
#    LIBRARIES
#      rocblas
#    LIBRARY_PATHS
#      ${ROCBLAS_ROOT}/lib
#    HEADERS
#      rocblas.h
#    HEADER_PATHS
#      ${ROCBLAS_ROOT}/include
#  )
#ELSE()
#  MESSAGE(ERROR "rocBLAS was not detected properly, please disable it or provide sufficient information at configure time.")
#  # Todo: figure out how to use the target defined during rocblas installation
#  # FIND_PACKAGE(ROCBLAS REQUIRED)
#  # KOKKOSKERNELS_CREATE_IMPORTED_TPL(ROCBLAS INTERFACE LINK_LIBRARIES ${ROCBLAS_LIBRARIES})
#  # GET_TARGET_PROPERTY(ROCBLAS_LINK_LIBRARIES ${ROCBLAS_LIBRARIES} IMPORTED_LINK_INTERFACE_LIBRARIES)
#ENDIF()
