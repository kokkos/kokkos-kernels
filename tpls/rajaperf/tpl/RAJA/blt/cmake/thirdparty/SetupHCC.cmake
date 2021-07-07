# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

################################
# ROCM
################################

if (ENABLE_HCC)
    set (CMAKE_MODULE_PATH "${BLT_ROOT_DIR}/cmake/thirdparty;${CMAKE_MODULE_PATH}")
    find_package(ROCm REQUIRED)

    if (ROCM_FOUND)
        message(STATUS "ROCM  Compile Flags:  ${ROCM_CXX_COMPILE_FLAGS}")
        message(STATUS "ROCM  Include Path:   ${ROCM_INCLUDE_PATH}")
        message(STATUS "ROCM  Link Flags:     ${ROCM_CXX_LINK_FLAGS}")
        message(STATUS "ROCM  Libraries:      ${ROCM_CXX_LIBRARIES}")
        message(STATUS "ROCM  Device Arch:    ${ROCM_ARCH}")

        if (ENABLE_FORTRAN)
             message(ERROR "ROCM does not support Fortran at this time")
        endif()
    else()
        message(ERROR "ROCM Executable not found")
    endif()
endif()



# register ROCM with blt
blt_register_library(NAME rocm
                     INCLUDES ${ROCM_CXX_INCLUDE_PATH}  
                     LIBRARIES ${ROCM_CXX_LIBRARIES}  
                     COMPILE_FLAGS ${ROCM_CXX_COMPILE_FLAGS}
                     LINK_FLAGS    ${ROCM_CXX_LINK_FLAGS} 
                     DEFINES USE_ROCM)


