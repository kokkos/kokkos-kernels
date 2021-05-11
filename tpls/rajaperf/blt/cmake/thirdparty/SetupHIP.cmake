# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
#
# SPDX-License-Identifier: (BSD-3-Clause)

# Author: Noel Chalmers @ Advanced Micro Devices, Inc.
# Date: March 11, 2019

################################
# HIP
################################
set (CMAKE_MODULE_PATH "${BLT_ROOT_DIR}/cmake/thirdparty;${CMAKE_MODULE_PATH}")
find_package(HIP REQUIRED)

message(STATUS "HIP version:      ${HIP_VERSION_STRING}")
message(STATUS "HIP platform:     ${HIP_PLATFORM}")

if(${HIP_PLATFORM} STREQUAL "hcc")
	set(HIP_RUNTIME_DEFINE "__HIP_PLATFORM_HCC__")
elseif(${HIP_PLATFORM} STREQUAL "nvcc")
	set(HIP_RUNTIME_DEFINE "__HIP_PLATFORM_NVCC__")
endif()
if ( IS_DIRECTORY "${HIP_ROOT_DIR}/hcc/include" ) # this path only exists on older rocm installs
        set(HIP_RUNTIME_INCLUDE_DIRS "${HIP_ROOT_DIR}/include;${HIP_ROOT_DIR}/hcc/include" CACHE STRING "")
else()
        set(HIP_RUNTIME_INCLUDE_DIRS "${HIP_ROOT_DIR}/include" CACHE STRING "")
endif()
set(HIP_RUNTIME_COMPILE_FLAGS "${HIP_RUNTIME_COMPILE_FLAGS};-D${HIP_RUNTIME_DEFINE};-Wno-unused-parameter")

# depend on 'hip', if you need to use hip
# headers, link to hip libs, and need to run your source
# through a hip compiler (hipcc)
# This is currently used only as an indicator for blt_add_hip* -- FindHIP/hipcc will handle resolution
# of all required HIP-related includes/libraries/flags.
blt_import_library(NAME      hip)

# depend on 'hip_runtime', if you only need to use hip
# headers or link to hip libs, but don't need to run your source
# through a hip compiler (hipcc)
blt_import_library(NAME          hip_runtime
                   INCLUDES      ${HIP_RUNTIME_INCLUDE_DIRS}
                   DEFINES       ${HIP_RUNTIME_DEFINES}
                   COMPILE_FLAGS ${HIP_RUNTIME_COMPILE_FLAGS}
                   TREAT_INCLUDES_AS_SYSTEM ON
                   EXPORTABLE    ${BLT_EXPORT_THIRDPARTY})
