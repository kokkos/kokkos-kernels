# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

# Author: Chip Freitag @ Advanced Micro Devices, Inc.
# Date: February 14, 2018

find_path(ROCM_PATH
     NAMES bin/hcc
     PATHS
       ENV ROCM_DIR
       ${ROCM_ROOT_DIR}
       /opt/rocm
     DOC "Path to ROCm hcc executable")


if(ROCM_PATH)
    message(STATUS "ROCM_PATH:  ${ROCM_PATH}")
    set(CMAKE_CXX_COMPILER_ID "HCC")

    set(ROCM_FOUND TRUE)

else()
    set(ROCM_FOUND FALSE)
    message(WARNING "ROCm hcc executable not found")
endif()
