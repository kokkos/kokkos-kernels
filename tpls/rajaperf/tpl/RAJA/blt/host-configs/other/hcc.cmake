# Copyright (c) 2017-2019, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

# The CMake command line to use this host config file must have certain
# defines enabled.  Here is an example script for an out-of-place build of the
# tutorial blank project template:
## hcc.sh
##!/bin/bash
#rm -rf build-hcc-release 2>/dev/null
#mkdir build-hcc-release && cd build-hcc-release

#BLT_DIR=$(git rev-parse --show-toplevel)

#cmake \
#  -DCMAKE_BUILD_TYPE=Release \
#  -DENABLE_HCC=ON -DBLT_SOURCE_DIR=${BLT_DIR} \
#  -DROCM_ARCH=gfx900 \
#  -C ${BLT_DIR}/host-configs/hcc.cmake \
#  -DCMAKE_INSTALL_PREFIX=../install-hcc-release \
#  "$@" \
#  ${BLT_DIR}/docs/tutorial/blank_project


###########################################################
# standard ROCm HCC compiler
###########################################################

set(ENABLE_HCC ON CACHE BOOL "")
set(ENABLE_OPENMP OFF CACHE BOOL "")

set(ROCM_ROOT_DIR "/opt/rocm" CACHE PATH "ROCm ROOT directory path")

set(ROCM_INCLUDE_PATH "${ROCM_ROOT_DIR}/hcc/include;${ROCM_ROOT_DIR}/include"  CACHE PATH "")
set(ROCM_CXX_LIBRARIES "-L${ROCM_ROOT_DIR}/lib -lhc_am -lhip_hcc" CACHE STRING "")

###########################################################
# specify the target architecture
#  Default with ROCm 1.7 is gfx803 (Fiji)
#  Other options:
#    gfx700  Hawaii
#    gfx803  Polaris (RX580)
#    gfx900  Vega
#    gfx901  
###########################################################
set(ROCM_ARCH_FLAG "-amdgpu-target=${ROCM_ARCH}" CACHE STRING "")

###########################################################
# get compile/link flags from hcc-config
###########################################################
# execute_process(COMMAND ${ROCM_ROOT_DIR}/hcc/bin/hcc-config --cxxflags OUTPUT_VARIABLE ROCM_CXX_COMPILE_FLAGS)
# execute_process(COMMAND ${ROCM_ROOT_DIR}/hcc/bin/hcc-config --ldflags OUTPUT_VARIABLE ROCM_CXX_LINK_FLAGS)
set(ROCM_CXX_COMPILE_FLAGS "-hc;-std=c++amp")
set(ROCM_CXX_LINK_FLAGS "-hc -std=c++amp")

set(ROCM_CXX_COMPILE_FLAGS "${ROCM_CXX_COMPILE_FLAGS};-Wno-unused-parameter;-DHCC_ENABLE_ACCELERATOR_PRINTF" CACHE STRING "")
set(ROCM_CXX_LINK_FLAGS "${ROCM_CXX_LINK_FLAGS} ${ROCM_ARCH_FLAG} ${ROCM_CXX_LIBRARIES}" CACHE STRING "")

###########################################################
# set CMake cache variables
###########################################################
set(CMAKE_CXX_COMPILER "${ROCM_ROOT_DIR}/hcc/bin/hcc" CACHE FILEPATH "ROCm HCC compiler")
# set(BLT_CXX_FLAGS "${ROCM_CXX_COMPILE_FLAGS}" CACHE STRING "HCC compiler flags")

set(CMAKE_CXX_LINK_EXECUTABLE "${CMAKE_CXX_COMPILER} ${ROCM_CXX_LINK_FLAGS} <OBJECTS> <LINK_LIBRARIES> -o <TARGET>" CACHE STRING "HCC linker command line")
