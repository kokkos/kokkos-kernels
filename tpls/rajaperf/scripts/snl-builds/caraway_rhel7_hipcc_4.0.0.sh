#!/usr/bin/env bash

###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors.
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
#################################################################################

BUILD_SUFFIX=snl_rhel7-hipcc-4.0.0

rm -rf build_${BUILD_SUFFIX} 2>/dev/null
mkdir build_${BUILD_SUFFIX} && cd build_${BUILD_SUFFIX}

##################################
#Caraway Build (AMD)
#################################
module purge

module load cmake/3.19.3

module load git/2.9.4

##################################
# FOR COMPUTE NODE (caraway04 GPU):

module load rocm/4.0.0

module load python/3.7.3

cmake \
-DCMAKE_BUILD_TYPE=Release \
-DENABLE_KOKKOS=ON \
-DENABLE_HIP=ON \
-DKokkos_ARCH_VEGA900=ON \
-DCMAKE_CXX_FLAGS="--gcc-toolchain=/home/projects/x86-64/gcc/8.2.0/" \
-DHIP_HIPCC_FLAGS="--gcc-toolchain=/home/projects/x86-64/gcc/8.2.0/ -std=c++17" \
-DCMAKE_CXX_STANDARD=17 \
-DCMAKE_CXX_COMPILER=hipcc .. \

make -j24;make

cd bin/
./raja-perf.exe


 
