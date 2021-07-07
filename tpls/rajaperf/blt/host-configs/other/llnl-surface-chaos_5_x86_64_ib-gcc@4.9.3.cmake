# Copyright (c) 2017-2021, Lawrence Livermore National Security, LLC and
# other BLT Project Developers. See the top-level COPYRIGHT file for details
# 
# SPDX-License-Identifier: (BSD-3-Clause)

#------------------------------------------------------------------------------
# Example host-config file for the surface cluster at LLNL
#------------------------------------------------------------------------------
#
# This file provides CMake with paths / details for:
#  C,C++, & Fortran compilers + MPI & CUDA
#------------------------------------------------------------------------------

#------------------------------------------------------------------------------
# gcc@4.9.3 compilers
#------------------------------------------------------------------------------
# _blt_tutorial_surface_compiler_config_start
set(CMAKE_C_COMPILER   "/usr/apps/gnu/4.9.3/bin/gcc" CACHE PATH "")
set(CMAKE_CXX_COMPILER "/usr/apps/gnu/4.9.3/bin/g++" CACHE PATH "")

# Fortran support
set(ENABLE_FORTRAN ON CACHE BOOL "")
set(CMAKE_Fortran_COMPILER "/usr/apps/gnu/4.9.3/bin/gfortran" CACHE PATH "")
# _blt_tutorial_surface_compiler_config_end

#------------------------------------------------------------------------------
# MPI Support
#------------------------------------------------------------------------------
# _blt_tutorial_surface_mpi_config_start
set(ENABLE_MPI ON CACHE BOOL "")

set(MPI_C_COMPILER "/usr/local/tools/mvapich2-gnu-2.0/bin/mpicc" CACHE PATH "")

set(MPI_CXX_COMPILER "/usr/local/tools/mvapich2-gnu-2.0/bin/mpicc" CACHE PATH "")

set(MPI_Fortran_COMPILER "/usr/local/tools/mvapich2-gnu-2.0/bin/mpif90" CACHE PATH "")
# _blt_tutorial_surface_mpi_config_end

#------------------------------------------------------------------------------
# CUDA support
#------------------------------------------------------------------------------
# _blt_tutorial_surface_cuda_config_start
set(ENABLE_CUDA ON CACHE BOOL "")

set(CUDA_TOOLKIT_ROOT_DIR "/opt/cudatoolkit-8.0" CACHE PATH "")
set(CMAKE_CUDA_COMPILER "/opt/cudatoolkit-8.0/bin/nvcc" CACHE PATH "")
set(CMAKE_CUDA_HOST_COMPILER "${CMAKE_CXX_COMPILER}" CACHE PATH "")
set(CUDA_SEPARABLE_COMPILATION ON CACHE BOOL "")
# _blt_tutorial_surface_cuda_config_end

