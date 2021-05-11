###############################################################################
# Copyright (c) 2017-20, Lawrence Livermore National Security, LLC
# and RAJA Performance Suite project contributors. 
# See the RAJAPerf/COPYRIGHT file for details.
#
# SPDX-License-Identifier: (BSD-3-Clause)
###############################################################################

project(RAJAPerfSuite CXX)
cmake_minimum_required(VERSION 3.9)

option(ENABLE_RAJA_SEQUENTIAL "Run sequential variants of RAJA kernels. Disable
this, and all other variants, to run _only_ raw C loops." On)
option(ENABLE_KOKKOS "Include Kokkos implementations of the kernels in the RAJA Perfsuite" Off)
option(INFRASTRUCTURE_ONLY "Only build the infrastructure, don't build kernels" ON)

if(INFRASTRUCTURE_ONLY)
  add_definitions("-DRAJAPERF_INFRASTRUCTURE_ONLY")
endif()
#
# Initialize the BLT build system
#

if (PERFSUITE_ENABLE_WARNINGS)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wextra -Werror")
endif()

set(ENABLE_TESTS Off CACHE BOOL "Enable BLT and RAJA tests")
set(ENABLE_EXAMPLES Off CACHE BOOL "Enable RAJA examples")
set(ENABLE_EXERCISES Off CACHE BOOL "Enable RAJA exercises")
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER})
set(CMAKE_CUDA_HOST_LINK_LAUNCHER ${CMAKE_CUDA_COMPILER})
include(blt/SetupBLT.cmake)

set(CMAKE_CXX_STANDARD 14)
set(BLT_CXX_STANDARD 14)
include(blt/SetupBLT.cmake)

#
# Define RAJA settings...
#

set(ENABLE_TESTS Off CACHE BOOL "")
set(ENABLE_EXAMPLES Off CACHE BOOL "")
set(ENABLE_DOCUMENTATION Off CACHE BOOL "")

set(ENABLE_TBB Off CACHE BOOL "")

set(RAJA_USE_CHRONO On CACHE BOOL "")

set(RAJA_RANGE_ALIGN 4)
set(RAJA_RANGE_MIN_LENGTH 32)
set(RAJA_DATA_ALIGN 64)


# exclude RAJA make targets from top-level build...
if(NOT INFRASTRUCTURE_ONLY)
add_subdirectory(tpl/RAJA)

get_property(RAJA_INCLUDE_DIRS DIRECTORY tpl/RAJA PROPERTY INCLUDE_DIRECTORIES)
include_directories(${RAJA_INCLUDE_DIRS})
endif()
#
# Setup variables to pass to Perf suite
#

#
# These (hopefully temporary) macro constants are needed to work-around
# performance issues in the xl compiler.
#
if (ENABLE_RAJA_SEQUENTIAL)
  add_definitions(-DRUN_RAJA_SEQ)
endif ()
if (ENABLE_OPENMP)
  add_definitions(-DRUN_OPENMP)
endif ()

set(RAJA_PERFSUITE_VERSION_MAJOR 0)
set(RAJA_PERFSUITE_VERSION_MINOR 10)
set(RAJA_PERFSUITE_VERSION_PATCHLEVEL 0)

if(NOT INFRASTUCTURE_ONLY)
set(RAJA_PERFSUITE_DEPENDS RAJA)
endif()

if (ENABLE_OPENMP)
  list(APPEND RAJA_PERFSUITE_DEPENDS openmp)
endif()
if (ENABLE_CUDA)
  list(APPEND RAJA_PERFSUITE_DEPENDS cuda)
endif()
# HIP is used with AMD / VEGA GPU
# Neatly separate RAJAPerf Suite and Kokkos handling of HIP
if ((ENABLE_HIP) AND (NOT ENABLE_KOKKOS))
#if (ENABLE_HIP)
  list(APPEND RAJA_PERFSUITE_DEPENDS hip)
endif()

set(RAJAPERF_BUILD_SYSTYPE $ENV{SYS_TYPE})
set(RAJAPERF_BUILD_HOST $ENV{HOSTNAME})

if (ENABLE_CUDA)
  set(CMAKE_CUDA_STANDARD 14)
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -restrict -arch ${CUDA_ARCH} --expt-extended-lambda --expt-relaxed-constexpr")

  set(RAJAPERF_COMPILER "${CUDA_NVCC_EXECUTABLE}")
  list(APPEND RAJAPERF_COMPILER ${CMAKE_CXX_COMPILER})
  set(RAJAPERF_COMPILER_OPTIONS "${CUDA_NVCC_FLAGS}")
elseif (ENABLE_HIP)
  set(RAJAPERF_COMPILER "${HIP_HIPCC_EXECUTABLE}")
  list(APPEND RAJAPERF_COMPILER ${CMAKE_CXX_COMPILER})
  set(RAJAPERF_COMPILER_OPTIONS "${HIP_HIPCC_FLAGS}")
else()
  set(RAJAPERF_COMPILER "${CMAKE_CXX_COMPILER}")
  string(TOUPPER ${CMAKE_BUILD_TYPE} RAJAPERF_BUILD_TYPE)
  set(RAJAPERF_COMPILER_OPTIONS "${CMAKE_CXX_FLAGS_${RAJAPERF_BUILD_TYPE}}")
  list(APPEND RAJAPERF_COMPILER_OPTIONS ${CMAKE_CXX_FLAGS})
endif()

configure_file(${CMAKE_CURRENT_SOURCE_DIR}/src/rajaperf_config.hpp.in
  ${CMAKE_CURRENT_BINARY_DIR}/bin/rajaperf_config.hpp)

# Make sure RAJA flag propagate
set (CUDA_NVCC_FLAGS ${RAJA_NVCC_FLAGS})
set (HIP_HIPCC_FLAGS ${RAJA_HIPCC_FLAGS})

# HACKS TO FIX COMPILATION ISSUES
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/tpl/RAJA/include/)


if(NOT INFRASTRUCTURE_ONLY)
# ENABLE KOKKOS IS A RAJA PERFSUITE OPTION
if(ENABLE_KOKKOS)
  add_definitions(-DRUN_KOKKOS)
  if(ENABLE_HIP)
    set(Kokkos_ENABLE_HIP ON CACHE BOOL "Kokkos builds with AMD HIP require a ... build...AJP FINISH")	
    set(Kokkos_ARCH_VEGA900 ON CACHE BOOL "Docstring") #TODO: better
    #set_property(GLOBAL PROPERTY RULE_LAUNCH_COMPILE /ascldap/users/ajpowel/RAJAPerf/amd_build/compiler_unscrewer)
  endif()
  if(ENABLE_TARGET_OPENMP)
    set(Kokkos_ENABLE_OPENMPTARGET ON CACHE BOOL "Docstring")	
    set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "Docstring") #TODO: better
    set(CMAKE_CXX_STANDARD 17)
    set(BLT_CXX_STANDARD 17)
    set(RAJA_ENABLE_TARGET_OPENMP ON CACHE BOOL "Docstring")
    if(NOT CMAKE_BUILD_TYPE MATCHES Debug)
      if(NOT EXPERIMENTAL_BUILD)
	      message(FATAL_ERROR "Kokkos builds with OpenMPTarget require a Debug build to succeed at the moment. Rebuild with CMAKE_BUILD_TYPE=Debug. If you're a compiler developer, rebuild with -DEXPERIMENTAL_BUILD=ON")
      endif()
    endif()







    #add_definitions(-DRAJA_ENABLE_TARGET_OPENMP)
  endif()

# ENABLE_CUDA IS A RAJA PERFSUITE OPTION
  if(ENABLE_CUDA)
	  set(Kokkos_ENABLE_CUDA ON CACHE BOOL "Docstring")
	  set(Kokkos_ENABLE_CUDA_LAMBDA ON CACHE BOOL "Docstring")
	  set(Kokkos_ARCH_VOLTA70 ON CACHE BOOL "Docstring") #TODO: better
    enable_language(CUDA)
  endif()
  if(ENABLE_OPENMP)
    #set(Kokkos_ENABLE_OPENMP CACHE BOOL ON)
    set(Kokkos_ENABLE_OPENMP ON CACHE BOOL "Docstring")
  endif()

  add_subdirectory(tpl/kokkos)
  if(ENABLE_CUDA)
	  #get_target_property(kokkos_core_files kokkoscore SOURCES)
	  #get_target_property(kokkos_container_files kokkoscontainers SOURCES)
	  ##message(STATUS "KOKKOS FILES: ${kokkos_core_files}")
	  #foreach(kokkos_core_file IN LISTS kokkos_core_files)
	  #        set_source_files_properties(${kokkos_core_file} PROPERTIES COMPILE_LANGUAGE CUDA)
	  #        get_source_file_property(local ${kokkos_core_file} LANGUAGE) 
	  #        set(remote "")
	  #        #get_source_file_property(remote ${kokkos_core_file} DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/tpl/kokkos/core/src LANGUAGE) 
	  #        message(STATUS "Kokkos file: ${kokkos_core_file} ${local} ${remote}")
	  #        
	  #endforeach()
    #set_target_properties(kokkoscore kokkoscontainers PROPERTIES LANGUAGE CUDA)
  endif()
  get_property(KOKKOS_INCLUDE_DIRS DIRECTORY tpl/kokkos PROPERTY INCLUDE_DIRECTORIES)
  include_directories(${KOKKOS_INCLUDE_DIRS})
  
  list(APPEND RAJA_PERFSUITE_DEPENDS kokkos)
endif()
endif() # INFRASTRUCTURE_ONLY

#
# Each directory in the perf suite has its own CMakeLists.txt file.
# DZP, AJP, DB, DA fixes
add_subdirectory(src)
