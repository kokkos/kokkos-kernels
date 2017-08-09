#=======================================================================
#=================== Settings ==========================================
#=======================================================================

ifndef KOKKOSKERNELS_PATH
  MAKEFILE_PATH := $(abspath $(lastword $(MAKEFILE_LIST)))
  KOKKOSKERNELS_PATH = $(subst Makefile.minimal,,$(MAKEFILE_PATH))..
endif

KOKKOS_PATH = ${KOKKOSKERNELS_PATH}/../kokkos


KOKKOS_DEVICES=OpenMP

KOKKOSKERNELS_SCALARS="'double,complex<double>'"
KOKKOSKERNELS_OPTIONS=eti-only

ifndef PREFIX
  PREFIX = ${CURDIR}/install
endif
KOKKOSKERNELS_INSTALL_PATH = ${PREFIX}

default: build

CXX = g++

CXXFLAGS = -O3 -g
LINK = ${CXX}
LINKFLAGS =

DEPFLAGS = -M

OBJ =
LIB = -lpthread


#=======================================================================
#=================== Include Kokkos and KokkosKernels ==================
#=======================================================================

include $(KOKKOSKERNELS_PATH)/Makefile.kokkos-kernels

build: kokkoskernels-build-lib

install: kokkoskernels-install
 
