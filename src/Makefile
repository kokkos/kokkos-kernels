#Settings for what to build
KOKKOS_DEVICES = "Serial"
KOKKOS_ARCH = "SNB,Kepler35"

# To enable Trilinos you need to copy KokkosKernels_config.h and KokkosKernels_ETIHelperMacros.h 
# from an appropriately configure Trilinos build directory (you don't need to build).
# The rest is accessed from Trilinos source.
BUILD_CUSPARSE = 0
BUILD_MKL = 0

#Path settings
KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOS_KERNELS_PATH = ${HOME}/Kokkos/kokkos-kernels
KOKKOS_KERNELS_INSTALL_PATH = ${HOME}/Kokkos/kokkos-kernels-install

#What to instantiate
KOKKOS_KERNELS_SCALARS = double
KOKKOS_KERNELS_LAYOUTS = LayoutRight,LayoutLeft
KOKKOS_KERNELS_SPACES = Serial,HostSpace

#MKL_PATH=/home/projects/install/rhel6-x86_64/sems/compiler/intel/16.0.1/base/mkl
#On Sandia Testbeds this should work:
#MKL_PATH=${MKLROOT}
#On Sandia SEMS modules this should work:
MKL_PATH=${SEMS_INTEL_ROOT}/mkl
MKL_LIBS=-L${MKL_PATH}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core

SRC_BLAS = $(wildcard ${KOKKOS_KERNELS_PATH}/src/impl/generated_specializations/*/KokkosBlas*.cpp)
SRC_SPARSE = $(wildcard ${KOKKOS_KERNELS_PATH}/src/impl/generated_specializations/*/KokkosSparse*.cpp)
HEADERS = $(wildcard ${KOKKOS_KERNELS_PATH}/src/impl/*.hpp) 
HEADERS += $(wildcard ${KOKKOS_KERNELS_PATH}/src/impl/generated_specializations/*/*.hpp) 

default: build
	echo "Start Build"

config_headers: 
	touch KokkosKernels_ETIHelperMacros.h
	touch KokkosKernels_config.h


CXXFLAGS = -g -O3 -I./ -I${KOKKOS_KERNELS_PATH}/src -I${KOKKOS_KERNELS_PATH}/src/impl
LINKFLAGS = -g -O3
LIB = 

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/config/nvcc_wrapper
else
CXX = g++
endif

LINK = ${CXX}

ifeq (${BUILD_MKL}, 1)
CXXFLAGS += -I${MKL_PATH}/include -DHAVE_MKL -mkl
LIB += ${MKL_LIBS} -mkl
endif

ifeq (${BUILD_CUSPARSE}, 1)
CXXFLAGS += -I${CUDA_ROOT}/include -DHAVE_CUSPARSE -DKOKKOS_USE_CUSPARSE
LIB += -lcusparse
endif

tmp := $(shell echo "/* ---------------------------------------------" > KokkosKernels_config.tmp)
tmp := $(shell echo "Makefile constructed configuration:" >> KokkosKernels_config.tmp)
tmp := $(shell date >> KokkosKernels_config.tmp)
tmp := $(shell echo "----------------------------------------------*/" >> KokkosKernels_config.tmp)

#ETI Macros Scalars
KOKKOS_KERNELS_INTERNAL_INST_SCALAR_DOUBLE=$(strip $(shell echo $(KOKKOS_KERNELS_SCALARS) | grep "double" | wc -l))
#ETI Macros Layouts
KOKKOS_KERNELS_INTERNAL_INST_LAYOUT_LAYOUTLEFT=$(strip $(shell echo $(KOKKOS_KERNELS_LAYOUTS) | grep "LayoutLeft" | wc -l))
KOKKOS_KERNELS_INTERNAL_INST_LAYOUT_LAYOUTRIGHT=$(strip $(shell echo $(KOKKOS_KERNELS_LAYOUTS) | grep "LayoutRight" | wc -l))
#ETI Macros Execution Spaces
KOKKOS_KERNELS_INTERNAL_INST_EXECSPACE_SERIAL=$(strip $(shell echo $(KOKKOS_KERNELS_SPACES) | grep "Serial" | wc -l))
#ETI Macros Memory Spaces
KOKKOS_KERNELS_INTERNAL_INST_MEMSPACE_HOSTSPACE=$(strip $(shell echo $(KOKKOS_KERNELS_SPACES) | grep "HostSpace" | wc -l))

ifeq ($(KOKKOS_KERNELS_INTERNAL_INST_SCALAR_DOUBLE), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_SCALAR_DOUBLE" >> KokkosKernels_config.tmp )
endif

ifeq ($(KOKKOS_KERNELS_INTERNAL_INST_LAYOUT_LAYOUTLEFT), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_LAYOUT_LAYOUTLEFT" >> KokkosKernels_config.tmp )
endif

ifeq ($(KOKKOS_KERNELS_INTERNAL_INST_LAYOUT_LAYOUTRIGHT), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_LAYOUT_LAYOUTRIGHT" >> KokkosKernels_config.tmp )
endif

ifeq ($(KOKKOS_KERNELS_INTERNAL_INST_EXECSPACE_SERIAL), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_EXECSPACE_SERIAL" >> KokkosKernels_config.tmp )
endif

ifeq ($(KOKKOS_KERNELS_INTERNAL_INST_MEMSPACE_HOSTSPACE), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_EXECSPACE_HOSTSPACE" >> KokkosKernels_config.tmp )
endif

KOKKOS_KERNELS_INTERNAL_LS_CONFIG := $(shell ls KokkosKernels_config.h)
ifeq ($(KOKKOS_KERNELS_INTERNAL_LS_CONFIG), KokkosKernels_config.h)
  KOKKOS_KERNELS_INTERNAL_NEW_CONFIG := $(strip $(shell diff KokkosKernels_config.h KokkosKernels_config.tmp | grep define | wc -l))
else
  KOKKOS_KERNELS_INTERNAL_NEW_CONFIG := 1
endif

ifneq ($(KOKKOS_KERNELS_INTERNAL_NEW_CONFIG), 0)
  tmp := $(shell cp KokkosKernels_config.tmp KokkosKernels_config.h)
endif

vpath %.cpp $(sort $(dir $(SRC_SPARSE)))
vpath %.cpp $(sort $(dir $(SRC_BLAS)))

DEPFLAGS = -M

SRC_BLAS_NODIR = $(notdir $(SRC_BLAS))
SRC_SPARSE_NODIR = $(notdir $(SRC_SPARSE))
OBJ_BLAS = $(SRC_BLAS_NODIR:.cpp=.o)
OBJ_SPARSE = $(SRC_SPARSE_NODIR:.cpp=.o)

include $(KOKKOS_PATH)/Makefile.kokkos

LIBRARY=kokkos_kernels.a

build: $(LIBRARY)

$(LIBRARY): $(OBJ_SPARSE) $(OBJ_BLAS) 
	ar cr ${LIBRARY} $(OBJ_SPARSE) $(OBJ_BLAS)
	ranlib ${LIBRARY}

clean: kokkos-clean 
	rm -f *.o *.cuda *.host KokkosKernels_config.h KokkosKernels_ETIHelperMacros.h

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) ${HEADERS} KokkosKernels_config.h 
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
