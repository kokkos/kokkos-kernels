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
KOKKOSKERNELS_PATH = ${HOME}/Kokkos/kokkos-kernels
KOKKOSKERNELS_INSTALL_PATH = ${HOME}/Kokkos/kokkos-kernels-install

#What to instantiate
#
# TODO CMake: Connect CMake to these variables
#
KOKKOSKERNELS_SCALARS = double
KOKKOSKERNELS_LAYOUTS = LayoutRight,LayoutLeft
KOKKOSKERNELS_SPACES = Serial,HostSpace

#MKL_PATH=/home/projects/install/rhel6-x86_64/sems/compiler/intel/16.0.1/base/mkl
#On Sandia Testbeds this should work:
#MKL_PATH=${MKLROOT}
#On Sandia SEMS modules this should work:
MKL_PATH=${SEMS_INTEL_ROOT}/mkl
MKL_LIBS=-L${MKL_PATH}/lib/intel64 -lmkl_intel_lp64 -lmkl_intel_thread -lmkl_core

default: build-lib
	echo "Start Build"


CXXFLAGS = -g -O3 -I./ -I${KOKKOSKERNELS_PATH}/src -I${KOKKOSKERNELS_PATH}/src/impl
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

#-------------------------------------------------------------------------
#-----Generate KokkosKernels_config.h-------------------------------------
#-------------------------------------------------------------------------

tmp := $(shell echo "/* ---------------------------------------------" > KokkosKernels_config.tmp)
tmp := $(shell echo "Makefile constructed configuration:" >> KokkosKernels_config.tmp)
tmp := $(shell date >> KokkosKernels_config.tmp)
tmp := $(shell echo "----------------------------------------------*/" >> KokkosKernels_config.tmp)

#ETI Macros Scalars
KOKKOSKERNELS_INTERNAL_INST_DOUBLE=$(strip $(shell echo $(KOKKOSKERNELS_SCALARS) | grep "double" | wc -l))
KOKKOSKERNELS_INTERNAL_INST_FLOAT=$(strip $(shell echo $(KOKKOSKERNELS_SCALARS) | grep "double" | wc -l))
KOKKOSKERNELS_INTERNAL_INST_COMPLEX_DOUBLE_=$(strip $(shell echo $(KOKKOSKERNELS_SCALARS) | grep "complex<double>" | wc -l))
KOKKOSKERNELS_INTERNAL_INST_COMPLEX_FLOAT_=$(strip $(shell echo $(KOKKOSKERNELS_SCALARS) | grep "complex<double>" | wc -l))

#
# TODO CMake: create macros corresponding to CMake options
#
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_DOUBLE), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_DOUBLE" >> KokkosKernels_config.tmp )
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_FLOAT), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_FLOAT" >> KokkosKernels_config.tmp )
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_COMPLEX_DOUBLE_), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_COMPLEX_DOUBLE" >> KokkosKernels_config.tmp )
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_COMPLEX_FLOAT_), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_COMPLEX_FLOAT" >> KokkosKernels_config.tmp )
endif
tmp := $(shell echo "\#if defined(KOKKOSKERNELS_INST_COMPLEX_DOUBLE)" >> KokkosKernels_config.tmp )
tmp := $(shell echo "\#define KOKKOSKERNELS_INST_KOKKOS_COMPLEX_DOUBLE_)" >> KokkosKernels_config.tmp )
tmp := $(shell echo "\#endif" >> KokkosKernels_config.tmp )
tmp := $(shell echo "\#if defined(KOKKOSKERNELS_INST_COMPLEX_FLOAT)" >> KokkosKernels_config.tmp )
tmp := $(shell echo "\#define KOKKOSKERNELS_INST_KOKKOS_COMPLEX_FLOAT_)" >> KokkosKernels_config.tmp )
tmp := $(shell echo "\#endif" >> KokkosKernels_config.tmp )


#ETI Macros Layouts
KOKKOSKERNELS_INTERNAL_INST_LAYOUTLEFT=$(strip $(shell echo $(KOKKOSKERNELS_LAYOUTS) | grep "LayoutLeft" | wc -l))
KOKKOSKERNELS_INTERNAL_INST_LAYOUTRIGHT=$(strip $(shell echo $(KOKKOSKERNELS_LAYOUTS) | grep "LayoutRight" | wc -l))

ifeq ($(KOKKOSKERNELS_INTERNAL_INST_LAYOUTLEFT), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_LAYOUTLEFT" >> KokkosKernels_config.tmp )
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_LAYOUTRIGHT), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_LAYOUTRIGHT" >> KokkosKernels_config.tmp )
endif

#ETI Macros Execution Spaces
KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_SERIAL=$(strip $(shell echo $(KOKKOSKERNELS_SPACES) | grep "Serial" | wc -l))
KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_OPENMP=$(strip $(shell echo $(KOKKOSKERNELS_SPACES) | grep "OpenMP" | wc -l))
KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_THREADS=$(strip $(shell echo $(KOKKOSKERNELS_SPACES) | grep "Pthread" | wc -l))
KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_CUDA=$(strip $(shell echo $(KOKKOSKERNELS_SPACES) | grep "Cuda" | wc -l))

ifeq ($(KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_SERIAL), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_EXECSPACE_SERIAL" >> KokkosKernels_config.tmp )
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_OPENMP), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_EXECSPACE_OPENMP" >> KokkosKernels_config.tmp )
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_THREADS), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_EXECSPACE_THREADS" >> KokkosKernels_config.tmp )
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_CUDA), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_EXECSPACE_CUDA" >> KokkosKernels_config.tmp )
endif

#ETI Macros Memory Spaces
KOKKOSKERNELS_INTERNAL_INST_MEMSPACE_HOSTSPACE=$(strip $(shell echo $(KOKKOSKERNELS_SPACES) | grep "HostSpace" | wc -l))
KOKKOSKERNELS_INTERNAL_INST_MEMSPACE_HBWSPACE=$(strip $(shell echo $(KOKKOSKERNELS_SPACES) | grep "HBWSpace" | wc -l))
KOKKOSKERNELS_INTERNAL_INST_MEMSPACE_CUDASPACE=$(strip $(shell echo $(KOKKOSKERNELS_SPACES) | grep "CudaSpace" | wc -l))
KOKKOSKERNELS_INTERNAL_INST_MEMSPACE_CUDAUVMSPACE=$(strip $(shell echo $(KOKKOSKERNELS_SPACES) | grep "CudaUVMSpace" | wc -l))

#
# TODO CMake: use MEMSPACE here, not EXECSPACE
#
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_MEMSPACE_HOSTSPACE), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_MEMSPACE_HOSTSPACE" >> KokkosKernels_config.tmp )
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_MEMSPACE_HBWSPACE), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_MEMSPACE_HBWSPACE" >> KokkosKernels_config.tmp )
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_MEMSPACE_CUDASPACE), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_MEMSPACE_CUDASPACE" >> KokkosKernels_config.tmp )
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_MEMSPACE_CUDAUVMSPACE), 1)
  tmp := $(shell echo "\#define KOKKOSKERNELS_INST_MEMSPACE_CUDAUVMSPACE" >> KokkosKernels_config.tmp )
endif

KOKKOSKERNELS_INTERNAL_LS_CONFIG := $(shell ls KokkosKernels_config.h)
ifeq ($(KOKKOSKERNELS_INTERNAL_LS_CONFIG), KokkosKernels_config.h)
  KOKKOSKERNELS_INTERNAL_NEW_CONFIG := $(strip $(shell diff KokkosKernels_config.h KokkosKernels_config.tmp | grep define | wc -l))
else
  KOKKOSKERNELS_INTERNAL_NEW_CONFIG := 1
endif

ifneq ($(KOKKOSKERNELS_INTERNAL_NEW_CONFIG), 0)
  tmp := $(shell cp KokkosKernels_config.tmp KokkosKernels_config.h)
endif


#-------------------------------------------------------------------------
#-----Add src Files-------------------------------------
#-------------------------------------------------------------------------

SRC_BLAS = 
SRC_SPARSE = 

ifeq ($(KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_SERIAL), 1)
SRC_BLAS += $(wildcard ${KOKKOSKERNELS_PATH}/src/impl/generated_specializations_cpp/*/KokkosBlas*Serial*.cpp)
SRC_SPARSE += $(wildcard ${KOKKOSKERNELS_PATH}/src/impl/generated_specializations_cpp/*/KokkosSparse*Serial*.cpp)
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_OPENMP), 1)
SRC_BLAS += $(wildcard ${KOKKOSKERNELS_PATH}/src/impl/generated_specializations_cpp/*/KokkosBlas*OpenMP*.cpp)
SRC_SPARSE += $(wildcard ${KOKKOSKERNELS_PATH}/src/impl/generated_specializations_cpp/*/KokkosSparse*OpenMP*.cpp)
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_THREADS), 1)
SRC_BLAS += $(wildcard ${KOKKOSKERNELS_PATH}/src/impl/generated_specializations_cpp/*/KokkosBlas*Threads*.cpp)
SRC_SPARSE += $(wildcard ${KOKKOSKERNELS_PATH}/src/impl/generated_specializations_cpp/*/KokkosSparse*Threads*.cpp)
endif
ifeq ($(KOKKOSKERNELS_INTERNAL_INST_EXECSPACE_CUDA), 1)
SRC_BLAS += $(wildcard ${KOKKOSKERNELS_PATH}/src/impl/generated_specializations_cpp/*/KokkosBlas*Cuda*.cpp)
SRC_SPARSE += $(wildcard ${KOKKOSKERNELS_PATH}/src/impl/generated_specializations_cpp/*/KokkosSparse*Cuda*.cpp)
endif

HEADERS = $(wildcard ${KOKKOSKERNELS_PATH}/src/impl/*.hpp)
HEADERS += $(wildcard ${KOKKOSKERNELS_PATH}/src/impl/generated_specializations_hpp/*/*.hpp)

vpath %.cpp $(sort $(dir $(SRC_SPARSE)))
vpath %.cpp $(sort $(dir $(SRC_BLAS)))

DEPFLAGS = -M

SRC_BLAS_NODIR = $(notdir $(SRC_BLAS))
SRC_SPARSE_NODIR = $(notdir $(SRC_SPARSE))
OBJ_BLAS = $(SRC_BLAS_NODIR:.cpp=.o)
OBJ_SPARSE = $(SRC_SPARSE_NODIR:.cpp=.o)

include $(KOKKOS_PATH)/Makefile.kokkos

LIBRARY=libkokkos_kernels.a

build-lib: $(LIBRARY)

install: build-lib
	mkdir -p $(KOKKOSKERNELS_INSTALL_PATH)
	mkdir -p $(KOKKOSKERNELS_INSTALL_PATH)/include
	mkdir -p $(KOKKOSKERNELS_INSTALL_PATH)/lib
	cp $(COPY_FLAG) KokkosKernels_config.h $(KOKKOSKERNELS_INSTALL_PATH)/include
	cp $(COPY_FLAG) ${KOKKOSKERNELS_PATH}/src/*.hpp $(KOKKOSKERNELS_INSTALL_PATH)/include
	cp $(COPY_FLAG) ${KOKKOSKERNELS_PATH}/src/impl/*.hpp $(KOKKOSKERNELS_INSTALL_PATH)/include
	cp $(COPY_FLAG) -r ${KOKKOSKERNELS_PATH}/src/impl/generated_specializations_hpp $(KOKKOSKERNELS_INSTALL_PATH)/include
	cp $(COPY_FLAG) libkokkos_kernels.a $(KOKKOSKERNELS_INSTALL_PATH)/lib

$(LIBRARY): $(OBJ_SPARSE) $(OBJ_BLAS) 
	ar cr ${LIBRARY} $(OBJ_SPARSE) $(OBJ_BLAS)
	ranlib ${LIBRARY}

clean: 
	rm -f *.o libkokkos_kernels.a KokkosKernels_config.h 

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) ${HEADERS} KokkosKernels_config.h 
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<
