
#=======================================================================
#=================== Settings ==========================================
#=======================================================================

KOKKOS_PATH = ${HOME}/Kokkos/kokkos
KOKKOSKERNELS_PATH = ${HOME}/Kokkos/kokkos-kernels
GTEST_PATH = ${KOKKOS_PATH}/tpls/gtest

KOKKOS_DEVICES=OpenMP
KOKKOS_CUDA_OPTIONS=enable_lambda

KOKKOSKERNELS_OPTIONS=eti-only

default: build

ifneq (,$(findstring Cuda,$(KOKKOS_DEVICES)))
CXX = ${KOKKOS_PATH}/config/nvcc_wrapper
else
CXX = g++
endif

CXXFLAGS = -O3 -g
LINK = ${CXX}
LINKFLAGS =  

DEPFLAGS = -M

OBJ =
LIB =

#=======================================================================
#=================== Include Kokkos and KokkosKernels ==================
#=======================================================================

#include $(KOKKOS_PATH)/Makefile.kokkos
include $(KOKKOSKERNELS_PATH)/Makefile.kokkos-kernels

#=======================================================================
#=================== Add common source files ===========================
#=======================================================================

CXXFLAGS += -I${GTEST_PATH} -I$(KOKKOSKERNELS_PATH)/unit_test/
CXXFLAGS += -I$(KOKKOSKERNELS_PATH)/unit_test/blas
#CXXFLAGS += -DKOKKOSKERNELS_IMPL_CHECK_ETI_CALLS

vpath %.cpp ${KOKKOSKERNELS_PATH}/unit_test

TEST_HEADERS =  $(wildcard $(KOKKOSKERNELS_PATH)/unit_test/*.hpp)
TEST_HEADERS += $(wildcard $(KOKKOSKERNELS_PATH)/unit_test/blas/*.hpp)

TEST_TARGETS =
TARGETS =

#=======================================================================
#=================== OpenMP BACKEND ====================================
#=======================================================================

vpath %.cpp ${KOKKOSKERNELS_PATH}/unit_test/openmp

ifeq ($(KOKKOS_INTERNAL_USE_OPENMP), 1)
  CXXFLAGS += -I${KOKKOSKERNELS_PATH}/unit_test/openmp
  OBJ_OPENMP = Test_Main.o gtest-all.o
  OBJ_OPENMP += Test_OpenMP_Blas1_AXPBY.o
  OBJ_OPENMP += Test_OpenMP_Blas1_Dot.o
  TARGETS += KokkosKernels_UnitTest_OpenMP
  TEST_TARGETS += test-openmp
endif

KokkosKernels_UnitTest_OpenMP: $(OBJ_OPENMP) $(KOKKOS_LINK_DEPENDS) $(KOKKOSKERNELS_LINK_DEPENDS) $(TEST_HEADERS)
	$(LINK) $(EXTRA_PATH) $(OBJ_OPENMP) $(KOKKOSKERNELS_LIBS) $(KOKKOS_LIBS) $(LIB) $(KOKKOS_LDFLAGS) $(LDFLAGS) -o KokkosKernels_UnitTest_OpenMP

test-openmp: KokkosKernels_UnitTest_OpenMP
	./KokkosKernels_UnitTest_OpenMP

#=======================================================================
#=================== Cuda BACKEND ====================================
#=======================================================================

vpath %.cpp ${KOKKOSKERNELS_PATH}/unit_test/cuda

ifeq ($(KOKKOS_INTERNAL_USE_CUDA), 1)
  CXXFLAGS += -I${KOKKOSKERNELS_PATH}/unit_test/cuda
  OBJ_CUDA = Test_Main.o gtest-all.o
  OBJ_CUDA += Test_Cuda_Blas1_AXPBY.o
  OBJ_CUDA += Test_Cuda_Blas1_Dot.o
  TARGETS += KokkosKernels_UnitTest_Cuda
  TEST_TARGETS += test-cuda
endif

KokkosKernels_UnitTest_Cuda: $(OBJ_CUDA) $(KOKKOS_LINK_DEPENDS) $(KOKKOSKERNELS_LINK_DEPENDS) $(TEST_HEADERS)
	$(LINK) $(EXTRA_PATH) $(OBJ_CUDA) $(KOKKOSKERNELS_LIBS) $(KOKKOS_LIBS) $(LIB) $(KOKKOS_LDFLAGS) $(LDFLAGS) -o KokkosKernels_UnitTest_Cuda

test-cuda: KokkosKernels_UnitTest_Cuda
	./KokkosKernels_UnitTest_Cuda

#=======================================================================
#=================== Serial BACKEND ====================================
#=======================================================================

vpath %.cpp ${KOKKOSKERNELS_PATH}/unit_test/serial

ifeq ($(KOKKOS_INTERNAL_USE_SERIAL), 1)
  CXXFLAGS += -I${KOKKOSKERNELS_PATH}/unit_test/serial
  OBJ_SERIAL = Test_Main.o gtest-all.o
  OBJ_SERIAL += Test_Serial_Blas1_AXPBY.o
  OBJ_SERIAL += Test_Serial_Blas1_Dot.o
  TARGETS += KokkosKernels_UnitTest_Serial
  TEST_TARGETS += test-serial
endif

KokkosKernels_UnitTest_Serial: $(OBJ_SERIAL) $(KOKKOS_LINK_DEPENDS) $(KOKKOSKERNELS_LINK_DEPENDS) $(TEST_HEADERS)
	$(LINK) $(EXTRA_PATH) $(OBJ_SERIAL) $(KOKKOSKERNELS_LIBS) $(KOKKOS_LIBS) $(LIB) $(KOKKOS_LDFLAGS) $(LDFLAGS) -o KokkosKernels_UnitTest_Serial

test-serial: KokkosKernels_UnitTest_Serial
	./KokkosKernels_UnitTest_Serial

#=======================================================================
#=================== Threads BACKEND ===================================
#=======================================================================

vpath %.cpp ${KOKKOSKERNELS_PATH}/unit_test/threads

ifeq ($(KOKKOS_INTERNAL_USE_PTHREADS), 1)
  CXXFLAGS += -I${KOKKOSKERNELS_PATH}/unit_test/threads
  OBJ_THREADS = Test_Main.o gtest-all.o
  OBJ_THREADS += Test_Threads_Blas1_AXPBY.o
  OBJ_THREADS += Test_Threads_Blas1_Dot.o
  TARGETS += KokkosKernels_UnitTest_Threads
  TEST_TARGETS += test-threads
endif

KokkosKernels_UnitTest_Threads: $(OBJ_THREADS) $(KOKKOS_LINK_DEPENDS) $(KOKKOSKERNELS_LINK_DEPENDS) $(TEST_HEADERS)
	$(LINK) $(EXTRA_PATH) $(OBJ_THREADS) $(KOKKOSKERNELS_LIBS) $(KOKKOS_LIBS) $(LIB) $(KOKKOS_LDFLAGS) $(LDFLAGS) -o KokkosKernels_UnitTest_Threads

test-threads: KokkosKernels_UnitTest_Threads
	./KokkosKernels_UnitTest_Threads

#=======================================================================
#=================== Make Rules ========================================
#=======================================================================

test: $(TEST_TARGETS)

build: $(TARGETS)
	
clean: kokkos-clean kokkoskernels-clean
	rm -f *.o

# Compilation rules

%.o:%.cpp $(KOKKOS_CPP_DEPENDS) $(KOKKOSKERNELS_CPP_DEPENDS) $(TEST_HEADERS)
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOSKERNELS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $<

gtest-all.o:$(GTEST_PATH)/gtest/gtest-all.cc
	$(CXX) $(KOKKOS_CPPFLAGS) $(KOKKOS_CXXFLAGS) $(CXXFLAGS) $(EXTRA_INC) -c $(GTEST_PATH)/gtest/gtest-all.cc
	
