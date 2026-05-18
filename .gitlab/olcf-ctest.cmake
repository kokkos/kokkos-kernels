set(CTEST_SOURCE_DIRECTORY $ENV{PWD}/kokkos-kernels)
set(CTEST_BINARY_DIRECTORY $ENV{PWD}/kokkos-kernels/build)

set(CTEST_SITE "OLCF-Frontier")
set(CTEST_BUILD_NAME "Linux-hipcc")

set(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY TRUE)

set(CTEST_UPDATE_COMMAND /usr/bin/git)
set(CTEST_CONFIGURE_COMMAND "cmake -S $ENV{PWD}/kokkos-kernels \
			    -B $ENV{PWD}/kokkos-kernels/build \
			    -DCMAKE_CXX_COMPILER=hipcc \
			    -DCMAKE_INSTALL_PREFIX=$ENV{PWD}/kokkos-kernels/install
			    -DCMAKE_BUILD_TYPE="Release"
			    -DCMAKE_VERBOSE_MAKEFILE=ON
			    -DKokkos_ROOT=$ENV{PWD}/kokkos-kernels/kokkos/install
			    -DKokkosKernels_INST_COMPLEX_DOUBLE=ON
			    -DKokkosKernels_ENABLE_TPL_ROCSOLVER=ON
			    -DKokkosKernels_ENABLE_TPL_ROCSPARSE=ON
			    -DKokkosKernels_ENABLE_TPL_ROCBLAS=ON
			    -DKokkosKernels_ENABLE_TESTS=ON
			    -DKokkosKernels_ENABLE_EXAMPLES:BOOL=ON
			    -DKokkosKernels_ENABLE_BENCHMARKS:BOOL=ON
      			    -DKokkosKernels_RUN_BENCHMARKS:BOOL=ON")

set(CTEST_BUILD_COMMAND "cmake --build $ENV{PWD}/kokkos-kernels/build -j 48")

ctest_start(Nightly)
ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}")
ctest_configure(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_build(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_test(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_submit()
