set(CTEST_SOURCE_DIRECTORY $ENV{PWD})
set(CTEST_BINARY_DIRECTORY $ENV{PWD}/kokkos-kernels_build)

set(CTEST_SITE "ALCF-Polaris")
set(CTEST_BUILD_NAME "Linux-g++")

set(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY TRUE)

set(CTEST_UPDATE_COMMAND /usr/bin/git -v)

set(CTEST_CONFIGURE_COMMAND "cmake -S $ENV{PWD} \
			    -B kokkos-kernels_build \
			    -DCMAKE_INSTALL_PREFIX=kokkos-kernels_install \
			    -DBUILD_SHARED_LIBS=ON \
			    -DCMAKE_BUILD_TYPE=Release \
			    -DCMAKE_VERBOSE_MAKEFILE=OFF \
			    -DKokkos_ROOT=$ENV{PWD}/kokkos_install \
			    -DKokkosKernels_ENABLE_TESTS:BOOL=ON \
			    -DKokkosKernels_ENABLE_EXAMPLES:BOOL=ON \
			    -DKokkosKernels_ENABLE_BENCHMARKS:BOOL=ON \
			    -DKokkosKernels_RUN_BENCHMARKS:BOOL=ON")

set(CTEST_BUILD_COMMAND "cmake --build $ENV{PWD}/kokkos-kernels_build --parallel")

ctest_start(Nightly)
ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}")
ctest_configure(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_build(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_test(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_submit()
