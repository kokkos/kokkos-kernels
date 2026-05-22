set(CTEST_SOURCE_DIRECTORY $ENV{PWD})
set(CTEST_BINARY_DIRECTORY $ENV{PWD}/kokkos-kernels_build)

set(CTEST_SITE "ALCF-Aurora")

set(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY TRUE)

set(CTEST_UPDATE_COMMAND "/usr/bin/git")

set(CTEST_CONFIGURE_COMMAND "cmake -S ${CTEST_SOURCE_DIRECTORY} \
			    	   -B ${CTEST_BINARY_DIRECTORY} \
            			   -DCMAKE_INSTALL_PREFIX=${CTEST_SOURCE_DIRECTORY}/kokkos-kernels-install \
            			   -DBUILD_SHARED_LIBS=ON \
            			   -DCMAKE_CXX_FLAGS="-fsycl-device-code-split=per_kernel -fp-model=precise" \
            			   -DCMAKE_BUILD_TYPE=Release \
            			   -DCMAKE_VERBOSE_MAKEFILE=OFF \
            			   -DCMAKE_CXX_COMPILER=icpx \
            			   -DSITE=ALCF-Aurora \
            			   -DKokkos_ROOT=${CTEST_SOURCE_DIRECTORY}/kokkos-install \
            			   -DKokkosKernels_INST_COMPLEX_DOUBLE:BOOL=ON \
            			   -DKokkosKernels_ENABLE_TESTS:BOOL=ON \
            			   -DKokkosKernels_ENABLE_EXAMPLES:BOOL=ON \
            			   -DKokkosKernels_ENABLE_BENCHMARKS:BOOL=ON \
            			   -DKokkosKernels_RUN_BENCHMARKS:BOOL=ON \
            			   -DKokkosKernels_ENABLE_TPL_MKL=ON")

set(CTEST_BUILD_COMMAND "cmake --build ${CTEST_BINARY_DIRECTORY} --parallel")

ctest_start(Nightly)
ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}")
ctest_configure(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_build(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_test(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_submit()
