set(CTEST_SOURCE_DIRECTORY $ENV{PWD})
set(CTEST_BINARY_DIRECTORY $ENV{PWD}/kokkos-kernels_build)

set(CTEST_SITE "ALCF-Polaris")
set(CTEST_BUILD_NAME "Linux-g++")

set(CTEST_START_WITH_EMPTY_BINARY_DIRECTORY TRUE)

set(CTEST_UPDATE_COMMAND /usr/bin/git)

set(CTEST_BUILD_COMMAND "cmake --build $ENV{PWD}/kokkos-kernels/build -j 48")

ctest_start(Nightly)
ctest_update(SOURCE "${CTEST_SOURCE_DIRECTORY}")
ctest_configure(BUILD "${CTEST_BINARY_DIRECTORY}")
ctest_build(BUILD "${CTEST_BINARY_DIRECTORY}")
# ------------------------------------------------------------------------------
# Update source tree
# ------------------------------------------------------------------------------

ctest_update(
  SOURCE "${CTEST_SOURCE_DIRECTORY}"
)

# ------------------------------------------------------------------------------
# Configure project
# ------------------------------------------------------------------------------

ctest_configure(
  BUILD "${CTEST_BINARY_DIRECTORY}"
  SOURCE "${CTEST_SOURCE_DIRECTORY}"
  OPTIONS
    "-DCMAKE_INSTALL_PREFIX=kokkos-kernels_install"
	"-DBUILD_SHARED_LIBS=ON"
	"-DCMAKE_BUILD_TYPE=Release"
	"-DCMAKE_VERBOSE_MAKEFILE=OFF"
	"-DKokkos_ROOT=kokkos_install"
	"-DKokkosKernels_ENABLE_TESTS:BOOL=ON"
	"-DKokkosKernels_ENABLE_EXAMPLES:BOOL=ON"
	"-DKokkosKernels_ENABLE_BENCHMARKS:BOOL=ON"
	"-DKokkosKernels_RUN_BENCHMARKS:BOOL=ON"
  RETURN_VALUE configure_result
)

# ------------------------------------------------------------------------------
# Build project
# ------------------------------------------------------------------------------

ctest_build(
  BUILD "${CTEST_BINARY_DIRECTORY}"
  NUMBER_ERRORS build_errors
  RETURN_VALUE build_result
)

# ------------------------------------------------------------------------------
# Run tests
# ------------------------------------------------------------------------------

ctest_test(
  BUILD "${CTEST_BINARY_DIRECTORY}"
  RETURN_VALUE test_result
)

# ------------------------------------------------------------------------------
# Submit results to CDash
# ------------------------------------------------------------------------------

ctest_submit()

# ------------------------------------------------------------------------------
# Failure handling
# ------------------------------------------------------------------------------

if(configure_result)
  message(FATAL_ERROR "Error during configuration! Exit code: ${configure_result}")
endif()

if(build_result)
  message(FATAL_ERROR "Error during build! Exit code: ${build_result}" with ${build_errors} errors)
endif()

if(test_result)
  message(FATAL_ERROR "Error during testing! Exit code: ${test_result}")
endif()
ctest_submit()
