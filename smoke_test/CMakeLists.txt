# First copy the CMakeList.txt so we can build the test
file(COPY ${PACKAGE_SOURCE_DIR}/smoke_test/CMakeLists.txt.in DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/source)
file(RENAME ${CMAKE_CURRENT_BINARY_DIR}/source/CMakeLists.txt.in ${CMAKE_CURRENT_BINARY_DIR}/source/CMakeLists.txt)

# Second copy the source files needed to the build area
file(COPY ${PACKAGE_SOURCE_DIR}/perf_test/sparse/KokkosSparse_pcg.hpp DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/source)
file(COPY ${PACKAGE_SOURCE_DIR}/perf_test/sparse/KokkosSparse_block_pcg.cpp DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/source)

# Third write a configure file that can be invoked to test the library installation
configure_file(${PACKAGE_SOURCE_DIR}/smoke_test/run_smoke_test.sh.in ${CMAKE_CURRENT_BINARY_DIR}/run_smoke_test.sh @ONLY)

add_test(NAME SMOKE_TEST COMMAND /bin/bash ${CMAKE_CURRENT_BINARY_DIR}/run_smoke_test.sh)
# KOKKOSKERNELS_ADD_TEST(NAME "SMOKE_TEST" COMMAND "${CMAKE_CURRENT_BINARY_DIR}/run_smoke_test.sh" COMPONENTS Sparse)
