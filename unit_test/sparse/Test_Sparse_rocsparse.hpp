// Note: Luc Berger-Vergiat 10/25/21
//       Only include this test if compiling
//       the cuda sparse tests and cuSPARSE
//       is enabled.
#if defined(TEST_HIP_SPARSE_CPP) && defined(KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE)

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <rocsparse.h>

void test_rocsparse_version() {
  // Print version
  rocsparse_handle handle;
  rocsparse_create_handle(&handle);

  int ver;
  char rev[64];

  rocsparse_get_version(handle, &ver);
  rocsparse_get_git_rev(handle, rev);

  std::cout << "rocSPARSE version: " << ver / 100000 << "." << ver / 100 % 1000
            << "." << ver % 100 << "-" << rev << std::endl;

  rocsparse_destroy_handle(handle);
}

TEST_F(TestCategory, sparse_rocsparse_version) { test_rocsparse_version(); }

#endif  // check for HIP and rocSPARSE
