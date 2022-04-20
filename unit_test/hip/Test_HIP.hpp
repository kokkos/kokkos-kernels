#ifndef TEST_HIP_HPP
#define TEST_HIP_HPP

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <KokkosKernels_config.h>

#if defined(KOKKOSKERNELS_TEST_ETI_ONLY) && !defined(KOKKOSKERNELS_ETI_ONLY)
#define KOKKOSKERNELS_ETI_ONLY
#endif

class hip : public ::testing::Test {
 protected:
  static void SetUpTestCase() {}

  static void TearDownTestCase() {}
};

#define TestCategory hip
#define TestExecSpace Kokkos::Experimental::HIP
#define TEST_BACKEND_IS_HIP  // tests can check this to disable themselves if
                             // desired

#endif  // TEST_HIP_HPP
