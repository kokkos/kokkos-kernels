#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

class threads : public ::testing::Test {
protected:
  static void SetUpTestCase()
  {
  }

  static void TearDownTestCase()
  {
  }
};

#define TestCategory threads
#define TestExecSpace Kokkos::Threads
