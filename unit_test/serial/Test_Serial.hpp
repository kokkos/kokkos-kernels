#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

class serial : public ::testing::Test {
protected:
  static void SetUpTestCase()
  {
  }

  static void TearDownTestCase()
  {
  }
};

#define TestCategory serial
#define TestExecSpace Kokkos::Serial
