#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

class cuda : public ::testing::Test {
protected:
  static void SetUpTestCase()
  {
  }

  static void TearDownTestCase()
  {
  }
};

#define TestCategory cuda
#define TestExecSpace Kokkos::Cuda
