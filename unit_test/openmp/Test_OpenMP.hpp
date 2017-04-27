#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

class openmp : public ::testing::Test {
protected:
  static void SetUpTestCase()
  {
  }

  static void TearDownTestCase()
  {
  }
};

#define TestCategory openmp
#define TestExecSpace Kokkos::OpenMP
