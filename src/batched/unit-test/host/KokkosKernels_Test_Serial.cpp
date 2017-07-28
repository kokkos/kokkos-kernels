#include <gtest/gtest.h>

#include <Kokkos_Core.hpp>
#include <impl/Kokkos_Timer.hpp>

typedef Kokkos::DefaultHostExecutionSpace HostSpaceType;
typedef Kokkos::Serial                    DeviceSpaceType;

#include "KokkosKernels_Test_Batched.hpp"

//using namespace KokkosKernels::Experimental;

int main (int argc, char *argv[]) {

  const bool detail = false;

  printf("KokkosKernels::Test::Serial::Begin\n");

  //  printExecSpaceConfiguration<DeviceSpaceType>("DeviceSpace", detail);
  //printExecSpaceConfiguration<HostSpaceType>  ("HostSpace",   detail);
  
  Kokkos::initialize(argc, argv);

  ::testing::InitGoogleTest(&argc, argv);
  const int r_val = RUN_ALL_TESTS();

  Kokkos::finalize();

  printf("KokkosKernels::Test::Serial::End\n");

  return r_val;
}
