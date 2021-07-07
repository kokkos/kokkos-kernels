//
// Created by Poliakoff, David Zoeller on 4/27/21.
//

#ifndef KOKKOSKERNELS_KOKKOSSPARSE_SPGEMM_TEST_HPP
#define KOKKOSKERNELS_KOKKOSSPARSE_SPGEMM_TEST_HPP

#include <PerfTestUtilities.hpp>
test_list make_spgemm_kernel_base(int argc, char* argv[], const rajaperf::RunParams& params);

#endif  // KOKKOSKERNELS_KOKKOSSPARSE_SPGEMM_TEST_HPP
