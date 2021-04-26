//
// Created by Poliakoff, David Zoeller on 4/26/21.
//

#ifndef KOKKOSKERNELS_KOKKOSSPARSE_SPMV_TEST_HPP
#define KOKKOSKERNELS_KOKKOSSPARSE_SPMV_TEST_HPP
#include <common/Executor.hpp>
#include <common/KernelBase.hpp>
rajaperf::KernelBase* make_spmv_kernel_base(int argc, char* argv[], const rajaperf::RunParams& params);
#endif  // KOKKOSKERNELS_KOKKOSSPARSE_SPMV_HPP
