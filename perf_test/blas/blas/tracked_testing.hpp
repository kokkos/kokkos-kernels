//
// Created by Poliakoff, David Zoeller on 4/26/21.
//
#ifndef KOKKOSKERNELS_BLAS_TRACKED_TESTING_HPP
#define KOKKOSKERNELS_BLAS_TRACKED_TESTING_HPP

#include <common/RAJAPerfSuite.hpp>
#include <common/Executor.hpp>
//#include "KokkosSparse_spmv_test.hpp"
#include "KokkosBlas_dot_perf_test.hpp"


namespace test {
namespace blas {

void build_executor(rajaperf::Executor& exec,
                    int argc,
                    char* argv[],
                    const rajaperf::RunParams& params) {
        exec.registerGroup("BLAS");
        for(auto* kernel : construct_dot_kernel_base(params)) {
                exec.registerKernel("BLAS", kernel);
        }
}

}
}  // namespace test
#endif  // KOKKOSKERNELS_TRACKED_TESTING_HPP
