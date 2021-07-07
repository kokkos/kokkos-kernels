//
// Created by Poliakoff, David Zoeller on 4/26/21.
//
#ifndef KOKKOSKERNELS_TRACKED_TESTING_HPP
#define KOKKOSKERNELS_TRACKED_TESTING_HPP
#include <common/RAJAPerfSuite.hpp>
#include <common/Executor.hpp>
#include "KokkosDot_test.hpp"
//#include "KokkosSparse_spmv_test.hpp"
//#include "KokkosSparse_spgemm_test.hpp"
namespace test {

        /*
namespace sparse {
void build_executor(rajaperf::Executor& exec, int argc, char* argv[], const rajaperf::RunParams& params) {
  exec.registerGroup("Sparse");
  for(auto* kernel : make_spmv_kernel_base(params)) {
    exec.registerKernel("Sparse", kernel);
  }

  //for(auto* kernel : make_spgemm_kernel_base(argc, argv, params)) {
  //  exec.registerKernel("Sparse", kernel);
  //}
}

}
*/

namespace dot {
        void build_executor(rajaperf::Executor& exec, int argc, char* argv[], const rajaperf::RunParams& params)
        {
                exec.registerGroup("DotProducts");
                for (auto* kernel: make_dot_kernel_base(params)) {
                        exec.registerKernel("DotProducts", kernel);
                }

        }

}

}  // namespace test
#endif  // KOKKOSKERNELS_TRACKED_TESTING_HPP
