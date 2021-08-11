/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#include <Kokkos_Core.hpp>
#include <KokkosBlas2_gemv.hpp>
#include <Kokkos_Random.hpp>

// For RPS implementation
//#include "KokkosBlas_dot_perf_test.hpp"
#include "KokkosBlas2_gemv_perf_test.hpp"
#ifdef KOKKOSKERNELS_ENABLE_TESTS_AND_PERFSUITE
#include <PerfTestUtilities.hpp>
#endif


//////////////////////////////////////////////////////////////////////////////////////////////////////////////
  // https://github.com/kokkos/kokkos-kernels/wiki/BLAS-2%3A%3Agemv
  // Header File: KokkosBlas2_gemv.hpp
  // Usage: KokkosBlas::gemv (mode, alpha, A, x, beta, y);
  // Interface Single Vector only
  //
  // Matrix Vector Multiplication y[i] = beta * y[i] + alpha * SUM_j(A[i,j] * x[j])
  //
  // Parameters:
  // AViewType: A rank-2 Kokkos::View
  // XViewType: A rank-1 Kokkos::View
  // YViewType: A rank-1 Kokkos::View
  //
  // Arguments:
  // trans [in] "N" for non-transpose,
  //            "T" for transpose,
  //            "C" for conjugate transpose. 
  //            All characters after the first are ignored. This works just like the BLAS routines.
  //
  // alpha [in] Input coefficient of A*x
  // A     [in] Input matrix, as a 2-D Kokkos::View
  // x     [in] Input vector, as a 1-D Kokkos::View
  // beta  [in] Input coefficient of y
  // y     [in/out] Output vector, as a nonconst 1-D Kokkos::View

  // Requirements:
  // If mode == "N": A.extent(0) == y.extent(0) && A.extent(1) == x.extent(0)
  // If mode == "C" || mode == "T": A.extent(1) == y.extent(0) && A.extent(0) == x.extent(0)
 
  // EXAMPLE:  Warm-up run
  // params:          mode, alpha, A, x, beta, y)
  // KokkosBlas::gemv("N", 1.0, A, x, 0.0, y); 
//////////////////////////////////////////////////////////////////////////////////////////////////////////////



// Recall --testData is a tempated class, 
// setup_test is a templated function
template<class ExecSpace, class Layout>
testData_gemv<ExecSpace, Layout> setup_test(int m,
                                            int n,
                                            int repeat
                                            )
{
        // use constructor to generate test data
        testData_gemv<ExecSpace, Layout> testData_gemv_obj(m,n,repeat);

        // set a field in the struct
        //testData_gemv_obj.A = A;
        //testData_gemv_obj.x = x;
        //testData_gemv_obj.y = y;
        //testData_gemv_obj.m = m;
        //testData_gemv_obj.n = n;
        //testData_gemv_obj.repeat = repeat;

        return testData_gemv_obj;
}


test_list construct_gemv_kernel_base(const rajaperf::RunParams& run_params)

{
        // instantiate test_list as kernel_base_vector
        test_list kernel_base_vector;


kernel_base_vector.push_back(rajaperf::make_kernel_base(
        "BLAS2_GEMV",
        run_params,
        // setup lambda by value
        [=](const int repeat, const int m) {
          // returns a tuple of testData_obj
          return std::make_tuple(
                          setup_test<Kokkos::DefaultExecutionSpace, Kokkos::DefaultExecutionSpace::array_layout>(m, m/10, repeat));
          },
        // run lambda will take the returned setup tuple
        [&](const int iteration, const int runsize, auto& data) {
        KokkosBlas::gemv("N", 1.0, data.A, data.x, 0.0, data.y); 
        }));


        // return a vector of kernel base objects
        // of type test_list
        return kernel_base_vector;
}
