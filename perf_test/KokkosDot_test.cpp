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
#include <cstdio>

#include <ctime>
#include <cstring>
#include <cstdlib>
#include <limits>
#include <limits.h>
#include <cmath>
#include <unordered_map>
#include <Kokkos_Core.hpp>
#include <KokkosSparse_CrsMatrix.hpp>
#include <KokkosKernels_IOUtils.hpp>
#include "KokkosKernels_default_types.hpp"
#include <common/RunParams.hpp>
#include <common/QuickKernelBase.hpp>
#include <PerfTestUtilities.hpp>

#ifdef HAVE_CUSPARSE
#include <CuSparse_SPMV.hpp>
#endif

#ifdef HAVE_MKL
#include <MKL_SPMV.hpp>
#endif

#ifdef KOKKOS_ENABLE_OPENMP
#include <OpenMPStatic_SPMV.hpp>
#include <OpenMPDynamic_SPMV.hpp>
#include <OpenMPSmartStatic_SPMV.hpp>
#endif

///////////////////////////////////////////////////////////////
// RAJAPerf Suite interface for Kokkos, Kokkos-Kernels
#include <common/Executor.hpp>

// Headers from dot test prog

#include <iostream>
#include "KokkosBlas1_dot.hpp"
#include "KokkosKernels_Utils.hpp"
// in test_common
//#include "KokkosKernels_TestUtils.hpp"


  //FUNCTION THAT IS PART OF KK for generating test matrices
  //create_random_x_vector and create_random_y_vector can be used together to generate a random 
  //linear system Ax = y.
  template<typename vec_t>
  vec_t create_random_x_vector(vec_t& kok_x, double max_value = 10.0) {
    typedef typename vec_t::value_type scalar_t;
    auto h_x = Kokkos::create_mirror_view (kok_x);
    for (size_t j = 0; j < h_x.extent(1); ++j){
      for (size_t i = 0; i < h_x.extent(0); ++i){
        scalar_t r =
            static_cast <scalar_t> (rand()) /
            static_cast <scalar_t> (RAND_MAX / max_value);
        h_x.access(i, j) = r;
      }
    }
    Kokkos::deep_copy (kok_x, h_x);
    return kok_x;
  }


DotTestData setup_test(DotTestData::matrix_type A_matrix, DotTestData::matrix_type B_matrix) {
          DotTestData test_data;
          using matrix_type   = DotTestData::matrix_type;
          test_data.A_matrix = create_random_x_vector(A_matrix);
          // rm line 113 if build is good
          //test.B_matrix = create_random_x_vector(B_matrix);
          test_data.B_matrix = create_random_x_vector(B_matrix);

        return test_data;
}



void run_benchmark(DotTestData& test_data) {

        Kokkos::Timer timer;
        Kokkos::fence();
        timer.reset();
        double result_1D = KokkosBlas::dot(test_data.A_matrix, test_data.B_matrix);
        double elapsed = timer.seconds();
}
