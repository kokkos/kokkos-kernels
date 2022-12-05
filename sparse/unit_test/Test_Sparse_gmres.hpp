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

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>

#include <string>
#include <stdexcept>

#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_IOUtils.hpp"
#include "KokkosBlas1_nrm2.hpp"
#include "KokkosSparse_spmv.hpp"
#include "KokkosSparse_gmres.hpp"

#include <gtest/gtest.h>

using namespace KokkosSparse;
using namespace KokkosSparse::Experimental;
using namespace KokkosKernels;
using namespace KokkosKernels::Experimental;

namespace Test {

template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void run_test_gmres() {
  using exe_space      = typename device::execution_space;
  using mem_space      = typename device::memory_space;
  using sp_matrix_type = KokkosSparse::CrsMatrix<scalar_t, lno_t, device, void, size_type>;
  using KernelHandle =
    KokkosKernels::Experimental::KokkosKernelsHandle<
      size_type, lno_t, scalar_t, exe_space, mem_space, mem_space>;

  // Create a diagonally dominant sparse matrix to test:
  constexpr auto n             = 5000;
  constexpr auto m             = 15;
  constexpr auto numRows       = n;
  constexpr auto numCols       = n;
  constexpr auto diagDominance = 1;
  constexpr bool verbose       = false;

  typename sp_matrix_type::non_const_size_type nnz = 10 * numRows;
  auto A =
    KokkosSparse::Impl::kk_generate_diagonally_dominant_sparse_matrix<
      sp_matrix_type>(numRows, numCols, nnz, 0, lno_t(0.01 * numRows),
                      diagDominance);

  // Make kernel handles
  KernelHandle kh;
  kh.create_gmres_handle(n, m);
  auto gmres_handle = kh.get_gmres_handle();
  using GMRESHandle = typename std::remove_reference<decltype(*gmres_handle)>::type;
  using ViewVectorType = typename GMRESHandle::nnz_value_view_t;
  using float_t        = typename GMRESHandle::float_t;

  // Set initial vectors:
  ViewVectorType X("X", n);    // Solution and initial guess
  ViewVectorType Wj("Wj", n);  // For checking residuals at end.
  ViewVectorType B(Kokkos::view_alloc(Kokkos::WithoutInitializing, "B"),
                   n);  // right-hand side vec

  gmres_handle->set_verbose(verbose);

  // Make rhs ones so that results are repeatable:
  {
    Kokkos::deep_copy(B, 1.0);

    gmres_numeric(&kh, A, B, X);

    // Double check residuals at end of solve:
    float_t nrmB = KokkosBlas::nrm2(B);
    KokkosSparse::spmv("N", 1.0, A, X, 0.0, Wj);  // wj = Ax
    KokkosBlas::axpy(-1.0, Wj, B);                // b = b-Ax.
    float_t endRes = KokkosBlas::nrm2(B) / nrmB;

    const auto num_iters   = gmres_handle->get_num_iters();
    const auto conv_flag   = gmres_handle->get_conv_flag_val();

    EXPECT_LT(num_iters, 40);
    EXPECT_GT(num_iters, 20);
    EXPECT_LT(endRes, gmres_handle->get_tol());
    EXPECT_EQ(conv_flag, GMRESHandle::Flag::Conv);
  }

  {
    gmres_handle->reset_handle(n, m);
    gmres_handle->set_ortho(GMRESHandle::Ortho::MGS);
    gmres_handle->set_verbose(verbose);
    Kokkos::deep_copy(X, 0.0);
    Kokkos::deep_copy(B, 1.0);

    gmres_numeric(&kh, A, B, X);

    // Double check residuals at end of solve:
    float_t nrmB = KokkosBlas::nrm2(B);
    KokkosSparse::spmv("N", 1.0, A, X, 0.0, Wj);  // wj = Ax
    KokkosBlas::axpy(-1.0, Wj, B);                // b = b-Ax.
    float_t endRes = KokkosBlas::nrm2(B) / nrmB;

    const auto num_iters   = gmres_handle->get_num_iters();
    const auto conv_flag   = gmres_handle->get_conv_flag_val();

    EXPECT_LT(num_iters, 40);
    EXPECT_GT(num_iters, 20);
    EXPECT_LT(endRes, gmres_handle->get_tol());
    EXPECT_EQ(conv_flag, GMRESHandle::Flag::Conv);
  }
}

}  // namespace Test

template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void test_gmres() {
  Test::run_test_gmres<scalar_t, lno_t, size_type, device>();
}

#define KOKKOSKERNELS_EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE)          \
  TEST_F(TestCategory,                                                       \
         sparse##_##gmres##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) { \
    test_gmres<SCALAR, ORDINAL, OFFSET, DEVICE>();                        \
  }

#include <Test_Common_Test_All_Type_Combos.hpp>

#undef KOKKOSKERNELS_EXECUTE_TEST
