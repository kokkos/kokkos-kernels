//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.4
//       Copyright (2021) National Technology & Engineering
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
#ifndef __KOKKOSBATCHED_GEMM_SERIAL_IMPL_HPP__
#define __KOKKOSBATCHED_GEMM_SERIAL_IMPL_HPP__

#include "KokkosBatched_Util.hpp"
#include "KokkosBlas3_gemm.hpp"

namespace KokkosBatched {
namespace Impl {
/********************* BEGIN non-functor-level routines *********************/
template <class ArgTransA, class ArgTransB, class ArgMode, class ArgBatchSzDim,
          class ArgResultsPerThread, class ScalarType, class AViewType,
          class BViewType, class CViewType>
class BatchedSerialGemm {
 private:
  AViewType A;
  BViewType B;
  CViewType C;
  ScalarType alpha, beta;
  size_t divisor, c_cols, batch_size;
  ArgBatchSzDim batch_layout_tag;
  ArgTransA transA_tag;
  ArgTransB transB_tag;

  void run() {
    using execution_space = typename CViewType::device_type::execution_space;
    using policy_type =
        Kokkos::RangePolicy<ArgResultsPerThread, execution_space>;
    Kokkos::parallel_for("BatchedSerialGemm", policy_type(0, batch_size),
                         *this);
  }

 public:
  int invoke() {
    if (std::is_same<ArgResultsPerThread, ResultsPerThread::Rank0>::value) {
      // Set members for ResultsPerThread::Rank0 operator; these members allow
      // each thread to calculate its C output index
      if (std::is_same<ArgBatchSzDim, BatchLayout::Left>::value) {
        batch_size = C.extent(0);
        divisor    = C.extent(1) * C.extent(2);
        c_cols     = C.extent(2);
      } else {
        batch_size = C.extent(2);
        divisor    = C.extent(0) * C.extent(1);
        c_cols     = C.extent(1);
      }

      // Increase the number of threads by the divisor
      batch_size *= divisor;

      run();
    } else if (std::is_same<ArgResultsPerThread,
                            ResultsPerThread::Rank2>::value) {
      if (std::is_same<ArgBatchSzDim, BatchLayout::Left>::value)
        batch_size = C.extent(0);
      else
        batch_size = C.extent(2);

      run();
    } else {
      std::cerr << "Error: ArgResultsPerThread not supported" << std::endl;
      return -1;
    }
    return 0;
  }

  BatchedSerialGemm(ScalarType _alpha, AViewType _A, BViewType _B,
                    ScalarType _beta, CViewType _C)
      : A(_A), B(_B), C(_C), alpha(_alpha), beta(_beta) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const ResultsPerThread::Rank0 &, const int &i) const {
    // Here, the batch_idx is strided by c_rows * c_cols
    auto batch_idx = i / divisor;
    // For every batch, we need mod in [0, c_rows*c_cols-1]
    auto mod = i % divisor;
    // For every mod, we need a column index in [0, c_cols-1]
    auto col_idx = mod % c_cols;
    // For every mod, we need a row index in [0, c_rows-1]
    auto row_idx = mod / c_cols;

    // Due to taking 1-rank subviews out, we must handle transpose here.
    // Use overloads of subview_wrapper to handle transpose at compile time.
    auto svA_row = subview_wrapper(A, batch_idx, row_idx, Kokkos::ALL(),
                                   batch_layout_tag, transA_tag);
    auto svB_col = subview_wrapper(B, batch_idx, Kokkos::ALL(), col_idx,
                                   batch_layout_tag, transB_tag);
    auto svC_ele =
        subview_wrapper(C, batch_idx, row_idx, col_idx, batch_layout_tag);

    // Kokkos::subview(scalar, ALL) or Kokkos::subview(ALL, scalar) always
    // returns a column vector. Since the subviews above handle the
    // matrix transpositions, here we must perform the GEMM on:
    // row_vec x col_vec, which is svA_row' x svB_col to compute the element
    // of C.
    KokkosBlas::SerialGemm<Trans::Transpose, Trans::NoTranspose,
                           ArgMode>::invoke(alpha, svA_row, svB_col, beta,
                                            svC_ele);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ResultsPerThread::Rank2 &, const int &i) const {
    auto svA =
        subview_wrapper(A, i, Kokkos::ALL(), Kokkos::ALL(), batch_layout_tag);
    auto svB =
        subview_wrapper(B, i, Kokkos::ALL(), Kokkos::ALL(), batch_layout_tag);
    auto svC =
        subview_wrapper(C, i, Kokkos::ALL(), Kokkos::ALL(), batch_layout_tag);

    KokkosBlas::SerialGemm<ArgTransA, ArgTransB, ArgMode>::invoke(
        alpha, svA, svB, beta, svC);
  }
};
/********************* END non-functor-level routines *********************/
}  // namespace Impl

}  // namespace KokkosBatched

#endif
