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

#ifndef KOKKOSBLAS3_SERIAL_GEMM_IMPL_HPP_
#define KOKKOSBLAS3_SERIAL_GEMM_IMPL_HPP_

#include "KokkosBlas3_serial_gemm_internal.hpp"
#include "KokkosBlas3_serial_gemm_tpl_spec_decl.hpp"
#include "KokkosBlas_util.hpp"

namespace KokkosBlas {

///
/// Serial Impl
/// ===========

///
/// Implemented:
/// NT/NT, T/NT, NT/T, T/T
///
/// Not yet immplemented (ConjTranspose):
/// CT/NT, NT/CT, CT/CT
///

template <typename ArgTransA, typename ArgTransB, typename ArgAlgo>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int SerialGemm<ArgTransA, ArgTransB, ArgAlgo>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  static_assert(std::is_same<ArgAlgo, Algo::Gemm::Unblocked>::value ||
                    std::is_same<ArgAlgo, Algo::Gemm::Blocked>::value ||
                    std::is_same<ArgAlgo, Algo::Gemm::CompactMKL>::value,
                "Algorithm not supported");

  using TransA   = Impl::MatrixModeInfo<ArgTransA>;
  using TransB   = Impl::MatrixModeInfo<ArgTransB>;
  const auto ae1 = TransA::extent(A, 1);
  const auto as0 = TransA::stride_0(A);
  const auto as1 = TransA::stride_1(A);
  const auto bs0 = TransB::stride_0(B);
  const auto bs1 = TransB::stride_1(B);

  return Impl::SerialGemmInternal<ArgAlgo>::invoke(
      C.extent(0), C.extent(1), ae1, alpha, A.data(), as0, as1, B.data(), bs0,
      bs1, beta, C.data(), C.stride_0(), C.stride_1());
}
}  // namespace KokkosBlas

#endif