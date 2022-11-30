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

#ifndef KOKKOSBLAS3_INNER_GEMM_FIXC_TEAM_IMPL_HPP
#define KOKKOSBLAS3_INNER_GEMM_FIXC_TEAM_IMPL_HPP

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

namespace KokkosBlas {

template <int mb, int nb>
template <typename MemberType, typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<mb, nb>::team_invoke(
    const MemberType &member, const ScalarType alpha,
    const ValueType *KOKKOS_RESTRICT A, const ValueType *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueType *KOKKOS_RESTRICT C) {
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, 0, mb * nb), [&](const int &ij) {
        const int i = ij / nb, j = ij % nb;

        const ValueType *KOKKOS_RESTRICT pA                  = A + i * _as0,
                                         *KOKKOS_RESTRICT pB = B + j * _bs1;

        ValueType c = 0;
        for (int p = 0; p < k; ++p) c += pA[p * _as1] * pB[p * _bs0];
        C[i * _cs0 + j * _cs1] += alpha * c;
      });
  return 0;
}

template <int mb, int nb>
template <typename MemberType, typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<mb, nb>::team_invoke(
    const MemberType &member, const ScalarType alpha,
    const ValueType *KOKKOS_RESTRICT A, const ValueType *KOKKOS_RESTRICT B,
    const int m, const int n, const int k,
    /**/ ValueType *KOKKOS_RESTRICT C) {
  Kokkos::parallel_for(
      Kokkos::TeamThreadRange(member, 0, m * n), [&](const int &ij) {
        const int i = ij / n, j = ij % n;

        const ValueType *KOKKOS_RESTRICT pA                  = A + i * _as0,
                                         *KOKKOS_RESTRICT pB = B + j * _bs1;

        ValueType c = 0;
        for (int p = 0; p < k; ++p) c += pA[p * _as1] * pB[p * _bs0];
        C[i * _cs0 + j * _cs1] += alpha * c;
      });
  return 0;
}

}  // namespace KokkosBlas

#endif
