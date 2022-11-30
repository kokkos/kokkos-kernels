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
#ifndef KOKKOSBLAS3_SERIAL_GEMM_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS3_SERIAL_GEMM_TPL_SPEC_DECL_HPP_

#include "KokkosBlas_util.hpp"
#include "KokkosBatched_Vector.hpp"
#include "KokkosKernels_MKLUtils.hpp"

#ifdef __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__

namespace KokkosBlas {

namespace Impl {

template <typename ArgTrans>
constexpr MKL_TRANSPOSE trans2mkl =
    std::is_same<ArgTrans, Trans::NoTranspose>::value
        ? MKL_NOTRANS
        : (std::is_same<ArgTrans, Trans::Transpose>::value
               ? MKL_TRANS
               : (std::is_same<ArgTrans, Trans::ConjTranspose>::value
                      ? MKL_CONJTRANS
                      : MKL_CONJ));  // Note: CONJ is not supported by MKL GEMM

}
///
/// Serial Impl
/// ===========

template <typename ArgTransA, typename ArgTransB>
struct SerialGemm<ArgTransA, ArgTransB, Algo::Gemm::CompactMKL> {
  template <typename ScalarType, typename AViewType, typename BViewType,
            typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType alpha,
                                           const AViewType &A,
                                           const BViewType &B,
                                           const ScalarType beta,
                                           const CViewType &C) {
    typedef typename CViewType::value_type vector_type;
    // typedef typename vector_type::value_type value_type;

    const int m = C.extent(0), n = C.extent(1);
    const int k =
        A.extent(std::is_same<ArgTransA, Trans::NoTranspose>::value ? 1 : 0);
    const MKL_TRANSPOSE trans_A = Impl::trans2mkl<ArgTransA>;
    const MKL_TRANSPOSE trans_B = Impl::trans2mkl<ArgTransB>;

    static_assert(KokkosBatched::is_vector<vector_type>::value,
                  "value type is not vector type");
    static_assert(
        vector_type::vector_length == 4 || vector_type::vector_length == 8,
        "AVX, AVX2 and AVX512 is supported");
    const MKL_COMPACT_PACK format =
        vector_type::vector_length == 8 ? MKL_COMPACT_AVX512 : MKL_COMPACT_AVX;

    // no error check
    int r_val = 0;
    if (A.stride_0() == 1 && B.stride_0() == 1 && C.stride_0() == 1) {
      mkl_dgemm_compact(MKL_COL_MAJOR, trans_A, trans_B, m, n, k, alpha,
                        (const double *)A.data(), A.stride_1(),
                        (const double *)B.data(), B.stride_1(), beta,
                        (double *)C.data(), C.stride_1(), format,
                        (MKL_INT)vector_type::vector_length);
    } else if (A.stride_1() == 1 && B.stride_1() == 1 && C.stride_1() == 1) {
      mkl_dgemm_compact(MKL_ROW_MAJOR, trans_A, trans_B, m, n, k, alpha,
                        (const double *)A.data(), A.stride_0(),
                        (const double *)B.data(), B.stride_0(), beta,
                        (double *)C.data(), C.stride_0(), format,
                        (MKL_INT)vector_type::vector_length);
    } else {
      r_val = -1;
    }
    return r_val;
  }
};

}  // namespace KokkosBlas

#endif  // __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__
#endif  // KOKKOSBLAS2_SERIAL_GEMV_TPL_SPEC_DECL_HPP_
