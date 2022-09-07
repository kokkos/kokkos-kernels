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

///
/// NT/NT
///
#if defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL__) &&         \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_BATCHED__) && \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_COMPACT_BATCHED__)
template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::NoTranspose,
           Algo::Gemm::CompactMKL>::invoke(const ScalarType alpha,
                                           const AViewType &A,
                                           const BViewType &B,
                                           const ScalarType beta,
                                           const CViewType &C) {
  typedef typename CViewType::value_type vector_type;
  // typedef typename vector_type::value_type value_type;

  const int m = C.extent(0), n = C.extent(1), k = A.extent(1);

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
    mkl_dgemm_compact(MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_1(),
                      (const double *)B.data(), B.stride_1(), beta,
                      (double *)C.data(), C.stride_1(), format,
                      (MKL_INT)vector_type::vector_length);
  } else if (A.stride_1() == 1 && B.stride_1() == 1 && C.stride_1() == 1) {
    mkl_dgemm_compact(MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_0(),
                      (const double *)B.data(), B.stride_0(), beta,
                      (double *)C.data(), C.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else {
    r_val = -1;
  }
  return r_val;
}
#endif

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::NoTranspose,
           Algo::Gemm::Unblocked>::invoke(const ScalarType alpha,
                                          const AViewType &A,
                                          const BViewType &B,
                                          const ScalarType beta,
                                          const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return Impl::SerialGemmInternal<Algo::Gemm::Unblocked>::invoke(
      C.extent(0), C.extent(1), A.extent(1), alpha, A.data(), A.stride_0(),
      A.stride_1(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Blocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return Impl::SerialGemmInternal<Algo::Gemm::Blocked>::invoke(
      C.extent(0), C.extent(1), A.extent(1), alpha, A.data(), A.stride_0(),
      A.stride_1(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

///
/// T/NT
///

#if defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL__) &&         \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_BATCHED__) && \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_COMPACT_BATCHED__)
template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::NoTranspose,
           Algo::Gemm::CompactMKL>::invoke(const ScalarType alpha,
                                           const AViewType &A,
                                           const BViewType &B,
                                           const ScalarType beta,
                                           const CViewType &C) {
  typedef typename CViewType::value_type vector_type;
  // typedef typename vector_type::value_type value_type;

  const int m = C.extent(0), n = C.extent(1), k = A.extent(0);

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
    mkl_dgemm_compact(MKL_COL_MAJOR, MKL_TRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_1(),
                      (const double *)B.data(), B.stride_1(), beta,
                      (double *)C.data(), C.stride_1(), format,
                      (MKL_INT)vector_type::vector_length);
  } else if (A.stride_1() == 1 && B.stride_1() == 1 && C.stride_1() == 1) {
    mkl_dgemm_compact(MKL_ROW_MAJOR, MKL_TRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_0(),
                      (const double *)B.data(), B.stride_0(), beta,
                      (double *)C.data(), C.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else {
    r_val = -1;
  }
  return r_val;
}
#endif

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Unblocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return Impl::SerialGemmInternal<Algo::Gemm::Unblocked>::invoke(
      C.extent(0), C.extent(1), A.extent(0), alpha, A.data(), A.stride_1(),
      A.stride_0(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Blocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return Impl::SerialGemmInternal<Algo::Gemm::Blocked>::invoke(
      C.extent(0), C.extent(1), A.extent(0), alpha, A.data(), A.stride_1(),
      A.stride_0(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

///
/// NT/T
///

#if defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL__) &&         \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_BATCHED__) && \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_COMPACT_BATCHED__)
template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::Transpose,
           Algo::Gemm::CompactMKL>::invoke(const ScalarType alpha,
                                           const AViewType &A,
                                           const BViewType &B,
                                           const ScalarType beta,
                                           const CViewType &C) {
  typedef typename CViewType::value_type vector_type;
  // typedef typename vector_type::value_type value_type;

  const int m = C.extent(0), n = C.extent(1), k = A.extent(1);

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
    mkl_dgemm_compact(MKL_COL_MAJOR, MKL_NOTRANS, MKL_TRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_1(),
                      (const double *)B.data(), B.stride_1(), beta,
                      (double *)C.data(), C.stride_1(), format,
                      (MKL_INT)vector_type::vector_length);
  } else if (A.stride_1() == 1 && B.stride_1() == 1 && C.stride_1() == 1) {
    mkl_dgemm_compact(MKL_ROW_MAJOR, MKL_NOTRANS, MKL_TRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_0(),
                      (const double *)B.data(), B.stride_0(), beta,
                      (double *)C.data(), C.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else {
    r_val = -1;
  }
  return r_val;
}
#endif

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Unblocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return Impl::SerialGemmInternal<Algo::Gemm::Unblocked>::invoke(
      C.extent(0), C.extent(1), A.extent(1), alpha, A.data(), A.stride_0(),
      A.stride_1(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Blocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return Impl::SerialGemmInternal<Algo::Gemm::Blocked>::invoke(
      C.extent(0), C.extent(1), A.extent(1), alpha, A.data(), A.stride_0(),
      A.stride_1(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

///
/// T/T
///

#if defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL__) &&         \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_BATCHED__) && \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_COMPACT_BATCHED__)
template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::Transpose, Algo::Gemm::CompactMKL>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  typedef typename CViewType::value_type vector_type;
  // typedef typename vector_type::value_type value_type;

  const int m = C.extent(0), n = C.extent(1), k = A.extent(0);

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
    mkl_dgemm_compact(MKL_COL_MAJOR, MKL_TRANS, MKL_TRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_1(),
                      (const double *)B.data(), B.stride_1(), beta,
                      (double *)C.data(), C.stride_1(), format,
                      (MKL_INT)vector_type::vector_length);
  } else if (A.stride_1() == 1 && B.stride_1() == 1 && C.stride_1() == 1) {
    mkl_dgemm_compact(MKL_ROW_MAJOR, MKL_TRANS, MKL_TRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_0(),
                      (const double *)B.data(), B.stride_0(), beta,
                      (double *)C.data(), C.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else {
    r_val = -1;
  }
  return r_val;
}
#endif

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::Transpose, Algo::Gemm::Unblocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return Impl::SerialGemmInternal<Algo::Gemm::Unblocked>::invoke(
      C.extent(0), C.extent(1), A.extent(0), alpha, A.data(), A.stride_1(),
      A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::Transpose, Algo::Gemm::Blocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return Impl::SerialGemmInternal<Algo::Gemm::Blocked>::invoke(
      C.extent(0), C.extent(1), A.extent(0), alpha, A.data(), A.stride_1(),
      A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

}  // namespace KokkosBlas

#endif
