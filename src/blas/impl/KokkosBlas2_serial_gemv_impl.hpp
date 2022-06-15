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
#ifndef __KOKKOSBLAS_GEMV_SERIAL_IMPL_HPP__
#define __KOKKOSBLAS_GEMV_SERIAL_IMPL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBlas_util.hpp"
#include "KokkosBlas2_serial_gemv_internal.hpp"

namespace KokkosBlas {

template <typename ArgTrans, typename ArgAlgo>
struct SerialGemv {
  template <typename ScalarType, typename AViewType, typename xViewType,
            typename yViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType /*alpha*/,
                                           const AViewType & /*A*/,
                                           const xViewType & /*x*/,
                                           const ScalarType /*beta*/,
                                           const yViewType & /*y*/) {
    assert(false && "Error: encounter dummy impl");
    return 0;
  }
};

///
/// Serial Impl
/// ===========
/// CompactMKL does not exist on Gemv

///
/// Implemented:
/// NT, T
///
/// Not yet implemented
/// CT

///
/// NT
///

#ifdef __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__
template <>
template <typename ScalarType, typename AViewType, typename xViewType,
          typename yViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemv<Trans::NoTranspose, Algo::Gemv::CompactMKL>::invoke(
    const ScalarType alpha, const AViewType &A, const xViewType &x,
    const ScalarType beta, const yViewType &y) {
  typedef typename yViewType::value_type vector_type;
  // typedef typename vector_type::value_type value_type;

  const int m = A.extent(0), n = 1, k = A.extent(1);

  static_assert(is_vector<vector_type>::value, "value type is not vector type");
  static_assert(
      vector_type::vector_length == 4 || vector_type::vector_length == 8,
      "AVX, AVX2 and AVX512 is supported");
  const MKL_COMPACT_PACK format =
      vector_type::vector_length == 8 ? MKL_COMPACT_AVX512 : MKL_COMPACT_AVX;

  // no error check
  int r_val = 0;
  if (A.stride_0() == 1) {
    mkl_dgemm_compact(MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_1(),
                      (const double *)x.data(), x.stride_0(), beta,
                      (double *)y.data(), y.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else if (A.stride_1() == 1) {
    mkl_dgemm_compact(MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_0(),
                      (const double *)x.data(), x.stride_0(), beta,
                      (double *)y.data(), y.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else {
    r_val = -1;
  }
  return r_val;
}
#endif

template <>
template <typename ScalarType, typename AViewType, typename xViewType,
          typename yViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemv<Trans::NoTranspose, Algo::Gemv::Unblocked>::invoke(
    const ScalarType alpha, const AViewType &A, const xViewType &x,
    const ScalarType beta, const yViewType &y) {
  return Impl::SerialGemvInternal<Algo::Gemv::Unblocked>::invoke(
      A.extent(0), A.extent(1), alpha, A.data(), A.stride_0(), A.stride_1(),
      x.data(), x.stride_0(), beta, y.data(), y.stride_0());
}

template <>
template <typename ScalarType, typename AViewType, typename xViewType,
          typename yViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemv<Trans::NoTranspose, Algo::Gemv::Blocked>::invoke(
    const ScalarType alpha, const AViewType &A, const xViewType &x,
    const ScalarType beta, const yViewType &y) {
  return Impl::SerialGemvInternal<Algo::Gemv::Blocked>::invoke(
      A.extent(0), A.extent(1), alpha, A.data(), A.stride_0(), A.stride_1(),
      x.data(), x.stride_0(), beta, y.data(), y.stride_0());
}
///
/// T
///

#ifdef __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__
template <>
template <typename ScalarType, typename AViewType, typename xViewType,
          typename yViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemv<Trans::Transpose, Algo::Gemv::CompactMKL>::invoke(
    const ScalarType alpha, const AViewType &A, const xViewType &x,
    const ScalarType beta, const yViewType &y) {
  typedef typename yViewType::value_type vector_type;
  // typedef typename vector_type::value_type value_type;

  const int m = A.extent(0), n = 1, k = A.extent(1);

  static_assert(is_vector<vector_type>::value, "value type is not vector type");
  static_assert(
      vector_type::vector_length == 4 || vector_type::vector_length == 8,
      "AVX, AVX2 and AVX512 is supported");
  const MKL_COMPACT_PACK format =
      vector_type::vector_length == 8 ? MKL_COMPACT_AVX512 : MKL_COMPACT_AVX;

  // no error check
  int r_val = 0;
  if (A.stride_0() == 1) {
    mkl_dgemm_compact(MKL_COL_MAJOR, MKL_TRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_1(),
                      (const double *)x.data(), x.stride_0(), beta,
                      (double *)y.data(), y.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else if (A.stride_1() == 1) {
    mkl_dgemm_compact(MKL_ROW_MAJOR, MKL_TRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_0(),
                      (const double *)x.data(), x.stride_0(), beta,
                      (double *)y.data(), y.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else {
    r_val = -1;
  }
  return r_val;
}
#endif

template <>
template <typename ScalarType, typename AViewType, typename xViewType,
          typename yViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemv<Trans::Transpose, Algo::Gemv::Unblocked>::invoke(
    const ScalarType alpha, const AViewType &A, const xViewType &x,
    const ScalarType beta, const yViewType &y) {
  return Impl::SerialGemvInternal<Algo::Gemv::Unblocked>::invoke(
      A.extent(1), A.extent(0), alpha, A.data(), A.stride_1(), A.stride_0(),
      x.data(), x.stride_0(), beta, y.data(), y.stride_0());
}

template <>
template <typename ScalarType, typename AViewType, typename xViewType,
          typename yViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemv<Trans::Transpose, Algo::Gemv::Blocked>::invoke(
    const ScalarType alpha, const AViewType &A, const xViewType &x,
    const ScalarType beta, const yViewType &y) {
  return Impl::SerialGemvInternal<Algo::Gemv::Blocked>::invoke(
      A.extent(1), A.extent(0), alpha, A.data(), A.stride_1(), A.stride_0(),
      x.data(), x.stride_0(), beta, y.data(), y.stride_0());
}

}  // namespace KokkosBlas

#endif
