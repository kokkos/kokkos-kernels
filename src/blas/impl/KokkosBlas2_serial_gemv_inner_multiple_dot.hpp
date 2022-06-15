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
#ifndef __KOKKOSBLAS_INNER_MULTIPLE_DOT_PRODUCT_SERIAL_IMPL_HPP__
#define __KOKKOSBLAS_INNER_MULTIPLE_DOT_PRODUCT_SERIAL_IMPL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosBlas {
namespace Impl {

template <int mb>
struct InnerMultipleDotProduct {
  const int _as0, _as1, _xs0, _ys0;

  KOKKOS_INLINE_FUNCTION
  InnerMultipleDotProduct(const int as0, const int as1, const int xs0,
                          const int ys0)
      : _as0(as0), _as1(as1), _xs0(xs0), _ys0(ys0) {}

  template <typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION int serial_invoke(const ScalarType alpha,
                                           const ValueType *KOKKOS_RESTRICT A,
                                           const ValueType *KOKKOS_RESTRICT x,
                                           const int n,
                                           /**/ ValueType *KOKKOS_RESTRICT y);

  template <typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION int serial_invoke(const ScalarType alpha,
                                           const ValueType *KOKKOS_RESTRICT A,
                                           const ValueType *KOKKOS_RESTRICT x,
                                           const int m, const int n,
                                           /**/ ValueType *KOKKOS_RESTRICT y);
};

///
/// Dot Product for GEMV
/// ====================

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerMultipleDotProduct<5>::serial_invoke(
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const ValueType *KOKKOS_RESTRICT x, const int n,
    /**/ ValueType *KOKKOS_RESTRICT y) {
  if (n <= 0) return 0;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0,
            i4 = 4 * _as0;

  // unroll by rows
  ValueType y_0 = 0, y_1 = 0, y_2 = 0, y_3 = 0, y_4 = 0;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int j = 0; j < n; ++j) {
    const int jj        = j * _as1;
    const ValueType x_j = x[j * _xs0];

    y_0 += A[i0 + jj] * x_j;
    y_1 += A[i1 + jj] * x_j;
    y_2 += A[i2 + jj] * x_j;
    y_3 += A[i3 + jj] * x_j;
    y_4 += A[i4 + jj] * x_j;
  }

  y[0 * _ys0] += alpha * y_0;
  y[1 * _ys0] += alpha * y_1;
  y[2 * _ys0] += alpha * y_2;
  y[3 * _ys0] += alpha * y_3;
  y[4 * _ys0] += alpha * y_4;

  return 0;
}

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerMultipleDotProduct<4>::serial_invoke(
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const ValueType *KOKKOS_RESTRICT x, const int n,
    /**/ ValueType *KOKKOS_RESTRICT y) {
  if (!n) return 0;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0;

  // unroll by rows
  ValueType y_0 = 0, y_1 = 0, y_2 = 0, y_3 = 0;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int j = 0; j < n; ++j) {
    const int jj        = j * _as1;
    const ValueType x_j = x[j * _xs0];

    y_0 += A[i0 + jj] * x_j;
    y_1 += A[i1 + jj] * x_j;
    y_2 += A[i2 + jj] * x_j;
    y_3 += A[i3 + jj] * x_j;
  }

  y[0 * _ys0] += alpha * y_0;
  y[1 * _ys0] += alpha * y_1;
  y[2 * _ys0] += alpha * y_2;
  y[3 * _ys0] += alpha * y_3;

  return 0;
}

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerMultipleDotProduct<3>::serial_invoke(
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const ValueType *KOKKOS_RESTRICT x, const int n,
    /**/ ValueType *KOKKOS_RESTRICT y) {
  if (n <= 0) return 0;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0;

  // unroll by rows
  ValueType y_0 = 0, y_1 = 0, y_2 = 0;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int j = 0; j < n; ++j) {
    const int jj        = j * _as1;
    const ValueType x_j = x[j * _xs0];

    y_0 += A[i0 + jj] * x_j;
    y_1 += A[i1 + jj] * x_j;
    y_2 += A[i2 + jj] * x_j;
  }

  y[0 * _ys0] += alpha * y_0;
  y[1 * _ys0] += alpha * y_1;
  y[2 * _ys0] += alpha * y_2;

  return 0;
}

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerMultipleDotProduct<2>::serial_invoke(
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const ValueType *KOKKOS_RESTRICT x, const int n,
    /**/ ValueType *KOKKOS_RESTRICT y) {
  if (n <= 0) return 0;

  const int i0 = 0 * _as0, i1 = 1 * _as0;

  // unroll by rows
  ValueType y_0 = 0, y_1 = 0;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int j = 0; j < n; ++j) {
    const int jj        = j * _as1;
    const ValueType x_j = x[j * _xs0];

    y_0 += A[i0 + jj] * x_j;
    y_1 += A[i1 + jj] * x_j;
  }

  y[0 * _ys0] += alpha * y_0;
  y[1 * _ys0] += alpha * y_1;

  return 0;
}

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerMultipleDotProduct<1>::serial_invoke(
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const ValueType *KOKKOS_RESTRICT x, const int n,
    /**/ ValueType *KOKKOS_RESTRICT y) {
  if (n <= 0) return 0;

  // unroll by rows
  ValueType y_0 = 0;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int j = 0; j < n; ++j) y_0 += A[j * _as1] * x[j * _xs0];

  y[0] += alpha * y_0;

  return 0;
}

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerMultipleDotProduct<5>::serial_invoke(
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const ValueType *KOKKOS_RESTRICT x, const int m, const int n,
    /**/ ValueType *KOKKOS_RESTRICT y) {
  if (m <= 0 || n <= 0) return 0;
  switch (m) {
    case 5: {
      InnerMultipleDotProduct<5> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
    case 4: {
      InnerMultipleDotProduct<4> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
    case 3: {
      InnerMultipleDotProduct<3> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
    case 2: {
      InnerMultipleDotProduct<2> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
    case 1: {
      InnerMultipleDotProduct<1> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
  }
  return 0;
}

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerMultipleDotProduct<4>::serial_invoke(
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const ValueType *KOKKOS_RESTRICT x, const int m, const int n,
    /**/ ValueType *KOKKOS_RESTRICT y) {
  if (m <= 0 || n <= 0) return 0;
  switch (m) {
    case 4: {
      InnerMultipleDotProduct<4> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
    case 3: {
      InnerMultipleDotProduct<3> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
    case 2: {
      InnerMultipleDotProduct<2> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
    case 1: {
      InnerMultipleDotProduct<1> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
  }
  return 0;
}

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerMultipleDotProduct<3>::serial_invoke(
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const ValueType *KOKKOS_RESTRICT x, const int m, const int n,
    /**/ ValueType *KOKKOS_RESTRICT y) {
  if (m <= 0 || n <= 0) return 0;
  switch (m) {
    case 3: {
      InnerMultipleDotProduct<3> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
    case 2: {
      InnerMultipleDotProduct<2> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
    case 1: {
      InnerMultipleDotProduct<1> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
  }
  return 0;
}

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerMultipleDotProduct<2>::serial_invoke(
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const ValueType *KOKKOS_RESTRICT x, const int m, const int n,
    /**/ ValueType *KOKKOS_RESTRICT y) {
  if (m <= 0 || n <= 0) return 0;
  switch (m) {
    case 2: {
      InnerMultipleDotProduct<2> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
    case 1: {
      InnerMultipleDotProduct<1> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
  }
  return 0;
}

template <>
template <typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int InnerMultipleDotProduct<1>::serial_invoke(
    const ScalarType alpha, const ValueType *KOKKOS_RESTRICT A,
    const ValueType *KOKKOS_RESTRICT x, const int m, const int n,
    /**/ ValueType *KOKKOS_RESTRICT y) {
  if (m <= 0 || n <= 0) return 0;
  switch (m) {
    case 1: {
      InnerMultipleDotProduct<1> inner(_as0, _as1, _xs0, _ys0);
      inner.serial_invoke(alpha, A, x, n, y);
      break;
    }
  }
  return 0;
}
}}  // namespace KokkosBlas::Impl

#endif // __KOKKOSBLAS_INNER_MULTIPLE_DOT_PRODUCT_SERIAL_IMPL_HPP__
