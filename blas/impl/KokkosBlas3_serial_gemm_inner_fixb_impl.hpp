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

#ifndef KOKKOSBLAS3_INNER_GEMM_FIXB_SERIAL_IMPL_HPP
#define KOKKOSBLAS3_INNER_GEMM_FIXB_SERIAL_IMPL_HPP

/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosBlas {

///
/// Inner kernel (5x5)
/// ==================

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<5, 5>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_03 = B[0 * _bs0 + 3 * _bs1],
                   b_04 = B[0 * _bs0 + 4 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1],
                   b_11 = B[1 * _bs0 + 1 * _bs1], b_12 = B[1 * _bs0 + 2 * _bs1],
                   b_13 = B[1 * _bs0 + 3 * _bs1], b_14 = B[1 * _bs0 + 4 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1],
                   b_22 = B[2 * _bs0 + 2 * _bs1], b_23 = B[2 * _bs0 + 3 * _bs1],
                   b_24 = B[2 * _bs0 + 4 * _bs1], b_30 = B[3 * _bs0 + 0 * _bs1],
                   b_31 = B[3 * _bs0 + 1 * _bs1], b_32 = B[3 * _bs0 + 2 * _bs1],
                   b_33 = B[3 * _bs0 + 3 * _bs1], b_34 = B[3 * _bs0 + 4 * _bs1],
                   b_40 = B[4 * _bs0 + 0 * _bs1], b_41 = B[4 * _bs0 + 1 * _bs1],
                   b_42 = B[4 * _bs0 + 2 * _bs1], b_43 = B[4 * _bs0 + 3 * _bs1],
                   b_44 = B[4 * _bs0 + 4 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2, a_p3, a_p4;
  ValueTypeC c_p0, c_p1, c_p2, c_p3, c_p4;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, ja3 = 3 * _as1,
            ja4 = 4 * _as1, jc0 = 0 * _cs1, jc1 = 1 * _cs1, jc2 = 2 * _cs1,
            jc3 = 3 * _cs1, jc4 = 4 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];
    a_p3 = A[p * _bs0 + ja3];
    a_p4 = A[p * _bs0 + ja4];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p3 = a_p0 * b_03;
    c_p4 = a_p0 * b_04;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;
    c_p3 += a_p1 * b_13;
    c_p4 += a_p1 * b_14;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;
    c_p2 += a_p2 * b_22;
    c_p3 += a_p2 * b_23;
    c_p4 += a_p2 * b_24;
    c_p0 += a_p3 * b_30;
    c_p1 += a_p3 * b_31;
    c_p2 += a_p3 * b_32;
    c_p3 += a_p3 * b_33;
    c_p4 += a_p3 * b_34;
    c_p0 += a_p4 * b_40;
    c_p1 += a_p4 * b_41;
    c_p2 += a_p4 * b_42;
    c_p3 += a_p4 * b_43;
    c_p4 += a_p4 * b_44;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
    C[p * _cs0 + jc3] += alpha * c_p3;
    C[p * _cs0 + jc4] += alpha * c_p4;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<5, 4>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_03 = B[0 * _bs0 + 3 * _bs1],
                   b_10 = B[1 * _bs0 + 0 * _bs1], b_11 = B[1 * _bs0 + 1 * _bs1],
                   b_12 = B[1 * _bs0 + 2 * _bs1], b_13 = B[1 * _bs0 + 3 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1],
                   b_22 = B[2 * _bs0 + 2 * _bs1], b_23 = B[2 * _bs0 + 3 * _bs1],
                   b_30 = B[3 * _bs0 + 0 * _bs1], b_31 = B[3 * _bs0 + 1 * _bs1],
                   b_32 = B[3 * _bs0 + 2 * _bs1], b_33 = B[3 * _bs0 + 3 * _bs1],
                   b_40 = B[4 * _bs0 + 0 * _bs1], b_41 = B[4 * _bs0 + 1 * _bs1],
                   b_42 = B[4 * _bs0 + 2 * _bs1], b_43 = B[4 * _bs0 + 3 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2, a_p3, a_p4;
  ValueTypeC c_p0, c_p1, c_p2, c_p3;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, ja3 = 3 * _as1,
            ja4 = 4 * _as1, jc0 = 0 * _cs1, jc1 = 1 * _cs1, jc2 = 2 * _cs1,
            jc3 = 3 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];
    a_p3 = A[p * _bs0 + ja3];
    a_p4 = A[p * _bs0 + ja4];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p3 = a_p0 * b_03;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;
    c_p3 += a_p1 * b_13;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;
    c_p2 += a_p2 * b_22;
    c_p3 += a_p2 * b_23;
    c_p0 += a_p3 * b_30;
    c_p1 += a_p3 * b_31;
    c_p2 += a_p3 * b_32;
    c_p3 += a_p3 * b_33;
    c_p0 += a_p4 * b_40;
    c_p1 += a_p4 * b_41;
    c_p2 += a_p4 * b_42;
    c_p3 += a_p4 * b_43;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
    C[p * _cs0 + jc3] += alpha * c_p3;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<5, 3>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1],
                   b_11 = B[1 * _bs0 + 1 * _bs1], b_12 = B[1 * _bs0 + 2 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1],
                   b_22 = B[2 * _bs0 + 2 * _bs1], b_30 = B[3 * _bs0 + 0 * _bs1],
                   b_31 = B[3 * _bs0 + 1 * _bs1], b_32 = B[3 * _bs0 + 2 * _bs1],
                   b_40 = B[4 * _bs0 + 0 * _bs1], b_41 = B[4 * _bs0 + 1 * _bs1],
                   b_42 = B[4 * _bs0 + 2 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2, a_p3, a_p4;
  ValueTypeC c_p0, c_p1, c_p2;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, ja3 = 3 * _as1,
            ja4 = 4 * _as1, jc0 = 0 * _cs1, jc1 = 1 * _cs1, jc2 = 2 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];
    a_p3 = A[p * _bs0 + ja3];
    a_p4 = A[p * _bs0 + ja4];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;
    c_p2 += a_p2 * b_22;
    c_p0 += a_p3 * b_30;
    c_p1 += a_p3 * b_31;
    c_p2 += a_p3 * b_32;
    c_p0 += a_p4 * b_40;
    c_p1 += a_p4 * b_41;
    c_p2 += a_p4 * b_42;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
  }

  return 0;
}
template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<5, 2>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_10 = B[1 * _bs0 + 0 * _bs1], b_11 = B[1 * _bs0 + 1 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1],
                   b_30 = B[3 * _bs0 + 0 * _bs1], b_31 = B[3 * _bs0 + 1 * _bs1],
                   b_40 = B[4 * _bs0 + 0 * _bs1], b_41 = B[4 * _bs0 + 1 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2, a_p3, a_p4;
  ValueTypeC c_p0, c_p1;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, ja3 = 3 * _as1,
            ja4 = 4 * _as1, jc0 = 0 * _cs1, jc1 = 1 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];
    a_p3 = A[p * _bs0 + ja3];
    a_p4 = A[p * _bs0 + ja4];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;
    c_p0 += a_p3 * b_30;
    c_p1 += a_p3 * b_31;
    c_p0 += a_p4 * b_40;
    c_p1 += a_p4 * b_41;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
  }

  return 0;
}
template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<5, 1>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_30 = B[3 * _bs0 + 0 * _bs1],
                   b_40 = B[4 * _bs0 + 0 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2, a_p3, a_p4;
  ValueTypeC c_p0;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, ja3 = 3 * _as1,
            ja4 = 4 * _as1, jc0 = 0 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];
    a_p3 = A[p * _bs0 + ja3];
    a_p4 = A[p * _bs0 + ja4];

    c_p0 = a_p0 * b_00;
    c_p0 += a_p1 * b_10;
    c_p0 += a_p2 * b_20;
    c_p0 += a_p3 * b_30;
    c_p0 += a_p4 * b_40;

    C[p * _cs0 + jc0] += alpha * c_p0;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<4, 5>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_03 = B[0 * _bs0 + 3 * _bs1],
                   b_04 = B[0 * _bs0 + 4 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1],
                   b_11 = B[1 * _bs0 + 1 * _bs1], b_12 = B[1 * _bs0 + 2 * _bs1],
                   b_13 = B[1 * _bs0 + 3 * _bs1], b_14 = B[1 * _bs0 + 4 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1],
                   b_22 = B[2 * _bs0 + 2 * _bs1], b_23 = B[2 * _bs0 + 3 * _bs1],
                   b_24 = B[2 * _bs0 + 4 * _bs1], b_30 = B[3 * _bs0 + 0 * _bs1],
                   b_31 = B[3 * _bs0 + 1 * _bs1], b_32 = B[3 * _bs0 + 2 * _bs1],
                   b_33 = B[3 * _bs0 + 3 * _bs1], b_34 = B[3 * _bs0 + 4 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2, a_p3;
  ValueTypeC c_p0, c_p1, c_p2, c_p3, c_p4;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, ja3 = 3 * _as1,
            jc0 = 0 * _cs1, jc1 = 1 * _cs1, jc2 = 2 * _cs1, jc3 = 3 * _cs1,
            jc4 = 4 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];
    a_p3 = A[p * _bs0 + ja3];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p3 = a_p0 * b_03;
    c_p4 = a_p0 * b_04;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;
    c_p3 += a_p1 * b_13;
    c_p4 += a_p1 * b_14;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;
    c_p2 += a_p2 * b_22;
    c_p3 += a_p2 * b_23;
    c_p4 += a_p2 * b_24;
    c_p0 += a_p3 * b_30;
    c_p1 += a_p3 * b_31;
    c_p2 += a_p3 * b_32;
    c_p3 += a_p3 * b_33;
    c_p4 += a_p3 * b_34;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
    C[p * _cs0 + jc3] += alpha * c_p3;
    C[p * _cs0 + jc4] += alpha * c_p4;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<3, 5>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_03 = B[0 * _bs0 + 3 * _bs1],
                   b_04 = B[0 * _bs0 + 4 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1],
                   b_11 = B[1 * _bs0 + 1 * _bs1], b_12 = B[1 * _bs0 + 2 * _bs1],
                   b_13 = B[1 * _bs0 + 3 * _bs1], b_14 = B[1 * _bs0 + 4 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1],
                   b_22 = B[2 * _bs0 + 2 * _bs1], b_23 = B[2 * _bs0 + 3 * _bs1],
                   b_24 = B[2 * _bs0 + 4 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2;
  ValueTypeC c_p0, c_p1, c_p2, c_p3, c_p4;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, jc0 = 0 * _cs1,
            jc1 = 1 * _cs1, jc2 = 2 * _cs1, jc3 = 3 * _cs1, jc4 = 4 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p3 = a_p0 * b_03;
    c_p4 = a_p0 * b_04;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;
    c_p3 += a_p1 * b_13;
    c_p4 += a_p1 * b_14;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;
    c_p2 += a_p2 * b_22;
    c_p3 += a_p2 * b_23;
    c_p4 += a_p2 * b_24;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
    C[p * _cs0 + jc3] += alpha * c_p3;
    C[p * _cs0 + jc4] += alpha * c_p4;
  }

  return 0;
}
template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<2, 5>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_03 = B[0 * _bs0 + 3 * _bs1],
                   b_04 = B[0 * _bs0 + 4 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1],
                   b_11 = B[1 * _bs0 + 1 * _bs1], b_12 = B[1 * _bs0 + 2 * _bs1],
                   b_13 = B[1 * _bs0 + 3 * _bs1], b_14 = B[1 * _bs0 + 4 * _bs1];

  ValueTypeA a_p0, a_p1;
  ValueTypeC c_p0, c_p1, c_p2, c_p3, c_p4;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, jc0 = 0 * _cs1, jc1 = 1 * _cs1,
            jc2 = 2 * _cs1, jc3 = 3 * _cs1, jc4 = 4 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p3 = a_p0 * b_03;
    c_p4 = a_p0 * b_04;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;
    c_p3 += a_p1 * b_13;
    c_p4 += a_p1 * b_14;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
    C[p * _cs0 + jc3] += alpha * c_p3;
    C[p * _cs0 + jc4] += alpha * c_p4;
  }

  return 0;
}
template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<1, 5>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_03 = B[0 * _bs0 + 3 * _bs1],
                   b_04 = B[0 * _bs0 + 4 * _bs1];

  ValueTypeA a_p0;
  ValueTypeC c_p0, c_p1, c_p2, c_p3, c_p4;

  const int ja0 = 0 * _as1, jc0 = 0 * _cs1, jc1 = 1 * _cs1, jc2 = 2 * _cs1,
            jc3 = 3 * _cs1, jc4 = 4 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p3 = a_p0 * b_03;
    c_p4 = a_p0 * b_04;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
    C[p * _cs0 + jc3] += alpha * c_p3;
    C[p * _cs0 + jc4] += alpha * c_p4;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<5, 5>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m, const int n, const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0 || n <= 0 || k <= 0) return 0;

  switch (k * 10 + n) {
    case 54: {
      InnerGemmFixB<5, 4> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 53: {
      InnerGemmFixB<5, 3> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 52: {
      InnerGemmFixB<5, 2> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 51: {
      InnerGemmFixB<5, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 45: {
      InnerGemmFixB<4, 5> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 35: {
      InnerGemmFixB<3, 5> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 25: {
      InnerGemmFixB<2, 5> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 15: {
      InnerGemmFixB<1, 5> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
  }

  return 0;
}

///
/// Inner kernel (4x4)
/// ==================

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<4, 4>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_03 = B[0 * _bs0 + 3 * _bs1],
                   b_10 = B[1 * _bs0 + 0 * _bs1], b_11 = B[1 * _bs0 + 1 * _bs1],
                   b_12 = B[1 * _bs0 + 2 * _bs1], b_13 = B[1 * _bs0 + 3 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1],
                   b_22 = B[2 * _bs0 + 2 * _bs1], b_23 = B[2 * _bs0 + 3 * _bs1],
                   b_30 = B[3 * _bs0 + 0 * _bs1], b_31 = B[3 * _bs0 + 1 * _bs1],
                   b_32 = B[3 * _bs0 + 2 * _bs1], b_33 = B[3 * _bs0 + 3 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2, a_p3;
  ValueTypeC c_p0, c_p1, c_p2, c_p3;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, ja3 = 3 * _as1,
            jc0 = 0 * _cs1, jc1 = 1 * _cs1, jc2 = 2 * _cs1, jc3 = 3 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];
    a_p3 = A[p * _bs0 + ja3];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p3 = a_p0 * b_03;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;
    c_p3 += a_p1 * b_13;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;
    c_p2 += a_p2 * b_22;
    c_p3 += a_p2 * b_23;
    c_p0 += a_p3 * b_30;
    c_p1 += a_p3 * b_31;
    c_p2 += a_p3 * b_32;
    c_p3 += a_p3 * b_33;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
    C[p * _cs0 + jc3] += alpha * c_p3;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<4, 3>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1],
                   b_11 = B[1 * _bs0 + 1 * _bs1], b_12 = B[1 * _bs0 + 2 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1],
                   b_22 = B[2 * _bs0 + 2 * _bs1], b_30 = B[3 * _bs0 + 0 * _bs1],
                   b_31 = B[3 * _bs0 + 1 * _bs1], b_32 = B[3 * _bs0 + 2 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2, a_p3;
  ValueTypeC c_p0, c_p1, c_p2;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, ja3 = 3 * _as1,
            jc0 = 0 * _cs1, jc1 = 1 * _cs1, jc2 = 2 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];
    a_p3 = A[p * _bs0 + ja3];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;
    c_p2 += a_p2 * b_22;
    c_p0 += a_p3 * b_30;
    c_p1 += a_p3 * b_31;
    c_p2 += a_p3 * b_32;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<4, 2>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_10 = B[1 * _bs0 + 0 * _bs1], b_11 = B[1 * _bs0 + 1 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1],
                   b_30 = B[3 * _bs0 + 0 * _bs1], b_31 = B[3 * _bs0 + 1 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2, a_p3;
  ValueTypeC c_p0, c_p1;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, ja3 = 3 * _as1,
            jc0 = 0 * _cs1, jc1 = 1 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];
    a_p3 = A[p * _bs0 + ja3];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;
    c_p0 += a_p3 * b_30;
    c_p1 += a_p3 * b_31;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<4, 1>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_30 = B[3 * _bs0 + 0 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2, a_p3;
  ValueTypeC c_p0;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, ja3 = 3 * _as1,
            jc0 = 0 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];
    a_p3 = A[p * _bs0 + ja3];

    c_p0 = a_p0 * b_00;
    c_p0 += a_p1 * b_10;
    c_p0 += a_p2 * b_20;
    c_p0 += a_p3 * b_30;

    C[p * _cs0 + jc0] += alpha * c_p0;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<3, 4>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_03 = B[0 * _bs0 + 3 * _bs1],
                   b_10 = B[1 * _bs0 + 0 * _bs1], b_11 = B[1 * _bs0 + 1 * _bs1],
                   b_12 = B[1 * _bs0 + 2 * _bs1], b_13 = B[1 * _bs0 + 3 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1],
                   b_22 = B[2 * _bs0 + 2 * _bs1], b_23 = B[2 * _bs0 + 3 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2;
  ValueTypeC c_p0, c_p1, c_p2, c_p3;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, jc0 = 0 * _cs1,
            jc1 = 1 * _cs1, jc2 = 2 * _cs1, jc3 = 3 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p3 = a_p0 * b_03;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;
    c_p3 += a_p1 * b_13;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;
    c_p2 += a_p2 * b_22;
    c_p3 += a_p2 * b_23;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
    C[p * _cs0 + jc3] += alpha * c_p3;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<2, 4>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_03 = B[0 * _bs0 + 3 * _bs1],
                   b_10 = B[1 * _bs0 + 0 * _bs1], b_11 = B[1 * _bs0 + 1 * _bs1],
                   b_12 = B[1 * _bs0 + 2 * _bs1], b_13 = B[1 * _bs0 + 3 * _bs1];

  ValueTypeA a_p0, a_p1;
  ValueTypeC c_p0, c_p1, c_p2, c_p3;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, jc0 = 0 * _cs1, jc1 = 1 * _cs1,
            jc2 = 2 * _cs1, jc3 = 3 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p3 = a_p0 * b_03;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;
    c_p3 += a_p1 * b_13;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
    C[p * _cs0 + jc3] += alpha * c_p3;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<1, 4>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_03 = B[0 * _bs0 + 3 * _bs1];

  ValueTypeA a_p0;
  ValueTypeC c_p0, c_p1, c_p2, c_p3;

  const int ja0 = 0 * _as1, jc0 = 0 * _cs1, jc1 = 1 * _cs1, jc2 = 2 * _cs1,
            jc3 = 3 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p3 = a_p0 * b_03;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
    C[p * _cs0 + jc3] += alpha * c_p3;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<4, 4>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m, const int n, const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0 || n <= 0 || k <= 0) return 0;

  switch (k * 10 + n) {
    case 43: {
      InnerGemmFixB<4, 3> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 42: {
      InnerGemmFixB<4, 2> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 41: {
      InnerGemmFixB<4, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 34: {
      InnerGemmFixB<3, 4> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 24: {
      InnerGemmFixB<2, 4> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 14: {
      InnerGemmFixB<1, 4> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
  }

  return 0;
}

///
/// Inner kernel (3x3)
/// ==================

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<3, 3>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1],
                   b_11 = B[1 * _bs0 + 1 * _bs1], b_12 = B[1 * _bs0 + 2 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1],
                   b_22 = B[2 * _bs0 + 2 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2;
  ValueTypeC c_p0, c_p1, c_p2;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, jc0 = 0 * _cs1,
            jc1 = 1 * _cs1, jc2 = 2 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;
    c_p2 += a_p2 * b_22;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<3, 2>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_10 = B[1 * _bs0 + 0 * _bs1], b_11 = B[1 * _bs0 + 1 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1], b_21 = B[2 * _bs0 + 1 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2;
  ValueTypeC c_p0, c_p1;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, jc0 = 0 * _cs1,
            jc1 = 1 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p0 += a_p2 * b_20;
    c_p1 += a_p2 * b_21;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<3, 1>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1],
                   b_20 = B[2 * _bs0 + 0 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2;
  ValueTypeC c_p0;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, jc0 = 0 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];
    a_p2 = A[p * _bs0 + ja2];

    c_p0 = a_p0 * b_00;
    c_p0 += a_p1 * b_10;
    c_p0 += a_p2 * b_20;

    C[p * _cs0 + jc0] += alpha * c_p0;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<2, 3>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 2 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1],
                   b_11 = B[1 * _bs0 + 1 * _bs1], b_12 = B[1 * _bs0 + 2 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2;
  ValueTypeC c_p0, c_p1, c_p2;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, jc0 = 0 * _cs1,
            jc1 = 1 * _cs1, jc2 = 2 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;
    c_p2 += a_p1 * b_12;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
  }

  return 0;
}
template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<1, 3>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_02 = B[0 * _bs0 + 1 * _bs1];

  ValueTypeA a_p0, a_p1, a_p2;
  ValueTypeC c_p0, c_p1, c_p2;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, ja2 = 2 * _as1, jc0 = 0 * _cs1,
            jc1 = 1 * _cs1, jc2 = 2 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p2 = a_p0 * b_02;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
    C[p * _cs0 + jc2] += alpha * c_p2;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<3, 3>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m, const int n, const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0 || n <= 0 || k <= 0) return 0;

  switch (k * 10 + n) {
    case 32: {
      InnerGemmFixB<3, 2> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 31: {
      InnerGemmFixB<3, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 23: {
      InnerGemmFixB<2, 3> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 13: {
      InnerGemmFixB<1, 3> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
  }

  return 0;
}

///
/// Inner kernel (2x2)
/// ==================

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<2, 2>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1],
                   b_10 = B[1 * _bs0 + 0 * _bs1], b_11 = B[1 * _bs0 + 1 * _bs1];

  ValueTypeA a_p0, a_p1;
  ValueTypeC c_p0, c_p1;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, jc0 = 0 * _cs1, jc1 = 1 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;
    c_p0 += a_p1 * b_10;
    c_p1 += a_p1 * b_11;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<2, 1>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_10 = B[1 * _bs0 + 0 * _bs1];

  ValueTypeA a_p0, a_p1;
  ValueTypeC c_p0;

  const int ja0 = 0 * _as1, ja1 = 1 * _as1, jc0 = 0 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];
    a_p1 = A[p * _bs0 + ja1];

    c_p0 = a_p0 * b_00;
    c_p0 += a_p1 * b_10;

    C[p * _cs0 + jc0] += alpha * c_p0;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<1, 2>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1], b_01 = B[0 * _bs0 + 1 * _bs1];

  ValueTypeA a_p0;
  ValueTypeC c_p0, c_p1;

  const int ja0 = 0 * _as1, jc0 = 0 * _cs1, jc1 = 1 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];

    c_p0 = a_p0 * b_00;
    c_p1 = a_p0 * b_01;

    C[p * _cs0 + jc0] += alpha * c_p0;
    C[p * _cs0 + jc1] += alpha * c_p1;
  }

  return 0;
}

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<2, 2>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m, const int n, const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0 || n <= 0 || k <= 0) return 0;

  switch (k * 10 + n) {
    case 21: {
      InnerGemmFixB<2, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
    case 12: {
      InnerGemmFixB<1, 2> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(alpha, A, B, m, C);
      break;
    }
  }

  return 0;
}

///
/// Inner kernel (1x1)
/// ==================

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<1, 1>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0) return 0;

  const ValueTypeB b_00 = B[0 * _bs0 + 0 * _bs1];

  ValueTypeA a_p0;
  ValueTypeC c_p0;

  const int ja0 = 0 * _as1, jc0 = 0 * _cs1;

  for (int p = 0; p < m; ++p) {
    a_p0 = A[p * _bs0 + ja0];

    c_p0 = a_p0 * b_00;

    C[p * _cs0 + jc0] += alpha * c_p0;
  }

  return 0;
}

///
/// Inner kernel (remainders)
/// =========================

template <>
template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
          typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixB<0, 0>::serial_invoke(
    const ScalarType alpha, const ValueTypeA *KOKKOS_RESTRICT A,
    const ValueTypeB *KOKKOS_RESTRICT B, const int m, const int n, const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0 || n <= 0 || k <= 0) return 0;

  if (k == n) {
    switch (k) {
      case 5: {
        InnerGemmFixB<5, 5> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
        inner.serial_invoke(alpha, A, B, m, C);
        break;
      }
      case 4: {
        InnerGemmFixB<4, 4> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
        inner.serial_invoke(alpha, A, B, m, C);
        break;
      }
      case 3: {
        InnerGemmFixB<3, 3> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
        inner.serial_invoke(alpha, A, B, m, C);
        break;
      }
      case 2: {
        InnerGemmFixB<2, 2> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
        inner.serial_invoke(alpha, A, B, m, C);
        break;
      }
      case 1: {
        InnerGemmFixB<1, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
        inner.serial_invoke(alpha, A, B, m, C);
        break;
      }
    }
  } else {
    for (int i = 0; i < m; ++i) {
      const ValueTypeA *KOKKOS_RESTRICT iA = A + i * _as0;
      /**/ ValueTypeC *KOKKOS_RESTRICT iC    = C + i * _cs0;
      for (int j = 0; j < n; ++j) {
        const ValueTypeB *KOKKOS_RESTRICT jB = B + j * _bs1;
        /**/ ValueTypeC tC                     = 0;
        for (int p = 0; p < k; ++p) tC += iA[p * _as1] * jB[p * _bs0];
        iC[j * _cs1] += alpha * tC;
      }
    }
  }
  return 0;
}

}  // namespace KokkosBlas

#endif
