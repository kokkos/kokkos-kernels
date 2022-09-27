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

#ifndef KOKKOSBLAS3_INNER_GEMM_FIXC_SERIAL_IMPL_HPP
#define KOKKOSBLAS3_INNER_GEMM_FIXC_SERIAL_IMPL_HPP

/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosBlas {

///
/// Inner kernel (5x5)
/// ==================

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<5, 5>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p, a_3p, a_4p;
  ValueTypeB b_p0, b_p1, b_p2, b_p3, b_p4;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_03 = zero, c_04 = zero,
             c_10 = zero, c_11 = zero, c_12 = zero, c_13 = zero, c_14 = zero,
             c_20 = zero, c_21 = zero, c_22 = zero, c_23 = zero, c_24 = zero,
             c_30 = zero, c_31 = zero, c_32 = zero, c_33 = zero, c_34 = zero,
             c_40 = zero, c_41 = zero, c_42 = zero, c_43 = zero, c_44 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0,
            i4 = 4 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1, j2 = 2 * _bs1,
            j3 = 3 * _bs1, j4 = 4 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    b_p1 = opB(B[p * _bs0 + j1]);
    a_2p = opA(A[i2 + p * _as1]);
    b_p2 = opB(B[p * _bs0 + j2]);
    a_3p = opA(A[i3 + p * _as1]);
    b_p3 = opB(B[p * _bs0 + j3]);
    a_4p = opA(A[i4 + p * _as1]);
    b_p4 = opB(B[p * _bs0 + j4]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_03 += a_0p * b_p3;
    c_04 += a_0p * b_p4;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
    c_13 += a_1p * b_p3;
    c_14 += a_1p * b_p4;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
    c_22 += a_2p * b_p2;
    c_23 += a_2p * b_p3;
    c_24 += a_2p * b_p4;
    c_30 += a_3p * b_p0;
    c_31 += a_3p * b_p1;
    c_32 += a_3p * b_p2;
    c_33 += a_3p * b_p3;
    c_34 += a_3p * b_p4;
    c_40 += a_4p * b_p0;
    c_41 += a_4p * b_p1;
    c_42 += a_4p * b_p2;
    c_43 += a_4p * b_p3;
    c_44 += a_4p * b_p4;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[0 * _cs0 + 3 * _cs1] += alpha * c_03;
  C[0 * _cs0 + 4 * _cs1] += alpha * c_04;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;
  C[1 * _cs0 + 3 * _cs1] += alpha * c_13;
  C[1 * _cs0 + 4 * _cs1] += alpha * c_14;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;
  C[2 * _cs0 + 2 * _cs1] += alpha * c_22;
  C[2 * _cs0 + 3 * _cs1] += alpha * c_23;
  C[2 * _cs0 + 4 * _cs1] += alpha * c_24;
  C[3 * _cs0 + 0 * _cs1] += alpha * c_30;
  C[3 * _cs0 + 1 * _cs1] += alpha * c_31;
  C[3 * _cs0 + 2 * _cs1] += alpha * c_32;
  C[3 * _cs0 + 3 * _cs1] += alpha * c_33;
  C[3 * _cs0 + 4 * _cs1] += alpha * c_34;
  C[4 * _cs0 + 0 * _cs1] += alpha * c_40;
  C[4 * _cs0 + 1 * _cs1] += alpha * c_41;
  C[4 * _cs0 + 2 * _cs1] += alpha * c_42;
  C[4 * _cs0 + 3 * _cs1] += alpha * c_43;
  C[4 * _cs0 + 4 * _cs1] += alpha * c_44;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<5, 4>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p, a_3p, a_4p;
  ValueTypeB b_p0, b_p1, b_p2, b_p3;
  ValueTypeB c_00 = zero, c_01 = zero, c_02 = zero, c_03 = zero, c_10 = zero,
             c_11 = zero, c_12 = zero, c_13 = zero, c_20 = zero, c_21 = zero,
             c_22 = zero, c_23 = zero, c_30 = zero, c_31 = zero, c_32 = zero,
             c_33 = zero, c_40 = zero, c_41 = zero, c_42 = zero, c_43 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0,
            i4 = 4 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1, j2 = 2 * _bs1,
            j3 = 3 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    b_p1 = opB(B[p * _bs0 + j1]);
    a_2p = opA(A[i2 + p * _as1]);
    b_p2 = opB(B[p * _bs0 + j2]);
    a_3p = opA(A[i3 + p * _as1]);
    b_p3 = opB(B[p * _bs0 + j3]);
    a_4p = opA(A[i4 + p * _as1]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_03 += a_0p * b_p3;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
    c_13 += a_1p * b_p3;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
    c_22 += a_2p * b_p2;
    c_23 += a_2p * b_p3;
    c_30 += a_3p * b_p0;
    c_31 += a_3p * b_p1;
    c_32 += a_3p * b_p2;
    c_33 += a_3p * b_p3;
    c_40 += a_4p * b_p0;
    c_41 += a_4p * b_p1;
    c_42 += a_4p * b_p2;
    c_43 += a_4p * b_p3;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[0 * _cs0 + 3 * _cs1] += alpha * c_03;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;
  C[1 * _cs0 + 3 * _cs1] += alpha * c_13;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;
  C[2 * _cs0 + 2 * _cs1] += alpha * c_22;
  C[2 * _cs0 + 3 * _cs1] += alpha * c_23;
  C[3 * _cs0 + 0 * _cs1] += alpha * c_30;
  C[3 * _cs0 + 1 * _cs1] += alpha * c_31;
  C[3 * _cs0 + 2 * _cs1] += alpha * c_32;
  C[3 * _cs0 + 3 * _cs1] += alpha * c_33;
  C[4 * _cs0 + 0 * _cs1] += alpha * c_40;
  C[4 * _cs0 + 1 * _cs1] += alpha * c_41;
  C[4 * _cs0 + 2 * _cs1] += alpha * c_42;
  C[4 * _cs0 + 3 * _cs1] += alpha * c_43;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<5, 3>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p, a_3p, a_4p;
  ValueTypeB b_p0, b_p1, b_p2;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_10 = zero, c_11 = zero,
             c_12 = zero, c_20 = zero, c_21 = zero, c_22 = zero, c_30 = zero,
             c_31 = zero, c_32 = zero, c_40 = zero, c_41 = zero, c_42 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0,
            i4 = 4 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1, j2 = 2 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    b_p1 = opB(B[p * _bs0 + j1]);
    a_2p = opA(A[i2 + p * _as1]);
    b_p2 = opB(B[p * _bs0 + j2]);
    a_3p = opA(A[i3 + p * _as1]);
    a_4p = opA(A[i4 + p * _as1]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
    c_22 += a_2p * b_p2;
    c_30 += a_3p * b_p0;
    c_31 += a_3p * b_p1;
    c_32 += a_3p * b_p2;
    c_40 += a_4p * b_p0;
    c_41 += a_4p * b_p1;
    c_42 += a_4p * b_p2;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;
  C[2 * _cs0 + 2 * _cs1] += alpha * c_22;
  C[3 * _cs0 + 0 * _cs1] += alpha * c_30;
  C[3 * _cs0 + 1 * _cs1] += alpha * c_31;
  C[3 * _cs0 + 2 * _cs1] += alpha * c_32;
  C[4 * _cs0 + 0 * _cs1] += alpha * c_40;
  C[4 * _cs0 + 1 * _cs1] += alpha * c_41;
  C[4 * _cs0 + 2 * _cs1] += alpha * c_42;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<5, 2>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p, a_3p, a_4p;
  ValueTypeA b_p0, b_p1;
  ValueTypeA c_00 = zero, c_01 = zero, c_10 = zero, c_11 = zero, c_20 = zero,
             c_21 = zero, c_30 = zero, c_31 = zero, c_40 = zero, c_41 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0,
            i4 = 4 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    b_p1 = opB(B[p * _bs0 + j1]);
    a_2p = opA(A[i2 + p * _as1]);
    a_3p = opA(A[i3 + p * _as1]);
    a_4p = opA(A[i4 + p * _as1]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
    c_30 += a_3p * b_p0;
    c_31 += a_3p * b_p1;
    c_40 += a_4p * b_p0;
    c_41 += a_4p * b_p1;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;
  C[3 * _cs0 + 0 * _cs1] += alpha * c_30;
  C[3 * _cs0 + 1 * _cs1] += alpha * c_31;
  C[4 * _cs0 + 0 * _cs1] += alpha * c_40;
  C[4 * _cs0 + 1 * _cs1] += alpha * c_41;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<5, 1>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p, a_3p, a_4p;
  ValueTypeB b_p0;
  ValueTypeC c_00 = zero, c_10 = zero, c_20 = zero, c_30 = zero, c_40 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0,
            i4 = 4 * _as0, j0 = 0 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    a_2p = opA(A[i2 + p * _as1]);
    a_3p = opA(A[i3 + p * _as1]);
    a_4p = opA(A[i4 + p * _as1]);

    c_00 += a_0p * b_p0;
    c_10 += a_1p * b_p0;
    c_20 += a_2p * b_p0;
    c_30 += a_3p * b_p0;
    c_40 += a_4p * b_p0;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[3 * _cs0 + 0 * _cs1] += alpha * c_30;
  C[4 * _cs0 + 0 * _cs1] += alpha * c_40;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<4, 5>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p, a_3p;
  ValueTypeB b_p0, b_p1, b_p2, b_p3, b_p4;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_03 = zero, c_04 = zero,
             c_10 = zero, c_11 = zero, c_12 = zero, c_13 = zero, c_14 = zero,
             c_20 = zero, c_21 = zero, c_22 = zero, c_23 = zero, c_24 = zero,
             c_30 = zero, c_31 = zero, c_32 = zero, c_33 = zero, c_34 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0,
            j0 = 0 * _bs1, j1 = 1 * _bs1, j2 = 2 * _bs1, j3 = 3 * _bs1,
            j4 = 4 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p    = opA(A[i0 + p * _as1]);
    b_p0    = opB(B[p * _bs0 + j0]);
    a_1p    = opA(A[i1 + p * _as1]);
    b_p1    = opB(B[p * _bs0 + j1]);
    a_2p    = opA(A[i2 + p * _as1]);
    b_p2    = opB(B[p * _bs0 + j2]);
    a_3p    = opA(A[i3 + p * _as1]);
    b_p3    = opB(B[p * _bs0 + j3]);
    /**/ b_p4 = opB(B[p * _bs0 + j4]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_03 += a_0p * b_p3;
    c_04 += a_0p * b_p4;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
    c_13 += a_1p * b_p3;
    c_14 += a_1p * b_p4;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
    c_22 += a_2p * b_p2;
    c_23 += a_2p * b_p3;
    c_24 += a_2p * b_p4;
    c_30 += a_3p * b_p0;
    c_31 += a_3p * b_p1;
    c_32 += a_3p * b_p2;
    c_33 += a_3p * b_p3;
    c_34 += a_3p * b_p4;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[0 * _cs0 + 3 * _cs1] += alpha * c_03;
  C[0 * _cs0 + 4 * _cs1] += alpha * c_04;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;
  C[1 * _cs0 + 3 * _cs1] += alpha * c_13;
  C[1 * _cs0 + 4 * _cs1] += alpha * c_14;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;
  C[2 * _cs0 + 2 * _cs1] += alpha * c_22;
  C[2 * _cs0 + 3 * _cs1] += alpha * c_23;
  C[2 * _cs0 + 4 * _cs1] += alpha * c_24;
  C[3 * _cs0 + 0 * _cs1] += alpha * c_30;
  C[3 * _cs0 + 1 * _cs1] += alpha * c_31;
  C[3 * _cs0 + 2 * _cs1] += alpha * c_32;
  C[3 * _cs0 + 3 * _cs1] += alpha * c_33;
  C[3 * _cs0 + 4 * _cs1] += alpha * c_34;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<3, 5>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p;
  ValueTypeB b_p0, b_p1, b_p2, b_p3, b_p4;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_03 = zero, c_04 = zero,
             c_10 = zero, c_11 = zero, c_12 = zero, c_13 = zero, c_14 = zero,
             c_20 = zero, c_21 = zero, c_22 = zero, c_23 = zero, c_24 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, j0 = 0 * _bs1,
            j1 = 1 * _bs1, j2 = 2 * _bs1, j3 = 3 * _bs1, j4 = 4 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p    = opA(A[i0 + p * _as1]);
    b_p0    = opB(B[p * _bs0 + j0]);
    a_1p    = opA(A[i1 + p * _as1]);
    b_p1    = opB(B[p * _bs0 + j1]);
    a_2p    = opA(A[i2 + p * _as1]);
    b_p2    = opB(B[p * _bs0 + j2]);
    /**/ b_p3 = opB(B[p * _bs0 + j3]);
    /**/ b_p4 = opB(B[p * _bs0 + j4]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_03 += a_0p * b_p3;
    c_04 += a_0p * b_p4;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
    c_13 += a_1p * b_p3;
    c_14 += a_1p * b_p4;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
    c_22 += a_2p * b_p2;
    c_23 += a_2p * b_p3;
    c_24 += a_2p * b_p4;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[0 * _cs0 + 3 * _cs1] += alpha * c_03;
  C[0 * _cs0 + 4 * _cs1] += alpha * c_04;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;
  C[1 * _cs0 + 3 * _cs1] += alpha * c_13;
  C[1 * _cs0 + 4 * _cs1] += alpha * c_14;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;
  C[2 * _cs0 + 2 * _cs1] += alpha * c_22;
  C[2 * _cs0 + 3 * _cs1] += alpha * c_23;
  C[2 * _cs0 + 4 * _cs1] += alpha * c_24;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<2, 5>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p;
  ValueTypeB b_p0, b_p1, b_p2, b_p3, b_p4;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_03 = zero, c_04 = zero,
             c_10 = zero, c_11 = zero, c_12 = zero, c_13 = zero, c_14 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1,
            j2 = 2 * _bs1, j3 = 3 * _bs1, j4 = 4 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p    = opA(A[i0 + p * _as1]);
    b_p0    = opB(B[p * _bs0 + j0]);
    a_1p    = opA(A[i1 + p * _as1]);
    b_p1    = opB(B[p * _bs0 + j1]);
    /**/ b_p2 = opB(B[p * _bs0 + j2]);
    /**/ b_p3 = opB(B[p * _bs0 + j3]);
    /**/ b_p4 = opB(B[p * _bs0 + j4]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_03 += a_0p * b_p3;
    c_04 += a_0p * b_p4;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
    c_13 += a_1p * b_p3;
    c_14 += a_1p * b_p4;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[0 * _cs0 + 3 * _cs1] += alpha * c_03;
  C[0 * _cs0 + 4 * _cs1] += alpha * c_04;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;
  C[1 * _cs0 + 3 * _cs1] += alpha * c_13;
  C[1 * _cs0 + 4 * _cs1] += alpha * c_14;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<1, 5>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p;
  ValueTypeB b_p0, b_p1, b_p2, b_p3, b_p4;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_03 = zero, c_04 = zero;

  const int i0 = 0 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1, j2 = 2 * _bs1,
            j3 = 3 * _bs1, j4 = 4 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p    = opA(A[i0 + p * _as1]);
    b_p0    = opB(B[p * _bs0 + j0]);
    /**/ b_p1 = opB(B[p * _bs0 + j1]);
    /**/ b_p2 = opB(B[p * _bs0 + j2]);
    /**/ b_p3 = opB(B[p * _bs0 + j3]);
    /**/ b_p4 = opB(B[p * _bs0 + j4]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_03 += a_0p * b_p3;
    c_04 += a_0p * b_p4;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[0 * _cs0 + 3 * _cs1] += alpha * c_03;
  C[0 * _cs0 + 4 * _cs1] += alpha * c_04;

  return 0;
}
///
/// Inner kernel (4x4)
/// ==================

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<4, 4>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p, a_3p;
  ValueTypeB b_p0, b_p1, b_p2, b_p3;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_03 = zero, c_10 = zero,
             c_11 = zero, c_12 = zero, c_13 = zero, c_20 = zero, c_21 = zero,
             c_22 = zero, c_23 = zero, c_30 = zero, c_31 = zero, c_32 = zero,
             c_33 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0,
            j0 = 0 * _bs1, j1 = 1 * _bs1, j2 = 2 * _bs1, j3 = 3 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    b_p1 = opB(B[p * _bs0 + j1]);
    a_2p = opA(A[i2 + p * _as1]);
    b_p2 = opB(B[p * _bs0 + j2]);
    a_3p = opA(A[i3 + p * _as1]);
    b_p3 = opB(B[p * _bs0 + j3]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_03 += a_0p * b_p3;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
    c_13 += a_1p * b_p3;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
    c_22 += a_2p * b_p2;
    c_23 += a_2p * b_p3;
    c_30 += a_3p * b_p0;
    c_31 += a_3p * b_p1;
    c_32 += a_3p * b_p2;
    c_33 += a_3p * b_p3;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[0 * _cs0 + 3 * _cs1] += alpha * c_03;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;
  C[1 * _cs0 + 3 * _cs1] += alpha * c_13;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;
  C[2 * _cs0 + 2 * _cs1] += alpha * c_22;
  C[2 * _cs0 + 3 * _cs1] += alpha * c_23;
  C[3 * _cs0 + 0 * _cs1] += alpha * c_30;
  C[3 * _cs0 + 1 * _cs1] += alpha * c_31;
  C[3 * _cs0 + 2 * _cs1] += alpha * c_32;
  C[3 * _cs0 + 3 * _cs1] += alpha * c_33;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<4, 3>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p, a_3p;
  ValueTypeB b_p0, b_p1, b_p2;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_10 = zero, c_11 = zero,
             c_12 = zero, c_20 = zero, c_21 = zero, c_22 = zero, c_30 = zero,
             c_31 = zero, c_32 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0,
            j0 = 0 * _bs1, j1 = 1 * _bs1, j2 = 2 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    b_p1 = opB(B[p * _bs0 + j1]);
    a_2p = opA(A[i2 + p * _as1]);
    b_p2 = opB(B[p * _bs0 + j2]);
    a_3p = opA(A[i3 + p * _as1]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
    c_22 += a_2p * b_p2;
    c_30 += a_3p * b_p0;
    c_31 += a_3p * b_p1;
    c_32 += a_3p * b_p2;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;
  C[2 * _cs0 + 2 * _cs1] += alpha * c_22;
  C[3 * _cs0 + 0 * _cs1] += alpha * c_30;
  C[3 * _cs0 + 1 * _cs1] += alpha * c_31;
  C[3 * _cs0 + 2 * _cs1] += alpha * c_32;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<4, 2>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p, a_3p;
  ValueTypeB b_p0, b_p1;
  ValueTypeC c_00 = zero, c_01 = zero, c_10 = zero, c_11 = zero, c_20 = zero,
             c_21 = zero, c_30 = zero, c_31 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0,
            j0 = 0 * _bs1, j1 = 1 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    b_p1 = opB(B[p * _bs0 + j1]);
    a_2p = opA(A[i2 + p * _as1]);
    a_3p = opA(A[i3 + p * _as1]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
    c_30 += a_3p * b_p0;
    c_31 += a_3p * b_p1;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;
  C[3 * _cs0 + 0 * _cs1] += alpha * c_30;
  C[3 * _cs0 + 1 * _cs1] += alpha * c_31;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<4, 1>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p, a_3p;
  ValueTypeB b_p0;
  ValueTypeC c_00 = zero, c_10 = zero, c_20 = zero, c_30 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, i3 = 3 * _as0,
            j0 = 0 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    a_2p = opA(A[i2 + p * _as1]);
    a_3p = opA(A[i3 + p * _as1]);

    c_00 += a_0p * b_p0;
    c_10 += a_1p * b_p0;
    c_20 += a_2p * b_p0;
    c_30 += a_3p * b_p0;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[3 * _cs0 + 0 * _cs1] += alpha * c_30;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<3, 4>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p;
  ValueTypeB b_p0, b_p1, b_p2, b_p3;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_03 = zero, c_10 = zero,
             c_11 = zero, c_12 = zero, c_13 = zero, c_20 = zero, c_21 = zero,
             c_22 = zero, c_23 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, j0 = 0 * _bs1,
            j1 = 1 * _bs1, j2 = 2 * _bs1, j3 = 3 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p    = opA(A[i0 + p * _as1]);
    b_p0    = opB(B[p * _bs0 + j0]);
    a_1p    = opA(A[i1 + p * _as1]);
    b_p1    = opB(B[p * _bs0 + j1]);
    a_2p    = opA(A[i2 + p * _as1]);
    b_p2    = opB(B[p * _bs0 + j2]);
    /**/ b_p3 = opB(B[p * _bs0 + j3]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_03 += a_0p * b_p3;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
    c_13 += a_1p * b_p3;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
    c_22 += a_2p * b_p2;
    c_23 += a_2p * b_p3;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[0 * _cs0 + 3 * _cs1] += alpha * c_03;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;
  C[1 * _cs0 + 3 * _cs1] += alpha * c_13;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;
  C[2 * _cs0 + 2 * _cs1] += alpha * c_22;
  C[2 * _cs0 + 3 * _cs1] += alpha * c_23;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<2, 4>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p;
  ValueTypeB b_p0, b_p1, b_p2, b_p3;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_03 = zero, c_10 = zero,
             c_11 = zero, c_12 = zero, c_13 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1,
            j2 = 2 * _bs1, j3 = 3 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p    = opA(A[i0 + p * _as1]);
    b_p0    = opB(B[p * _bs0 + j0]);
    a_1p    = opA(A[i1 + p * _as1]);
    b_p1    = opB(B[p * _bs0 + j1]);
    /**/ b_p2 = opB(B[p * _bs0 + j2]);
    /**/ b_p3 = opB(B[p * _bs0 + j3]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_03 += a_0p * b_p3;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
    c_13 += a_1p * b_p3;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[0 * _cs0 + 3 * _cs1] += alpha * c_03;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;
  C[1 * _cs0 + 3 * _cs1] += alpha * c_13;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<1, 4>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p;
  ValueTypeB b_p0, b_p1, b_p2, b_p3;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_03 = zero;

  const int i0 = 0 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1, j2 = 2 * _bs1,
            j3 = 3 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p    = opA(A[i0 + p * _as1]);
    b_p0    = opB(B[p * _bs0 + j0]);
    /**/ b_p1 = opB(B[p * _bs0 + j1]);
    /**/ b_p2 = opB(B[p * _bs0 + j2]);
    /**/ b_p3 = opB(B[p * _bs0 + j3]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_03 += a_0p * b_p3;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[0 * _cs0 + 3 * _cs1] += alpha * c_03;

  return 0;
}

///
/// Inner kernel (3x3)
/// ==================

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<3, 3>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p;
  ValueTypeB b_p0, b_p1, b_p2;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_10 = zero, c_11 = zero,
             c_12 = zero, c_20 = zero, c_21 = zero, c_22 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, j0 = 0 * _bs1,
            j1 = 1 * _bs1, j2 = 2 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    b_p1 = opB(B[p * _bs0 + j1]);
    a_2p = opA(A[i2 + p * _as1]);
    b_p2 = opB(B[p * _bs0 + j2]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
    c_22 += a_2p * b_p2;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;
  C[2 * _cs0 + 2 * _cs1] += alpha * c_22;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<3, 2>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p, a_2p;
  ValueTypeB b_p0, b_p1;
  ValueTypeC c_00 = zero, c_01 = zero, c_10 = zero, c_11 = zero, c_20 = zero,
             c_21 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, j0 = 0 * _bs1,
            j1 = 1 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    b_p1 = opB(B[p * _bs0 + j1]);
    a_2p = opA(A[i2 + p * _as1]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_20 += a_2p * b_p0;
    c_21 += a_2p * b_p1;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;
  C[2 * _cs0 + 1 * _cs1] += alpha * c_21;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<3, 1>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_2p, a_1p;
  ValueTypeB b_p0;
  ValueTypeC c_00 = zero, c_10 = zero, c_20 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, i2 = 2 * _as0, j0 = 0 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    a_2p = opA(A[i2 + p * _as1]);

    c_00 += a_0p * b_p0;
    c_10 += a_1p * b_p0;
    c_20 += a_2p * b_p0;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[2 * _cs0 + 0 * _cs1] += alpha * c_20;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<2, 3>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p;
  ValueTypeB b_p0, b_p1, b_p2;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero, c_10 = zero, c_11 = zero,
             c_12 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1,
            j2 = 2 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p    = opA(A[i0 + p * _as1]);
    b_p0    = opB(B[p * _bs0 + j0]);
    a_1p    = opA(A[i1 + p * _as1]);
    b_p1    = opB(B[p * _bs0 + j1]);
    /**/ b_p2 = opB(B[p * _bs0 + j2]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
    c_12 += a_1p * b_p2;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;
  C[1 * _cs0 + 2 * _cs1] += alpha * c_12;

  return 0;
}
template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<1, 3>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p;
  ValueTypeB b_p0, b_p1, b_p2;
  ValueTypeC c_00 = zero, c_01 = zero, c_02 = zero;

  const int i0 = 0 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1, j2 = 2 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p    = opA(A[i0 + p * _as1]);
    b_p0    = opB(B[p * _bs0 + j0]);
    /**/ b_p1 = opB(B[p * _bs0 + j1]);
    /**/ b_p2 = opB(B[p * _bs0 + j2]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_02 += a_0p * b_p2;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[0 * _cs0 + 2 * _cs1] += alpha * c_02;

  return 0;
}

///
/// Inner kernel (2x2)
/// ==================

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<2, 2>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p;
  ValueTypeB b_p0, b_p1;
  ValueTypeC c_00 = zero, c_01 = zero, c_10 = zero, c_11 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);
    b_p1 = opB(B[p * _bs0 + j1]);

    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
    c_10 += a_1p * b_p0;
    c_11 += a_1p * b_p1;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;
  C[1 * _cs0 + 1 * _cs1] += alpha * c_11;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<2, 1>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p, a_1p;
  ValueTypeB b_p0;
  ValueTypeC c_00 = zero, c_10 = zero;

  const int i0 = 0 * _as0, i1 = 1 * _as0, j0 = 0 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p = opA(A[i0 + p * _as1]);
    b_p0 = opB(B[p * _bs0 + j0]);
    a_1p = opA(A[i1 + p * _as1]);

    c_00 += a_0p * b_p0;
    c_10 += a_1p * b_p0;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[1 * _cs0 + 0 * _cs1] += alpha * c_10;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<1, 2>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  const auto zero = static_cast<ValueTypeC>(0);
  ValueTypeA a_0p;
  ValueTypeB b_p0, b_p1;
  ValueTypeC c_00 = zero, c_01 = zero;
  const int i0 = 0 * _as0, j0 = 0 * _bs1, j1 = 1 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    a_0p       = opA(A[i0 + p * _as1]);
    b_p0       = opB(B[p * _bs0 + j0]);
    /* */ b_p1 = opB(B[p * _bs0 + j1]);
    c_00 += a_0p * b_p0;
    c_01 += a_0p * b_p1;
  }

  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;
  C[0 * _cs0 + 1 * _cs1] += alpha * c_01;

  return 0;
}

///
/// Inner kernel (1x1)
/// ==================

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<1, 1>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (k <= 0) return 0;

  ValueTypeC c_00 = static_cast<ValueTypeC>(0);

  const int i0 = 0 * _as0, j0 = 0 * _bs1;

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif
  for (int p = 0; p < k; ++p) {
    const ValueTypeA a_0p = opA(A[i0 + p * _as1]);
    const ValueTypeA b_p0 = opB(B[p * _bs0 + j0]);
    c_00 += a_0p * b_p0;
  }
  C[0 * _cs0 + 0 * _cs1] += alpha * c_00;

  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<0, 1>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int m, const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0 || k <= 0) return 0;

  switch (m) {
    case 5: {
      InnerGemmFixC<5, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 4: {
      InnerGemmFixC<4, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 3: {
      InnerGemmFixC<3, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 2: {
      InnerGemmFixC<2, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 1: {
      InnerGemmFixC<1, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    default: {
      Kokkos::abort("InnerGemmFixC<0,1>::serial_invoke, assert failure (m<=5)");
      break;
    }
  }
  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<5, 5>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int m, const int n, const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0 || n <= 0 || k <= 0) return 0;
  if (!(m <= 5 && n <= 5))
    Kokkos::abort(
        "InnerGemmFixC<5,5>::serial_invoke, assert failure (m<=5 && n<=5)");

  switch (m * 10 + n) {
    case 55: {
      InnerGemmFixC<5, 5> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 54: {
      InnerGemmFixC<5, 4> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 53: {
      InnerGemmFixC<5, 3> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 52: {
      InnerGemmFixC<5, 2> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 51: {
      InnerGemmFixC<5, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 45: {
      InnerGemmFixC<4, 5> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 35: {
      InnerGemmFixC<3, 5> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 25: {
      InnerGemmFixC<2, 5> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 15: {
      InnerGemmFixC<1, 5> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    default: {
      InnerGemmFixC<4, 4> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, m, n, k, C);
      break;
    }
  }
  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<4, 4>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int m, const int n, const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0 || n <= 0 || k <= 0) return 0;
  if (!(m <= 4 && n <= 4))
    Kokkos::abort(
        "InnerGemmFixC<4,4>::serial_invoke, assert failure (m<=4 && n<=4)");

  switch (m * 10 + n) {
    case 44: {
      InnerGemmFixC<4, 4> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 43: {
      InnerGemmFixC<4, 3> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 42: {
      InnerGemmFixC<4, 2> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 41: {
      InnerGemmFixC<4, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 34: {
      InnerGemmFixC<3, 4> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 24: {
      InnerGemmFixC<2, 4> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 14: {
      InnerGemmFixC<1, 4> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    default: {
      InnerGemmFixC<3, 3> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, m, n, k, C);
      break;
    }
  }
  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<3, 3>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int m, const int n, const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0 || n <= 0 || k <= 0) return 0;
  if (!(m <= 3 && n <= 3))
    Kokkos::abort(
        "InnerGemmFixC<3,3>::serial_invoke, assert failure (m<=3 && n<=3)");

  switch (m * 10 + n) {
    case 33: {
      InnerGemmFixC<3, 3> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 32: {
      InnerGemmFixC<3, 2> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 31: {
      InnerGemmFixC<3, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 23: {
      InnerGemmFixC<2, 3> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 13: {
      InnerGemmFixC<1, 3> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    default: {
      InnerGemmFixC<2, 2> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, m, n, k, C);
      break;
    }
  }
  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<2, 2>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int m, const int n, const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0 || n <= 0 || k <= 0) return 0;
  if (!(m <= 2 && n <= 2))
    Kokkos::abort(
        "InnerGemmFixC<2,2>::serial_invoke, assert failure (m<=2 && n<=2)");

  switch (m * 10 + n) {
    case 22: {
      InnerGemmFixC<2, 2> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 21: {
      InnerGemmFixC<2, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 12: {
      InnerGemmFixC<1, 2> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
    case 11: {
      InnerGemmFixC<1, 1> inner(_as0, _as1, _bs0, _bs1, _cs0, _cs1);
      inner.serial_invoke(opA, opB, alpha, A, B, k, C);
      break;
    }
  }
  return 0;
}

template <>
template <typename OpA, typename OpB, typename ScalarType, typename ValueTypeA,
          typename ValueTypeB, typename ValueTypeC>
KOKKOS_INLINE_FUNCTION int InnerGemmFixC<1, 1>::serial_invoke(
    OpA opA, OpB opB, const ScalarType alpha,
    const ValueTypeA *KOKKOS_RESTRICT A, const ValueTypeB *KOKKOS_RESTRICT B,
    const int m, const int n, const int k,
    /**/ ValueTypeC *KOKKOS_RESTRICT C) {
  if (m <= 0 || n <= 0 || k <= 0) return 0;
  if (!(m <= 1 && n <= 1))
    Kokkos::abort(
        "InnerGemmFixC<1,1>::serial_invoke, assert failure (m<=1 && n<=1)");

  return serial_invoke(alpha, A, B, k, C);
  ;
}

}  // namespace KokkosBlas

#endif
