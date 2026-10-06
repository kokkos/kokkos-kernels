//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
#ifndef KOKKOSBATCHED_QR_WITH_COLUMNPIVOTING_SERIAL_INTERNAL_HPP
#define KOKKOSBATCHED_QR_WITH_COLUMNPIVOTING_SERIAL_INTERNAL_HPP

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"

#include "KokkosBatched_FindAmax_Internal.hpp"
#include "KokkosBatched_Dot.hpp"
#include "KokkosBatched_ApplyPivot_Internal.hpp"

#include "KokkosBatched_Householder_Serial_Internal.hpp"
#include "KokkosBatched_ApplyHouseholder_Serial_Internal.hpp"

namespace KokkosBatched {

///
/// Serial Internal
/// ===================
///
/// this impl follows the flame interface of householder transformation
///
struct SerialUpdateColumnNormsInternal {
  template <typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int n, const ValueType *KOKKOS_RESTRICT a, const int as0,
                                           /* */ ValueType *KOKKOS_RESTRICT norm, const int ns0) {
    using ats = Kokkos::ArithTraits<ValueType>;
    for (int j = 0; j < n; ++j) {
      const int idx_a = j * as0, idx_n = j * ns0;
      norm[idx_n] -= ats::conj(a[idx_a]) * a[idx_a];
    }
    return 0;
  }
};

struct SerialQR_WithColumnPivotingInternal {
  template <typename ValueType, typename IntType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int m,  // m = NumRows(A)
                                           const int n,  // n = NumCols(A)
                                           /* */ ValueType *A, const int as0, const int as1,
                                           /* */ ValueType *t, const int ts0,
                                           /* */ IntType *p, const int ps0,
                                           /* */ ValueType *w,
                                           /* */ int &matrix_rank) {
    using value_type = ValueType;
    using int_type   = IntType;
    using ats        = Kokkos::ArithTraits<value_type>;

    /// Given a matrix A, it computes QR decomposition of the matrix
    ///  - t is to store tau and w is for workspace

    // partitions used for loop iteration
    Partition2x2<value_type> A_part2x2(as0, as1);
    Partition3x3<value_type> A_part3x3(as0, as1);

    // column vector of tau (size of min_mn)
    Partition2x1<value_type> t_part2x1(ts0);
    Partition3x1<value_type> t_part3x1(ts0);

    // row vector for norm and p (size of n)
    Partition1x2<int_type> p_part1x2(ps0);
    Partition1x3<int_type> p_part1x3(ps0);

    Partition1x2<value_type> norm_part1x2(1);
    Partition1x3<value_type> norm_part1x3(1);

    // loop size
    const int min_mn = m < n ? m : n;

    // workspace (norm and householder application, 2*max(m,n) is needed)
    value_type *norm = w;
    w += n;

    // initial partition of A where ATL has a zero dimension
    A_part2x2.partWithATL(A, m, n, 0, 0);
    t_part2x1.partWithAT(t, min_mn, 0);

    p_part1x2.partWithAL(p, n, 0);
    norm_part1x2.partWithAL(norm, n, 0);

    // compute initial column norms (replaced by dot product)
    SerialDotInternal::invoke(m, n, A, as0, as1, A, as0, as1, norm, 1);

    const bool finish_when_rank_found = (matrix_rank == -1);

    matrix_rank = min_mn;
    value_type max_diag(0);
    for (int m_atl = 0; m_atl < min_mn; ++m_atl) {
      const int n_AR = n - m_atl;

      // part 2x2 into 3x3
      A_part3x3.partWithABR(A_part2x2, 1, 1);
      const int m_A22 = m - m_atl - 1;
      const int n_A22 = n - m_atl - 1;

      t_part3x1.partWithAB(t_part2x1, 1);
      value_type *tau = t_part3x1.A1;

      p_part1x3.partWithAR(p_part1x2, 1);
      int_type *pividx = p_part1x3.A1;

      norm_part1x3.partWithAR(norm_part1x2, 1);

      /// -----------------------------------------------------
      // find max location
      SerialFindAmaxInternal::invoke(n_AR, norm_part1x2.AR, 1, pividx);

      // apply pivot
      SerialApplyPivotVectorForwardInternal::invoke(*pividx, norm_part1x2.AR, 1);
      SerialApplyPivotMatrixForwardInternal::invoke(m, *pividx, A_part2x2.ATR, as1, as0);

      // perform householder transformation
      SerialApplyLeftHouseholderInternal::invoke(m_A22, A_part3x3.A11, A_part3x3.A21, as0, tau);

      // left apply householder to A22
      SerialApplyLeftHouseholderInternal::invoke(m_A22, n_A22, tau, A_part3x3.A21, as0, A_part3x3.A12, as1,
                                                 A_part3x3.A22, as0, as1, w);

      // break condition
      if (matrix_rank == min_mn) {
        if (m_atl == 0) max_diag = ats::abs(A[0]);
        const value_type val_diag = ats::abs(A_part3x3.A11[0]), threshold(10 * max_diag * ats::epsilon());
        if (val_diag < threshold) {
          matrix_rank = m_atl;
          if (finish_when_rank_found) break;
        }
      }

      // norm update
      SerialUpdateColumnNormsInternal::invoke(n_A22, A_part3x3.A12, as1, norm_part1x3.A2, 1);
      /// -----------------------------------------------------
      A_part2x2.mergeToATL(A_part3x3);
      t_part2x1.mergeToAT(t_part3x1);
      p_part1x2.mergeToAL(p_part1x3);
      norm_part1x2.mergeToAL(norm_part1x3);
    }

    return 0;
  }
};

}  // end namespace KokkosBatched

#endif
