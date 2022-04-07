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
#ifndef __KOKKOSBATCHED_GESV_IMPL_HPP__
#define __KOKKOSBATCHED_GESV_IMPL_HPP__

/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include <KokkosBatched_LU_Decl.hpp>
#include "KokkosBatched_Trsm_Decl.hpp"

namespace KokkosBatched {

struct SerialStaticPivoting {
  template <class MatrixType1, class MatrixType2, class VectorType1,
            class VectorType2>
  KOKKOS_INLINE_FUNCTION static void invoke(
      const MatrixType1 A, const MatrixType2 PDAD, const VectorType1 Y,
      const VectorType2 PDY, const VectorType2 D2, const VectorType2 tmp_v_1,
      const VectorType2 tmp_v_2);
};

template <typename MemberType>
struct TeamStaticPivoting {
  template <class MatrixType1, class MatrixType2, class VectorType1,
            class VectorType2>
  KOKKOS_INLINE_FUNCTION static void invoke(
      const MemberType &member, const MatrixType1 A, const MatrixType2 PDAD,
      const VectorType1 Y, const VectorType2 PDY, const VectorType2 D2,
      const VectorType2 tmp_v_1, const VectorType2 tmp_v_2);
};

template <typename MemberType>
struct TeamVectorStaticPivoting {
  template <class MatrixType1, class MatrixType2, class VectorType1,
            class VectorType2>
  KOKKOS_INLINE_FUNCTION static void invoke(
      const MemberType &member, const MatrixType1 A, const MatrixType2 PDAD,
      const VectorType1 Y, const VectorType2 PDY, const VectorType2 D2,
      const VectorType2 tmp_v_1, const VectorType2 tmp_v_2);
};

template <class MatrixType1, class MatrixType2, class VectorType1,
          class VectorType2>
KOKKOS_INLINE_FUNCTION void SerialStaticPivoting::invoke(
    const MatrixType1 A, const MatrixType2 PDAD, const VectorType1 Y,
    const VectorType2 PDY, const VectorType2 D2, const VectorType2 tmp_v_1,
    const VectorType2 tmp_v_2) {
  using value_type = typename MatrixType1::non_const_value_type;
  const int n      = A.extent(0);

  for (int i = 0; i < n; ++i) {
    D2(i)      = 0.;
    tmp_v_1(i) = 0;
    tmp_v_2(i) = 1.;
    for (int j = 0; j < n; ++j) {
      if (D2(i) < Kokkos::abs(A(j, i))) D2(i) = Kokkos::abs(A(j, i));
      if (tmp_v_1(i) < Kokkos::abs(A(i, j))) tmp_v_1(i) = Kokkos::abs(A(i, j));
    }
    D2(i) = 1. / D2(i);
  }

  for (int i = 0; i < n; ++i) {
    for (int j = 0; j < n; ++j) {
      A(i, j) *= D2(j);
    }
  }

  for (int i = 0; i < n; ++i) {
    value_type D1_i = 0.;
    for (int j = 0; j < n; ++j) {
      if (D1_i < Kokkos::abs(A(i, j))) D1_i = Kokkos::abs(A(i, j));
    }
    D1_i = 1. / D1_i;
    for (int j = 0; j < n; ++j) {
      A(i, j) *= D1_i;
    }
    Y(i) *= D1_i;
  }

  for (int i = 0; i < n; ++i) {
    int row_index    = 0;
    int col_index    = 0;
    value_type tmp_0 = 0.;
    value_type tmp_1 = 0.;
    for (int j = 0; j < n; ++j) {
      if (tmp_0 < tmp_v_1(j)) {
        tmp_0     = tmp_v_1(j);
        row_index = j;
      }
    }
    for (int j = 0; j < n; ++j) {
      if (tmp_1 < Kokkos::abs(A(row_index, j) * tmp_v_2(j))) {
        tmp_1     = Kokkos::abs(A(row_index, j) * tmp_v_2(j));
        col_index = j;
      }
    }
    tmp_v_1(row_index) = 0.;
    tmp_v_2(col_index) = 0.;

    for (int j = 0; j < n; ++j) {
      PDAD(col_index, j) = A(row_index, j);
    }
    PDY(col_index) = Y(row_index);
  }
}

template <typename MemberType>
template <class MatrixType1, class MatrixType2, class VectorType1,
          class VectorType2>
KOKKOS_INLINE_FUNCTION void TeamStaticPivoting<MemberType>::invoke(
    const MemberType &member, const MatrixType1 A, const MatrixType2 PDAD,
    const VectorType1 Y, const VectorType2 PDY, const VectorType2 D2,
    const VectorType2 tmp_v_1, const VectorType2 tmp_v_2) {
  using value_type = typename MatrixType1::non_const_value_type;
  using reducer_value_type =
      typename Kokkos::MaxLoc<value_type, int>::value_type;
  // Made this non-const in order to WORKAROUND issue #349 (Credit to C. Trott)
  int n = A.extent(0);

  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n), [&](const int &i) {
    D2(i)      = 0.;
    tmp_v_1(i) = 0;
    tmp_v_2(i) = 1.;
    for (int j = 0; j < n; ++j) {
      if (D2(i) < Kokkos::abs(A(j, i))) D2(i) = Kokkos::abs(A(j, i));
      if (tmp_v_1(i) < Kokkos::abs(A(i, j))) tmp_v_1(i) = Kokkos::abs(A(i, j));
    }
    D2(i) = 1. / D2(i);
  });

  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n), [&](const int &i) {
    for (int j = 0; j < n; ++j) {
      A(i, j) *= D2(j);
    }
  });

  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n), [&](const int &i) {
    value_type D1_i = 0.;
    for (int j = 0; j < n; ++j) {
      if (D1_i < Kokkos::abs(A(i, j))) D1_i = Kokkos::abs(A(i, j));
    }
    D1_i = 1. / D1_i;
    for (int j = 0; j < n; ++j) {
      A(i, j) *= D1_i;
    }
    Y(i) *= D1_i;
  });

  for (int i = 0; i < n; ++i) {
    int row_index, col_index;
    reducer_value_type value;
    Kokkos::MaxLoc<value_type, int> reducer_value(value);
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, n),
        [&](const int &j, reducer_value_type &update) {
          if (tmp_v_1(j) > update.val) {
            update.val = tmp_v_1(j);
            update.loc = j;
          }
        },
        reducer_value);
    row_index = value.loc;
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, n),
        [&](const int &j, reducer_value_type &update) {
          if (Kokkos::abs(A(row_index, j) * tmp_v_2(j)) > update.val) {
            update.val = Kokkos::abs(A(row_index, j) * tmp_v_2(j));
            update.loc = j;
          }
        },
        reducer_value);
    col_index          = value.loc;
    tmp_v_1(row_index) = 0.;
    tmp_v_2(col_index) = 0.;

    for (int j = 0; j < n; ++j) {
      PDAD(col_index, j) = A(row_index, j);
    }
    PDY(col_index) = Y(row_index);
  }
}

template <typename MemberType>
template <class MatrixType1, class MatrixType2, class VectorType1,
          class VectorType2>
KOKKOS_INLINE_FUNCTION void TeamVectorStaticPivoting<MemberType>::invoke(
    const MemberType &member, const MatrixType1 A, const MatrixType2 PDAD,
    const VectorType1 Y, const VectorType2 PDY, const VectorType2 D2,
    const VectorType2 tmp_v_1, const VectorType2 tmp_v_2) {
  using value_type = typename MatrixType1::non_const_value_type;
  using reducer_value_type =
      typename Kokkos::MaxLoc<value_type, int>::value_type;
  const int n = A.extent(0);

  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n), [&](const int &i) {
    D2(i)      = 0.;
    tmp_v_1(i) = 0;
    tmp_v_2(i) = 1.;
    reducer_value_type value;
    Kokkos::MaxLoc<value_type, int> reducer_value(value);
    Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(member, n),
        [&](const int &j, reducer_value_type &update) {
          if (Kokkos::abs(A(j, i)) > update.val) {
            update.val = Kokkos::abs(A(j, i));
            update.loc = j;
          }
        },
        reducer_value);
    D2(i) = 1. / value.val;
    Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(member, n),
        [&](const int &j, reducer_value_type &update) {
          if (Kokkos::abs(A(i, j)) > update.val) {
            update.val = Kokkos::abs(A(i, j));
            update.loc = j;
          }
        },
        reducer_value);
    tmp_v_1(i) = value.val;
  });

  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n), [&](const int &i) {
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                         [&](const int &j) { A(i, j) *= D2(j); });
  });

  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n), [&](const int &i) {
    value_type D1_i = 0.;
    reducer_value_type value;
    Kokkos::MaxLoc<value_type, int> reducer_value(value);
    Kokkos::parallel_reduce(
        Kokkos::ThreadVectorRange(member, n),
        [&](const int &j, reducer_value_type &update) {
          if (Kokkos::abs(A(i, j)) > update.val) {
            update.val = Kokkos::abs(A(i, j));
            update.loc = j;
          }
        },
        reducer_value);
    D1_i = 1. / value.val;
    Kokkos::parallel_for(Kokkos::ThreadVectorRange(member, n),
                         [&](const int &j) { A(i, j) *= D1_i; });
    Y(i) *= D1_i;
  });

  for (int i = 0; i < n; ++i) {
    int row_index, col_index;
    reducer_value_type value;
    Kokkos::MaxLoc<value_type, int> reducer_value(value);
    Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, n),
        [&](const int &j, reducer_value_type &update) {
          if (tmp_v_1(j) > update.val) {
            update.val = tmp_v_1(j);
            update.loc = j;
          }
        },
        reducer_value);
    row_index = value.loc;
    Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, n),
        [&](const int &j, reducer_value_type &update) {
          if (Kokkos::abs(A(row_index, j) * tmp_v_2(j)) > update.val) {
            update.val = Kokkos::abs(A(row_index, j) * tmp_v_2(j));
            update.loc = j;
          }
        },
        reducer_value);
    col_index          = value.loc;
    tmp_v_1(row_index) = 0.;
    tmp_v_2(col_index) = 0.;

    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n), [&](const int &j) {
      PDAD(col_index, j) = A(row_index, j);
    });
    PDY(col_index) = Y(row_index);
  }
}

template <class VectorType1, class VectorType2, class VectorType3>
KOKKOS_INLINE_FUNCTION void SerialHadamard1D(const VectorType1 X,
                                             const VectorType2 D,
                                             const VectorType3 DX) {
  const int n = X.extent(0);

  for (int i = 0; i < n; ++i) {
    DX(i) = D(i) * X(i);
  }
}

template <typename MemberType, class VectorType1, class VectorType2,
          class VectorType3>
KOKKOS_INLINE_FUNCTION void TeamHadamard1D(const MemberType &member,
                                           const VectorType1 X,
                                           const VectorType2 D,
                                           const VectorType3 DX) {
  const int n = X.extent(0);

  Kokkos::parallel_for(Kokkos::TeamThreadRange(member, n),
                       [&](const int &i) { DX(i) = D(i) * X(i); });
}

template <typename MemberType, class VectorType1, class VectorType2,
          class VectorType3>
KOKKOS_INLINE_FUNCTION void TeamVectorHadamard1D(const MemberType &member,
                                                 const VectorType1 X,
                                                 const VectorType2 D,
                                                 const VectorType3 DX) {
  const int n = X.extent(0);

  Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n),
                       [&](const int &i) { DX(i) = D(i) * X(i); });
}

///
/// Serial Impl
/// ===========
template <typename MatrixType, typename VectorType>
KOKKOS_INLINE_FUNCTION int SerialGesv::invoke(const MatrixType A,
                                              const VectorType X,
                                              const VectorType Y,
                                              const MatrixType tmp) {
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  static_assert(Kokkos::is_view<MatrixType>::value,
                "KokkosBatched::gesv: MatrixType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<VectorType>::value,
                "KokkosBatched::gesv: VectorType is not a Kokkos::View.");
  static_assert(MatrixType::Rank == 2,
                "KokkosBatched::gesv: MatrixType must have rank 2.");
  static_assert(VectorType::Rank == 1,
                "KokkosBatched::gesv: VectorType must have rank 1.");

  // Check compatibility of dimensions at run time.

  if (A.extent(0) != tmp.extent(0) || A.extent(1) + 4 != tmp.extent(1)) {
    KOKKOS_IMPL_DO_NOT_USE_PRINTF(
        "KokkosBatched::gesv: dimensions of A and tmp do not match: A: "
        "%d x %d, tmp (note: its second dimension should be the second "
        "dimension of A + 4): %d x %d\n",
        (int)A.extent(0), (int)A.extent(1), (int)tmp.extent(0),
        (int)tmp.extent(1));
    return 1;
  }

  if (A.extent(0) != X.extent(0) || A.extent(1) != X.extent(0) ||
      A.extent(0) != Y.extent(0)) {
    KOKKOS_IMPL_DO_NOT_USE_PRINTF(
        "KokkosBatched::gesv: dimensions of A and X and Y do not match: A: "
        "%d x %d, X: %d, Y: %d\n",
        (int)A.extent(0), (int)A.extent(1), (int)X.extent(0), (int)Y.extent(0));
    return 1;
  }
#endif

  const int n = A.extent(0);

  auto PDAD    = Kokkos::subview(tmp, Kokkos::ALL, Kokkos::make_pair(0, n));
  auto PDY     = Kokkos::subview(tmp, Kokkos::ALL, n);
  auto D2      = Kokkos::subview(tmp, Kokkos::ALL, n + 1);
  auto tmp_v_1 = Kokkos::subview(tmp, Kokkos::ALL, n + 2);
  auto tmp_v_2 = Kokkos::subview(tmp, Kokkos::ALL, n + 3);

  SerialStaticPivoting::invoke(A, PDAD, Y, PDY, D2, tmp_v_1, tmp_v_2);

  SerialLU<Algo::Level3::Unblocked>::invoke(PDAD);

  SerialTrsm<Side::Left, Uplo::Lower, Trans::NoTranspose, Diag::Unit,
             Algo::Level3::Unblocked>::invoke(1.0, PDAD, PDY);

  SerialTrsm<Side::Left, Uplo::Upper, Trans::NoTranspose, Diag::NonUnit,
             Algo::Level3::Unblocked>::invoke(1.0, PDAD, PDY);

  SerialHadamard1D(PDY, D2, X);
  return 0;
}

///
/// Team Impl
/// =========

template <typename MemberType>
template <typename MatrixType, typename VectorType>
KOKKOS_INLINE_FUNCTION int TeamGesv<MemberType>::invoke(
    const MemberType &member, const MatrixType A, const VectorType X,
    const VectorType Y) {
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  static_assert(Kokkos::is_view<MatrixType>::value,
                "KokkosBatched::gesv: MatrixType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<VectorType>::value,
                "KokkosBatched::gesv: VectorType is not a Kokkos::View.");
  static_assert(MatrixType::Rank == 2,
                "KokkosBatched::gesv: MatrixType must have rank 2.");
  static_assert(VectorType::Rank == 1,
                "KokkosBatched::gesv: VectorType must have rank 1.");

  // Check compatibility of dimensions at run time.
  if (A.extent(0) != X.extent(0) || A.extent(1) != X.extent(0) ||
      A.extent(0) != Y.extent(0)) {
    KOKKOS_IMPL_DO_NOT_USE_PRINTF(
        "KokkosBatched::gesv: dimensions of A and X and Y do not match: A: "
        "%d x %d, X: %d, Y: %d\n",
        (int)A.extent(0), (int)A.extent(1), (int)X.extent(0), (int)Y.extent(0));
    return 1;
  }
#endif
  using ScratchPadMatrixViewType =
      Kokkos::View<typename MatrixType::non_const_value_type **,
                   typename MatrixType::array_layout,
                   typename MatrixType::execution_space::scratch_memory_space>;

  const int n = A.extent(0);

  ScratchPadMatrixViewType tmp(member.team_scratch(0), n, n + 4);
  auto PDAD    = Kokkos::subview(tmp, Kokkos::ALL, Kokkos::make_pair(0, n));
  auto PDY     = Kokkos::subview(tmp, Kokkos::ALL, n);
  auto D2      = Kokkos::subview(tmp, Kokkos::ALL, n + 1);
  auto tmp_v_1 = Kokkos::subview(tmp, Kokkos::ALL, n + 2);
  auto tmp_v_2 = Kokkos::subview(tmp, Kokkos::ALL, n + 3);

  TeamStaticPivoting<MemberType>::invoke(member, A, PDAD, Y, PDY, D2, tmp_v_1,
                                         tmp_v_2);
  member.team_barrier();

  TeamLU<MemberType, Algo::Level3::Unblocked>::invoke(member, PDAD);
  member.team_barrier();

  TeamTrsm<MemberType, Side::Left, Uplo::Lower, Trans::NoTranspose, Diag::Unit,
           Algo::Level3::Unblocked>::invoke(member, 1.0, PDAD, PDY);
  member.team_barrier();

  TeamTrsm<MemberType, Side::Left, Uplo::Upper, Trans::NoTranspose,
           Diag::NonUnit, Algo::Level3::Unblocked>::invoke(member, 1.0, PDAD,
                                                           PDY);
  member.team_barrier();

  TeamHadamard1D(member, PDY, D2, X);
  member.team_barrier();
  return 0;
}

///
/// TeamVector Impl
/// =========

template <typename MemberType>
template <typename MatrixType, typename VectorType>
KOKKOS_INLINE_FUNCTION int TeamVectorGesv<MemberType>::invoke(
    const MemberType &member, const MatrixType A, const VectorType X,
    const VectorType Y) {
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  static_assert(Kokkos::is_view<MatrixType>::value,
                "KokkosBatched::gesv: MatrixType is not a Kokkos::View.");
  static_assert(Kokkos::is_view<VectorType>::value,
                "KokkosBatched::gesv: VectorType is not a Kokkos::View.");
  static_assert(MatrixType::Rank == 2,
                "KokkosBatched::gesv: MatrixType must have rank 2.");
  static_assert(VectorType::Rank == 1,
                "KokkosBatched::gesv: VectorType must have rank 1.");

  // Check compatibility of dimensions at run time.
  if (A.extent(0) != X.extent(0) || A.extent(1) != X.extent(0) ||
      A.extent(0) != Y.extent(0)) {
    KOKKOS_IMPL_DO_NOT_USE_PRINTF(
        "KokkosBatched::gesv: dimensions of A and X and Y do not match: A: "
        "%d x %d, X: %d, Y: %d\n",
        (int)A.extent(0), (int)A.extent(1), (int)X.extent(0), (int)Y.extent(0));
    return 1;
  }
#endif
  using ScratchPadMatrixViewType =
      Kokkos::View<typename MatrixType::non_const_value_type **,
                   typename MatrixType::array_layout,
                   typename MatrixType::execution_space::scratch_memory_space>;

  const int n = A.extent(0);

  ScratchPadMatrixViewType tmp(member.team_scratch(0), n, n + 4);
  auto PDAD    = Kokkos::subview(tmp, Kokkos::ALL, Kokkos::make_pair(0, n));
  auto PDY     = Kokkos::subview(tmp, Kokkos::ALL, n);
  auto D2      = Kokkos::subview(tmp, Kokkos::ALL, n + 1);
  auto tmp_v_1 = Kokkos::subview(tmp, Kokkos::ALL, n + 2);
  auto tmp_v_2 = Kokkos::subview(tmp, Kokkos::ALL, n + 3);

  TeamVectorStaticPivoting<MemberType>::invoke(member, A, PDAD, Y, PDY, D2,
                                               tmp_v_1, tmp_v_2);
  member.team_barrier();

  TeamLU<MemberType, Algo::Level3::Unblocked>::invoke(member, PDAD);
  member.team_barrier();

  TeamTrsm<MemberType, Side::Left, Uplo::Lower, Trans::NoTranspose, Diag::Unit,
           Algo::Level3::Unblocked>::invoke(member, 1.0, PDAD, PDY);
  member.team_barrier();

  TeamTrsm<MemberType, Side::Left, Uplo::Upper, Trans::NoTranspose,
           Diag::NonUnit, Algo::Level3::Unblocked>::invoke(member, 1.0, PDAD,
                                                           PDY);
  member.team_barrier();

  TeamVectorHadamard1D(member, PDY, D2, X);
  member.team_barrier();
  return 0;
}

}  // namespace KokkosBatched

#endif
