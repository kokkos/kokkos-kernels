// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBATCHED_SYPMV_INTERNAL_HPP_
#define KOKKOSBATCHED_SYPMV_INTERNAL_HPP_
#include <concepts>
#include <Kokkos_Core.hpp>
#include "KokkosBlas1_set_impl.hpp"
#include "KokkosBlas1_serial_scal_impl.hpp"
#include "KokkosBatched_Util.hpp"

namespace KokkosBatched {
namespace Impl {

///
/// Serial Internal Impl
/// ====================

struct SerialSypmvInternal {
  template <typename ArgUplo, typename Op, typename SymOp, typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(Op op, SymOp sym_op, const int n, const ScalarType alpha,
                                           const ValueType *KOKKOS_RESTRICT ap, const int as0,
                                           const ValueType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
                                           ValueType *KOKKOS_RESTRICT y, const int ys0);
};

template <typename ArgUplo, typename Op, typename SymOp, typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int SerialSypmvInternal::invoke(Op op, SymOp sym_op, const int n, const ScalarType alpha,
                                                       const ValueType *KOKKOS_RESTRICT ap, const int as0,
                                                       const ValueType *KOKKOS_RESTRICT x, const int xs0,
                                                       const ScalarType beta, ValueType *KOKKOS_RESTRICT y,
                                                       const int ys0) {
  if (beta == ScalarType(0)) {
    KokkosBlas::Impl::SerialSetInternal::invoke(n, ScalarType(0), y, ys0);
  } else {
    KokkosBlas::Impl::SerialScaleInternal::invoke(n, beta, y, ys0);
  }

  if (alpha != ScalarType(0)) {
    if constexpr (std::same_as<ArgUplo, KokkosBatched::Uplo::Lower>) {
      // Lower triangular specific implementation
      for (int j = 0; j < n; j++) {
        const int offset = j * (2 * n - j + 1) / 2;
        auto temp1       = alpha * x[j * xs0];
        ValueType temp2(0);
        y[j * ys0] += temp1 * sym_op(ap[offset * as0]);
        for (int i = j + 1; i < n; i++) {
          y[i * ys0] += temp1 * ap[(offset + i - j) * as0];
          temp2 += op(ap[(offset + i - j) * as0]) * x[i * xs0];
        }
        y[j * ys0] += alpha * temp2;
      }
    } else {
      // Upper triangular specific implementation
      for (int j = 0; j < n; j++) {
        const int offset = j * (j + 1) / 2;
        auto temp1       = alpha * x[j * xs0];
        ValueType temp2(0);
        for (int i = 0; i < j; i++) {
          y[i * ys0] += temp1 * ap[(offset + i) * as0];
          temp2 += op(ap[(offset + i) * as0]) * x[i * xs0];
        }
        y[j * ys0] += temp1 * sym_op(ap[(offset + j) * as0]) + alpha * temp2;
      }
    }
  }

  return 0;
}

///
/// Team Internal Impl
/// ====================

struct TeamSypmvInternal {
  template <typename ArgUplo, typename MemberType, typename Op, typename SymOp, typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, Op op, SymOp sym_op, const int n,
                                           const ScalarType alpha, const ValueType *KOKKOS_RESTRICT ap, const int as0,
                                           const ValueType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
                                           ValueType *KOKKOS_RESTRICT y, const int ys0);
};

template <typename ArgUplo, typename MemberType, typename Op, typename SymOp, typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int TeamSypmvInternal::invoke(const MemberType &member, Op op, SymOp sym_op, const int n,
                                                     const ScalarType alpha, const ValueType *KOKKOS_RESTRICT ap,
                                                     const int as0, const ValueType *KOKKOS_RESTRICT x, const int xs0,
                                                     const ScalarType beta, ValueType *KOKKOS_RESTRICT y,
                                                     const int ys0) {
  if constexpr (std::same_as<ArgUplo, KokkosBatched::Uplo::Lower>) {
    // Lower triangular specific implementation
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n), [&](const int i) {
      auto index = [&](const int i0, const int i1) { return (i1 * (2 * n - i1 - 1) / 2 + i0) * as0; };
      auto sum   = sym_op(ap[index(i, i)]) * x[i * xs0];
      // Lower triangle:
      // A[i, j] * x[j], j < i
      ValueType sum_lower(0);
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(member, i),
          [&](const int j, ValueType &update) { update += ap[index(i, j)] * x[j * xs0]; },
          sum_lower);  // end of parallel_reduce

      // Hermitian contribution:
      // conj(A[j, i]) * x[j], j > i
      ValueType sum_upper(0);
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(member, (n - i - 1)),
          [&](const int j, ValueType &update) { update += op(ap[index(j + i + 1, i)]) * x[(j + i + 1) * xs0]; },
          sum_upper);

      y[i * ys0] = beta * y[i * ys0] + alpha * (sum + sum_lower + sum_upper);
    });
  } else {
    // Upper triangular specific implementation
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n), [&](const int i) {
      auto index    = [&](const int i0, const int i1) { return (i1 * (i1 + 1) / 2 + i0) * as0; };
      ValueType sum = sym_op(ap[index(i, i)]) * x[i * xs0];
      // Upper triangle:
      // A[i, j] * x[j], j > i
      ValueType sum_upper(0);
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(member, (n - i - 1)),
          [&](const int j, ValueType &update) { update += ap[index(i, j + i + 1)] * x[(j + i + 1) * xs0]; },
          sum_upper);  // end of parallel_reduce

      // Hermitian contribution:
      // conj(A[j, i]) * x[j], j < i
      ValueType sum_lower(0);
      Kokkos::parallel_reduce(
          Kokkos::ThreadVectorRange(member, i),
          [&](const int j, ValueType &update) { update += op(ap[index(j, i)]) * x[j * xs0]; },
          sum_lower);  // end of parallel_reduce

      y[i * ys0] = beta * y[i * ys0] + alpha * (sum + sum_lower + sum_upper);
    });
  }

  return 0;
}

///
/// TeamVector Internal Impl
/// ====================

struct TeamVectorSypmvInternal {
  template <typename ArgUplo, typename MemberType, typename Op, typename SymOp, typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, Op op, SymOp sym_op, const int n,
                                           const ScalarType alpha, const ValueType *KOKKOS_RESTRICT ap, const int as0,
                                           const ValueType *KOKKOS_RESTRICT x, const int xs0, const ScalarType beta,
                                           ValueType *KOKKOS_RESTRICT y, const int ys0);
};

template <typename ArgUplo, typename MemberType, typename Op, typename SymOp, typename ScalarType, typename ValueType>
KOKKOS_INLINE_FUNCTION int TeamVectorSypmvInternal::invoke(const MemberType &member, Op op, SymOp sym_op, const int n,
                                                           const ScalarType alpha, const ValueType *KOKKOS_RESTRICT ap,
                                                           const int as0, const ValueType *KOKKOS_RESTRICT x,
                                                           const int xs0, const ScalarType beta,
                                                           ValueType *KOKKOS_RESTRICT y, const int ys0) {
  if constexpr (std::same_as<ArgUplo, KokkosBatched::Uplo::Lower>) {
    // Lower triangular specific implementation
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n), [&](const int i) {
      auto index = [&](const int i0, const int i1) { return (i1 * (2 * n - i1 - 1) / 2 + i0) * as0; };
      auto sum   = sym_op(ap[index(i, i)]) * x[i * xs0];
      // Lower triangle:
      // A[i, j] * x[j], j < i
      for (int j = 0; j < i; j++) {
        sum += ap[index(i, j)] * x[j * xs0];
      }
      // Hermitian contribution:
      // conj(A[j, i]) * x[j], j > i
      for (int j = i + 1; j < n; j++) {
        sum += op(ap[index(j, i)]) * x[j * xs0];
      }
      y[i * ys0] = beta * y[i * ys0] + alpha * sum;
    });
  } else {
    // Upper triangular specific implementation
    Kokkos::parallel_for(Kokkos::TeamVectorRange(member, n), [&](const int i) {
      auto index = [&](const int i0, const int i1) { return (i1 * (i1 + 1) / 2 + i0) * as0; };
      auto sum   = sym_op(ap[index(i, i)]) * x[i * xs0];
      // Upper triangle:
      // A[i, j] * x[j], j > i
      for (int j = i + 1; j < n; j++) {
        sum += ap[index(i, j)] * x[j * xs0];
      }
      // Hermitian contribution:
      // conj(A[j, i]) * x[j], j < i
      for (int j = 0; j < i; j++) {
        sum += op(ap[index(j, i)]) * x[j * xs0];
      }
      y[i * ys0] = beta * y[i * ys0] + alpha * sum;
    });
  }

  return 0;
}

}  // namespace Impl
}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_SYPMV_INTERNAL_HPP_
