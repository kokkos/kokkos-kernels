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

#ifndef KOKKOSBLAS2_SYR_IMPL_HPP_
#define KOKKOSBLAS2_SYR_IMPL_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"
#include "Kokkos_ArithTraits.hpp"

namespace KokkosBlas {
namespace Impl {

// Functor for a single-level parallel_for version of nontranspose SYR.
// The functor parallelizes over rows of the input matrix A.
template <class XViewType, class AViewType, class IndexType>
struct SingleLevelSYR {
  using AlphaCoeffType = typename AViewType::non_const_value_type;
  using XComponentType = typename XViewType::non_const_value_type;
  using AComponentType = typename AViewType::non_const_value_type;

  SingleLevelSYR(const bool justTranspose, const bool justUp,
                 const AlphaCoeffType& alpha, const XViewType& x,
                 const AViewType& A)
      : justTranspose_(justTranspose),
        justUp_(justUp),
        alpha_(alpha),
        x_(x),
        A_(A) {
    // Nothing to do
  }

  KOKKOS_INLINE_FUNCTION void operator()(const IndexType& i) const {
    if (alpha_ == Kokkos::ArithTraits<AlphaCoeffType>::zero()) {
      // Nothing to do
    } else if (x_(i) == Kokkos::ArithTraits<XComponentType>::zero()) {
      // Nothing to do
    } else {
      const XComponentType x_fixed(x_(i));
      const IndexType N(A_.extent(1));

      if (justTranspose_) {
        for (IndexType j = 0; j < N; ++j) {
          if (((justUp_ == true) && (i <= j)) ||
              ((justUp_ == false) && (i >= j))) {
            A_(i, j) += AComponentType(alpha_ * x_fixed * x_(j));
          }
        }
      } else {
        for (IndexType j = 0; j < N; ++j) {
          if (((justUp_ == true) && (i <= j)) ||
              ((justUp_ == false) && (i >= j))) {
            A_(i, j) += AComponentType(
                alpha_ * x_fixed *
                Kokkos::ArithTraits<XComponentType>::conj(x_(j)));
          }
        }
      }
    }
  }

 private:
  bool justTranspose_;
  bool justUp_;
  AlphaCoeffType alpha_;
  typename XViewType::const_type x_;
  AViewType A_;
};

// Single-level parallel version of SYR.
template <class ExecutionSpace, class XViewType, class AViewType,
          class IndexType = typename AViewType::size_type>
void singleLevelSyr(const ExecutionSpace& space, const char trans[],
                    const char uplo[],
                    const typename AViewType::const_value_type& alpha,
                    const XViewType& x, const AViewType& A) {
  static_assert(std::is_integral<IndexType>::value,
                "IndexType must be an integer");

  using AlphaCoeffType = typename AViewType::non_const_value_type;

  if (x.extent(0) == 0) {
    // no entries to update
  } else if (alpha == Kokkos::ArithTraits<AlphaCoeffType>::zero()) {
    // no entries to update
  } else {
    Kokkos::RangePolicy<ExecutionSpace, IndexType> rangePolicy(space, 0,
                                                               A.extent(0));
    SingleLevelSYR<XViewType, AViewType, IndexType> functor(
        (trans[0] == 'T') || (trans[0] == 't'),
        (uplo[0] == 'U') || (uplo[0] == 'u'), alpha, x, A);
    Kokkos::parallel_for("KokkosBlas::syr[SingleLevel]", rangePolicy, functor);
  }
}

struct TwoLevelSYR_LayoutLeftTag {};
struct TwoLevelSYR_LayoutRightTag {};

// ---------------------------------------------------------------------------------------------

// Functor for a two-level parallel_reduce version of SYR, designed for
// performance on GPU. Kernel depends on the layout of A.
template <class ExecutionSpace, class XViewType, class AViewType,
          class IndexType>
struct TwoLevelSYR {
  using AlphaCoeffType = typename AViewType::non_const_value_type;
  using XComponentType = typename XViewType::non_const_value_type;
  using AComponentType = typename AViewType::non_const_value_type;

  using policy_type = Kokkos::TeamPolicy<ExecutionSpace>;
  using member_type = typename policy_type::member_type;

  TwoLevelSYR(const bool justTranspose, const bool justUp,
              const AlphaCoeffType& alpha, const XViewType& x,
              const AViewType& A)
      : justTranspose_(justTranspose),
        justUp_(justUp),
        alpha_(alpha),
        x_(x),
        A_(A) {
    // Nothing to do
  }

 public:
  // LayoutLeft version: one team per column
  KOKKOS_INLINE_FUNCTION void operator()(TwoLevelSYR_LayoutLeftTag,
                                         const member_type& team) const {
    if (alpha_ == Kokkos::ArithTraits<AlphaCoeffType>::zero()) {
      // Nothing to do
    } else {
      const IndexType j(team.league_rank());
      if (x_(j) == Kokkos::ArithTraits<XComponentType>::zero()) {
        // Nothing to do
      } else {
        const IndexType M(A_.extent(0));
        if (justTranspose_) {
          const XComponentType x_fixed(x_(j));
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, M), [&](const IndexType& i) {
                if (((justUp_ == true) && (i <= j)) ||
                    ((justUp_ == false) && (i >= j))) {
                  A_(i, j) += AComponentType(alpha_ * x_(i) * x_fixed);
                }
              });
        } else {
          const XComponentType x_fixed(
              Kokkos::ArithTraits<XComponentType>::conj(x_(j)));
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, M), [&](const IndexType& i) {
                if (((justUp_ == true) && (i <= j)) ||
                    ((justUp_ == false) && (i >= j))) {
                  A_(i, j) += AComponentType(alpha_ * x_(i) * x_fixed);
                }
              });
        }
      }
    }
  }

  // LayoutRight version: one team per row
  KOKKOS_INLINE_FUNCTION void operator()(TwoLevelSYR_LayoutRightTag,
                                         const member_type& team) const {
    if (alpha_ == Kokkos::ArithTraits<AlphaCoeffType>::zero()) {
      // Nothing to do
    } else {
      const IndexType i(team.league_rank());
      if (x_(i) == Kokkos::ArithTraits<XComponentType>::zero()) {
        // Nothing to do
      } else {
        const IndexType N(A_.extent(1));
        const XComponentType x_fixed(x_(i));
        if (justTranspose_) {
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, N), [&](const IndexType& j) {
                if (((justUp_ == true) && (i <= j)) ||
                    ((justUp_ == false) && (i >= j))) {
                  A_(i, j) += AComponentType(alpha_ * x_fixed * x_(j));
                }
              });
        } else {
          Kokkos::parallel_for(
              Kokkos::TeamThreadRange(team, N), [&](const IndexType& j) {
                if (((justUp_ == true) && (i <= j)) ||
                    ((justUp_ == false) && (i >= j))) {
                  A_(i, j) += AComponentType(
                      alpha_ * x_fixed *
                      Kokkos::ArithTraits<XComponentType>::conj(x_(j)));
                }
              });
        }
      }
    }
    team.team_barrier();
  }

 private:
  bool justTranspose_;
  bool justUp_;
  AlphaCoeffType alpha_;
  typename XViewType::const_type x_;
  AViewType A_;
};

// Two-level parallel version of SYR.
template <class ExecutionSpace, class XViewType, class AViewType,
          class IndexType = typename AViewType::size_type>
void twoLevelSyr(const ExecutionSpace& space, const char trans[],
                 const char uplo[],
                 const typename AViewType::const_value_type& alpha,
                 const XViewType& x, const AViewType& A) {
  static_assert(std::is_integral<IndexType>::value,
                "IndexType must be an integer");

  using AlphaCoeffType = typename AViewType::non_const_value_type;

  if (x.extent(0) == 0) {
    // no entries to update
    return;
  } else if (alpha == Kokkos::ArithTraits<AlphaCoeffType>::zero()) {
    // no entries to update
    return;
  }

  constexpr bool isLayoutLeft =
      std::is_same<typename AViewType::array_layout, Kokkos::LayoutLeft>::value;
  using layout_tag =
      typename std::conditional<isLayoutLeft, TwoLevelSYR_LayoutLeftTag,
                                TwoLevelSYR_LayoutRightTag>::type;
  using TeamPolicyType = Kokkos::TeamPolicy<ExecutionSpace, layout_tag>;
  TeamPolicyType teamPolicy;
  if (isLayoutLeft) {
    // LayoutLeft: one team per column
    teamPolicy = TeamPolicyType(space, A.extent(1), Kokkos::AUTO);
  } else {
    // LayoutRight: one team per row
    teamPolicy = TeamPolicyType(space, A.extent(0), Kokkos::AUTO);
  }

  TwoLevelSYR<ExecutionSpace, XViewType, AViewType, IndexType> functor(
      (trans[0] == 'T') || (trans[0] == 't'),
      (uplo[0] == 'U') || (uplo[0] == 'u'), alpha, x, A);
  Kokkos::parallel_for("KokkosBlas::syr[twoLevel]", teamPolicy, functor);
}

// ---------------------------------------------------------------------------------------------

// generalSyr: use 1 level (Range) or 2 level (Team) implementation,
// depending on whether execution space is CPU or GPU.
// The 'enable_if' makes sure unused kernels are not instantiated.

template <class ExecutionSpace, class XViewType, class AViewType,
          class IndexType,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              ExecutionSpace>()>::type* = nullptr>
void generalSyrImpl(const ExecutionSpace& space, const char trans[],
                    const char uplo[],
                    const typename AViewType::const_value_type& alpha,
                    const XViewType& x, const AViewType& A) {
  singleLevelSyr(space, trans, uplo, alpha, x, A);
}

template <class ExecutionSpace, class XViewType, class AViewType,
          class IndexType,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              ExecutionSpace>()>::type* = nullptr>
void generalSyrImpl(const ExecutionSpace& space, const char trans[],
                    const char uplo[],
                    const typename AViewType::const_value_type& alpha,
                    const XViewType& x, const AViewType& A) {
  twoLevelSyr(space, trans, uplo, alpha, x, A);
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS2_SYR_IMPL_HPP_
