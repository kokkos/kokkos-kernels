/*
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
*/
#ifndef KOKKOSBLAS2_GER_IMPL_HPP_
#define KOKKOSBLAS2_GER_IMPL_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include "KokkosKernels_ExecSpaceUtils.hpp"
#include "Kokkos_ArithTraits.hpp"

namespace KokkosBlas {
namespace Impl {

// Functor for a single-level parallel_for version of nontranspose GER.
// The functor parallelizes over rows of the input matrix A.
template <class XViewType, class YViewType, class AViewType,
          class IndexType = typename AViewType::size_type>
struct SingleLevelGER {
  using AlphaCoeffType = typename AViewType::non_const_value_type;
  using A_value_type   = typename AViewType::non_const_value_type;

  SingleLevelGER( const AlphaCoeffType & alpha
                , const XViewType      & x
                , const YViewType      & y
                , const AViewType      & A
                )
      : alpha_(alpha)
      , x_    (x)
      , y_    (y)
      , A_    (A)
  {
    static_assert(Kokkos::is_view<AViewType>::value,
                  "AViewType must be a Kokkos::View.");
    static_assert(Kokkos::is_view<XViewType>::value,
                  "XViewType must be a Kokkos::View.");
    static_assert(Kokkos::is_view<YViewType>::value,
                  "YViewType must be a Kokkos::View.");
    static_assert(static_cast<int>(AViewType::rank) == 2,
                  "AViewType must have rank 2.");
    static_assert(static_cast<int>(XViewType::rank) == 1,
                  "XViewType must have rank 1.");
    static_assert(static_cast<int>(YViewType::rank) == 1,
                  "YViewType must have rank 1.");
    static_assert(std::is_integral<IndexType>::value,
                  "IndexType must be an integer.");
  }

  KOKKOS_INLINE_FUNCTION void operator()(const IndexType & i) const { // EEP
    using AlphaKAT = Kokkos::Details::ArithTraits<typename XViewType::non_const_value_type>;

    if (alpha_ == AlphaKAT::zero()) {
      // Nothing to do
    }
    else {
      const IndexType numCols = A_.extent(1);
      for (IndexType j = 0; j < numCols; ++j) {
	A_(i,j) += A_value_type( alpha_ * x_(i) * y_(j) );
      }
    }
  }

private:
  AlphaCoeffType                 alpha_;
  typename XViewType::const_type x_;
  typename YViewType::const_type y_;
  AViewType                      A_;
};

// Single-level parallel version of GER.
template <class XViewType, class YViewType, class AViewType,
          class IndexType = typename AViewType::size_type>
void singleLevelGer( const typename AViewType::execution_space  & space
                   , const typename AViewType::const_value_type & alpha
                   , const          XViewType                   & x
                   , const          YViewType                   & y
                   , const          AViewType                   & A
                   ) {
  static_assert(Kokkos::is_view<AViewType>::value,
                "AViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<XViewType>::value,
                "XViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<YViewType>::value,
                "YViewType must be a Kokkos::View.");
  static_assert(static_cast<int>(AViewType::rank) == 2,
                "AViewType must have rank 2.");
  static_assert(static_cast<int>(XViewType::rank) == 1,
                "XViewType must have rank 1.");
  static_assert(static_cast<int>(YViewType::rank) == 1,
                "YViewType must have rank 1.");
  static_assert(std::is_integral<IndexType>::value,
                "IndexType must be an integer");

  using AlphaKAT = Kokkos::Details::ArithTraits<typename XViewType::non_const_value_type>;

  if (y.extent(0) == 0) {
    // no entries to update
  }
  else if (x.extent(0) == 0) {
    // no entries to update
  }
  else if (alpha == AlphaKAT::zero()) {
    // no entries to update
  }
  else {
    using execution_space = typename AViewType::execution_space;
    Kokkos::RangePolicy<execution_space, IndexType>            rangePolicy(space, 0, A.extent(0));
    SingleLevelGER<XViewType, YViewType, AViewType, IndexType> functor    (alpha, x, y, A);
    Kokkos::parallel_for("KokkosBlas::ger[SingleLevel]", rangePolicy, functor);
  }
}

struct TwoLevelGER_LayoutLeftTag {};
struct TwoLevelGER_LayoutRightTag {};

// ---------------------------------------------------------------------------------------------

// Functor for a two-level parallel_reduce version of GER, designed for performance on GPU.
// Kernel depends on the layout of A.
template <class XViewType, class YViewType, class AViewType, class IndexType = typename AViewType::size_type>
struct TwoLevelGER {
  using AlphaCoeffType = typename AViewType::non_const_value_type;
  using A_value_type   = typename AViewType::non_const_value_type;

  using execution_space = typename AViewType::execution_space;
  using policy_type     = Kokkos::TeamPolicy<execution_space>;
  using member_type     = typename policy_type::member_type;

  TwoLevelGER( const AlphaCoeffType & alpha
             , const XViewType      & x
             , const YViewType      & y
             , const AViewType      & A
             )
      : alpha_(alpha)
      , x_    (x)
      , y_    (y)
      , A_    (A)
  {
    static_assert(Kokkos::is_view<AViewType>::value,
                  "AViewType must be a Kokkos::View.");
    static_assert(Kokkos::is_view<XViewType>::value,
                  "XViewType must be a Kokkos::View.");
    static_assert(Kokkos::is_view<YViewType>::value,
                  "YViewType must be a Kokkos::View.");
    static_assert(static_cast<int>(AViewType::rank) == 2,
                  "AViewType must have rank 2.");
    static_assert(static_cast<int>(XViewType::rank) == 1,
                  "XViewType must have rank 1.");
    static_assert(static_cast<int>(YViewType::rank) == 1,
                  "YViewType must have rank 1.");
    static_assert(std::is_integral<IndexType>::value,
                  "IndexType must be an integer.");
  }

 public:
  // LayoutLeft version: 32xK blocks.
  //  -Each team handles block rows.
  //  -Groups of 32 threads handle N/teamsize columns sequentially
  KOKKOS_INLINE_FUNCTION void operator()( TwoLevelGER_LayoutLeftTag // EEP
                                        , const member_type & team
                                        ) const {
    // Which block this thread will work on
    int block = team.team_rank() / 32;
    // Which row in the block this thread will work on
    IndexType row           = team.league_rank() * 32 + team.team_rank() % 32;
    IndexType blockColStart = columnsPerThread * block;
    // Compute local sum
    if (row < (IndexType)A_.extent(0)) {
      for (IndexType col = blockColStart; (col < blockColStart + columnsPerThread) && (col < A_.extent(1)); ++col) {
        A_(row, col) += A_value_type( alpha_ * x_(row) * y_(col) );
      }
    }
    team.team_barrier();
  }

  // LayoutRight version: one team per row
  KOKKOS_INLINE_FUNCTION void operator()( TwoLevelGER_LayoutRightTag // EEP
                                        , const member_type & team
                                        ) const {
    using AlphaKAT = Kokkos::Details::ArithTraits<typename XViewType::non_const_value_type>;

    if (alpha_ == AlphaKAT::zero()) {
      // Nothing to do
    }
    else {
      const IndexType i = team.league_rank();
      const IndexType j = team.team_rank();
      A_(i,j) += A_value_type( alpha_ * x_(i) * y_(j) );
    }
  }

  IndexType columnsPerThread;

private:
  AlphaCoeffType                 alpha_;
  typename XViewType::const_type x_;
  typename YViewType::const_type y_;
  AViewType                      A_;
};

// Two-level parallel version of GER.
template <class XViewType, class YViewType, class AViewType,
          class IndexType = typename AViewType::size_type>
void twoLevelGer( const typename AViewType::execution_space  & space
                , const typename AViewType::const_value_type & alpha
                , const          XViewType                   & x
                , const          YViewType                   & y
                , const          AViewType                   & A
                ) {
  static_assert(Kokkos::is_view<AViewType>::value,
                "AViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<XViewType>::value,
                "XViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<YViewType>::value,
                "YViewType must be a Kokkos::View.");
  static_assert(static_cast<int>(AViewType::rank) == 2,
                "AViewType must have rank 2.");
  static_assert(static_cast<int>(XViewType::rank) == 1,
                "XViewType must have rank 1.");
  static_assert(static_cast<int>(YViewType::rank) == 1,
                "YViewType must have rank 1.");
  static_assert(std::is_integral<IndexType>::value,
                "IndexType must be an integer");

  using A_value_type    = typename AViewType::non_const_value_type;
  using execution_space = typename AViewType::execution_space;

  using AlphaKAT = Kokkos::Details::ArithTraits<typename XViewType::non_const_value_type>;

  if (y.extent(0) == 0) {
    // no entries to update
    return;
  }
  else if (x.extent(0) == 0) {
    // no entries to update
    return;
  }
  else if (alpha == AlphaKAT::zero()) {
    // no entries to update
    return;
  }

  constexpr bool isLayoutLeft = std::is_same<typename AViewType::array_layout, Kokkos::LayoutLeft>::value;
  // Both kernels work for both layouts - the only difference is access pattern.
  using layout_tag  = typename std::conditional<isLayoutLeft, TwoLevelGER_LayoutLeftTag, TwoLevelGER_LayoutRightTag>::type;
  using TeamPolicyType = Kokkos::TeamPolicy<execution_space, layout_tag>;

  TwoLevelGER<XViewType, YViewType, AViewType, IndexType> functor(alpha, x, y, A);

  TeamPolicyType teamPolicy;
  if (isLayoutLeft) {
    size_t sharedPerTeam = 32 * sizeof(A_value_type);
    IndexType numTeams   = (A.extent(0) + 31) / 32;

    TeamPolicyType tempPolicy(1, 1);
    tempPolicy.set_scratch_size(0, Kokkos::PerTeam(sharedPerTeam));
    int teamSize = tempPolicy.team_size_recommended(functor, Kokkos::ParallelForTag());
    // make sure teamSize is a multiple of 32
    teamSize -= teamSize % 32;
    // don't make teamSize larger than what's useful
    if ((size_t)teamSize > 32 * A.extent(1)) teamSize = 32 * A.extent(1);

    // FIXME SYCL: team_size_recommended() returns too big of a team size.
    // Kernel hangs with 1024 threads on XEHP.
#ifdef KOKKOS_ENABLE_SYCL
    if (std::is_same<execution_space, Kokkos::Experimental::SYCL>::value) {
      if (teamSize > 256) teamSize = 256;
    }
#endif
    int numBlocks            = teamSize / 32;
    functor.columnsPerThread = (A.extent(1) + numBlocks - 1) / numBlocks;
    teamPolicy               = TeamPolicyType(space, numTeams, teamSize).set_scratch_size(0, Kokkos::PerTeam(sharedPerTeam));
  }
  else {
    // LayoutRight: one team per row
    teamPolicy = TeamPolicyType(space, A.extent(0), Kokkos::AUTO);
  }

  Kokkos::parallel_for("KokkosBlas::ger[twoLevel]", teamPolicy, functor);
}

// ---------------------------------------------------------------------------------------------

// generalGer: use 1 level (Range) or 2 level (Team) implementation,
// depending on whether execution space is CPU or GPU.
// The 'enable_if' makes sure unused kernels are not instantiated.

template < class XViewType
         , class YViewType
         , class AViewType
         , class IndexType
         , typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space< typename AViewType::execution_space>() >::type* = nullptr
         >
void generalGerImpl( const typename AViewType::execution_space  & space
                   , const typename AViewType::const_value_type & alpha
                   , const          XViewType                   & x
                   , const          YViewType                   & y
                   , const          AViewType                   & A
                   ) {
  singleLevelGer(space, alpha, x, y, A);
}

template < class XViewType
         , class YViewType
         , class AViewType
         , class IndexType
         , typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space< typename AViewType::execution_space>()>::type* = nullptr
         >
void generalGerImpl( const typename AViewType::execution_space  & space
                   , const typename AViewType::const_value_type & alpha
                   , const          XViewType                   & x
                   , const          YViewType                   & y
                   , const          AViewType                   & A
                   ) {
  twoLevelGer(space, alpha, x, y, A);
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS2_GER_IMPL_HPP_
