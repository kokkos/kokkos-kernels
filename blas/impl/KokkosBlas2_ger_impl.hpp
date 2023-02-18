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
  using y_value_type   = typename YViewType::non_const_value_type;
  using AccumScalar    = typename std::conditional< std::is_same<y_value_type, Kokkos::Experimental::half_t>::value || std::is_same<y_value_type, Kokkos::Experimental::bhalf_t>::value
                                                  , float
                                                  , y_value_type
                                                  >::type;

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
    if ((i == 0)) {
      printf("Aqui 001\n");
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

  using Kokkos::Details::ArithTraits;
  using AlphaKAT = ArithTraits<typename XViewType::non_const_value_type>;

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
    using policy_type = Kokkos::RangePolicy<execution_space, IndexType>;
    policy_type range(space, 0, A.extent(0));

    using functor_type = SingleLevelGER<XViewType, YViewType, AViewType, IndexType>;
    functor_type functor(alpha, x, y, A);
    Kokkos::parallel_for("KokkosBlas::ger[SingleLevel]", range, functor);
  }
}

struct TwoLevelGER_LayoutLeftTag {};
struct TwoLevelGER_LayoutRightTag {};

// ---------------------------------------------------------------------------------------------

// Functor for a two-level parallel_reduce version of GER, designed for performance on GPU.
// Kernel depends on the layout of A.
template <class XViewType, class YViewType, class AViewType, class IndexType = typename AViewType::size_type>
struct TwoLevelGER {
  using y_value_type   = typename YViewType::non_const_value_type;
  using AlphaCoeffType = typename AViewType::non_const_value_type;
  using BetaCoeffType  = typename YViewType::non_const_value_type;
  using AccumScalar    = typename std::conditional< std::is_same<y_value_type, Kokkos::Experimental::half_t>::value || std::is_same<y_value_type, Kokkos::Experimental::bhalf_t>::value
                                                  , float
                                                  , y_value_type
                                                  >::type;

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
  //  -Groups of 32 threads handle N/teamsize columns sequentially, placing results into shared.
  //  -Then individual thread results are combined with parallel_reduce.
  KOKKOS_INLINE_FUNCTION void operator()( TwoLevelGER_LayoutLeftTag
                                        , const member_type & team
                                        ) const {
    // Allocate a Scalar in shared for each thread
    AccumScalar* blockResult = (AccumScalar*)team.team_shmem().get_shmem(32 * sizeof(AccumScalar));
  }

  // LayoutRight version: one team per row
  KOKKOS_INLINE_FUNCTION void operator()( TwoLevelGER_LayoutRightTag
                                        , const member_type & team
                                        ) const {
    const int i       = team.league_rank();  // batch id
    if (i == 0) {
      printf("Aqui 002\n");
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

  using y_value_type      = typename YViewType::non_const_value_type;
  using execution_space   = typename AViewType::execution_space;
  //using team_policy_type  = Kokkos::TeamPolicy<execution_space>;
  //using range_policy_type = Kokkos::RangePolicy<execution_space, IndexType>;

  using Kokkos::Details::ArithTraits;
  //using KAT      = ArithTraits<typename AViewType::non_const_value_type>;
  using AlphaKAT = ArithTraits<typename XViewType::non_const_value_type>;

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
    using layout_tag = typename std::conditional<isLayoutLeft, TwoLevelGER_LayoutLeftTag, TwoLevelGER_LayoutRightTag>::type;
    using tagged_policy = Kokkos::TeamPolicy<execution_space, layout_tag>;
    using functor_type  = TwoLevelGER<XViewType, YViewType, AViewType, IndexType>;
    functor_type functor(alpha, x, y, A);
    tagged_policy team;
    if (isLayoutLeft) {
      using AccumScalar = typename std::conditional< std::is_same<y_value_type, Kokkos::Experimental::half_t>::value || std::is_same<y_value_type, Kokkos::Experimental::bhalf_t>::value
                                                   , float
                                                   , y_value_type
                                                   >::type;
      size_t sharedPerTeam = 32 * sizeof(AccumScalar);
      IndexType numTeams   = (A.extent(0) + 31) / 32;
      tagged_policy temp(1, 1);
      temp.set_scratch_size(0, Kokkos::PerTeam(sharedPerTeam));
      int teamSize = temp.team_size_recommended(functor, Kokkos::ParallelForTag());
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
      team                     = tagged_policy(space, numTeams, teamSize).set_scratch_size(0, Kokkos::PerTeam(sharedPerTeam));
    }
    else {
      // LayoutRight: one team per row
      team = tagged_policy(space, A.extent(0), Kokkos::AUTO);
    }
    Kokkos::parallel_for("KokkosBlas::ger[twoLevel]", team, functor);
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
