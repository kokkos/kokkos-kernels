<<<<<<< HEAD
=======
/*
>>>>>>> d868047b (Renamed files)
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
<<<<<<< HEAD

#ifndef KOKKOSBLAS2_GER_IMPL_HPP_
#define KOKKOSBLAS2_GER_IMPL_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"
#include "Kokkos_ArithTraits.hpp"
=======
*/
#ifndef KOKKOSBLAS1_GER_IMPL_HPP_
#define KOKKOSBLAS1_GER_IMPL_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
>>>>>>> d868047b (Renamed files)

namespace KokkosBlas {
namespace Impl {

<<<<<<< HEAD
// Functor for a single-level parallel_for version of nontranspose GER.
// The functor parallelizes over rows of the input matrix A.
template <class XViewType, class YViewType, class AViewType,
          class IndexType = typename AViewType::size_type>
struct SingleLevelGER {
  using AlphaCoeffType = typename AViewType::non_const_value_type;
  using A_value_type   = typename AViewType::non_const_value_type;

  SingleLevelGER( const bool             justTranspose
                , const AlphaCoeffType & alpha
                , const XViewType      & x
                , const YViewType      & y
                , const AViewType      & A
                )
      : justTranspose_(justTranspose)
      , alpha_        (alpha)
      , x_            (x)
      , y_            (y)
      , A_            (A)
  {
    static_assert(Kokkos::is_view<XViewType>::value, "XViewType must be a Kokkos::View.");
    static_assert(Kokkos::is_view<YViewType>::value, "YViewType must be a Kokkos::View.");
    static_assert(Kokkos::is_view<AViewType>::value, "AViewType must be a Kokkos::View.");

    static_assert(static_cast<int>(XViewType::rank) == 1, "XViewType must have rank 1.");
    static_assert(static_cast<int>(YViewType::rank) == 1, "YViewType must have rank 1.");
    static_assert(static_cast<int>(AViewType::rank) == 2, "AViewType must have rank 2.");

    static_assert(std::is_integral<IndexType>::value, "IndexType must be an integer.");
  }

  KOKKOS_INLINE_FUNCTION void operator()(const IndexType & i) const {
    using KAT = Kokkos::Details::ArithTraits<typename AViewType::non_const_value_type>;

    if (alpha_ == KAT::zero()) {
      // Nothing to do
    }
    else {
      const IndexType    N      ( A_.extent(1) );
      const A_value_type x_fixed( x_(i) );

      if (justTranspose_) {
        for (IndexType j = 0; j < N; ++j) {
          A_(i,j) += A_value_type( alpha_ * x_fixed * y_(j) );
        }
      }
      else {
        for (IndexType j = 0; j < N; ++j) {
          A_(i,j) += A_value_type( alpha_ * x_fixed * KAT::conj( y_(j) ) );
        }
      }
    }
  }

private:
  bool                           justTranspose_;
  AlphaCoeffType                 alpha_;
  typename XViewType::const_type x_;
  typename YViewType::const_type y_;
  AViewType                      A_;
};

// Single-level parallel version of GER.
template <class XViewType, class YViewType, class AViewType,
          class IndexType = typename AViewType::size_type>
void singleLevelGer( const typename AViewType::execution_space  & space
                   , const          char                          trans[]
                   , const typename AViewType::const_value_type & alpha
                   , const          XViewType                   & x
                   , const          YViewType                   & y
                   , const          AViewType                   & A
                   ) {
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Entering IMPL singleLevelGer(), AViewType = %s\n", typeid(AViewType).name() );
  static_assert(Kokkos::is_view<XViewType>::value, "XViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<YViewType>::value, "YViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<AViewType>::value, "AViewType must be a Kokkos::View.");

  static_assert(static_cast<int>(XViewType::rank) == 1, "XViewType must have rank 1.");
  static_assert(static_cast<int>(YViewType::rank) == 1, "YViewType must have rank 1.");
  static_assert(static_cast<int>(AViewType::rank) == 2, "AViewType must have rank 2.");

  static_assert(std::is_integral<IndexType>::value, "IndexType must be an integer");

  using KAT = Kokkos::Details::ArithTraits<typename AViewType::non_const_value_type>;

  if (y.extent(0) == 0) {
    // no entries to update
  }
  else if (x.extent(0) == 0) {
    // no entries to update
  }
  else if (alpha == KAT::zero()) {
    // no entries to update
  }
  else {
    using execution_space = typename AViewType::execution_space;
    Kokkos::RangePolicy<execution_space, IndexType>            rangePolicy(space, 0, A.extent(0));
    SingleLevelGER<XViewType, YViewType, AViewType, IndexType> functor( (trans[0] == 'T') || (trans[0] == 't')
                                                                      , alpha
                                                                      , x
                                                                      , y
                                                                      , A
                                                                      );
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

  TwoLevelGER( const bool             justTranspose
             , const AlphaCoeffType & alpha
             , const XViewType      & x
             , const YViewType      & y
             , const AViewType      & A
             )
      : justTranspose_(justTranspose)
      , alpha_        (alpha)
      , x_            (x)
      , y_            (y)
      , A_            (A)
  {
    static_assert(Kokkos::is_view<XViewType>::value, "XViewType must be a Kokkos::View.");
    static_assert(Kokkos::is_view<YViewType>::value, "YViewType must be a Kokkos::View.");
    static_assert(Kokkos::is_view<AViewType>::value, "AViewType must be a Kokkos::View.");

    static_assert(static_cast<int>(XViewType::rank) == 1, "XViewType must have rank 1.");
    static_assert(static_cast<int>(YViewType::rank) == 1, "YViewType must have rank 1.");
    static_assert(static_cast<int>(AViewType::rank) == 2, "AViewType must have rank 2.");

    static_assert(std::is_integral<IndexType>::value, "IndexType must be an integer.");
  }

public:
  // LayoutLeft version: one team per column
  KOKKOS_INLINE_FUNCTION void operator()( TwoLevelGER_LayoutLeftTag
                                        , const member_type & team
                                        ) const {
    using KAT = Kokkos::Details::ArithTraits<typename AViewType::non_const_value_type>;

    if (alpha_ == KAT::zero()) {
      // Nothing to do
    }
    else {
      const IndexType M ( A_.extent(0) );
      const IndexType j ( team.league_rank() );
      if (justTranspose_) {
        const A_value_type y_fixed( y_(j) );
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, M), [&](const IndexType & i) {
          A_(i,j) += A_value_type( alpha_ * x_(i) * y_fixed );
        });
      }
      else {
        const A_value_type y_fixed( KAT::conj( y_(j) ) );
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, M), [&](const IndexType & i) {
          A_(i,j) += A_value_type( alpha_ * x_(i) * y_fixed );
        });
      }
    }
  }

  // LayoutRight version: one team per row
  KOKKOS_INLINE_FUNCTION void operator()( TwoLevelGER_LayoutRightTag
                                        , const member_type & team
                                        ) const {
    using KAT = Kokkos::Details::ArithTraits<typename AViewType::non_const_value_type>;

    if (alpha_ == KAT::zero()) {
      // Nothing to do
    }
    else {
      const IndexType    N      ( A_.extent(1) );
      const IndexType    i      ( team.league_rank() );
      const A_value_type x_fixed( x_(i) );
      if (justTranspose_) {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const IndexType & j) {
          A_(i,j) += A_value_type( alpha_ * x_fixed * y_(j) );
        });
      }
      else {
        Kokkos::parallel_for(Kokkos::TeamThreadRange(team, N), [&](const IndexType & j) {
          A_(i,j) += A_value_type( alpha_ * x_fixed * KAT::conj( y_(j) ) );
        });
      }
    }
    team.team_barrier();
  }

private:
  bool                           justTranspose_;
  AlphaCoeffType                 alpha_;
  typename XViewType::const_type x_;
  typename YViewType::const_type y_;
  AViewType                      A_;
};

// Two-level parallel version of GER.
template <class XViewType, class YViewType, class AViewType,
          class IndexType = typename AViewType::size_type>
void twoLevelGer( const typename AViewType::execution_space  & space
                , const          char                          trans[]
                , const typename AViewType::const_value_type & alpha
                , const          XViewType                   & x
                , const          YViewType                   & y
                , const          AViewType                   & A
                ) {
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Entering IMPL twoLevelGer(), AViewType = %s\n", typeid(AViewType).name() );
  static_assert(Kokkos::is_view<XViewType>::value, "XViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<YViewType>::value, "YViewType must be a Kokkos::View.");
  static_assert(Kokkos::is_view<AViewType>::value, "AViewType must be a Kokkos::View.");

  static_assert(static_cast<int>(XViewType::rank) == 1, "XViewType must have rank 1.");
  static_assert(static_cast<int>(YViewType::rank) == 1, "YViewType must have rank 1.");
  static_assert(static_cast<int>(AViewType::rank) == 2, "AViewType must have rank 2.");

  static_assert(std::is_integral<IndexType>::value, "IndexType must be an integer");

  using KAT = Kokkos::Details::ArithTraits<typename AViewType::non_const_value_type>;

  if (y.extent(0) == 0) {
    // no entries to update
    return;
  }
  else if (x.extent(0) == 0) {
    // no entries to update
    return;
  }
  else if (alpha == KAT::zero()) {
    // no entries to update
    return;
  }

  using execution_space = typename AViewType::execution_space;
  constexpr bool isLayoutLeft = std::is_same<typename AViewType::array_layout, Kokkos::LayoutLeft>::value;
  using layout_tag  = typename std::conditional<isLayoutLeft, TwoLevelGER_LayoutLeftTag, TwoLevelGER_LayoutRightTag>::type;
  using TeamPolicyType = Kokkos::TeamPolicy<execution_space, layout_tag>;
  TeamPolicyType teamPolicy;
  if (isLayoutLeft) {
    // LayoutLeft: one team per column
    teamPolicy = TeamPolicyType(space, A.extent(1), Kokkos::AUTO);
  }
  else {
    // LayoutRight: one team per row
    teamPolicy = TeamPolicyType(space, A.extent(0), Kokkos::AUTO);
  }

  TwoLevelGER<XViewType, YViewType, AViewType, IndexType> functor( (trans[0] == 'T') || (trans[0] == 't')
                                                                 , alpha
                                                                 , x
                                                                 , y
                                                                 , A
                                                                 );
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
                   , const          char                          trans[]
                   , const typename AViewType::const_value_type & alpha
                   , const          XViewType                   & x
                   , const          YViewType                   & y
                   , const          AViewType                   & A
                   ) {
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Entering IMPL generalGerImpl(CPU), AViewType = %s\n", typeid(AViewType).name() );
  singleLevelGer(space, trans, alpha, x, y, A);
}

template < class XViewType
         , class YViewType
         , class AViewType
         , class IndexType
         , typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space< typename AViewType::execution_space>()>::type* = nullptr
         >
void generalGerImpl( const typename AViewType::execution_space  & space
                   , const          char                          trans[]
                   , const typename AViewType::const_value_type & alpha
                   , const          XViewType                   & x
                   , const          YViewType                   & y
                   , const          AViewType                   & A
                   ) {
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Entering IMPL generalGerImpl(GPU), AViewType = %s\n", typeid(AViewType).name() );
  twoLevelGer(space, trans, alpha, x, y, A);
}
=======
  // EEP
>>>>>>> d868047b (Renamed files)

}  // namespace Impl
}  // namespace KokkosBlas

<<<<<<< HEAD
#endif  // KOKKOSBLAS2_GER_IMPL_HPP_
=======
#endif  // KOKKOSBLAS1_GER_IMPL_HPP_
>>>>>>> d868047b (Renamed files)
