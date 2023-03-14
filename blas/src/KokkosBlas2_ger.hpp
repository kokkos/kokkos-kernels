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

#ifndef KOKKOSBLAS2_GER_HPP_
#define KOKKOSBLAS2_GER_HPP_

#include <KokkosBlas2_ger_spec.hpp>

namespace KokkosBlas {

/// \brief Rank-1 update of a general matrix: A = A + alpha * x * y^{T,H}.
///
/// \tparam XViewType Input vector, as a 1-D Kokkos::View
/// \tparam YViewType Input vector, as a 1-D Kokkos::View
/// \tparam AViewType Input/Output matrix, as a 2-D Kokkos::View
///
/// \param space [in]     Execution space instance on which to run the kernel.
///                       This may contain information about which stream to
///                       run on.
/// \param trans [in]     "T" or "t" for transpose, "H" or "h" for Hermitian.
///                       Only the first character is taken into account.
/// \param alpha [in]     Input coefficient of x * y^{T,H}
/// \param x     [in]     Input vector, as a 1-D Kokkos::View
/// \param y     [in]     Input vector, as a 1-D Kokkos::View
/// \param A     [in/out] Output matrix, as a nonconst 2-D Kokkos::View
template <class XViewType, class YViewType, class AViewType>
void ger( const typename AViewType::execution_space  & space
        , const          char                          trans[]
        , const typename AViewType::const_value_type & alpha
        , const          XViewType                   & x
        , const          YViewType                   & y
        , const          AViewType                   & A
        ) {
  KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Entering SRC KokkosBlas::ger(), AViewType = %s\n", typeid(AViewType).name() );

  static_assert( Kokkos::is_view<AViewType>::value, "AViewType must be a Kokkos::View." );
  static_assert( Kokkos::is_view<XViewType>::value, "XViewType must be a Kokkos::View." );
  static_assert( Kokkos::is_view<YViewType>::value, "YViewType must be a Kokkos::View." );

  static_assert( static_cast<int>(AViewType::rank) == 2, "AViewType must have rank 2." );
  static_assert( static_cast<int>(XViewType::rank) == 1, "XViewType must have rank 1." );
  static_assert( static_cast<int>(YViewType::rank) == 1, "YViewType must have rank 1." );

  // Check compatibility of dimensions at run time.
  if (( A.extent(0) != x.extent(0) ) ||
      ( A.extent(1) != y.extent(0) )) {
    std::ostringstream os;
    os << "KokkosBlas::ger: Dimensions of A, x, and y do not match: "
       << "A is " << A.extent(0) << " by " << A.extent(1)
       << ", x has size " << x.extent(0)
       << ", y has size " << y.extent(0);
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  using ALayout = typename AViewType::array_layout;

  // Minimize the number of Impl::GER instantiations, by standardizing 
  // on particular View specializations for its template parameters.
  typedef Kokkos::View< typename XViewType::const_value_type*
                      , typename KokkosKernels::Impl::GetUnifiedLayoutPreferring<XViewType, ALayout>::array_layout
                      , typename XViewType::device_type
                      , Kokkos::MemoryTraits<Kokkos::Unmanaged>
                      > XVT;

  typedef Kokkos::View< typename YViewType::const_value_type*
                      , typename KokkosKernels::Impl::GetUnifiedLayoutPreferring<YViewType, ALayout>::array_layout
                      , typename YViewType::device_type
                      , Kokkos::MemoryTraits<Kokkos::Unmanaged>
                      > YVT;

  typedef Kokkos::View< typename AViewType::non_const_value_type**
                      , ALayout
                      , typename AViewType::device_type
                      , Kokkos::MemoryTraits<Kokkos::Unmanaged>
                      > AVT;

  if (( A.extent(0) == 0 ) ||
      ( A.extent(1) == 0 )) {
    // For degenerate cases, use fallback implementation to avoid potential
    // (unlikely?) circular dependence issues by including other KokkosBlas
    // headers.
    const bool eti_spec_avail = KokkosBlas::Impl::ger_eti_spec_avail<XVT, YVT, AVT>::value;
    Impl::GER<XVT, YVT, AVT, false, eti_spec_avail>::ger( space
                                                        , trans
                                                        , alpha
                                                        , x
                                                        , y
                                                        , A
                                                        );
  }
  else {
    Impl::GER<XVT, YVT, AVT>::ger( space
                                 , trans
                                 , alpha
                                 , x
                                 , y
                                 , A
                                 );
  }
}

/// \brief Rank-1 update of a general matrix: A = A + alpha * x * y^{T,H}.
///
/// \tparam XViewType Input vector, as a 1-D Kokkos::View
/// \tparam YViewType Input vector, as a 1-D Kokkos::View
/// \tparam AViewType Input/Output matrix, as a 2-D Kokkos::View
///
/// \param trans [in]     "T" or "t" for transpose, "H" or "h" for Hermitian.
///                       Only the first character is taken into account.
/// \param alpha [in]     Input coefficient of x * y^{T,H}
/// \param x     [in]     Input vector, as a 1-D Kokkos::View
/// \param y     [in]     Input vector, as a 1-D Kokkos::View
/// \param A     [in/out] Output matrix, as a nonconst 2-D Kokkos::View
template <class XViewType, class YViewType, class AViewType>
void ger( const          char                          trans[]
        , const typename AViewType::const_value_type & alpha
        , const          XViewType                   & x
        , const          YViewType                   & y
        , const          AViewType                   & A
        ) {
  const typename AViewType::execution_space space = typename AViewType::execution_space();
  ger( space
     , trans
     , alpha
     , x
     , y
     , A
     );
}

} // namespace KokkosBlas

#endif // KOKKOSBLAS2_GER_HPP_
