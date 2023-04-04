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

#ifndef KOKKOSBLAS2_SYR_SPEC_HPP_
#define KOKKOSBLAS2_SYR_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosBlas2_syr_impl.hpp>
#endif

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class XMV, class ZMV>
struct syr_eti_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization availability KokkosBlas::Impl::SYR.
// This is NOT for users!!!
// All the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or more .cpp files.
//
#define KOKKOSBLAS2_SYR_ETI_SPEC_AVAIL(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE)      \
  template <>                                                                      \
  struct syr_eti_spec_avail< Kokkos::View< const SCALAR*                           \
                                         , LAYOUT                                  \
                                         , Kokkos::Device<EXEC_SPACE, MEM_SPACE>   \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged> \
                                         >                                         \
                           , Kokkos::View< SCALAR**                                \
                                         , LAYOUT                                  \
                                         , Kokkos::Device<EXEC_SPACE, MEM_SPACE>   \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged> \
                                         >                                         \
                           > {                                                     \
    enum : bool { value = true };                                                  \
  };

// Include the actual specialization declarations
#include <KokkosBlas2_syr_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosBlas2_syr_eti_spec_avail.hpp>

namespace KokkosBlas {
namespace Impl {

//
// syr
//

// Implementation of KokkosBlas::syr.
template < class XViewType
         , class AViewType
         , bool tpl_spec_avail = syr_tpl_spec_avail<XViewType, AViewType>::value
         , bool eti_spec_avail = syr_eti_spec_avail<XViewType, AViewType>::value
         >
struct SYR {
  static void syr( const typename AViewType::execution_space  & space
                 , const          char                          trans[]
                 , const          char                          uplo[]
                 , const typename AViewType::const_value_type & alpha
                 , const          XViewType                   & x
                 , const          AViewType                   & A
                 )
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
  {
    KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Entering KokkosBlas::Impl::Syr::syr()\n" );

    static_assert(Kokkos::is_view<XViewType>::value, "XViewType must be a Kokkos::View.");
    static_assert(Kokkos::is_view<AViewType>::value, "AViewType must be a Kokkos::View.");

    static_assert(static_cast<int>(XViewType::rank) == 1, "XViewType must have rank 1.");
    static_assert(static_cast<int>(AViewType::rank) == 2, "AViewType must have rank 2.");

    if ((trans[0] == 'T') ||
        (trans[0] == 't') ||
        (trans[0] == 'H') ||
        (trans[0] == 'h')) {
      // Ok
    }
    else {
      std::ostringstream oss;
      oss << "In impl of KokkosBlas2::syr(): invalid trans[0] = " << trans[0];
      throw std::runtime_error(oss.str());
    }

    if ((uplo[0] == 'U') ||
        (uplo[0] == 'u') ||
        (uplo[0] == 'L') ||
        (uplo[0] == 'l')) {
      // Ok
    }
    else {
      std::ostringstream oss;
      oss << "In impl of KokkosBlas2::syr(): invalid uplo[0] = " << uplo[0];
      throw std::runtime_error(oss.str());
    }

    if (A.extent(0) != x.extent(0)) {
      std::ostringstream oss;
      oss << "In impl of KokkosBlas2::syr(): A.extent(0) = " << A.extent(0)
	  << ", but x.extent(0) = " << x.extent(0);
      throw std::runtime_error(oss.str());
    }

    if (A.extent(1) != x.extent(0)) {
      std::ostringstream oss;
      oss << "In impl of KokkosBlas2::syr(): A.extent(1) = " << A.extent(1)
	  << ", but x.extent(0) = " << x.extent(0);
      throw std::runtime_error(oss.str());
    }

    Kokkos::Profiling::pushRegion(KOKKOSKERNELS_IMPL_COMPILE_LIBRARY ? "KokkosBlas::syr[ETI]" : "KokkosBlas::syr[noETI]");

    typedef typename AViewType::size_type size_type;
    const size_type numRows = A.extent(0);
    const size_type numCols = A.extent(1);

    // Prefer int as the index type, but use a larsyr type if needed.
    if (( numRows < static_cast<size_type>(INT_MAX) ) &&
        ( numCols < static_cast<size_type>(INT_MAX) )) {
      generalSyrImpl<XViewType, AViewType, int>( space
                                               , trans
                                               , uplo
                                               , alpha
                                               , x
                                               , A
                                               );
    }
    else {
      generalSyrImpl<XViewType, AViewType, int64_t>( space
                                                   , trans
                                                   , uplo
                                                   , alpha
                                                   , x
                                                   , A
                                                   );
    }

    Kokkos::Profiling::popRegion();
  }
#else
  ;
#endif // if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
};

}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization of KokkosBlas::Impl::SYR.
// This is NOT for users!!!
// All the declarations of full specializations go in this header file.
// We may spread out definitions (see _DEF macro below) across one or more .cpp files.
//
#define KOKKOSBLAS2_SYR_ETI_SPEC_DECL(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE)        \
  extern template struct SYR< Kokkos::View< const SCALAR*                           \
                                          , LAYOUT                                  \
                                          , Kokkos::Device<EXEC_SPACE, MEM_SPACE>   \
                                          , Kokkos::MemoryTraits<Kokkos::Unmanaged> \
                                          >                                         \
                            , Kokkos::View< SCALAR**                                \
                                          , LAYOUT                                  \
                                          , Kokkos::Device<EXEC_SPACE, MEM_SPACE>   \
                                          , Kokkos::MemoryTraits<Kokkos::Unmanaged> \
                                          >                                         \
                            , false                                                 \
                            , true                                                  \
                            >;

#define KOKKOSBLAS2_SYR_ETI_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template struct SYR< Kokkos::View< const SCALAR*                           \
                                   , LAYOUT                                  \
                                   , Kokkos::Device<EXEC_SPACE, MEM_SPACE>   \
                                   , Kokkos::MemoryTraits<Kokkos::Unmanaged> \
                                   >                                         \
                     , Kokkos::View< SCALAR**                                \
                                   , LAYOUT                                  \
                                   , Kokkos::Device<EXEC_SPACE, MEM_SPACE>   \
                                   , Kokkos::MemoryTraits<Kokkos::Unmanaged> \
                                   >                                         \
                     , false                                                 \
                     , true                                                  \
                     >;

#include <KokkosBlas2_syr_tpl_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosBlas2_syr_eti_spec_decl.hpp>

#endif // KOKKOSBLAS2_SYR_SPEC_HPP_
