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

#ifndef KOKKOSBLAS2_GER_SPEC_HPP_
#define KOKKOSBLAS2_GER_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>

// Include the actual functors
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosBlas2_ger_impl.hpp>
#endif

namespace KokkosBlas {
namespace Impl {

// Specialization struct which defines whether a specialization exists
template <class XMV, class YMV, class ZMV>
struct ger_eti_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

//
// Macro for declaration of full specialization availability KokkosBlas::Impl::GER.
// This is NOT for users!!!
// All the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or more .cpp files.
//
#define KOKKOSBLAS2_GER_ETI_SPEC_AVAIL(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template <>                                                                 \
  struct ger_eti_spec_avail<                                                  \
      Kokkos::View<const SCALAR*, LAYOUT,                                     \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<const SCALAR*, LAYOUT,                                     \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,		      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<SCALAR**, LAYOUT,                                          \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {              \
    enum : bool { value = true };                                             \
  };

// Include the actual specialization declarations
#include <KokkosBlas2_ger_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosBlas2_ger_eti_spec_avail.hpp>

namespace KokkosBlas {
namespace Impl {

//
// ger
//

// Implementation of KokkosBlas::ger.
template <class XViewType, class YViewType, class AViewType,
          bool tpl_spec_avail = ger_tpl_spec_avail<XViewType, YViewType, AViewType>::value,
          bool eti_spec_avail = ger_eti_spec_avail<XViewType, YViewType, AViewType>::value>
struct GER {
  static void ger( const typename AViewType::execution_space  & space
                 , const typename AViewType::const_value_type & alpha
                 , const          XViewType                   & x
                 , const          YViewType                   & y
                 , const          AViewType                   & A
                 )
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
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

    Kokkos::Profiling::pushRegion(KOKKOSKERNELS_IMPL_COMPILE_LIBRARY ? "KokkosBlas::ger[ETI]" : "KokkosBlas::ger[noETI]");

    typedef typename AViewType::size_type size_type;
    const size_type numRows = A.extent(0);
    const size_type numCols = A.extent(1);

    // Prefer int as the index type, but use a larger type if needed.
    if (numRows < static_cast<size_type>(INT_MAX) &&
        numCols < static_cast<size_type>(INT_MAX)) {
      generalGerImpl<XViewType, YViewType, AViewType, int>(space, alpha, x, y, A);
    }
    else {
      generalGerImpl<XViewType, YViewType, AViewType, int64_t>(space, alpha, x, y, A);
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
// Macro for declaration of full specialization of KokkosBlas::Impl::GER.
// This is NOT for users!!!
// All the declarations of full specializations go in this header file.
// We may spread out definitions (see _DEF macro below) across one or more .cpp files.
//
#define KOKKOSBLAS2_GER_ETI_SPEC_DECL(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  extern template struct GER<                                                \
      Kokkos::View<const SCALAR*, LAYOUT,                                    \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<const SCALAR*, LAYOUT,                                    \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,		     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<SCALAR**, LAYOUT,                                         \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      false, true>;

#define KOKKOSBLAS2_GER_ETI_SPEC_INST(SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE) \
  template struct GER<                                                       \
      Kokkos::View<const SCALAR*, LAYOUT,                                    \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<const SCALAR*, LAYOUT,                                    \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<SCALAR**, LAYOUT,                                         \
                   Kokkos::Device<EXEC_SPACE, MEM_SPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      false, true>;

#include <KokkosBlas2_ger_tpl_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosBlas2_ger_eti_spec_decl.hpp>

#endif // KOKKOSBLAS2_GER_SPEC_HPP_
