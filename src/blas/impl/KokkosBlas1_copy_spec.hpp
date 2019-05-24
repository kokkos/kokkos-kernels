/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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
*/
#ifndef KOKKOS_BLAS1_IMPL_COPY_SPEC_HPP_
#define KOKKOS_BLAS1_IMPL_COPY_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>

// Include the actual functors
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY 
#include <KokkosBlas1_copy_impl.hpp>
#endif

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template<class XMV, class YMV, int rank = YMV::rank>
struct copy_eti_spec_avail {
  enum : bool { value = false };
};
}
}

//
// Macro for declaration of full specialization availability
// KokkosBlas::Impl::Copy for rank == 1.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS1_COPY_ETI_SPEC_AVAIL_LAYOUTS( SCALAR, LAYOUTX, LAYOUTY, EXEC_SPACE, MEM_SPACE ) \
    template<> \
    struct copy_eti_spec_avail< \
        Kokkos::View<const SCALAR*, LAYOUTX, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<SCALAR*, LAYOUTY, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        1> { enum : bool { value = true }; };

#define KOKKOSBLAS1_COPY_ETI_SPEC_AVAIL( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSBLAS1_COPY_ETI_SPEC_AVAIL_LAYOUTS( SCALAR, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_ETI_SPEC_AVAIL_LAYOUTS( SCALAR, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_ETI_SPEC_AVAIL_LAYOUTS( SCALAR, Kokkos::LayoutStride, LAYOUT, EXEC_SPACE, MEM_SPACE)

//
// Macro for declaration of full specialization availability
// KokkosBlas::Impl::Copy for rank == 2.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _DEF macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS1_COPY_MV_ETI_SPEC_AVAIL_LAYOUTS( SCALAR, LAYOUTX, LAYOUTY, EXEC_SPACE, MEM_SPACE ) \
    template<> \
    struct copy_eti_spec_avail< \
        Kokkos::View<const SCALAR**, LAYOUTX, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        Kokkos::View<SCALAR**, LAYOUTY, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
        2> { enum : bool { value = true }; };

#define KOKKOSBLAS1_COPY_MV_ETI_SPEC_AVAIL( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSBLAS1_COPY_MV_ETI_SPEC_AVAIL_LAYOUTS( SCALAR, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_MV_ETI_SPEC_AVAIL_LAYOUTS( SCALAR, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_MV_ETI_SPEC_AVAIL_LAYOUTS( SCALAR, Kokkos::LayoutStride, LAYOUT, EXEC_SPACE, MEM_SPACE)

// Include the actual specialization declarations
#include<KokkosBlas1_copy_tpl_spec_avail.hpp>
#include<generated_specializations_hpp/KokkosBlas1_copy_eti_spec_avail.hpp>
#include<generated_specializations_hpp/KokkosBlas1_copy_mv_eti_spec_avail.hpp>

namespace KokkosBlas {
namespace Impl {

// Unification layer
template<class XMV, class YMV, int rank = YMV::rank,
         bool tpl_spec_avail = copy_tpl_spec_avail<XMV,YMV>::value,
         bool eti_spec_avail = copy_eti_spec_avail<XMV,YMV>::value>
struct Copy {
  static void copy (const XMV& X, const YMV& Y);
};

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
//! Full specialization of Copy for single vectors (1-D Views).
template<class XMV, class YMV>
struct Copy<XMV, YMV, 1, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY>
{
  typedef typename XMV::size_type size_type;

  static void copy (const XMV& X, const YMV& Y)
  {
    static_assert (Kokkos::Impl::is_view<XMV>::value, "KokkosBlas::Impl::"
                   "Copy<1-D>: XMV is not a Kokkos::View.");
    static_assert (Kokkos::Impl::is_view<YMV>::value, "KokkosBlas::Impl::"
                   "Copy<1-D>: YMV is not a Kokkos::View.");
    static_assert (XMV::rank == 1, "KokkosBlas::Impl::Copy<1-D>: "
                   "XMV is not rank 1.");
    static_assert (YMV::rank == 1, "KokkosBlas::Impl::Copy<1-D>: "
                   "YMV is not rank 1.");
    Kokkos::Profiling::pushRegion(KOKKOSKERNELS_IMPL_COMPILE_LIBRARY?"KokkosBlas::copy[ETI]":"KokkosBlas::copy[noETI]");
    #ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
    if(KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
      printf("KokkosBlas1::copy<> ETI specialization for < %s , %s >\n",typeid(XMV).name(),typeid(YMV).name());
    else {
      printf("KokkosBlas1::copy<> non-ETI specialization for < %s , %s >\n",typeid(XMV).name(),typeid(YMV).name());
    }
    #endif
    const size_type numRows = X.extent(0);

    if (numRows < static_cast<size_type> (INT_MAX)) {
      typedef int index_type;
      V_Copy_Generic<XMV, YMV, index_type> (X, Y);
    }
    else {
      typedef std::int64_t index_type;
      V_Copy_Generic<XMV, YMV, index_type> (X, Y);
    }
    Kokkos::Profiling::popRegion();
  }
};
  
template<class XMV, class YMV>
struct Copy<XMV, YMV, 2, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  typedef typename XMV::size_type size_type;

  static void copy (const XMV& X, const YMV& Y)
  {
    static_assert (Kokkos::Impl::is_view<XMV>::value, "KokkosBlas::Impl::"
                   "Copy<2-D>: XMV is not a Kokkos::View.");
    static_assert (Kokkos::Impl::is_view<YMV>::value, "KokkosBlas::Impl::"
                   "Copy<2-D>: YMV is not a Kokkos::View.");
    static_assert (XMV::rank == 2, "KokkosBlas::Impl::Copy<2-D>: "
                   "XMV is not rank 2.");
    static_assert (YMV::rank == 2, "KokkosBlas::Impl::Copy<2-D>: "
                   "YMV is not rank 2.");
    Kokkos::Profiling::pushRegion(KOKKOSKERNELS_IMPL_COMPILE_LIBRARY?"KokkosBlas::copy[ETI]":"KokkosBlas::copy[noETI]");
    #ifdef KOKKOSKERNELS_ENABLE_CHECK_SPECIALIZATION
    if(KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
      printf("KokkosBlas1::copy<> ETI specialization for < %s , %s >\n",typeid(XMV).name(),typeid(YMV).name());
    else {
      printf("KokkosBlas1::copy<> non-ETI specialization for < %s , %s >\n",typeid(XMV).name(),typeid(YMV).name());
    }
    #endif

    const size_type numRows = X.extent(0);
    const size_type numCols = X.extent(1);
    if (numRows < static_cast<size_type> (INT_MAX) &&
        numRows * numCols < static_cast<size_type> (INT_MAX)) {
      typedef int index_type;
      MV_Copy_Generic<XMV, YMV, index_type> (X, Y);
    }
    else {
      typedef std::int64_t index_type;
      MV_Copy_Generic<XMV, YMV, index_type> (X, Y);
    }
    Kokkos::Profiling::popRegion();
  }
};
#endif

}
}

//
// Macro for declaration of full specialization of
// KokkosBlas::Impl::Copy for rank == 1.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _DEF macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS1_COPY_ETI_SPEC_DECL_LAYOUTS( SCALAR, LAYOUTX, LAYOUTY, EXEC_SPACE, MEM_SPACE ) \
extern template struct Copy< \
         Kokkos::View<const SCALAR*, LAYOUTX, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         Kokkos::View<SCALAR*, LAYOUTY, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         1, false, true>;

#define KOKKOSBLAS1_COPY_ETI_SPEC_DECL( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSBLAS1_COPY_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutStride, LAYOUT, EXEC_SPACE, MEM_SPACE)

//
// Macro for definition of full specialization of
// KokkosBlas::Impl::Copy for rank == 1.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define KOKKOSBLAS1_COPY_ETI_SPEC_INST_LAYOUTS( SCALAR, LAYOUTX, LAYOUTY, EXEC_SPACE, MEM_SPACE ) \
template struct Copy< \
         Kokkos::View<const SCALAR*, LAYOUTX, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         Kokkos::View<SCALAR*, LAYOUTY, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         1, false, true>;

#define KOKKOSBLAS1_COPY_ETI_SPEC_INST( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSBLAS1_COPY_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutStride, LAYOUT, EXEC_SPACE, MEM_SPACE)

//
// Macro for declaration of full specialization of
// KokkosBlas::Impl::Copy for rank == 2.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _DEF macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS1_COPY_MV_ETI_SPEC_DECL_LAYOUTS( SCALAR, LAYOUTX, LAYOUTY, EXEC_SPACE, MEM_SPACE ) \
extern template struct Copy< \
         Kokkos::View<const SCALAR**, LAYOUTX, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         Kokkos::View<SCALAR**, LAYOUTY, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         2, false, true>;

#define KOKKOSBLAS1_COPY_MV_ETI_SPEC_DECL( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSBLAS1_COPY_MV_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_MV_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_MV_ETI_SPEC_DECL_LAYOUTS(SCALAR, Kokkos::LayoutStride, LAYOUT, EXEC_SPACE, MEM_SPACE)

//
// Macro for definition of full specialization of
// KokkosBlas::Impl::Copy for rank == 2.  This is NOT for users!!!  We
// use this macro in one or more .cpp files in this directory.
//
#define KOKKOSBLAS1_COPY_MV_ETI_SPEC_INST_LAYOUTS( SCALAR, LAYOUTX, LAYOUTY, EXEC_SPACE, MEM_SPACE ) \
template struct Copy< \
         Kokkos::View<const SCALAR**, LAYOUTX, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         Kokkos::View<SCALAR**, LAYOUTY, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         2, false, true>;

#define KOKKOSBLAS1_COPY_MV_ETI_SPEC_INST( SCALAR, LAYOUT, EXEC_SPACE, MEM_SPACE ) \
    KOKKOSBLAS1_COPY_MV_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutLeft, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_MV_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutRight, LAYOUT, EXEC_SPACE, MEM_SPACE) \
    KOKKOSBLAS1_COPY_MV_ETI_SPEC_INST_LAYOUTS(SCALAR, Kokkos::LayoutStride, LAYOUT, EXEC_SPACE, MEM_SPACE)

#include<KokkosBlas1_copy_tpl_spec_decl.hpp>
#include<generated_specializations_hpp/KokkosBlas1_copy_eti_spec_decl.hpp>
#include<generated_specializations_hpp/KokkosBlas1_copy_mv_eti_spec_decl.hpp>

#endif // KOKKOS_BLAS1_IMPL_COPY_SPEC_HPP_
