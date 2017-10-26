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
#ifndef KOKKOSBLAS3_GEMM_SPEC_HPP_
#define KOKKOSBLAS3_GEMM_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "Kokkos_InnerProductSpaceTraits.hpp"

//#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
//#include<KokkosBlas3_gemm_impl.hpp>
//#endif

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template<class AVT, class BVT, class CVT>
struct gemm_eti_spec_avail {
  enum : bool { value = false };
};
}
}


//
// Macro for declaration of full specialization availability
// KokkosBlas::Impl::GEMM.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or
// more .cpp files.
//
#define KOKKOSBLAS3_GEMM_ETI_SPEC_AVAIL( SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, EXEC_SPACE, MEM_SPACE ) \
    template<> \
    struct gemm_eti_spec_avail< \
         Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         Kokkos::View<const SCALAR**, LAYOUTB, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
         Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                      Kokkos::MemoryTraits<Kokkos::Unmanaged> > \
         > { enum : bool { value = true }; };


// Include the actual specialization declarations
#include<KokkosBlas3_gemm_tpl_spec_avail.hpp>
// No Native Kokkos Variant available right now
//#include<generated_specializations_hpp/KokkosBlas3_gemm_eti_spec_avail.hpp>

namespace KokkosBlas {
namespace Impl {

//
// gemm
//

// Implementation of KokkosBlas::gemm.
template<class AViewType,
         class BViewType,
         class CViewType,
         bool tpl_spec_avail = gemm_tpl_spec_avail<AViewType, BViewType, CViewType>::value,
         bool eti_spec_avail = gemm_eti_spec_avail<AViewType, BViewType, CViewType>::value
         >
struct GEMM {
  static void
  gemm (const char transA[],
        const char transB[],
        typename AViewType::const_value_type& alpha,
        const AViewType& A,
        const BViewType& B,
        typename CViewType::const_value_type& beta,
        const CViewType& C);
};




} // namespace Impl
} // namespace KokkosBlas


//
// Macro for declaration of full specialization of
// KokkosBlas::Impl::GEMM.  This is NOT for users!!!
// All the declarations of full specializations go in this header
// file.  We may spread out definitions (see _DEF macro below) across
// one or more .cpp files.
//

#define KOKKOSBLAS3_GEMM_ETI_SPEC_DECL( SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, EXEC_SPACE, MEM_SPACE ) \
extern template struct GEMM< \
     Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const SCALAR**, LAYOUTB, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     false, true>;

#define KOKKOSBLAS3_GEMM_ETI_SPEC_INST( SCALAR, LAYOUTA, LAYOUTB, LAYOUTC, EXEC_SPACE, MEM_SPACE ) \
template struct GEMM< \
     Kokkos::View<const SCALAR**, LAYOUTA, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<const SCALAR**, LAYOUTB, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >, \
     Kokkos::View<SCALAR**, LAYOUTC, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged> >,  \
     false, true>;


#include<KokkosBlas3_gemm_tpl_spec_decl.hpp>
// No native Kokkos variant available right now
//#include<generated_specializations_hpp/KokkosBlas3_gemm_eti_spec_decl.hpp>

#endif // KOKKOSBLAS3_GEMM_SPEC_HPP_
