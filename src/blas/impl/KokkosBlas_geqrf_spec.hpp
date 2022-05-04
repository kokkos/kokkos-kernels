/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
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
#ifndef KOKKOSBLAS_GEQRF_SPEC_HPP_
#define KOKKOSBLAS_GEQRF_SPEC_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include "KokkosBlas_geqrf_impl.hpp"
#endif

namespace KokkosBlas {
namespace Impl {

template <class AVT, class TVT, class WVT>
struct geqrf_eti_spec_avail {
  enum : bool { value = false };
};

template <class AVT, class TVT>
struct geqrf_workspace_eti_spec_avail {
  enum : bool { value = false };
};

}  // namespace Impl
}  // namespace KokkosBlas

#define KOKKOSBLAS_GEQRF_ETI_SPEC_AVAIL(SCALAR_TYPE, LAYOUT_TYPE,        \
                                        EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
  template <>                                                            \
  struct geqrf_eti_spec_avail<                                           \
      Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE,                           \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,            \
      Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE,                            \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,            \
      Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE,                            \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {         \
    enum : bool { value = true };                                        \
  };

#define KOKKOSBLAS_GEQRF_WORKSPACE_ETI_SPEC_AVAIL(                  \
    SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE)      \
  template <>                                                       \
  struct geqrf_workspace_eti_spec_avail<                            \
      Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE,                      \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,       \
      Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE,                       \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {    \
    enum : bool { value = true };                                   \
  };

#include <KokkosBlas_geqrf_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosBlas_geqrf_eti_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosBlas_geqrf_workspace_eti_spec_avail.hpp>

namespace KokkosBlas {
namespace Impl {
// Unification Layer

template <class AVT, class TVT, class WVT,
          bool tpl_spec_avail = geqrf_tpl_spec_avail<AVT, TVT, WVT>::value,
          bool eti_spec_avail = geqrf_eti_spec_avail<AVT, TVT, WVT>::value>
struct GEQRF {
  static void geqrf(AVT& A, TVT& tau, WVT& workspace);
};

template <class AVT, class TVT,
          bool tpl_spec_avail = geqrf_workspace_tpl_spec_avail<AVT, TVT>::value,
          bool eti_spec_avail = geqrf_workspace_eti_spec_avail<AVT, TVT>::value>
struct GEQRF_WORKSPACE {
  static int64_t geqrf_workspace(AVT& A, TVT& tau);
};

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
// specialization layer for no TPL
template <class AVT, class TVT, class WVT>
struct GEQRF<AVT, TVT, WVT, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  static void geqrf(AVT& A, TVT& tau, WVT& workspace) {
    execute_geqrf<AVT, TVT, WVT>(A, tau, workspace);
  }
};
#endif

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
// specialization layer for no TPL
template <class AVT, class TVT>
struct GEQRF_WORKSPACE<AVT, TVT, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  static int64_t geqrf_workspace(AVT& A, TVT& tau) {
    return execute_geqrf_workspace<AVT, TVT>(A, tau);
  }
};
#endif

}  // namespace Impl
}  // namespace KokkosBlas

#define KOKKOSBLAS_GEQRF_ETI_SPEC_DECL(SCALAR_TYPE, LAYOUT_TYPE,        \
                                       EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
  extern template struct GEQRF<                                         \
      Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE,                          \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,           \
      Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE,                           \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,           \
      Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE,                           \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,           \
      false, true>;

#define KOKKOSBLAS_GEQRF_WORKSPACE_ETI_SPEC_DECL(                   \
    SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE)      \
  extern template struct GEQRF_WORKSPACE<                           \
      Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE,                      \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,       \
      Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE,                       \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,       \
      false, true>;

#define KOKKOSBLAS_GEQRF_ETI_SPEC_INST(SCALAR_TYPE, LAYOUT_TYPE,        \
                                       EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
  template struct GEQRF<                                                \
      Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE,                          \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,           \
      Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE,                           \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,           \
      Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE,                           \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,           \
      false, true>;

#define KOKKOSBLAS_GEQRF_WORKSPACE_ETI_SPEC_INST(                   \
    SCALAR_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, MEM_SPACE_TYPE)      \
  template struct GEQRF_WORKSPACE<                                  \
      Kokkos::View<SCALAR_TYPE**, LAYOUT_TYPE,                      \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,       \
      Kokkos::View<SCALAR_TYPE*, LAYOUT_TYPE,                       \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,       \
      false, true>;

#include <KokkosBlas_geqrf_tpl_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosBlas_geqrf_eti_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosBlas_geqrf_workspace_eti_spec_decl.hpp>

#endif  // KOKKOSBLAS_IMPL_GEQRF_HPP_
