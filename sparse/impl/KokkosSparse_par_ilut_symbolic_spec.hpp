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
#ifndef KOKKOSSPARSE_IMPL_PAR_ILUT_SYMBOLIC_SPEC_HPP_
#define KOKKOSSPARSE_IMPL_PAR_ILUT_SYMBOLIC_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_Handle.hpp"

// Include the actual functors
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosSparse_par_ilut_symbolic_impl.hpp>
#endif

namespace KokkosSparse {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class KernelHandle, class ARowMapType, class AEntriesType,
          class LRowMapType, class URowMapType>
struct par_ilut_symbolic_eti_spec_avail {
  enum : bool { value = false };
};

}  // namespace Impl
}  // namespace KokkosSparse

#define KOKKOSSPARSE_PAR_ILUT_SYMBOLIC_ETI_SPEC_AVAIL(                         \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE,      \
    MEM_SPACE_TYPE)                                                            \
  template <>                                                                  \
  struct par_ilut_symbolic_eti_spec_avail<                                     \
      KokkosKernels::Experimental::KokkosKernelsHandle<                        \
          const OFFSET_TYPE, const ORDINAL_TYPE, const SCALAR_TYPE,            \
          EXEC_SPACE_TYPE, MEM_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::View<                                                            \
          const OFFSET_TYPE *, LAYOUT_TYPE,                                    \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                     \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >,    \
      Kokkos::View<                                                            \
          const ORDINAL_TYPE *, LAYOUT_TYPE,                                   \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                     \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >,    \
      Kokkos::View<                                                            \
          OFFSET_TYPE *, LAYOUT_TYPE,                                          \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                     \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >,    \
      Kokkos::View<                                                            \
          OFFSET_TYPE *, LAYOUT_TYPE,                                          \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                     \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> > > { \
    enum : bool { value = true };                                              \
  };

// Include the actual specialization declarations
#include <KokkosSparse_par_ilut_symbolic_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosSparse_par_ilut_symbolic_eti_spec_avail.hpp>

namespace KokkosSparse {
namespace Impl {

// Unification layer
/// \brief Implementation of KokkosSparse::par_ilut_symbolic

template <class KernelHandle, class ARowMapType, class AEntriesType,
          class LRowMapType, class URowMapType,
          bool tpl_spec_avail = par_ilut_symbolic_tpl_spec_avail<
              KernelHandle, ARowMapType, AEntriesType, LRowMapType,
              URowMapType>::value,
          bool eti_spec_avail = par_ilut_symbolic_eti_spec_avail<
              KernelHandle, ARowMapType, AEntriesType, LRowMapType,
              URowMapType>::value>
struct PAR_ILUT_SYMBOLIC {
  static void par_ilut_symbolic(KernelHandle *handle,
                                const ARowMapType &A_row_map,
                                const AEntriesType &A_entries,
                                LRowMapType &L_row_map, URowMapType &U_row_map);
};

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
//! Full specialization of par_ilut_symbolic
// Unification layer
template <class KernelHandle, class ARowMapType, class AEntriesType,
          class LRowMapType, class URowMapType>
struct PAR_ILUT_SYMBOLIC<KernelHandle, ARowMapType, AEntriesType, LRowMapType,
                         URowMapType, false,
                         KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  static void par_ilut_symbolic(KernelHandle *handle,
                                const ARowMapType &A_row_map,
                                const AEntriesType &A_entries,
                                LRowMapType &L_row_map,
                                URowMapType &U_row_map) {
    auto par_ilut_handle = handle->get_par_ilut_handle();

    Experimental::ilut_symbolic(*par_ilut_handle, A_row_map, A_entries,
                                L_row_map, U_row_map);
    par_ilut_handle->set_symbolic_complete();
  }
};
#endif
}  // namespace Impl
}  // namespace KokkosSparse

//
// Macro for declaration of full specialization of
// This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _DEF macro below) across one or
// more .cpp files.
//
#define KOKKOSSPARSE_PAR_ILUT_SYMBOLIC_ETI_SPEC_DECL(                       \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE,   \
    MEM_SPACE_TYPE)                                                         \
  extern template struct PAR_ILUT_SYMBOLIC<                                 \
      KokkosKernels::Experimental::KokkosKernelsHandle<                     \
          const OFFSET_TYPE, const ORDINAL_TYPE, const SCALAR_TYPE,         \
          EXEC_SPACE_TYPE, MEM_SPACE_TYPE, MEM_SPACE_TYPE>,                 \
      Kokkos::View<                                                         \
          const OFFSET_TYPE *, LAYOUT_TYPE,                                 \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      Kokkos::View<                                                         \
          const ORDINAL_TYPE *, LAYOUT_TYPE,                                \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      Kokkos::View<                                                         \
          OFFSET_TYPE *, LAYOUT_TYPE,                                       \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      Kokkos::View<                                                         \
          OFFSET_TYPE *, LAYOUT_TYPE,                                       \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      false, true>;

#define KOKKOSSPARSE_PAR_ILUT_SYMBOLIC_ETI_SPEC_INST(                       \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE,   \
    MEM_SPACE_TYPE)                                                         \
  template struct PAR_ILUT_SYMBOLIC<                                        \
      KokkosKernels::Experimental::KokkosKernelsHandle<                     \
          const OFFSET_TYPE, const ORDINAL_TYPE, const SCALAR_TYPE,         \
          EXEC_SPACE_TYPE, MEM_SPACE_TYPE, MEM_SPACE_TYPE>,                 \
      Kokkos::View<                                                         \
          const OFFSET_TYPE *, LAYOUT_TYPE,                                 \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      Kokkos::View<                                                         \
          const ORDINAL_TYPE *, LAYOUT_TYPE,                                \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      Kokkos::View<                                                         \
          OFFSET_TYPE *, LAYOUT_TYPE,                                       \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      Kokkos::View<                                                         \
          OFFSET_TYPE *, LAYOUT_TYPE,                                       \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      false, true>;

#include <KokkosSparse_par_ilut_symbolic_tpl_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosSparse_par_ilut_symbolic_eti_spec_decl.hpp>

#endif
