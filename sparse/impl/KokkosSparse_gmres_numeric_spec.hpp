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
#ifndef KOKKOSSPARSE_IMPL_GMRES_NUMERIC_SPEC_HPP_
#define KOKKOSSPARSE_IMPL_GMRES_NUMERIC_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosKernels_Handle.hpp"

// Include the actual functors
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosSparse_gmres_numeric_impl.hpp>
#endif

namespace KokkosSparse {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class KernelHandle, class AT, class AO, class AD, class AM, class AS, class BType, class XType>
struct gmres_numeric_eti_spec_avail {
  enum : bool { value = false };
};

}  // namespace Impl
}  // namespace KokkosSparse

#define KOKKOSSPARSE_GMRES_NUMERIC_ETI_SPEC_AVAIL(                       \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE,   \
    MEM_SPACE_TYPE)                                                         \
  template <>                                                               \
  struct gmres_numeric_eti_spec_avail<                                       \
      KokkosKernels::Experimental::KokkosKernelsHandle<                     \
          const OFFSET_TYPE, const ORDINAL_TYPE, const SCALAR_TYPE,         \
          EXEC_SPACE_TYPE, MEM_SPACE_TYPE, MEM_SPACE_TYPE>,                 \
      const SCALAR_TYPE, const ORDINAL_TYPE,                                \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                      \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,           \
      Kokkos::View<                                                         \
          const SCALAR_TYPE *, LAYOUT_TYPE,                                 \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      Kokkos::View<                                                         \
          SCALAR_TYPE *, LAYOUT_TYPE,                                       \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> > > { \
    enum : bool { value = true };                                           \
  };

// Include the actual specialization declarations
#include <KokkosSparse_gmres_numeric_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosSparse_gmres_numeric_eti_spec_avail.hpp>

namespace KokkosSparse {
namespace Impl {

// Unification layer
/// \brief Implementation of KokkosSparse::gmres_numeric

template <class KernelHandle, class AT, class AO, class AD, class AM, class AS, class BType, class XType,
          bool tpl_spec_avail = gmres_numeric_tpl_spec_avail<
            KernelHandle, AT, AO, AD, AM, AS, BType, XType>::value,
          bool eti_spec_avail = gmres_numeric_eti_spec_avail<
            KernelHandle, AT, AO, AD, AM, AS, BType, XType>::value>
struct GMRES_NUMERIC {
  using AMatrix = CrsMatrix<AT, AO, AD, AM, AS>;
  static void gmres_numeric(KernelHandle *handle,
                            const AMatrix &A,
                            const BType &B, XType &X);
};

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
//! Full specialization of gmres_numeric
// Unification layer
template <class KernelHandle, class AT, class AO, class AD, class AM, class AS, class BType, class XType>
struct GMRES_NUMERIC<KernelHandle, AT, AO, AD, AM, AS,
                     BType, XType, false, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  using AMatrix = CrsMatrix<AT, AO, AD, AM, AS>;
  static void gmres_numeric(KernelHandle *handle,
                            const AMatrix &A,
                            BType &B, XType &X) {
    auto gmres_handle = handle->get_gmres_handle();
    using Gmres           = Experimental::GmresWrap<
        typename std::remove_pointer<decltype(gmres_handle)>::type>;

    Gmres::gmres_numeric(*handle, *gmres_handle, A, B, X);
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
#define KOKKOSSPARSE_GMRES_NUMERIC_ETI_SPEC_DECL(                        \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE,   \
    MEM_SPACE_TYPE)                                                         \
  extern template struct GMRES_NUMERIC<                                  \
      KokkosKernels::Experimental::KokkosKernelsHandle<                     \
          const OFFSET_TYPE, const ORDINAL_TYPE, const SCALAR_TYPE,         \
          EXEC_SPACE_TYPE, MEM_SPACE_TYPE, MEM_SPACE_TYPE>,                 \
      const SCALAR_TYPE, const ORDINAL_TYPE,                                \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                      \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,           \
      Kokkos::View<                                                         \
          const SCALAR_TYPE *, LAYOUT_TYPE,                                 \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      Kokkos::View<                                                         \
          SCALAR_TYPE *, LAYOUT_TYPE,                                       \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      false, true>;

#define KOKKOSSPARSE_GMRES_NUMERIC_ETI_SPEC_INST(                        \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE,   \
    MEM_SPACE_TYPE)                                                         \
  template struct GMRES_NUMERIC<                                         \
      KokkosKernels::Experimental::KokkosKernelsHandle<                     \
          const OFFSET_TYPE, const ORDINAL_TYPE, const SCALAR_TYPE,         \
          EXEC_SPACE_TYPE, MEM_SPACE_TYPE, MEM_SPACE_TYPE>,                 \
      const SCALAR_TYPE, const ORDINAL_TYPE,                                \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                      \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,           \
      Kokkos::View<                                                         \
          const SCALAR_TYPE *, LAYOUT_TYPE,                                 \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      Kokkos::View<                                                         \
          SCALAR_TYPE *, LAYOUT_TYPE,                                       \
          Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                  \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      false, true>;

#include <KokkosSparse_gmres_numeric_tpl_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosSparse_gmres_numeric_eti_spec_decl.hpp>

#endif