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
#ifndef KOKKOSSPARSE_IMPL_SPGEMM_SYMBOLIC_SPEC_HPP_
#define KOKKOSSPARSE_IMPL_SPGEMM_SYMBOLIC_SPEC_HPP_

#include <KokkosKernels_config.h>

#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include "KokkosKernels_Handle.hpp"
// Include the actual functors
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include "KokkosSparse_spgemm_cuSPARSE_impl.hpp"
#include "KokkosSparse_spgemm_rocSPARSE_impl.hpp"
#include "KokkosSparse_spgemm_CUSP_impl.hpp"
#include "KokkosSparse_spgemm_impl.hpp"
#include "KokkosSparse_spgemm_impl_seq.hpp"
#include "KokkosSparse_spgemm_mkl_impl.hpp"
#include "KokkosSparse_spgemm_mkl2phase_impl.hpp"
#include "KokkosSparse_spgemm_viennaCL_impl.hpp"
#endif

namespace KokkosSparse {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class KernelHandle, class a_size_view_t_, class a_lno_view_t,
          class b_size_view_t_, class b_lno_view_t, class c_size_view_t_>
struct spgemm_symbolic_eti_spec_avail {
  enum : bool { value = false };
};

}  // namespace Impl
}  // namespace KokkosSparse

#define KOKKOSSPARSE_SPGEMM_SYMBOLIC_ETI_SPEC_AVAIL(                      \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE, SLOW_MEM_SPACE_TYPE)                                  \
  template <>                                                             \
  struct spgemm_symbolic_eti_spec_avail<                                  \
      KokkosKernels::Experimental::KokkosKernelsHandle<                   \
          const OFFSET_TYPE, const ORDINAL_TYPE, const SCALAR_TYPE,       \
          EXEC_SPACE_TYPE, MEM_SPACE_TYPE, SLOW_MEM_SPACE_TYPE>,          \
      Kokkos::View<const OFFSET_TYPE *, LAYOUT_TYPE,                      \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<const ORDINAL_TYPE *, LAYOUT_TYPE,                     \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<const OFFSET_TYPE *, LAYOUT_TYPE,                      \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<const ORDINAL_TYPE *, LAYOUT_TYPE,                     \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<OFFSET_TYPE *, LAYOUT_TYPE,                            \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {          \
    enum : bool { value = true };                                         \
  };

// Include the actual specialization declarations
#include <KokkosSparse_spgemm_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosSparse_spgemm_symbolic_eti_spec_avail.hpp>

namespace KokkosSparse {
namespace Impl {

// Unification layer
/// \brief Implementation of KokkosSparse::spgemm (sparse matrix - sparse
///   matrix multiply)
///
template <class KernelHandle, class a_size_view_t_, class a_lno_view_t,
          class b_size_view_t_, class b_lno_view_t, class c_size_view_t_,
          bool tpl_spec_avail = spgemm_symbolic_tpl_spec_avail<
              KernelHandle, a_size_view_t_, a_lno_view_t, b_size_view_t_,
              b_lno_view_t, c_size_view_t_>::value,
          bool eti_spec_avail = spgemm_symbolic_eti_spec_avail<
              KernelHandle, a_size_view_t_, a_lno_view_t, b_size_view_t_,
              b_lno_view_t, c_size_view_t_>::value>
struct SPGEMM_SYMBOLIC {
  static void spgemm_symbolic(KernelHandle *handle,
                              typename KernelHandle::const_nnz_lno_t m,
                              typename KernelHandle::const_nnz_lno_t n,
                              typename KernelHandle::const_nnz_lno_t k,
                              a_size_view_t_ row_mapA, a_lno_view_t entriesA,
                              bool transposeA, b_size_view_t_ row_mapB,
                              b_lno_view_t entriesB, bool transposeB,
                              c_size_view_t_ row_mapC);
};

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
//! Full specialization of spgemm for single vectors (1-D Views).
// Unification layer
template <class KernelHandle, class a_size_view_t_, class a_lno_view_t,
          class b_size_view_t_, class b_lno_view_t, class c_size_view_t_>
struct SPGEMM_SYMBOLIC<KernelHandle, a_size_view_t_, a_lno_view_t,
                       b_size_view_t_, b_lno_view_t, c_size_view_t_, false,
                       KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  static void spgemm_symbolic(KernelHandle *handle,
                              typename KernelHandle::nnz_lno_t m,
                              typename KernelHandle::nnz_lno_t n,
                              typename KernelHandle::nnz_lno_t k,
                              a_size_view_t_ row_mapA, a_lno_view_t entriesA,
                              bool transposeA, b_size_view_t_ row_mapB,
                              b_lno_view_t entriesB, bool transposeB,
                              c_size_view_t_ row_mapC) {
    typedef typename KernelHandle::SPGEMMHandleType spgemmHandleType;
    spgemmHandleType *sh = handle->get_spgemm_handle();
    switch (sh->get_algorithm_type()) {
      case SPGEMM_CUSPARSE:
#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSPARSE)
        cuSPARSE_symbolic<spgemmHandleType, a_size_view_t_, a_lno_view_t,
                          b_size_view_t_, b_lno_view_t, c_size_view_t_>(
            sh, m, n, k, row_mapA, entriesA, transposeA, row_mapB, entriesB,
            transposeB, row_mapC);
#else
        throw std::runtime_error(
            "Requiring SPGEMM_CUSPARSE but TPL_CUSPARSE was not enabled!");
#endif
        break;
      case SPGEMM_ROCSPARSE:
#if defined(KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE)
        rocsparse_spgemm_symbolic<spgemmHandleType, a_size_view_t_,
                                  a_lno_view_t, b_size_view_t_, b_lno_view_t,
                                  c_size_view_t_>(
            sh, m, n, k, row_mapA, entriesA, transposeA, row_mapB, entriesB,
            transposeB, row_mapC);
#else
        throw std::runtime_error(
            "Requiring SPGEMM_ROCSPARSE but TPL_ROCSPARSE was not enabled!");
#endif
        break;
      case SPGEMM_CUSP:
      case SPGEMM_VIENNA: break;

      case SPGEMM_MKL2PHASE:
        mkl2phase_symbolic(sh, m, n, k, row_mapA, entriesA, transposeA,
                           row_mapB, entriesB, transposeB, row_mapC,
                           handle->get_verbose());
        break;

      default: {
        KokkosSPGEMM<KernelHandle, a_size_view_t_, a_lno_view_t,
                     typename KernelHandle::in_scalar_nnz_view_t,
                     b_size_view_t_, b_lno_view_t,
                     typename KernelHandle::in_scalar_nnz_view_t>
            kspgemm(handle, m, n, k, row_mapA, entriesA, transposeA, row_mapB,
                    entriesB, transposeB);
        kspgemm.KokkosSPGEMM_symbolic(row_mapC);
      } break;
      case SPGEMM_SERIAL:
      case SPGEMM_DEBUG:
        spgemm_debug_symbolic(handle, m, n, k, row_mapA, entriesA,

                              transposeA, row_mapB, entriesB, transposeB,
                              row_mapC);
        break;
      case SPGEMM_MKL:
#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
        mkl_symbolic(sh, m, n, k, row_mapA, entriesA, transposeA, row_mapB,
                     entriesB, transposeB, row_mapC, handle->get_verbose());
        break;
#else
        throw std::runtime_error("MKL was not enabled in this build!");
#endif
    }
    sh->set_call_symbolic();
  }
};

#endif

}  // namespace Impl
}  // namespace KokkosSparse

//
// Macro for declaration of full specialization of
// KokkosSparse::Impl::Dot for rank == 2.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _DEF macro below) across one or
// more .cpp files.
//
#define KOKKOSSPARSE_SPGEMM_SYMBOLIC_ETI_SPEC_DECL(                       \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE, SLOW_MEM_SPACE_TYPE)                                  \
  extern template struct SPGEMM_SYMBOLIC<                                 \
      KokkosKernels::Experimental::KokkosKernelsHandle<                   \
          const OFFSET_TYPE, const ORDINAL_TYPE, const SCALAR_TYPE,       \
          EXEC_SPACE_TYPE, MEM_SPACE_TYPE, SLOW_MEM_SPACE_TYPE>,          \
      Kokkos::View<const OFFSET_TYPE *, LAYOUT_TYPE,                      \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<const ORDINAL_TYPE *, LAYOUT_TYPE,                     \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<const OFFSET_TYPE *, LAYOUT_TYPE,                      \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<const ORDINAL_TYPE *, LAYOUT_TYPE,                     \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<OFFSET_TYPE *, LAYOUT_TYPE,                            \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      false, true>;

#define KOKKOSSPARSE_SPGEMM_SYMBOLIC_ETI_SPEC_INST(                       \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE, SLOW_MEM_SPACE_TYPE)                                  \
  template struct SPGEMM_SYMBOLIC<                                        \
      KokkosKernels::Experimental::KokkosKernelsHandle<                   \
          const OFFSET_TYPE, const ORDINAL_TYPE, const SCALAR_TYPE,       \
          EXEC_SPACE_TYPE, MEM_SPACE_TYPE, SLOW_MEM_SPACE_TYPE>,          \
      Kokkos::View<const OFFSET_TYPE *, LAYOUT_TYPE,                      \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<const ORDINAL_TYPE *, LAYOUT_TYPE,                     \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<const OFFSET_TYPE *, LAYOUT_TYPE,                      \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<const ORDINAL_TYPE *, LAYOUT_TYPE,                     \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<OFFSET_TYPE *, LAYOUT_TYPE,                            \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      false, true>;

#include <KokkosSparse_spgemm_tpl_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosSparse_spgemm_symbolic_eti_spec_decl.hpp>

#endif  // KOKKOS_BLAS1_MV_IMPL_DOT_HPP_
