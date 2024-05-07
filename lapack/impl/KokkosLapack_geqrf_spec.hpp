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
#ifndef KOKKOSLAPACK_IMPL_GEQRF_SPEC_HPP_
#define KOKKOSLAPACK_IMPL_GEQRF_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>

// Include the actual functors
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosLapack_geqrf_impl.hpp>
#endif

namespace KokkosLapack {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class ExecutionSpace, class AVT, class TWVT>
struct geqrf_eti_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosLapack

//
// Macro for declaration of full specialization availability
// KokkosLapack::Impl::GEQRF.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _INST macro below) across one or
// more .cpp files.
//
#define KOKKOSLAPACK_GEQRF_ETI_SPEC_AVAIL(SCALAR_TYPE, LAYOUT_TYPE,       \
                                         EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
  template <>                                                             \
  struct geqrf_eti_spec_avail<                                            \
      EXEC_SPACE_TYPE,                                                    \
      Kokkos::View<SCALAR_TYPE **, LAYOUT_TYPE,                           \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,              \
      Kokkos::View<SCALAR_TYPE **, LAYOUT_TYPE,                           \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,              \
      Kokkos::View<int *, LAYOUT_TYPE,                                    \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {            \
    enum : bool { value = true };                                         \
  };

// Include the actual specialization declarations
#include <KokkosLapack_geqrf_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosLapack_geqrf_eti_spec_avail.hpp>

namespace KokkosLapack {
namespace Impl {

// Unification layer
/// \brief Implementation of KokkosLapack::geqrf.

template <class ExecutionSpace, class AMatrix, class TWArray,
          bool tpl_spec_avail =
              geqrf_tpl_spec_avail<ExecutionSpace, AMatrix, TWArray>::value,
          bool eti_spec_avail =
              geqrf_eti_spec_avail<ExecutionSpace, AMatrix, TWArray>::value>
struct GEQRF {
  static void geqrf(const ExecutionSpace &space, const AMatrix &A, const TWArray &Tau,
                   const TWArray &Work);
};

#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
//! Full specialization of geqrf for multi vectors.
// Unification layer
template <class ExecutionSpace, class AMatrix, class TWArray>
struct GEQRF<ExecutionSpace, AMatrix, TWArray, TWArray, false,
            KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  static void geqrf(const ExecutionSpace & /* space */, const AMatrix & /* A */,
                   const TWArray & /* Tau */, const TWArray & /* Work */) {
    // NOTE: Might add the implementation of KokkosLapack::geqrf later
    throw std::runtime_error(
        "No fallback implementation of GEQRF (general QR factorization) "
        "exists. Enable LAPACK, CUSOLVER, ROCSOLVER or MAGMA TPL.");
  }
};

#endif
}  // namespace Impl
}  // namespace KokkosLapack

//
// Macro for declaration of full specialization of
// KokkosLapack::Impl::GEQRF.  This is NOT for users!!!  All
// the declarations of full specializations go in this header file.
// We may spread out definitions (see _DEF macro below) across one or
// more .cpp files.
//
#define KOKKOSLAPACK_GEQRF_ETI_SPEC_DECL(SCALAR_TYPE, LAYOUT_TYPE,       \
                                        EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
  extern template struct GEQRF<                                          \
      EXEC_SPACE_TYPE,                                                   \
      Kokkos::View<SCALAR_TYPE **, LAYOUT_TYPE,                          \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,             \
      Kokkos::View<SCALAR_TYPE *, LAYOUT_TYPE,                           \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,             \
      Kokkos::View<SCALAR_TYPE *, LAYOUT_TYPE,                           \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,             \
      false, true>;

#define KOKKOSLAPACK_GEQRF_ETI_SPEC_INST(SCALAR_TYPE, LAYOUT_TYPE,       \
                                        EXEC_SPACE_TYPE, MEM_SPACE_TYPE) \
  template struct GEQRF<                                                 \
      EXEC_SPACE_TYPE,                                                   \
      Kokkos::View<SCALAR_TYPE **, LAYOUT_TYPE,                          \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,             \
      Kokkos::View<SCALAR_TYPE *, LAYOUT_TYPE,                           \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,             \
      Kokkos::View<SCALAR_TYPE *, LAYOUT_TYPE,                           \
                   Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,             \
      false, true>;

#include <KokkosLapack_geqrf_tpl_spec_decl.hpp>

#endif  // KOKKOSLAPACK_IMPL_GEQRF_SPEC_HPP_
