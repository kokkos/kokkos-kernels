// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_POTRF_TPL_SPEC_AVAIL_HPP_
#define KOKKOSLAPACK_POTRF_TPL_SPEC_AVAIL_HPP_

namespace KokkosLapack {
namespace Impl {

// Specialization struct which defines whether a TPL specialization exists
template <class AViewType>
struct potrf_tpl_spec_avail {
  enum : bool { value = false };
};

// TODO: Define TPL availability macros for your supported TPLs
// Example for LAPACK/BLAS:
// #ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK
// #define KOKKOSLAPACK_POTRF_TPL_SPEC_AVAIL_LAPACK(SCALAR, LAYOUT, MEMSPACE) \
//   template <class ExecSpace> \
//   struct potrf_tpl_spec_avail< \
//       Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> { \
//     enum : bool { value = true }; \
//   };
// #else
// #define KOKKOSLAPACK_POTRF_TPL_SPEC_AVAIL_LAPACK(SCALAR, LAYOUT, MEMSPACE)
// #endif

}  // namespace Impl
}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_POTRF_TPL_SPEC_AVAIL_HPP_
