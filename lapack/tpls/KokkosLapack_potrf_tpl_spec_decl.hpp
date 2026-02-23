// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_POTRF_TPL_SPEC_DECL_HPP_
#define KOKKOSLAPACK_POTRF_TPL_SPEC_DECL_HPP_

// TODO: Include TPL headers as needed
// Example:
// #ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK
// #include "KokkosLapack_Host_tpl.hpp"
// #endif

namespace KokkosLapack {
namespace Impl {

// TODO: Define TPL specializations for your supported TPLs
// Example for LAPACK:
// #ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK
// #define KOKKOSLAPACK_POTRF_TPL_LAPACK(SCALAR, LAYOUT, MEMSPACE, ETI_SPEC_AVAIL) \
//   template <class ExecSpace> \
//   struct Potrf<Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<EXEC_SPACE, MEM_SPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
//                          true, ETI_SPEC_AVAIL> { \
//     static void potrf(const char uplo[], const int& n, AViewType& A, const int& lda) { \
//       // TODO: Call TPL library here \
//     } \
//   };
// #else
// #define KOKKOSLAPACK_POTRF_TPL_LAPACK(SCALAR, LAYOUT, MEMSPACE, ETI_SPEC_AVAIL)
// #endif

}  // namespace Impl
}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_POTRF_TPL_SPEC_DECL_HPP_
