// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_POTRF_IMPL_HPP_
#define KOKKOSLAPACK_POTRF_IMPL_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"

namespace KokkosLapack {
namespace Impl {

// Implementation struct for potrf
template <class AViewType>
struct PotrfImpl {
  // TODO: Add your implementation here
  static void potrf(const char uplo[], const int& n, AViewType& A, const int& lda) {
    // TODO: Implement your kernel here
    // Kokkos::abort("KokkosLapack::potrf: Not yet implemented");
  }
};

}  // namespace Impl
}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_POTRF_IMPL_HPP_
