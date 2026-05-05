// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_POTRS_IMPL_HPP_
#define KOKKOSLAPACK_POTRS_IMPL_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"

namespace KokkosLapack {
namespace Impl {

// Implementation struct for potrs
template <class AViewType, class BViewType>
struct PotrsImpl {
  // TODO: Add your implementation here
  static void potrs(const char uplo[], const int& n, const int& nrhs, const AViewType& A, const int& lda, BViewType& B,
                    const int& ldb) {
    // TODO: Implement your kernel here
    Kokkos::abort("KokkosLapack::potrs: Not yet implemented");
  }
};

}  // namespace Impl
}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_POTRS_IMPL_HPP_
