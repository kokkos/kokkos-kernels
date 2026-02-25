// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_POTRF_HPP_
#define KOKKOSLAPACK_POTRF_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "KokkosLapack_potrf_spec.hpp"

namespace KokkosLapack {

/// \brief TODO: Add brief description
///
/// TODO: Add detailed description of the function
///
/// \tparam AViewType TODO: describe this parameter
///
template <class execution_space, class AViewType>
void potrf(const execution_space& space,
                const char uplo[],
                const int& n,
                AViewType& A,
                const int& lda) {
  static_assert(Kokkos::is_execution_space<execution_space>::value,
                "KokkosLapack::potrf: execution_space must be a valid Kokkos execution space");
  static_assert(Kokkos::is_view<AViewType>::value,
                "KokkosLapack::potrf: AViewType must be a Kokkos::View");

  // Convert views to unmanaged
  using AViewInternalType = Kokkos::View<typename AViewType::data_type, typename AViewType::array_layout,
                                         typename AViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  AViewInternalType uA(A);

  Impl::Potrf<AViewInternalType>::potrf(uplo, n, uA, lda);
}

// Overload without execution space (uses default)
template <class AViewType>
void potrf(const char uplo[],
                const int& n,
                AViewType& A,
                const int& lda) {
  potrf(typename AViewType::execution_space{}, uplo, n, A, lda);
}

}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_POTRF_HPP_
