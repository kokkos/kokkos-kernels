// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_POTRS_HPP_
#define KOKKOSLAPACK_POTRS_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "KokkosLapack_potrs_spec.hpp"

namespace KokkosLapack {

/// \brief TODO: Add brief description
///
/// TODO: Add detailed description of the function
///
/// \tparam AViewType TODO: describe this parameter
/// \tparam BViewType TODO: describe this parameter
///
template <class execution_space, class AViewType, class BViewType>
void potrs(const execution_space& space, const char uplo[], const int& n, const int& nrhs, const AViewType& A,
           const int& lda, BViewType& B, const int& ldb) {
  static_assert(Kokkos::is_execution_space<execution_space>::value,
                "KokkosLapack::potrs: execution_space must be a valid Kokkos execution space");
  static_assert(Kokkos::is_view<AViewType>::value, "KokkosLapack::potrs: AViewType must be a Kokkos::View");
  static_assert(Kokkos::is_view<BViewType>::value, "KokkosLapack::potrs: BViewType must be a Kokkos::View");

  // Convert views to unmanaged
  using AViewInternalType = Kokkos::View<typename AViewType::const_data_type, typename AViewType::array_layout,
                                         typename AViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;
  using BViewInternalType = Kokkos::View<typename BViewType::data_type, typename BViewType::array_layout,
                                         typename BViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  AViewInternalType uA(A);
  BViewInternalType uB(B);

  Impl::Potrs<execution_space, AViewInternalType, BViewInternalType>::potrs(space, uplo, n, nrhs, uA, lda, uB, ldb);
}

// Overload without execution space (uses default)
template <class AViewType, class BViewType>
void potrs(const char uplo[], const int& n, const int& nrhs, const AViewType& A, const int& lda, BViewType& B,
           const int& ldb) {
  potrs(typename AViewType::execution_space{}, uplo, n, nrhs, A, lda, B, ldb);
}

}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_POTRS_HPP_
