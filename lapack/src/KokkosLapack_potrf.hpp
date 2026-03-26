// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_POTRF_HPP_
#define KOKKOSLAPACK_POTRF_HPP_

#include "KokkosKernels_config.h"
#include "KokkosKernels_Error.hpp"
#include "Kokkos_Core.hpp"
#include "KokkosLapack_potrf_spec.hpp"

#include <type_traits>

namespace KokkosLapack {

/// \brief Computes the Cholesky factorization of a Hermitian positive definite matrix A.
///
/// POTRF computes the Cholesky factorization of a Hermitian
/// positive definite matrix A.
///
/// The factorization has the form
///    A = U**H * U,  if UPLO = 'U', or
///    A = L  * L**H,  if UPLO = 'L',
/// where U is an upper triangular matrix and L is lower triangular.
///
/// This is the block version of the algorithm, calling Level 3 BLAS if blas/lapack
/// is enabled.
///
/// \tparam execution_space The space where the kernel will run.
/// \tparam AViewType [in] Type of matrix A, as a 2-D Kokkos::View (LayoutLeft!)
///
/// \param space [in] Execution space instance used to specify how to execute
///                   the potrf kernels.
/// \param uplo  [in] 'U': upper triangle of A is stored; 'L': lower triangle.
/// \param A     [in,out] Square 2-D Kokkos::View of dimension N x N.
///                       On entry, the Hermitian positive-definite matrix A.
///                       The order N and leading dimension are derived from the
///                       view extents: N = A.extent(0), lda = A.stride(1).
///                       On exit, the factor U or L from the Cholesky
///                       factorization A = U**H*U or A = L*L**H.
///
template <class execution_space, class AViewType>
void potrf(const execution_space& space, const char uplo[], AViewType& A) {
  static_assert(Kokkos::is_execution_space<execution_space>::value,
                "KokkosLapack::potrf: execution_space must be a valid Kokkos execution space");
  static_assert(Kokkos::is_view<AViewType>::value, "KokkosLapack::potrf: AViewType must be a Kokkos::View");
  static_assert(static_cast<int>(AViewType::rank) == 2, "KokkosLapack::potrf: A must have rank 2.");
  static_assert(!std::is_const_v<typename AViewType::value_type>, "A should not have const value type");
  static_assert(std::is_same_v<typename AViewType::array_layout, Kokkos::LayoutLeft>,
                "KokkosLapack::potrf: A must have Kokkos::LayoutLeft (column-major) layout, "
                "as required by LAPACK/cuSOLVER/rocSOLVER.");

  if (A.extent(0) != A.extent(1)) {
    std::ostringstream os;
    os << "KokkosLapack::potrf: A must be square, got " << A.extent(0) << " x " << A.extent(1);
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  const int n   = static_cast<int>(A.extent(0));
  const int lda = static_cast<int>(A.stride(1));

  // Convert views to unmanaged
  using AViewInternalType = Kokkos::View<typename AViewType::data_type, typename AViewType::array_layout,
                                         typename AViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  AViewInternalType uA(A);

  Impl::Potrf<execution_space, AViewInternalType>::potrf(space, uplo, n, uA, lda);
}

// Overload without execution space (uses default)
template <class AViewType>
void potrf(const char uplo[], AViewType& A) {
  potrf(typename AViewType::execution_space{}, uplo, A);
}

}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_POTRF_HPP_
