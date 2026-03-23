// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_POTRF_HPP_
#define KOKKOSLAPACK_POTRF_HPP_

#include "KokkosKernels_config.h"
#include "Kokkos_Core.hpp"
#include "KokkosLapack_potrf_spec.hpp"

namespace KokkosLapack {

/// \brief Computes the Cholesky factorization of a complex Hermitian positive definite matrix A.
///
/// CPOTRF computes the Cholesky factorization of a complex Hermitian
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
/// \tparam AViewType [in] Type of matrix A, as a 2-D Kokkos::View.
///
/// \param space [in] Execution space instance used to specified how to execute
///                   the potrf kernels.
/// \param uplo  [in] 'U':  Upper triangle of A is stored, else lower triangle
/// \param n     [in] The order of the matrix A.  N >= 0.
/// \param A     [in,out] A is 2d kokkos view, dimension (LDA,N)
///                       On entry, the Hermitian matrix A.  If UPLO = 'U', the leading
///                       N-by-N upper triangular part of A contains the upper
///                       triangular part of the matrix A, and the strictly lower
///                       triangular part of A is not referenced.  If UPLO = 'L', the
///                       leading N-by-N lower triangular part of A contains the lower
///                       triangular part of the matrix A, and the strictly upper
///                       triangular part of A is not referenced.
///
///                       On exit, the factor U or L from the Cholesky
///                       factorization A = U**H*U or A = L*L**H.
/// \param                The leading dimension of matrix A.  LDA >= max(1,N).
///
template <class execution_space, class AViewType>
void potrf([[maybe_unused]] const execution_space& space, const char uplo[], const int& n, AViewType& A,
           const int& lda) {
  static_assert(Kokkos::is_execution_space<execution_space>::value,
                "KokkosLapack::potrf: execution_space must be a valid Kokkos execution space");
  static_assert(Kokkos::is_view<AViewType>::value, "KokkosLapack::potrf: AViewType must be a Kokkos::View");

  // Convert views to unmanaged
  using AViewInternalType = Kokkos::View<typename AViewType::data_type, typename AViewType::array_layout,
                                         typename AViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >;

  AViewInternalType uA(A);

  Impl::Potrf<AViewInternalType>::potrf(uplo, n, uA, lda);
}

// Overload without execution space (uses default)
template <class AViewType>
void potrf(const char uplo[], const int& n, AViewType& A, const int& lda) {
  potrf(typename AViewType::execution_space{}, uplo, n, A, lda);
}

}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_POTRF_HPP_
