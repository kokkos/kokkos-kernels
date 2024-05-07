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

/// \file KokkosLapack_geqrf.hpp
/// \brief Local dense linear solve
///
/// This file provides KokkosLapack::geqrf. This function performs a
/// local (no MPI) QR factorization of a M-by-N matrix A.

#ifndef KOKKOSLAPACK_GEQRF_HPP_
#define KOKKOSLAPACK_GEQRF_HPP_

#include <type_traits>

#include "KokkosLapack_geqrf_spec.hpp"
#include "KokkosKernels_Error.hpp"

namespace KokkosLapack {

/// \brief Computes a QR factorization of a matrix A
///
/// \tparam ExecutionSpace the space where the kernel will run.
/// \tparam AMatrix Type of matrix A, as a 2-D Kokkos::View.
/// \tparam TWArray Type of arrays Tau and Work, as a 1-D Kokkos::View.
///
/// \param space [in] Execution space instance used to specified how to execute
///                   the geqrf kernels.
/// \param A [in,out] On entry, the M-by-N matrix to be factorized.
///                   On exit, the elements on and above the diagonal contain
///                   the min(M,N)-by-N upper trapezoidal matrix R (R is
///                   upper triangular if M >= N); the elements below the
///                   diagonal, with the array Tau, represent the unitary
///                   matrix Q as a product of min(M,N) elementary reflectors.
/// \param Tau [out]  One-dimensional array of size min(M,N) that contain
///                   the scalar factors of the elementary reflectors.
/// \param Work [out] One-dimensional array of size max(1,LWORK).
///                   If min(M,N) == 0, then LWORK must be >= 1.
///                   If min(M,N) != 0, then LWORK must be >= N.
///                   If the QR factorization is successful, then the first
///                   position of Work contains the optimal LWORK.
///
template <class ExecutionSpace, class AMatrix, class TWArray>
void geqrf(const ExecutionSpace& space, const AMatrix& A, const TWArray& Tau,
          const TWArray& Work) {
  // NOTE: Currently, KokkosLapack::geqrf only supports LAPACK, MAGMA and
  // rocSOLVER TPLs.
  //       MAGMA/rocSOLVER TPL should be enabled to call the MAGMA/rocSOLVER GPU
  //       interface for device views LAPACK TPL should be enabled to call the
  //       LAPACK interface for host views

  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename AMatrix::memory_space>::accessible);
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename BXMV::memory_space>::accessible);
#if defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA)
  if constexpr (!std::is_same_v<ExecutionSpace, Kokkos::Cuda>) {
    static_assert(
        Kokkos::SpaceAccessibility<ExecutionSpace,
                                   typename IPIVV::memory_space>::accessible);
  }
#else
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename IPIVV::memory_space>::accessible);
#endif
  static_assert(Kokkos::is_view<AMatrix>::value,
                "KokkosLapack::geqrf: A must be a Kokkos::View.");
  static_assert(Kokkos::is_view<BXMV>::value,
                "KokkosLapack::geqrf: B must be a Kokkos::View.");
  static_assert(Kokkos::is_view<IPIVV>::value,
                "KokkosLapack::geqrf: IPIV must be a Kokkos::View.");
  static_assert(static_cast<int>(AMatrix::rank) == 2,
                "KokkosLapack::geqrf: A must have rank 2.");
  static_assert(
      static_cast<int>(BXMV::rank) == 1 || static_cast<int>(BXMV::rank) == 2,
      "KokkosLapack::geqrf: B must have either rank 1 or rank 2.");
  static_assert(static_cast<int>(IPIVV::rank) == 1,
                "KokkosLapack::geqrf: IPIV must have rank 1.");

  int64_t IPIV0 = IPIV.extent(0);
  int64_t A0    = A.extent(0);
  int64_t A1    = A.extent(1);
  int64_t B0    = B.extent(0);

  // Check validity of pivot argument
  bool valid_pivot =
      (IPIV0 == A1) || ((IPIV0 == 0) && (IPIV.data() == nullptr));
  if (!(valid_pivot)) {
    std::ostringstream os;
    os << "KokkosLapack::geqrf: IPIV: " << IPIV0 << ". "
       << "Valid options include zero-extent 1-D view (no pivoting), or 1-D "
          "View with size of "
       << A0 << " (partial pivoting).";
g    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  // Check for no pivoting case. Only MAGMA supports no pivoting interface
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA   // have MAGMA TPL
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK  // and have LAPACK TPL
  if ((!std::is_same<typename AMatrix::device_type::memory_space,
                     Kokkos::CudaSpace>::value) &&
      (IPIV0 == 0) && (IPIV.data() == nullptr)) {
    std::ostringstream os;
    os << "KokkosLapack::geqrf: IPIV: " << IPIV0 << ". "
       << "LAPACK TPL does not support no pivoting.";
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }
#endif
#else                                   // not have MAGMA TPL
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK  // but have LAPACK TPL
  if ((IPIV0 == 0) && (IPIV.data() == nullptr)) {
    std::ostringstream os;
    os << "KokkosLapack::geqrf: IPIV: " << IPIV0 << ". "
       << "LAPACK TPL does not support no pivoting.";
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }
#endif
#endif

  // Check compatibility of dimensions at run time.
  if ((A0 < A1) || (A0 != B0)) {
    std::ostringstream os;
    os << "KokkosLapack::geqrf: Dimensions of A, and B do not match: "
       << " A: " << A.extent(0) << " x " << A.extent(1) << " B: " << B.extent(0)
       << " x " << B.extent(1);
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  typedef Kokkos::View<
      typename AMatrix::non_const_value_type**, typename AMatrix::array_layout,
      typename AMatrix::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      AMatrix_Internal;
  typedef Kokkos::View<typename BXMV::non_const_value_type**,
                       typename BXMV::array_layout, typename BXMV::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      BXMV_Internal;
  typedef Kokkos::View<
      typename IPIVV::non_const_value_type*, typename IPIVV::array_layout,
      typename IPIVV::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      IPIVV_Internal;
  AMatrix_Internal A_i = A;
  // BXMV_Internal B_i = B;
  IPIVV_Internal IPIV_i = IPIV;

  if (BXMV::rank == 1) {
    auto B_i = BXMV_Internal(B.data(), B.extent(0), 1);
    KokkosLapack::Impl::GEQRF<ExecutionSpace, AMatrix_Internal, BXMV_Internal,
                             IPIVV_Internal>::geqrf(space, A_i, B_i, IPIV_i);
  } else {  // BXMV::rank == 2
    auto B_i = BXMV_Internal(B.data(), B.extent(0), B.extent(1));
    KokkosLapack::Impl::GEQRF<ExecutionSpace, AMatrix_Internal, BXMV_Internal,
                             IPIVV_Internal>::geqrf(space, A_i, B_i, IPIV_i);
  }
}

/// \brief Computes a QR factorization of a matrix A
///
/// \tparam AMatrix Type of matrix A, as a 2-D Kokkos::View.
/// \tparam TWArray Type of arrays Tau and Work, as a 1-D Kokkos::View.
///
/// \param A [in,out] On entry, the M-by-N matrix to be factorized.
///                   On exit, the elements on and above the diagonal contain
///                   the min(M,N)-by-N upper trapezoidal matrix R (R is
///                   upper triangular if M >= N); the elements below the
///                   diagonal, with the array Tau, represent the unitary
///                   matrix Q as a product of min(M,N) elementary reflectors.
/// \param Tau [out]  One-dimensional array of size min(M,N) that contain
///                   the scalar factors of the elementary reflectors.
/// \param Work [out] One-dimensional array of size max(1,LWORK).
///                   If min(M,N) == 0, then LWORK must be >= 1.
///                   If min(M,N) != 0, then LWORK must be >= N.
///                   If the QR factorization is successful, then the first
///                   position of Work contains the optimal LWORK.
///
template <class AMatrix, class TWArray>
void geqrf(const AMatrix& A, const TWArray& Tau, const TWArray& Work) {
  typename AMatrix::execution_space space{};
  geqrf(space, A, Tau, Work);
}

}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_GEQRF_HPP_
