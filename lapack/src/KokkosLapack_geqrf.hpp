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
/// \brief QR factorization
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
/// \tparam ExecutionSpace The space where the kernel will run.
/// \tparam AMatrix        Type of matrix A, as a 2-D Kokkos::View.
/// \tparam TWArray        Type of arrays Tau and Work, as a 1-D Kokkos::View.
///
/// \param space [in] Execution space instance used to specified how to execute
///                   the geqrf kernels.
/// \param A [in,out] On entry, the M-by-N matrix to be factorized.
///                   On exit, the elements on and above the diagonal contain
///                   the min(M,N)-by-N upper trapezoidal matrix R (R is upper
///                   triangular if M >= N); the elements below the diagonal,
///                   with the array Tau, represent the unitary matrix Q as a
///                   product of min(M,N) elementary reflectors. The matrix Q
///                   is represented as a product of elementary reflectors
///                     Q = H(1) H(2) . . . H(k), where k = min(M,N).
///                   Each H(i) has the form
///                     H(i) = I - Tau * v * v**H
///                   where tau is a complex scalar, and v is a complex vector
///                   with v(1:i-1) = 0 and v(i) = 1; v(i+1:M) is stored on
///                   exit in A(i+1:M,i), and tau in Tau(i).
/// \param Tau [out]  One-dimensional array of size min(M,N) that contains
///                   the scalar factors of the elementary reflectors.
/// \param Work [out] One-dimensional array of size max(1,LWORK).
///                   If min(M,N) == 0, then LWORK must be >= 1.
///                   If min(M,N) != 0, then LWORK must be >= N.
///                   If the QR factorization is successful, then the first
///                   position of Work contains the optimal LWORK.
///
/// \return           = 0: successfull exit
///                   < 0: if equal to '-i', the i-th argument had an illegal
///                        value
///
template <class ExecutionSpace, class AMatrix, class TWArray>
int geqrf(const ExecutionSpace& space, const AMatrix& A, const TWArray& Tau,
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
                                 typename TWArray::memory_space>::accessible);

  static_assert(Kokkos::is_view<AMatrix>::value,
                "KokkosLapack::geqrf: A must be a Kokkos::View.");
  static_assert(Kokkos::is_view<TWArray>::value,
                "KokkosLapack::geqrf: Tau and Work must be Kokkos::View.");
  static_assert(static_cast<int>(AMatrix::rank) == 2,
                "KokkosLapack::geqrf: A must have rank 2.");
  static_assert(static_cast<int>(TWArray::rank) == 1,
                "KokkosLapack::geqrf: Tau and Work must have rank 1.");

  int64_t m     = A.extent(0);
  int64_t n     = A.extent(1);
  int64_t tau0  = Tau.extent(0);
  int64_t work0 = Work.extent(0);

  // Check validity of dimensions
  if (tau0 != std::min(m, n)) {
    std::ostringstream os;
    os << "KokkosLapack::geqrf: length of Tau must be equal to min(m,n): "
       << " A: " << m << " x " << n << ", Tau length = " << tau0;
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }
  if ((m == 0) || (n == 0)) {
    if (work0 < 1) {
      std::ostringstream os;
      os << "KokkosLapack::geqrf: In case min(m,n) == 0, then Work must have "
            "length >= 1: "
         << " A: " << m << " x " << n << ", Work length = " << work0;
      KokkosKernels::Impl::throw_runtime_exception(os.str());
    }
  } else {
    if (work0 < n) {
      std::ostringstream os;
      os << "KokkosLapack::geqrf: In case min(m,n) != 0, then Work must have "
            "length >= n: "
         << " A: " << m << " x " << n << ", Work length = " << work0;
      KokkosKernels::Impl::throw_runtime_exception(os.str());
    }
  }

  using RetArray = Kokkos::View<int*, typename TWArray::array_layout, typename TWArray::device_type>;
  RetArray rc("rc", 1);

  using AMatrix_Internal = Kokkos::View<
      typename AMatrix::non_const_value_type**, typename AMatrix::array_layout,
      typename AMatrix::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using TWArray_Internal = Kokkos::View<
      typename TWArray::non_const_value_type*, typename TWArray::array_layout,
      typename TWArray::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;
  using RetArray_Internal = Kokkos::View<
      int*, typename TWArray::array_layout,
      typename TWArray::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  AMatrix_Internal  A_i    = A;
  TWArray_Internal  Tau_i  = Tau;
  TWArray_Internal  Work_i = Work;
  RetArray_Internal rc_i   = rc;

  KokkosLapack::Impl::GEQRF<ExecutionSpace, AMatrix_Internal, TWArray_Internal,
                            RetArray_Internal>::geqrf(space, A_i, Tau_i, Work_i,
                                                      rc_i);

  typename RetArray_Internal::HostMirror h_rc = Kokkos::create_mirror_view(rc_i);

  Kokkos::deep_copy(h_rc, rc_i);

  return h_rc[0];
}

/// \brief Computes a QR factorization of a matrix A
///
/// \tparam AMatrix Type of matrix A, as a 2-D Kokkos::View.
/// \tparam TWArray Type of arrays Tau and Work, as a 1-D Kokkos::View.
///
/// \param A [in,out] On entry, the M-by-N matrix to be factorized.
///                   On exit, the elements on and above the diagonal contain
///                   the min(M,N)-by-N upper trapezoidal matrix R (R is upper
///                   triangular if M >= N); the elements below the diagonal,
///                   with the array Tau, represent the unitary matrix Q as a
///                   product of min(M,N) elementary reflectors. The matrix Q
///                   is represented as a product of elementary reflectors
///                     Q = H(1) H(2) . . . H(k), where k = min(M,N).
///                   Each H(i) has the form
///                     H(i) = I - Tau * v * v**H
///                   where tau is a complex scalar, and v is a complex vector
///                   with v(1:i-1) = 0 and v(i) = 1; v(i+1:M) is stored on
///                   exit in A(i+1:M,i), and tau in Tau(i).
/// \param Tau [out]  One-dimensional array of size min(M,N) that contains
///                   the scalar factors of the elementary reflectors.
/// \param Work [out] One-dimensional array of size max(1,LWORK).
///                   If min(M,N) == 0, then LWORK must be >= 1.
///                   If min(M,N) != 0, then LWORK must be >= N.
///                   If the QR factorization is successful, then the first
///                   position of Work contains the optimal LWORK.
///
/// \return           = 0: successfull exit
///                   < 0: if equal to '-i', the i-th argument had an illegal
///                        value
///
template <class AMatrix, class TWArray>
int geqrf(const AMatrix& A, const TWArray& Tau, const TWArray& Work) {
  typename AMatrix::execution_space space{};
  return geqrf(space, A, Tau, Work);
}

}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_GEQRF_HPP_
