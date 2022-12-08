/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

/// \file KokkosSparse_gmres.hpp
/// \brief GMRES Ax = b solver
///
/// This file provides KokkosSparse::gmres_numeric.  This function performs a
/// local (no MPI) solve of Ax = b for sparse A. It is expected that A is in
/// compressed row sparse ("Crs") format.
///
/// This algorithm is described in the paper:
/// GMRES - A Generalized Minimal Residual Algorithm for Solving Nonsymmetric
/// Linear Systems - Saad, Schultz

#ifndef KOKKOSSPARSE_GMRES_HPP_
#define KOKKOSSPARSE_GMRES_HPP_

#include <type_traits>

#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Error.hpp"
#include "KokkosSparse_gmres_numeric_spec.hpp"
#include "KokkosSparse_Preconditioner.hpp"

namespace KokkosSparse {
namespace Experimental {

#define KOKKOSKERNELS_GMRES_SAME_TYPE(A, B)         \
  std::is_same<typename std::remove_const<A>::type, \
               typename std::remove_const<B>::type>::value

template <typename KernelHandle, typename AMatrix, typename BType,
          typename XType>
void gmres_numeric(KernelHandle* handle, AMatrix& A, BType& B, XType& X, Preconditioner<AMatrix>* precond = nullptr) {
  using scalar_type  = typename KernelHandle::nnz_scalar_t;
  using size_type    = typename KernelHandle::size_type;
  using ordinal_type = typename KernelHandle::nnz_lno_t;

  static_assert(
      KOKKOSKERNELS_GMRES_SAME_TYPE(typename BType::value_type, scalar_type),
      "gmres_numeric: B scalar type must match KernelHandle entry "
      "type (aka nnz_scalar_t, and const doesn't matter)");

  static_assert(
      KOKKOSKERNELS_GMRES_SAME_TYPE(typename XType::value_type, scalar_type),
      "gmres_numeric: X scalar type must match KernelHandle entry "
      "type (aka nnz_scalar_t, and const doesn't matter)");

  static_assert(
      KOKKOSKERNELS_GMRES_SAME_TYPE(typename AMatrix::value_type, scalar_type),
      "gmres_numeric: A scalar type must match KernelHandle entry "
      "type (aka nnz_scalar_t, and const doesn't matter)");

  static_assert(KOKKOSKERNELS_GMRES_SAME_TYPE(typename AMatrix::ordinal_type,
                                              ordinal_type),
                "gmres_numeric: A ordinal type must match KernelHandle entry "
                "type (aka nnz_lno_t, and const doesn't matter)");

  static_assert(
      KOKKOSKERNELS_GMRES_SAME_TYPE(typename AMatrix::size_type, size_type),
      "gmres_numeric: A size type must match KernelHandle entry "
      "type (aka size_type, and const doesn't matter)");

  static_assert(KokkosSparse::is_crs_matrix<AMatrix>::value,
                "gmres_numeric: A is not a CRS matrix.");
  static_assert(Kokkos::is_view<BType>::value,
                "gmres_numeric: B is not a Kokkos::View.");
  static_assert(Kokkos::is_view<XType>::value,
                "gmres_numeric: X is not a Kokkos::View.");

  static_assert(BType::rank == 1, "gmres_numeric: B must have rank 1");
  static_assert(XType::rank == 1, "gmres_numeric: X must have rank 1");

  static_assert(std::is_same<typename XType::value_type,
                             typename XType::non_const_value_type>::value,
                "gmres_numeric: The output X must be nonconst.");

  static_assert(std::is_same<typename XType::device_type,
                             typename BType::device_type>::value,
                "gmres_numeric: X and B have different device types.");

  static_assert(std::is_same<typename AMatrix::device_type,
                             typename BType::device_type>::value,
                "gmres_numeric: A and B have different device types.");

  using c_size_t   = typename KernelHandle::const_size_type;
  using c_lno_t    = typename KernelHandle::const_nnz_lno_t;
  using c_scalar_t = typename KernelHandle::const_nnz_scalar_t;

  using c_exec_t    = typename KernelHandle::HandleExecSpace;
  using c_temp_t    = typename KernelHandle::HandleTempMemorySpace;
  using c_persist_t = typename KernelHandle::HandlePersistentMemorySpace;

  if ((X.extent(0) != B.extent(0)) ||
      (static_cast<size_t>(A.numCols()) != static_cast<size_t>(X.extent(0))) ||
      (static_cast<size_t>(A.numRows()) != static_cast<size_t>(B.extent(0)))) {
    std::ostringstream os;
    os << "KokkosSparse::gmres: Dimensions do not match: "
       << ", A: " << A.numRows() << " x " << A.numCols()
       << ", x: " << X.extent(0) << ", b: " << B.extent(0);
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }

  using const_handle_type =
      typename KokkosKernels::Experimental::KokkosKernelsHandle<
          c_size_t, c_lno_t, c_scalar_t, c_exec_t, c_temp_t, c_persist_t>;

  const_handle_type tmp_handle(*handle);

  using AMatrix_Internal = KokkosSparse::CrsMatrix<
      typename AMatrix::const_value_type, typename AMatrix::const_ordinal_type,
      typename AMatrix::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      typename AMatrix::const_size_type>;

  using B_Internal = Kokkos::View<
      typename BType::const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<BType>::array_layout,
      typename BType::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

  using X_Internal = Kokkos::View<
      typename XType::non_const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<XType>::array_layout,
      typename XType::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

  using Precond_Internal = Preconditioner<AMatrix_Internal>;

  AMatrix_Internal A_i = A;
  B_Internal b_i       = B;
  X_Internal x_i       = X;

  Precond_Internal* precond_i = reinterpret_cast<Precond_Internal*>(precond);

  KokkosSparse::Impl::GMRES_NUMERIC<
      const_handle_type, typename AMatrix_Internal::value_type,
      typename AMatrix_Internal::ordinal_type,
      typename AMatrix_Internal::device_type,
      typename AMatrix_Internal::memory_traits,
      typename AMatrix_Internal::size_type, B_Internal,
    X_Internal>::gmres_numeric(&tmp_handle, A_i, b_i, x_i, precond_i);

}  // gmres_numeric

}  // namespace Experimental
}  // namespace KokkosSparse

#undef KOKKOSKERNELS_GMRES_SAME_TYPE

#endif  // KOKKOSSPARSE_GMRES_HPP_
