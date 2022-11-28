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
/// GMRES - A Generalized Minimal Residual Algorithm for Solving Nonsymmetric Linear Systems - Saad, Schultz

#ifndef KOKKOSSPARSE_GMRES_HPP_
#define KOKKOSSPARSE_GMRES_HPP_

#include <type_traits>

#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Error.hpp"
#include "KokkosSparse_gmres_numeric_spec.hpp"

namespace KokkosSparse {
namespace Experimental {

#define KOKKOSKERNELS_GMRES_SAME_TYPE(A, B)      \
  std::is_same<typename std::remove_const<A>::type, \
               typename std::remove_const<B>::type>::value

template <typename KernelHandle,
          typename ARowMapType, typename AEntriesType, typename AValuesType,
          typename BType, typename XType>
void gmres_numeric(KernelHandle* handle, ARowMapType& A_rowmap,
                   AEntriesType& A_entries, AValuesType& A_values,
                   BType& B, XType& X) {
  using size_type    = typename KernelHandle::size_type;
  using ordinal_type = typename KernelHandle::nnz_lno_t;
  using scalar_type  = typename KernelHandle::nnz_scalar_t;

  static_assert(
      KOKKOSKERNELS_GMRES_SAME_TYPE(
          typename ARowMapType::non_const_value_type, size_type),
      "gmres_numeric: A size_type must match KernelHandle size_type "
      "(const doesn't matter)");
  static_assert(KOKKOSKERNELS_GMRES_SAME_TYPE(
                  typename AEntriesType::non_const_value_type, ordinal_type),
                "gmres_numeric: A entry type must match KernelHandle entry "
                "type (aka nnz_lno_t, and const doesn't matter)");
  static_assert(KOKKOSKERNELS_GMRES_SAME_TYPE(
                  typename AValuesType::value_type, scalar_type),
                "gmres_numeric: A scalar type must match KernelHandle entry "
                "type (aka nnz_scalar_t, and const doesn't matter)");

  static_assert(KOKKOSKERNELS_GMRES_SAME_TYPE(
                  typename BType::value_type, scalar_type),
                "gmres_numeric: B scalar type must match KernelHandle entry "
                "type (aka nnz_scalar_t, and const doesn't matter)");

  static_assert(KOKKOSKERNELS_GMRES_SAME_TYPE(
                  typename XType::value_type, scalar_type),
                "gmres_numeric: X scalar type must match KernelHandle entry "
                "type (aka nnz_scalar_t, and const doesn't matter)");

  static_assert(Kokkos::is_view<ARowMapType>::value,
                "gmres_numeric: A_rowmap is not a Kokkos::View.");
  static_assert(Kokkos::is_view<AEntriesType>::value,
                "gmres_numeric: A_entries is not a Kokkos::View.");
  static_assert(Kokkos::is_view<AValuesType>::value,
                "gmres_numeric: A_values is not a Kokkos::View.");
  static_assert(Kokkos::is_view<BType>::value,
                "gmres_numeric: B is not a Kokkos::View.");
  static_assert(Kokkos::is_view<XType>::value,
                "gmres_numeric: X is not a Kokkos::View.");

  static_assert(
      (int)BType::rank == (int)ARowMapType::rank,
      "gmres_numeric: The ranks of B and A_rowmap do not match.");
  static_assert(
      (int)XType::rank == (int)AEntriesType::rank,
      "gmres_numeric: The ranks of X and A_entries do not match.");

  static_assert(ARowMapType::rank == 1,
                "gmres_numeric: A_rowmap must have rank 1.");
  static_assert(AEntriesType::rank == 1,
                "gmres_numeric: A_entries must have rank 1.");
  static_assert(AValuesType::rank == 1,
                "gmres_numeric: A_values must have rank 1.");

  static_assert(
      std::is_same<typename XType::value_type,
                   typename XType::non_const_value_type>::value,
      "gmres_numeric: The output X must be nonconst.");

  static_assert(
      std::is_same<
          typename ARowMapType::device_type::execution_space,
          typename KernelHandle::GMRESHandleType::execution_space>::value,
      "gmres_numeric: KernelHandle and Views have different execution "
      "spaces.");
  static_assert(
      std::is_same<
          typename AEntriesType::device_type::execution_space,
          typename KernelHandle::GMRESHandleType::execution_space>::value,
      "gmres_numeric: KernelHandle and Views have different execution "
      "spaces.");
  static_assert(
      std::is_same<
          typename AValuesType::device_type::execution_space,
          typename KernelHandle::GMRESHandleType::execution_space>::value,
      "gmres_numeric: KernelHandle and Views have different execution "
      "spaces.");

  static_assert(
      std::is_same<typename ARowMapType::device_type,
                   typename BType::device_type>::value,
      "gmres_numeric: rowmap and B have different device types.");
  static_assert(
      std::is_same<typename ARowMapType::device_type,
                   typename XType::device_type>::value,
      "gmres_numeric: rowmap and X have different device types.");

  using c_size_t   = typename KernelHandle::const_size_type;
  using c_lno_t    = typename KernelHandle::const_nnz_lno_t;
  using c_scalar_t = typename KernelHandle::const_nnz_scalar_t;

  using c_exec_t    = typename KernelHandle::HandleExecSpace;
  using c_temp_t    = typename KernelHandle::HandleTempMemorySpace;
  using c_persist_t = typename KernelHandle::HandlePersistentMemorySpace;

  using const_handle_type =
      typename KokkosKernels::Experimental::KokkosKernelsHandle<
          c_size_t, c_lno_t, c_scalar_t, c_exec_t, c_temp_t, c_persist_t>;

  const_handle_type tmp_handle(*handle);

  using ARowMap_Internal = Kokkos::View<
      typename ARowMapType::const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<ARowMapType>::array_layout,
      typename ARowMapType::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

  using AEntries_Internal = Kokkos::View<
      typename AEntriesType::const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<
          AEntriesType>::array_layout,
      typename AEntriesType::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

  using AValues_Internal = Kokkos::View<
      typename AValuesType::const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<AValuesType>::array_layout,
      typename AValuesType::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

  using B_Internal = Kokkos::View<
      typename BType::non_const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<BType>::array_layout,
      typename BType::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >;

  using X_Internal = Kokkos::View<
    typename XType::non_const_value_type*,
    typename KokkosKernels::Impl::GetUnifiedLayout<XType>::array_layout,
    typename XType::device_type,
    Kokkos::MemoryTraits<Kokkos::RandomAccess> >;

  ARowMap_Internal  A_rowmap_i  = A_rowmap;
  AEntries_Internal A_entries_i = A_entries;
  AValues_Internal  A_values_i  = A_values;
  B_Internal        b_i         = B;
  X_Internal        x_i         = X;

  KokkosSparse::Impl::GMRES_NUMERIC<
      const_handle_type, ARowMap_Internal, AEntries_Internal, AValues_Internal,
      B_Internal, X_Internal>::gmres_numeric(&tmp_handle, A_rowmap_i, A_entries_i,
                                             A_values_i, b_i, x_i);

}  // gmres_numeric

}  // namespace Experimental
}  // namespace KokkosSparse

#undef KOKKOSKERNELS_GMRES_SAME_TYPE

#endif  // KOKKOSSPARSE_GMRES_HPP_
