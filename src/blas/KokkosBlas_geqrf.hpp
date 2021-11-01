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

#ifndef KOKKOSBLAS_GEQRF_HPP_
#define KOKKOSBLAS_GEQRF_HPP_

/// \file KokkosBlas_qeqrf.hpp

#include "KokkosKernels_Macros.hpp"
#include "KokkosBlas_geqrf_spec.hpp"
#include "KokkosKernels_helpers.hpp"
#include <sstream>
#include <type_traits>

namespace KokkosBlas {

/// \brief Compute the QR factorization of M x N matrix A. (geqrf)

/// \tparam AViewType Input(A) / Output (Solution) M x N matrix       , as a 2-D
/// Kokkos::View \tparam TauViewType Input k vector     , as a 1-D Kokkos::View
/// \tparam WViewType Input Workspace, as a 1-D Kokkos::View
///
/// \param A [in, out]     Input matrix, as a 2-D Kokkos::View
///                   On entry, M-by-N matrix
///                   On exit, overwritten with the solution.
/// \param tau [in] Input vector, as a 1-D Kokkos::View. Scalar factors of
/// reflectors. \param workspace [in] Input vector, as a 1-D Kokkos::View.
/// Scratchspace for calculations.

template <class AViewType, class TauViewType, class WViewType>
void geqrf(AViewType& A, TauViewType& tau, WViewType& workspace) {
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  static_assert(Kokkos::Impl::is_view<AViewType>::value,
                "KokkosBlas::geqrf: A must be a Kokkos::View");
  static_assert(Kokkos::Impl::is_view<TauViewType>::value,
                "KokkosBlas::geqrf: tau must be a Kokkos::View");
  static_assert(Kokkos::Impl::is_view<WViewType>::value,
                "KokkosBlas::geqrf: workspace must be a Kokkos::View");

  static_assert(static_cast<int>(AViewType::rank) == 2,
                "KokkosBlas::geqrf: A must have rank 2");
  static_assert(static_cast<int>(TauViewType::rank) == 1,
                "KokkosBlas::geqrf: Tau must have rank 1");
  static_assert(static_cast<int>(WViewType::rank) == 1,
                "KokkosBlas::geqrf: Workspace must have rank 1");

  int64_t A0    = A.extent(0);  // M
  int64_t A1    = A.extent(1);  // N
  int64_t minmn = (A0 < A1) ? A0 : A1;

  int64_t tau0  = tau.extent(0);
  int64_t lwork = workspace.extent(0);

  // Check validity of Tau
  if (tau0 < minmn) {
    std::ostringstream os;
    os << "KokkosBlas::geqrf: Dimensions of tau and MIN(M, N) do not match "
          "(require len(tau) >= min(M, N) ): "
       << "min(M, N): " << minmn << "Tau: " << tau0;
    Kokkos::Impl::throw_runtime_exception(os.str());
  }

#endif  // KOKKOSKERNELS_DEBUG_LEVEL > 0

  // return if degenerate matrix provided
  if ((A.extent(0) == 0) || (A.extent(1) == 0)) return;

  // standardize particular View specializations
  typedef Kokkos::View<typename AViewType::non_const_value_type**,
                       typename AViewType::array_layout,
                       typename AViewType::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      AVT;

  typedef Kokkos::View<typename TauViewType::non_const_value_type*,
                       typename TauViewType::array_layout,
                       typename TauViewType::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      TVT;

  typedef Kokkos::View<typename WViewType::non_const_value_type*,
                       typename TauViewType::array_layout,
                       typename TauViewType::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      WVT;

  AVT A_i   = A;
  TVT tau_i = tau;
  WVT W_i   = workspace;

  typedef KokkosBlas::Impl::GEQRF<AVT, TVT, WVT> impl_type;
  impl_type::geqrf(A_i, tau_i, W_i);

}  // function geqrf

template <class AViewType, class TauViewType>
int64_t geqrf_workspace(AViewType& A, TauViewType& tau) {
  // return if degenerate matrix provided
  if ((A.extent(0) == 0) || (A.extent(1) == 0)) return 0;

  // standardize particular View specializations
  typedef Kokkos::View<typename AViewType::non_const_value_type**,
                       typename AViewType::array_layout,
                       typename AViewType::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      AVT;

  typedef Kokkos::View<typename TauViewType::non_const_value_type*,
                       typename TauViewType::array_layout,
                       typename TauViewType::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      TVT;

  AVT A_i   = A;
  TVT tau_i = tau;

  typedef KokkosBlas::Impl::GEQRF_WORKSPACE<AVT, TVT> impl_type;
  return impl_type::geqrf_workspace(A_i, tau_i);

}  // function geqrf_workspace

template <class AViewType, class TauViewType>
void geqrf(AViewType& A, TauViewType& tau) {
  int64_t lwork = geqrf_workspace(A, tau);
  TauViewType workspace("KokkosBlas::temporary_geqrf_workspace", lwork);
  geqrf(A, tau, workspace);

}  // function geqrf with temp workspace

}  // namespace KokkosBlas

#endif  // KOKKOSBLAS_GEQRF_HPP_
