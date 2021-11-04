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

#ifndef KOKKOSBLAS_UNMQR_HPP_
#define KOKKOSBLAS_UNMQR_HPP_

/// \file KokkosBlas_unmqr.hpp

#include "KokkosKernels_Macros.hpp"
#include "KokkosBlas_unmqr_spec.hpp"
#include "KokkosKernels_helpers.hpp"
#include <sstream>
#include <type_traits>

namespace KokkosBlas {

/// \brief Multiply rectangular matrix C by Q or Q^H (where Q is the unitary
/// output of QR by geqrf or geqp3)

/// \tparam AViewType Input matrix M-by-k matrix       , as a 2-D Kokkos::View
/// \tparam CViewType Input (RHS)/Output (Solution) M-by-N matrix, as a 2-D
/// Kokkos::View \tparam TauViewType Input k vector     , as a 1-D Kokkos::View
/// \tparam WViewType Input Workspace, as a 1-D Kokkos::View
///
/// \param side [in] "L" or "l" indicates matrix Q is applied on the left of C
///                   "R" or "r" indicates matrix Q is applied on the right of C
/// \param transpose [in] Specifies what op does to Q:
//                    "N" or "n" for non-transpose,
//                    "T" or "t" for transpose
/// \param k [in]     Number of elementary reflectors that define Q
/// \param A [in]     Input matrix, as a 2-D Kokkos::View, output of geqrf or
/// geqp3.
/// \param tau [in] Input vector, as a 1-D Kokkos::View. Scalar factors
/// of reflectors.
/// \param C [in,out] Input/Output matrix, as a 2-D Kokkos::View
///                   On entry, M-by-N matrix
///                   On exit, overwritten with the solution.
/// \param workspace [in] Input vector, as a 1-D Kokkos::View. Scratchspace for
/// calculations.

template <class AViewType, class TauViewType, class CViewType, class WViewType>
void unmqr(const char side[], const char trans[], int k, AViewType& A,
           TauViewType& tau, CViewType& C, WViewType& workspace) {
#if (KOKKOSKERNELS_DEBUG_LEVEL > 0)
  static_assert(Kokkos::Impl::is_view<AViewType>::value,
                "KokkosBlas::unmqr: A must be a Kokkos::View");
  static_assert(Kokkos::Impl::is_view<TauViewType>::value,
                "KokkosBlas::unmqr: tau must be a Kokkos::View");
  static_assert(Kokkos::Impl::is_view<CViewType>::value,
                "KokkosBlas::unmqr: C must be a Kokkos::View");
  static_assert(Kokkos::Impl::is_view<WViewType>::value,
                "KokkosBlas::unmqr: workspace must be a Kokkos::View");

  static_assert(static_cast<int>(AViewType::rank) == 2,
                "KokkosBlas::unmqr: A must have rank 2");
  static_assert(static_cast<int>(TauViewType::rank) == 1,
                "KokkosBlas::unmqr: Tau must have rank 1");
  static_assert(static_cast<int>(CViewType::rank) == 2,
                "KokkosBlas::unmqr: C must have rank 2");
  static_assert(static_cast<int>(WViewType::rank) == 1,
                "KokkosBlas::unmqr: Workspace must have rank 1");

  // Check validity of side argument
  bool valid_side = (side[0] == 'L') || (side[0] == 'l') || (side[0] == 'R') ||
                    (side[0] == 'r');

  bool valid_trans = (trans[0] == 'T') || (trans[0] == 't') ||
                     (trans[0] == 'C') || (trans[0] == 'c') ||
                     (trans[0] == 'N') || (trans[0] == 'n');

  if (!(valid_side)) {
    std::ostringstream os;
    os << "KokkosBlas::unmqr: side[0] = '" << side[0] << "'. "
       << "Valid values include 'L' or 'l' (Left), 'R' or 'r' (Right).";
    Kokkos::Impl::throw_runtime_exception(os.str());
  }
  if (!(valid_trans)) {
    std::ostringstream os;
    os << "KokkosBlas::unmqr: trans[0] = '" << trans[0] << "'. "
       << "Valid values include 'T' or 't' (Transpose), 'N' or 'n' (No "
          "transpose).";
    Kokkos::Impl::throw_runtime_exception(os.str());
  }

  int64_t A0   = A.extent(0);  // M if 'L', N if 'R'
  int64_t A1   = A.extent(1);  // > k
  int64_t C0   = C.extent(0);  // M
  int64_t C1   = C.extent(1);  // N
  int64_t tau0 = tau.extent(0);

  // Check validity of Tau
  if (tau0 < k) {
    std::ostringstream os;
    os << "KokkosBlas::unmqr: Dimensions of tau and k do not match (require "
          "len(tau) >=k ): "
       << "k: " << k << "Tau: " << tau0;
    Kokkos::Impl::throw_runtime_exception(os.str());
  }

  // Check validity of k
  if ((side[0] == 'L') || (side[0] == 'l')) {
    if ((k > C0) || (k < 0)) {
      std::ostringstream os;
      os << "KokkosBlas::unmqr: Number of reflectors k must not exceed M. "
         << "M: " << C0 << " "
         << "k: " << k;
      Kokkos::Impl::throw_runtime_exception(os.str());
    }
    if ((A0 != C0)) {
      std::ostringstream os;
      os << "KokkosBlas::unmqr: A must be of size M x k: "
         << "A: " << A0 << " x " << A1 << " "
         << "M: " << C0;
      Kokkos::Impl::throw_runtime_exception(os.str());
    }
  } else {
    if ((k > C1) || (k < 0)) {
      std::ostringstream os;
      os << "KokkosBlas::unmqr: Number of reflectors k must not exceed N. "
         << "N: " << C1 << " "
         << "k: " << k;
      Kokkos::Impl::throw_runtime_exception(os.str());
    }
    if ((A0 != C1)) {
      std::ostringstream os;
      os << "KokkosBlas::unmqr: A must be of size N x k: "
         << "A: " << A0 << " x " << A1 << " "
         << "N: " << C1;
      Kokkos::Impl::throw_runtime_exception(os.str());
    }
  }
#endif  // KOKKOSKERNELS_DEBUG_LEVEL > 0

  // return if degenerate matrix provided
  if ((A.extent(0) == 0) || (A.extent(1) == 0)) return;
  if ((C.extent(0) == 0) || (C.extent(1) == 0)) return;
  if ((k == 0)) return;

  // standardize particular View specializations
  typedef Kokkos::View<
      typename AViewType::const_value_type**, typename AViewType::array_layout,
      typename AViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      AVT;

  typedef Kokkos::View<typename TauViewType::const_value_type*,
                       typename TauViewType::array_layout,
                       typename TauViewType::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      TVT;

  typedef Kokkos::View<typename CViewType::non_const_value_type**,
                       typename CViewType::array_layout,
                       typename CViewType::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      CVT;

  typedef Kokkos::View<typename WViewType::non_const_value_type*,
                       typename TauViewType::array_layout,
                       typename TauViewType::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      WVT;

  AVT A_i   = A;
  TVT tau_i = tau;
  CVT C_i   = C;
  WVT W_i   = workspace;

  typedef KokkosBlas::Impl::UNMQR<AVT, TVT, CVT, WVT> impl_type;
  impl_type::unmqr(side[0], trans[0], k, A_i, tau_i, C_i, W_i);

}  // function unmqr

/// \brief Returns the length of workspace needed for unmqr (Multiply
/// rectangular matrix C by Q or Q^H (where Q is the unitary output of QR by
/// geqrf or geqp3)).

/// \tparam AViewType Input matrix M-by-k matrix       , as a 2-D Kokkos::View
/// \tparam CViewType Input (RHS)/Output (Solution) M-by-N matrix, as a 2-D
/// Kokkos::View \tparam TauViewType Input k vector     , as a 1-D Kokkos::View

/// \return int64_t length of required workspace
/// \param side [in] "L" or "l" indicates matrix Q is applied on the left of C
///                   "R" or "r" indicates matrix Q is applied on the right of C
/// \param transpose [in] Specifies what op does to Q:
//                    "N" or "n" for non-transpose,
//                    "T" or "t" for transpose
/// \param k [in]     Number of elementary reflectors that define Q
/// \param A [in]     Input matrix, as a 2-D Kokkos::View, output of geqrf or
/// geqp3. Can be empty for workspace queries, just needs to be the correct
/// size.
///\param tau [in] Input vector, as a 1-D Kokkos::View. Scalar factors
/// of reflectors. Can be empty for workspace queries.
/// \param C [in] Input/Output unmqr matrix, as a 2-D Kokkos::View. Can be empty
/// for workspace queries, just needs to be the correct size.

template <class AViewType, class TauViewType, class CViewType>
int64_t unmqr_workspace(const char side[], const char trans[], int k,
                        AViewType& A, TauViewType& tau, CViewType& C) {
  // return if degenerate matrix provided
  if ((A.extent(0) == 0) || (A.extent(1) == 0)) return 0;
  if ((C.extent(0) == 0) || (C.extent(1) == 0)) return 0;
  if ((k == 0)) return 0;

  // standardize particular View specializations
  typedef Kokkos::View<
      typename AViewType::const_value_type**, typename AViewType::array_layout,
      typename AViewType::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      AVT;

  typedef Kokkos::View<typename TauViewType::const_value_type*,
                       typename TauViewType::array_layout,
                       typename TauViewType::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      TVT;

  typedef Kokkos::View<typename CViewType::non_const_value_type**,
                       typename CViewType::array_layout,
                       typename CViewType::device_type,
                       Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      CVT;

  AVT A_i   = A;
  TVT tau_i = tau;
  CVT C_i   = C;

  typedef KokkosBlas::Impl::UNMQR_WORKSPACE<AVT, TVT, CVT> impl_type;
  return impl_type::unmqr_workspace(side[0], trans[0], k, A_i, tau_i, C_i);

}  // function unmqr_workspace

/// \brief Multiply rectangular matrix C by Q or Q^H (where Q is the unitary
/// output of QR by geqrf or geqp3). Allocates a workspace internally.

/// \tparam AViewType Input matrix M-by-k matrix       , as a 2-D Kokkos::View
/// \tparam CViewType Input (RHS)/Output (Solution) M-by-N matrix, as a 2-D
/// Kokkos::View \tparam TauViewType Input k vector     , as a 1-D Kokkos::View
///
/// \param side [in] "L" or "l" indicates matrix Q is applied on the left of C
///                   "R" or "r" indicates matrix Q is applied on the right of C
/// \param transpose [in] Specifies what op does to Q:
//                    "N" or "n" for non-transpose,
//                    "T" or "t" for transpose
/// \param k [in]     Number of elementary reflectors that define Q
/// \param A [in]     Input matrix, as a 2-D Kokkos::View, output of geqrf or
/// geqp3. \param tau [in] Input vector, as a 1-D Kokkos::View. Scalar factors
/// of reflectors. \param C [in,out] Input/Output matrix, as a 2-D Kokkos::View
///                   On entry, M-by-N matrix
///                   On exit, overwritten with the solution.

template <class AViewType, class TauViewType, class CViewType>
void unmqr(const char side[], const char trans[], int k, AViewType& A,
           TauViewType& tau, CViewType& C) {
  int64_t lwork = unmqr_workspace(side, trans, k, A, tau, C);
  TauViewType workspace("KokkosBlas::temporary_geqrf_workspace", lwork);
  unmqr(side, trans, k, A, tau, C, workspace);
}  // function unmqr with temp workspace

}  // namespace KokkosBlas

#endif  // KOKKOSBLAS_UNMQR_HPP_