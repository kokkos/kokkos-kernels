/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
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
#ifndef KOKKOSBLAS3_GEMV_HPP_
#define KOKKOSBLAS3_GEMV_HPP_

/// \file Kokkos_Blas3_MV.hpp
/// \brief BLAS 2 kernels specifically optimized for typical
///   Tpetra::MultiVector use cases.

#include <KokkosBlas3_gemm_spec.hpp>
#include <KokkosKernels_helpers.hpp>
#include <sstream>
#include <type_traits> // requires C++11, but so does Kokkos

namespace KokkosBlas {

/// \brief Dense matrix-vector multiply: y = beta*y + alpha*A*x.
///
/// \tparam AViewType Input matrix, as a 2-D Kokkos::View
/// \tparam XViewType Input vector, as a 1-D Kokkos::View
/// \tparam YViewType Output vector, as a nonconst 1-D Kokkos::View
/// \tparam AlphaCoeffType Type of input coefficient alpha
/// \tparam BetaCoeffType Type of input coefficient beta
///
/// \param trans [in] "N" for non-transpose, "T" for transpose, "C"
///   for conjugate transpose.  All characters after the first are
///   ignored.  This works just like the BLAS routines.
/// \param alpha [in] Input coefficient of A*x
/// \param A [in] Input matrix, as a 2-D Kokkos::View
/// \param x [in] Input vector, as a 1-D Kokkos::View
/// \param beta [in] Input coefficient of y
/// \param y [in/out] Output vector, as a nonconst 1-D Kokkos::View
template<class AViewType,
         class BViewType,
         class CViewType>
void
gemm (const char transA[],
      const char transB[],
      typename AViewType::const_value_type& alpha,
      const AViewType& A,
      const BViewType& B,
      typename CViewType::const_value_type& beta,
      const CViewType& C)
{
  static_assert (Kokkos::Impl::is_view<AViewType>::value,
                 "AViewType must be a Kokkos::View.");
  static_assert (Kokkos::Impl::is_view<BViewType>::value,
                 "BViewType must be a Kokkos::View.");
  static_assert (Kokkos::Impl::is_view<CViewType>::value,
                 "CViewType must be a Kokkos::View.");
  static_assert (static_cast<int> (AViewType::rank) == 2,
                 "AViewType must have rank 2.");
  static_assert (static_cast<int> (BViewType::rank) == 2,
                 "BViewType must have rank 2.");
  static_assert (static_cast<int> (CViewType::rank) == 2,
                 "CViewType must have rank 2.");

  // Check compatibility of dimensions at run time.
  /*if (transA[0] == 'N' || transA[0] == 'n') {
    if (A.dimension_0 () != y.dimension_0 () || A.dimension_1 () != x.dimension_0 ()) {
      std::ostringstream os;
      os << "KokkosBlas::gemv: Dimensions of A, x, and y do not match: "
         << "A: " << A.dimension_0 () << " x " << A.dimension_1 ()
         << ", x: " << x.dimension_0 () << ", y: " << y.dimension_0 ();
      Kokkos::Impl::throw_runtime_exception (os.str ());
    }
  }
  else if (trans[0] == 'T' || trans[0] == 't' ||
           trans[0] == 'C' || trans[0] == 'c' ||
           trans[0] == 'H' || trans[0] == 'h') {
    if (A.dimension_1 () != y.dimension_0 () || A.dimension_0 () != x.dimension_0 ()) {
      std::ostringstream os;
      os << "KokkosBlas::dot: Dimensions of A, x, and y do not match: "
         << "A: " << A.dimension_0 () << " x " << A.dimension_1 ()
         << ", x: " << x.dimension_0 () << ", y: " << y.dimension_0 ();
      Kokkos::Impl::throw_runtime_exception (os.str ());
    }
  }
  else {
    std::ostringstream os;
    os << "KokkosBlas::gemv: trans[0] = '" << trans[0] << "'.  Valid values "
      "include 'N' (No transpose), 'T' (Transpose), and 'C' (Conjugate "
      "transpose).";
    Kokkos::Impl::throw_runtime_exception (os.str ());
  }*/

  // Minimize the number of Impl::GEMV instantiations, by
  // standardizing on particular View specializations for its template
  // parameters.
  typedef Kokkos::View<typename AViewType::const_value_type**,
    typename AViewType::array_layout,
    typename AViewType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged> > AVT;
  typedef Kokkos::View<typename BViewType::const_value_type**,
    typename BViewType::array_layout,
    typename BViewType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged> > BVT;
  typedef Kokkos::View<typename CViewType::non_const_value_type**,
    typename CViewType::array_layout,
    typename CViewType::device_type,
    Kokkos::MemoryTraits<Kokkos::Unmanaged> > CVT;
  typedef Impl::GEMM<AVT, BVT, CVT> impl_type;
  impl_type::gemm (transA, transB, alpha, A, B, beta, C);
}

} // namespace KokkosBlas

#endif // KOKKOS_BLAS3_MV_HPP_
