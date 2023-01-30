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
// Questions? Contact Jennifer Loe (jloe@sandia.gov)
//
// ************************************************************************
//@HEADER
*/
/// @file KokkosKernels_LUPrec.hpp

#ifndef KK_LU_PREC_HPP
#define KK_LU_PREC_HPP

#include <KokkosSparse_Preconditioner.hpp>
#include <Kokkos_Core.hpp>
#include <KokkosBlas.hpp>
#include <KokkosSparse_spmv.hpp>
#include <KokkosBlas3_trsm_impl.hpp>

namespace KokkosSparse {

namespace Experimental {

/// \class LUPrec
/// \brief  This class is for applying LU preconditioning.
///         It takes L and U and the apply method returns U^inv L^inv x
/// \tparam CRS the CRS type of L and U
///
/// Preconditioner provides the following methods
///   - initialize() Does nothing; members initialized upon object construction.
///   - isInitialized() returns true
///   - compute() Does nothing; members initialized upon object construction.
///   - isComputed() returns true
///
template <class CRS>
class LUPrec : public KokkosSparse::Experimental::Preconditioner<CRS> {
 public:
  using ScalarType = typename std::remove_const<typename CRS::value_type>::type;
  using EXSP       = typename CRS::execution_space;
  using karith     = typename Kokkos::ArithTraits<ScalarType>;

 private:
  Kokkos::View<ScalarType**> _L, _U, _tmp;

 public:
  //! Constructor:
  template <class ViewArg>
  LUPrec(const ViewArg &L, const ViewArg &U) : _L(L), _U(U), _tmp("LUPrec::_tmp", _L.extent(0), 1) {}

  //! Destructor.
  virtual ~LUPrec() {}

  ///// \brief Apply the preconditioner to X, putting the result in Y.
  /////
  ///// \tparam XViewType Input vector, as a 1-D Kokkos::View
  ///// \tparam YViewType Output vector, as a nonconst 1-D Kokkos::View
  /////
  ///// \param transM [in] "N" for non-transpose, "T" for transpose, "C"
  /////   for conjugate transpose.  All characters after the first are
  /////   ignored.  This works just like the BLAS routines.
  ///// \param alpha [in] Input coefficient of M*x
  ///// \param beta [in] Input coefficient of Y
  /////
  ///// If the result of applying this preconditioner to a vector X is
  ///// \f$M \cdot X\f$, then this method computes \f$Y = \beta Y + \alpha M
  ///\cdot X\f$.
  ///// The typical case is \f$\beta = 0\f$ and \f$\alpha = 1\f$.
  //
  virtual void apply(const Kokkos::View<const ScalarType *, EXSP> &X,
                     const Kokkos::View<ScalarType *, EXSP> &Y,
                     const char transM[] = "N",
                     ScalarType alpha    = karith::one(),
                     ScalarType beta     = karith::zero()) const {


    // tmp = trsm(L, x); //Apply L^inv to x
    // y = trsm(U, tmp); //Apply U^inv to tmp
    auto tmpsv = Kokkos::subview(_tmp, Kokkos::ALL, 0);
    Kokkos::deep_copy(tmpsv, X);
    KokkosBlas::Impl::SerialTrsm_Invoke("L", "L", transM, "N", alpha, _L, _tmp);
    KokkosBlas::Impl::SerialTrsm_Invoke("L", "U", transM, "N", alpha, _U, _tmp);
    Kokkos::deep_copy(Y, tmpsv);
  }
  //@}

  //! Set this preconditioner's parameters.
  void setParameters() {}

  void initialize() {}

  //! True if the preconditioner has been successfully initialized, else false.
  bool isInitialized() const { return true; }

  void compute() {}

  //! True if the preconditioner has been successfully computed, else false.
  bool isComputed() const { return true; }

  //! True if the preconditioner implements a transpose operator apply.
  bool hasTransposeApply() const { return true; }
};
}  // namespace Experimental
}  // End namespace KokkosSparse

#endif
