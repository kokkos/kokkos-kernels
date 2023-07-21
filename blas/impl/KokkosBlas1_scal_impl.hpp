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
#ifndef KOKKOSBLAS1_SCAL_IMPL_HPP_
#define KOKKOSBLAS1_SCAL_IMPL_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_InnerProductSpaceTraits.hpp>
#include <KokkosBlas1_scal_spec.hpp>
#include <KokkosKernels_AlwaysFalse.hpp>
#include <KokkosBlas1_scal_unified_scalar_view_impl.hpp>
#include <KokkosKernels_ScalarHint.hpp>

#ifndef KOKKOSBLAS_OPTIMIZATION_LEVEL_SCAL
#define KOKKOSBLAS_OPTIMIZATION_LEVEL_SCAL 2
#endif  // KOKKOSBLAS_OPTIMIZATION_LEVEL_SCAL

namespace KokkosBlas {
namespace Impl {


// Single-vector version of MV_Scal_Functor.  
// a has been unified into either a scalar or a 0D view

// 1-D View.  Below is a partial specialization that lets a be a
// scalar.  This functor computes any of the following:
//
// 1. Y(i) = alpha*X(i) for alpha in -1,0,1
// 2. Y(i) = a(0)*X(i)
//
// The template parameter scalar_x corresponds to alpha in the
// operation y = alpha*x + beta*y.  The values -1, 0, and -1
// correspond to literal values of this coefficient.  The value 2
// tells the functor to use the corresponding vector of coefficients.
// Any literal coefficient of zero has BLAS semantics of ignoring the
// corresponding (multi)vector entry.  This does not apply to
// coefficients in the a vector, if used.
template <class RV, class AV, class XV, KokkosKernels::Impl::ScalarHint ALPHA_HINT, class SizeType>
struct V_Scal_Functor {
  typedef SizeType size_type;
  typedef Kokkos::ArithTraits<typename RV::non_const_value_type> ATS;

  RV m_r;
  XV m_x;
  AV m_a;

  V_Scal_Functor(const RV& r, const XV& x, const AV& a)
      : m_r(r), m_x(x), m_a(a) {
    static_assert(Kokkos::is_view<RV>::value,
                  "V_Scal_Functor: RV is not a Kokkos::View.");

    // TODO: static assert truths about AV

    static_assert(Kokkos::is_view<XV>::value,
                  "V_Scal_Functor: XV is not a Kokkos::View.");
    static_assert(RV::rank == 1, "V_Scal_Functor: RV is not rank 1.");
    static_assert(XV::rank == 1, "V_Scal_Functor: XV is not rank 1.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const size_type& i) const {

    using ScalarHint = KokkosKernels::Impl::ScalarHint;

    // scalar_a is a compile-time constant (since it is a template
    // parameter), so the compiler should evaluate these branches at
    // compile time.
    if constexpr (ALPHA_HINT == ScalarHint::zero) {
      m_r(i) = ATS::zero();
    }
    else if constexpr (ALPHA_HINT == ScalarHint::neg_one) {
      m_r(i) = -m_x(i);
    }
    else if constexpr (ALPHA_HINT == ScalarHint::pos_one) {
      m_r(i) = m_x(i);
    }
    else if constexpr (ALPHA_HINT == ScalarHint::none) {
      m_r(i) = KokkosBlas::Impl::as_scalar(m_a) * m_x(i);
    }
    else {
      static_assert(KokkosKernels::Impl::always_false_v<AV>, "Unexpected value for ALPHA_HINT");
    }
  }
};

/*! \brief 

 r(i) = av * x(i)
 r(i) = av() * x(i)

 \param space
 \param r
 \param av
 \param x
 \param alphaHint A KokkosKernels::Impl::ScalarHint corresponding to the value of av. If not KokkosKernels::Impl:ß:ScalarHint::none, may be used to optimize the implementation

 \tparam SizeType
 \tparam ExecutionSpace
 \tparam RV
 \tparam AV
 \tparam XV

*/
template <typename SizeType, typename ExecutionSpace, typename RV, typename AV, typename XV>
void V_Scal_Generic(const ExecutionSpace& space, const RV& r, const AV& av,
                    const XV& x,
                    const KokkosKernels::Impl::ScalarHint &alphaHint = KokkosKernels::Impl::ScalarHint::none) {

  // TODO: assert some things about AV

  static_assert(Kokkos::is_view<RV>::value,
                "V_Scal_Generic: RV is not a Kokkos::View.");
  static_assert(Kokkos::is_view<XV>::value,
                "V_Scal_Generic: XV is not a Kokkos::View.");
  static_assert(RV::rank == 1, "V_Scal_Generic: RV is not rank 1.");
  static_assert(XV::rank == 1, "V_Scal_Generic: XV is not rank 1.");

  const SizeType numRows = x.extent(0);
  Kokkos::RangePolicy<ExecutionSpace, SizeType> policy(space, 0, numRows);

  if (alphaHint == KokkosKernels::Impl::ScalarHint::zero) {
    V_Scal_Functor<RV, AV, XV, KokkosKernels::Impl::ScalarHint::zero, SizeType> op(r, x, av);
    Kokkos::parallel_for("KokkosBlas::Scal::0", policy, op);
    return;
  }
  else if (alphaHint == KokkosKernels::Impl::ScalarHint::neg_one) {
    V_Scal_Functor<RV, AV, XV, KokkosKernels::Impl::ScalarHint::neg_one, SizeType> op(r, x, av);
    Kokkos::parallel_for("KokkosBlas::Scal::-1", policy, op);
    return;
  }
  else if (alphaHint == KokkosKernels::Impl::ScalarHint::pos_one) {
    V_Scal_Functor<RV, AV, XV, KokkosKernels::Impl::ScalarHint::pos_one, SizeType> op(r, x, av);
    Kokkos::parallel_for("KokkosBlas::Scal::1", policy, op);
    return;
  }

  V_Scal_Functor<RV, AV, XV, KokkosKernels::Impl::ScalarHint::none, SizeType> op(r, x, av);
  Kokkos::parallel_for("KokkosBlas::Scal::none", policy, op);
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS1_SCAL_IMPL_HPP_
