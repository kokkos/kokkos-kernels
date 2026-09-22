// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSLAPACK_IMPL_GETRS_HPP_
#define KOKKOSLAPACK_IMPL_GETRS_HPP_

/// \file KokkosLapack_getrs_impl.hpp
/// \brief Implementation(s) of solve using LU factors.

#include <KokkosKernels_config.h>
#include <KokkosKernels_ArithTraits.hpp>
#include <KokkosBlas3_trsm.hpp>

namespace KokkosLapack {
namespace Impl {

template <class IpivView, class BMatrix>
struct laswp_functor {
  IpivView m_Ipiv;
  BMatrix m_B;

  laswp_functor(const IpivView& Ipiv, const BMatrix& B) : m_Ipiv(Ipiv), m_B(B) {}

  void KOKKOS_FUNCTION operator()(const int rowIdx) const {
    const int piv = m_Ipiv(rowIdx);
    typename BMatrix::non_const_value_type tmp;
    for (int colIdx = 0; colIdx < m_B.extent_int(1); ++colIdx) {
      tmp                 = m_B(rowIdx, colIdx);
      m_B(rowIdx, colIdx) = m_B(piv, colIdx);
      m_B(piv, colIdx)    = tmp;
    }
  }
};

template <class ExecutionSpace, class AMatrix, class IpivView, class BMatrix, class InfoView>
void getrs_impl(const ExecutionSpace& space, const char trans[], const AMatrix& A, const IpivView& Ipiv,
                const BMatrix& B, const InfoView& /* Info */) {
  auto one = KokkosKernels::ArithTraits<typename AMatrix::non_const_value_type>::one();

  laswp_functor swapper(Ipiv, B);
  if (trans[0] == 'N' || trans[0] == 'n') {
    Kokkos::parallel_for(Kokkos::RangePolicy(space, 0, B.extent(0)), swapper);
    KokkosBlas::trsm(space, "L", "L", "N", "U", one, A, B);
    KokkosBlas::trsm(space, "L", "U", "N", "N", one, A, B);
  } else {
    KokkosBlas::trsm(space, "L", "U", trans, "N", one, A, B);
    KokkosBlas::trsm(space, "L", "L", trans, "U", one, A, B);
    Kokkos::parallel_for(Kokkos::RangePolicy(space, 0, B.extent(0)), swapper);
  }
}

}  // namespace Impl
}  // namespace KokkosLapack

#endif  // KOKKOSLAPACK_IMPL_GETRS_HPP_
