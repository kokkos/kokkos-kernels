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

#ifndef KOKKOSBLAS1_ROTMG_HPP_
#define KOKKOSBLAS1_ROTMG_HPP_

#include <Kokkos_Core.hpp>
#include <KokkosBlas1_rotmg_spec.hpp>

namespace KokkosBlas {

/// \brief Compute the coefficients to apply a modified Givens rotation.
///
/// \tparam execution_space the execution space where the kernel will be
/// executed \tparam DXView a rank0 view type that hold non const data \tparam
/// YView a rank0 view type that holds const data \tparam PView a rank1 view of
/// static extent 5 that holds non const data
///
/// \param d1 [in/out]
/// \param d2 [in/out]
/// \param x1 [in/out]
/// \param y1 [in]
/// \param param [out]
///
template <class execution_space, class DXView, class YView, class PView>
void rotmg(execution_space const& space, DXView const& d1, DXView const& d2,
           DXView const& x1, YView const& y1, PView const& param) {
  static_assert(
      Kokkos::SpaceAccessibility<execution_space,
                                 typename DXView::memory_space>::accessible,
      "rotmg: execution_space cannot access data in DXView");

  using DXView_Internal = Kokkos::View<
      typename DXView::value_type,
      typename KokkosKernels::Impl::GetUnifiedLayout<DXView>::array_layout,
      Kokkos::Device<execution_space, typename DXView::memory_space>,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using YView_Internal = Kokkos::View<
      typename YView::value_type,
      typename KokkosKernels::Impl::GetUnifiedLayout<YView>::array_layout,
      Kokkos::Device<execution_space, typename YView::memory_space>,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using PView_Internal = Kokkos::View<
      typename PView::value_type[5],
      typename KokkosKernels::Impl::GetUnifiedLayout<PView>::array_layout,
      Kokkos::Device<execution_space, typename PView::memory_space>,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  DXView_Internal d1_(d1), d2_(d2), x1_(x1);
  YView_Internal y1_(y1);
  PView_Internal param_(param);

  Kokkos::Profiling::pushRegion("KokkosBlas::rotmg");
  Impl::Rotmg<execution_space, DXView_Internal, YView_Internal,
              PView_Internal>::rotmg(space, d1_, d2_, x1_, y1_, param_);
  Kokkos::Profiling::popRegion();
}

template <class DXView, class YView, class PView>
void rotmg(DXView const& d1, DXView const& d2, DXView const& x1,
           YView const& y1, PView const& param) {
  const typename PView::execution_space space =
      typename PView::execution_space();
  rotmg(space, d1, d2, x1, y1, param);
}

}  // namespace KokkosBlas

#endif  // KOKKOSBLAS1_ROTMG_HPP_
