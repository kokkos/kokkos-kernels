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
#ifndef KOKKOSBLAS1_SWAP_IMPL_HPP_
#define KOKKOSBLAS1_SWAP_IMPL_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
// #include <KokkosBlas1_rot_spec.hpp>

namespace KokkosBlas {
namespace Impl {

template <class XVector, class YVector>
struct swap_functor {
  using scalar_type = typename XVector::non_const_value_type;

  XVector X;
  YVector Y;

  swap_functor(XVector const& X_, YVector const& Y_) : X(X_), Y(Y_) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(int const entryIdx) const {
    scalar_type const temp = Y(entryIdx);
    Y(entryIdx)            = X(entryIdx);
    X(entryIdx)            = temp;
  }
};

template <class ExecutionSpace, class XVector, class YVector>
void Swap_Invoke(ExecutionSpace const& space, XVector const& X,
                 YVector const& Y) {
  Kokkos::RangePolicy<ExecutionSpace> swap_policy(space, 0, X.extent(0));
  swap_functor swap_func(X, Y);
  Kokkos::parallel_for("KokkosBlas::swap", swap_policy, swap_func);
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS1_SWAP_IMPL_HPP_
