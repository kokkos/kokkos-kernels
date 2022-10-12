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

#ifndef KOKKOSBLAS1_ROT_HPP_
#define KOKKOSBLAS1_ROT_HPP_

#include <KokkosBlas1_rot_spec.hpp>

namespace KokkosBlas {

template <class execution_space, class Vector, class Scalar>
void rot(execution_space const& space, Vector const& X, Vector const& Y,
         Scalar const& c, Scalar const& s) {
  static_assert(Kokkos::is_execution_space<execution_space>::value,
                "rot: execution_space template parameter is not a Kokkos "
                "execution space.");
  static_assert(Vector::rank == 1,
                "rot: Vector template parameter needs to be a rank 1 view");
  static_assert(Scalar::rank == 0,
                "rot: Scalar template parameter needs to be a rank 0 view");
  static_assert(
      Kokkos::SpaceAccessibility<execution_space,
                                 typename Vector::memory_space>::accessible,
      "rot: Vector template parameter memory space needs to be accessible from "
      "execution_space template parameter");
  static_assert(
      Kokkos::SpaceAccessibility<execution_space,
                                 typename Scalar::memory_space>::accessible,
      "rot: Vector template parameter memory space needs to be accessible from "
      "execution_space template parameter");
  static_assert(
      std::is_same<typename Vector::non_const_value_type,
                   typename Vector::value_type>::value,
      "rot: Vector template parameter needs to store non-const values");

  using Vector_Internal = Kokkos::View<
      typename Vector::non_const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<Vector>::array_layout,
      Kokkos::Device<execution_space, typename Vector::memory_space>,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using Scalar_Internal = Kokkos::View<
      typename Scalar::non_const_value_type,
      typename KokkosKernels::Impl::GetUnifiedLayout<Scalar>::array_layout,
      Kokkos::Device<execution_space, typename Scalar::memory_space>,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  Vector_Internal X_(X), Y_(Y);
  Scalar_Internal c_(c), s_(s);

  Kokkos::Profiling::pushRegion("KokkosBlas::rot");
  Impl::Rot<execution_space, Vector_Internal, Scalar_Internal>::rot(space, X_,
                                                                    Y_, c_, s_);
  Kokkos::Profiling::popRegion();
}

template <class Vector, class Scalar>
void rot(Vector const& X, Vector const& Y, Scalar const& c, Scalar const& s) {
  const typename Vector::execution_space space =
      typename Vector::execution_space();
  rot(space, X, Y, c, s);
}

}  // namespace KokkosBlas
#endif  // KOKKOSBLAS1_ROT_HPP_
