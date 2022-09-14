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
#ifndef KOKKOSBLAS1_ROTG_IMPL_HPP_
#define KOKKOSBLAS1_ROTG_IMPL_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <KokkosBlas1_rotg_spec.hpp>

namespace KokkosBlas {
namespace Impl {

/// \brief Compute Givens rotation coefficients.
template <class Scalar,
          typename std::enable_if<!Kokkos::ArithTraits<Scalar>::is_complex,
                                  bool>::type = true>
void Rotg_Invoke(Scalar& a, Scalar& b, Scalar& c, Scalar& s) {
  const Scalar one  = Kokkos::ArithTraits<Scalar>::one();
  const Scalar zero = Kokkos::ArithTraits<Scalar>::zero();

  const Scalar numerical_scaling = Kokkos::abs(a) + Kokkos::abs(b);
  if (numerical_scaling == zero) {
    c = one;
    s = zero;
    a = zero;
    b = zero;
  } else {
    const Scalar scaled_a = a / numerical_scaling;
    const Scalar scaled_b = b / numerical_scaling;
    Scalar norm = Kokkos::sqrt(scaled_a * scaled_a + scaled_b * scaled_b) *
                  numerical_scaling;
    Scalar sign = Kokkos::abs(a) > Kokkos::abs(b) ? a : b;
    norm        = Kokkos::copysign(norm, sign);
    c           = a / norm;
    s           = b / norm;

    Scalar z = one;
    if (Kokkos::abs(a) > Kokkos::abs(b)) {
      z = s;
    }
    if ((Kokkos::abs(b) >= Kokkos::abs(a)) && (c != zero)) {
      z = one / c;
    }
    a = norm;
    b = z;
  }
}

template <class Scalar,
          typename std::enable_if<Kokkos::ArithTraits<Scalar>::is_complex,
                                  bool>::type = true>
void Rotg_Invoke(Scalar& a, Scalar& b,
                 typename Kokkos::ArithTraits<Scalar>::mag_type& c, Scalar& s) {
  using mag_type = typename Kokkos::ArithTraits<Scalar>::mag_type;

  const Scalar one        = Kokkos::ArithTraits<Scalar>::one();
  const Scalar zero       = Kokkos::ArithTraits<Scalar>::zero();
  const mag_type mag_zero = Kokkos::ArithTraits<mag_type>::zero();

  const mag_type numerical_scaling = Kokkos::abs(a) + Kokkos::abs(b);
  if (Kokkos::abs(a) == zero) {
    c = mag_zero;
    s = one;
    a = b;
  } else {
    const Scalar scaled_a = Kokkos::abs(a / numerical_scaling);
    const Scalar scaled_b = Kokkos::abs(b / numerical_scaling);
    mag_type norm = Kokkos::sqrt(scaled_a * scaled_a + scaled_b * scaled_b) *
                    numerical_scaling;
    Scalar unit_a = a / Kokkos::abs(a);
    c             = Kokkos::abs(a) / norm;
    s             = unit_a * Kokkos::conj(b) / norm;
    a             = unit_a * norm;
  }
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS1_ROTG_IMPL_HPP_
