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

namespace KokkosBlas {
namespace Impl {

/// \brief Rotg: construct plane rotation such
/// that [ c  s] [a] = [r] with c**2 + s**2 = 1.
///      [-s  c] [b]   [0]
///
/// \tparam a input/output Scalar
/// \tparam b input/output Scalar
/// \tparam c output Scalar
/// \tparam s output Scalar
  template <class Scalar, class execution_space, class memory_space>
void Rotg_Invoke(execution_space space, Scalar& a, Scalar& b, Scalar& c, Scalar& s) {

  // For now we do not use the parameter
  // space but it should become more handy
  // once we write the stream interface...
  (void) space;
  const Scalar anorm = Kokkos::Experimental::fabs(a);
  const Scalar bnorm = Kokkos::Experimental::fabs(b);
  const Scalar zero  = Kokkos::reduction_identity<Scalar>::sum();
  const Scalar one   = Kokkos::reduction_identity<Scalar>::prod();

  if(b == zero) {
    c = one;
    s = zero;
    b = one;
  } else if (a == zero) {
    c = zero;
    s = one;
    a = b;
    b = one;
  } else {
    const Scalar scaling =
      Kokkos::Experimental::fmin(Kokkos::reduction_identity<Scalar>::max(),
				 Kokkos::Experimental::fmax(Kokkos::reduction_identity<Scalar>::min(),
							   Kokkos::Experimental::fmax(anorm, bnorm)));

    // Should ask for sign function
    // in Kokkos_MathematicalFunctions.hpp
    Scalar sigma;
    if(anorm > bnorm) {
      sigma = a / anorm;
    } else {
      sigma = b / bnorm;
    }
    const Scalar r = sigma*scaling*Kokkos::Experimental::sqrt(
							      Kokkos::Experimental::pow(a/scaling, 2)
							      +Kokkos::Experimental::pow(b/scaling, 2));
    c = a / r;
    s = b / r;
    Scalar z;
    if(anorm > bnorm) {
      z = s;
    } else if(c != zero) {
      z = one / c;
    } else {
      z = one;
    }
    a = r;
    b = z;
  }
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS1_IAMAX_IMPL_HPP_
