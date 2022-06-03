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

#ifndef KOKKOSBLAS2_SERIAL_GEMV_HPP_
#define KOKKOSBLAS2_SERIAL_GEMV_HPP_

#include "KokkosBatched_Gemv_Decl.hpp"
#include "KokkosBatched_Util.hpp"
#include "KokkosKernels_Error.hpp"

namespace KokkosBlas {
namespace Experimental {

template <class MatrixType, class XVector, class YVector,
          class ScalarType = typename MatrixType::non_const_value_type>
void KOKKOS_INLINE_FUNCTION
gemv(const char trans, const ScalarType& alpha, const MatrixType& A,
     const XVector& x, const typename YVector::non_const_value_type& beta,
     const YVector& y) {
  static_assert(
      std::is_same<ScalarType, typename XVector::non_const_value_type>::value &&
          std::is_same<ScalarType,
                       typename YVector::non_const_value_type>::value,
      "Serial GEMV requires A,x and y to have same scalar type");

  const auto run = [&](auto mode) {
    using algo        = KokkosBatched::Algo::Gemv::Default;
    using serial_impl = KokkosBatched::SerialGemv<decltype(mode), algo>;
    // ensure same type for alpha and beta (required by serial impl)
    const auto beta_ = static_cast<ScalarType>(beta);

    serial_impl::invoke(alpha, A, x, beta_, y);
  };

  if (trans == 'N' || trans == 'n') {
    return run(KokkosBatched::Trans::NoTranspose());
  } else if (trans == 'T' || trans == 't') {
    return run(KokkosBatched::Trans::Transpose());
    // } else if (trans == 'C' || trans == 'c') { // NOT implemented
    //   return run(KokkosBatched::Trans::ConjTranspose());
  } else {  // conjugate[no-transpose] not supported ?
    std::ostringstream os;
    os << "Matrix mode not supported: " << trans << std::endl;
    KokkosKernels::Impl::throw_runtime_exception(os.str());
  }
}

}  // namespace Experimental
}  // namespace KokkosBlas

#endif
