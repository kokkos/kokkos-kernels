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
#ifndef __KOKKOSBLAS_GEMV_SERIAL_IMPL_HPP__
#define __KOKKOSBLAS_GEMV_SERIAL_IMPL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBlas_util.hpp"
#include "KokkosBlas2_serial_gemv_internal.hpp"

#include "KokkosBlas2_serial_gemv_tpl_spec_decl.hpp"

namespace KokkosBlas {

///
/// Serial Impl
/// ===========

template <typename ArgTrans, typename ArgAlgo>
template <typename ScalarType, typename AViewType, typename xViewType,
          typename yViewType>
KOKKOS_INLINE_FUNCTION int SerialGemv<ArgTrans, ArgAlgo>::invoke(
    const ScalarType alpha, const AViewType &A, const xViewType &x,
    const ScalarType beta, const yViewType &y) {
  static_assert(std::is_same<ArgAlgo, Algo::Gemv::Unblocked>::value ||
                    std::is_same<ArgAlgo, Algo::Gemv::Blocked>::value ||
                    std::is_same<ArgAlgo, Algo::Gemv::CompactMKL>::value,
                "Algorithm not supported");

  using TransA   = Impl::MatrixModeInfo<ArgTrans>;
  const auto ae0 = TransA::extent(A, 0);
  const auto ae1 = TransA::extent(A, 1);
  const auto as0 = TransA::stride_0(A);
  const auto as1 = TransA::stride_1(A);

  return Impl::SerialGemvInternal<ArgAlgo>::invoke(
      ae0, ae1, alpha, A.data(), as0, as1, x.data(), x.stride_0(), beta,
      y.data(), y.stride_0());
}

}  // namespace KokkosBlas

#endif