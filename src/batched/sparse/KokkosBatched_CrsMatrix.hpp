//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.4
//       Copyright (2021) National Technology & Engineering
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
#ifndef __KOKKOSBATCHED_CRSMATRIX_HPP__
#define __KOKKOSBATCHED_CRSMATRIX_HPP__

/// \author Kim Liegeois (knliege@sandia.gov)

namespace KokkosBatched {

template <class ValuesViewType, class IntViewType>
class CrsMatrix {
 public:
  using ScalarType = typename ValuesViewType::non_const_value_type;
  using MagnitudeType =
      typename Kokkos::Details::ArithTraits<ScalarType>::mag_type;

 private:
  ValuesViewType values;
  IntViewType row_ptr;
  IntViewType colIndices;
  int n_operators;

 public:
  KOKKOS_INLINE_FUNCTION
  CrsMatrix(const ValuesViewType &_values, const IntViewType &_row_ptr,
            const IntViewType &_colIndices)
      : values(_values), row_ptr(_row_ptr), colIndices(_colIndices) {
    n_operators = _values.extent(0);
  }
  KOKKOS_INLINE_FUNCTION
  ~CrsMatrix() {}

  template <typename MemberType, typename XViewType, typename YViewType,
            typename ArgTrans, typename ArgMode>
  KOKKOS_INLINE_FUNCTION void apply(
      const MemberType &member, const XViewType &X, const YViewType &Y,
      MagnitudeType alpha = Kokkos::Details::ArithTraits<MagnitudeType>::one(),
      MagnitudeType beta =
          Kokkos::Details::ArithTraits<MagnitudeType>::zero()) const {
    if (beta == 0)
      KokkosBatched::Spmv<MemberType, ArgTrans, ArgMode>::template invoke<
          ValuesViewType, IntViewType, XViewType, YViewType, 0>(
          member, alpha, values, row_ptr, colIndices, X, beta, Y);
    else
      KokkosBatched::Spmv<MemberType, ArgTrans, ArgMode>::template invoke<
          ValuesViewType, IntViewType, XViewType, YViewType, 1>(
          member, alpha, values, row_ptr, colIndices, X, beta, Y);
  }

  template <typename MemberType, typename XViewType, typename YViewType,
            typename NormViewType, typename ArgTrans, typename ArgMode>
  KOKKOS_INLINE_FUNCTION void apply(const MemberType &member,
                                    const XViewType &X, const YViewType &Y,
                                    NormViewType alpha) const {
    KokkosBatched::Spmv<MemberType, ArgTrans, ArgMode>::template invoke<
        ValuesViewType, IntViewType, XViewType, YViewType, NormViewType,
        NormViewType, 0>(member, alpha, values, row_ptr, colIndices, X, alpha,
                         Y);
  }

  template <typename MemberType, typename XViewType, typename YViewType,
            typename NormViewType, typename ArgTrans, typename ArgMode>
  KOKKOS_INLINE_FUNCTION void apply(const MemberType &member,
                                    const XViewType &X, const YViewType &Y,
                                    const NormViewType &alpha,
                                    const NormViewType &beta) const {
    KokkosBatched::Spmv<MemberType, ArgTrans, ArgMode>::template invoke<
        ValuesViewType, IntViewType, XViewType, YViewType, NormViewType,
        NormViewType, 1>(member, alpha, values, row_ptr, colIndices, X, beta,
                         Y);
  }
};

}  // namespace KokkosBatched

#endif