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
#ifndef __KOKKOSBATCHED_INNER_GEMM_FIX_C_DECL_HPP__
#define __KOKKOSBATCHED_INNER_GEMM_FIX_C_DECL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosBatched {

namespace details {

template <typename ValueType>
struct identity {
  KOKKOS_FORCEINLINE_FUNCTION ValueType operator()(const ValueType &x) {
    return x;
  }
};

template <typename ValueType>
struct conj {
  KOKKOS_FORCEINLINE_FUNCTION ValueType operator()(const ValueType &x) {
    return Kokkos::ArithTraits<ValueType>::conj(x);
  }
};

}  // namespace details

template <int mb = 0, int nb = 0>
struct InnerGemmFixC {
  const int _as0, _as1, _bs0, _bs1, _cs0, _cs1;

  KOKKOS_INLINE_FUNCTION
  InnerGemmFixC(const int as0, const int as1, const int bs0, const int bs1,
                const int cs0, const int cs1)
      : _as0(as0), _as1(as1), _bs0(bs0), _bs1(bs1), _cs0(cs0), _cs1(cs1) {}

  // serial rank update
  template <typename ScalarType, typename ValueType,
            typename OpA = details::identity<ValueType> >
  KOKKOS_INLINE_FUNCTION int serial_invoke(const ScalarType alpha,
                                           const ValueType *KOKKOS_RESTRICT A,
                                           const ValueType *KOKKOS_RESTRICT B,
                                           const int k,
                                           /**/ ValueType *KOKKOS_RESTRICT C);

  // serial rank update for remainder
  template <typename ScalarType, typename ValueType,
            typename OpA = details::identity<ValueType> >
  KOKKOS_INLINE_FUNCTION int serial_invoke(const ScalarType alpha,
                                           const ValueType *KOKKOS_RESTRICT A,
                                           const ValueType *KOKKOS_RESTRICT B,
                                           const int m, const int k,
                                           /**/ ValueType *KOKKOS_RESTRICT C);

  // serial rank update for remainder
  template <typename ScalarType, typename ValueType,
            typename OpA = details::identity<ValueType> >
  KOKKOS_INLINE_FUNCTION int serial_invoke(const ScalarType alpha,
                                           const ValueType *KOKKOS_RESTRICT A,
                                           const ValueType *KOKKOS_RESTRICT B,
                                           const int m, const int n,
                                           const int k,
                                           /**/ ValueType *KOKKOS_RESTRICT C);

  template <typename MemberType, typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION int team_invoke(const MemberType &member,
                                         const ScalarType alpha,
                                         const ValueType *KOKKOS_RESTRICT A,
                                         const ValueType *KOKKOS_RESTRICT B,
                                         const int k,
                                         /**/ ValueType *KOKKOS_RESTRICT C);

  // team rank update for remainder
  template <typename MemberType, typename ScalarType, typename ValueType>
  KOKKOS_INLINE_FUNCTION int team_invoke(const MemberType &member,
                                         const ScalarType alpha,
                                         const ValueType *KOKKOS_RESTRICT A,
                                         const ValueType *KOKKOS_RESTRICT B,
                                         const int m, const int n, const int k,
                                         /**/ ValueType *KOKKOS_RESTRICT C);
};
}  // namespace KokkosBatched

#endif
