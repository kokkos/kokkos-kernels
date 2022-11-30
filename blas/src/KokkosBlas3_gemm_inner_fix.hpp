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

#ifndef KOKKOSBLAS3_GEMM_INNER_FIX_HPP
#define KOKKOSBLAS3_GEMM_INNER_FIX_HPP

/// \author Kyungjoo Kim (kyukim@sandia.gov)

namespace KokkosBlas {

template <int mb, int nb>
struct InnerGemmFixA {
  const int _as0, _as1, _bs0, _bs1, _cs0, _cs1;

  KOKKOS_INLINE_FUNCTION
  InnerGemmFixA(const int as0, const int as1, const int bs0, const int bs1,
                const int cs0, const int cs1)
      : _as0(as0), _as1(as1), _bs0(bs0), _bs1(bs1), _cs0(cs0), _cs1(cs1) {}

  // serial rank update
  template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
            typename ValueTypeC>
  KOKKOS_INLINE_FUNCTION int serial_invoke(const ScalarType alpha,
                                           const ValueTypeA *KOKKOS_RESTRICT A,
                                           const ValueTypeB *KOKKOS_RESTRICT B,
                                           const int n,
                                           /**/ ValueTypeC *KOKKOS_RESTRICT C);

  // serial rank update for remainder
  template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
            typename ValueTypeC>
  KOKKOS_INLINE_FUNCTION int serial_invoke(const ScalarType alpha,
                                           const ValueTypeA *KOKKOS_RESTRICT A,
                                           const ValueTypeB *KOKKOS_RESTRICT B,
                                           const int m, const int n,
                                           const int k,
                                           /**/ ValueTypeC *KOKKOS_RESTRICT C);
};

template <int mb, int nb>
struct InnerGemmFixB {
  const int _as0, _as1, _bs0, _bs1, _cs0, _cs1;

  KOKKOS_INLINE_FUNCTION
  InnerGemmFixB(const int as0, const int as1, const int bs0, const int bs1,
                const int cs0, const int cs1)
      : _as0(as0), _as1(as1), _bs0(bs0), _bs1(bs1), _cs0(cs0), _cs1(cs1) {}

  // serial rank update
  template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
            typename ValueTypeC>
  KOKKOS_INLINE_FUNCTION int serial_invoke(const ScalarType alpha,
                                           const ValueTypeA *KOKKOS_RESTRICT A,
                                           const ValueTypeB *KOKKOS_RESTRICT B,
                                           const int n,
                                           /**/ ValueTypeC *KOKKOS_RESTRICT C);

  // serial rank update for remainder
  template <typename ScalarType, typename ValueTypeA, typename ValueTypeB,
            typename ValueTypeC>
  KOKKOS_INLINE_FUNCTION int serial_invoke(const ScalarType alpha,
                                           const ValueTypeA *KOKKOS_RESTRICT A,
                                           const ValueTypeB *KOKKOS_RESTRICT B,
                                           const int m, const int n,
                                           const int k,
                                           /**/ ValueTypeC *KOKKOS_RESTRICT C);
};

template <int mb = 0, int nb = 0>
struct InnerGemmFixC {
  const int _as0, _as1, _bs0, _bs1, _cs0, _cs1;

  KOKKOS_INLINE_FUNCTION
  InnerGemmFixC(const int as0, const int as1, const int bs0, const int bs1,
                const int cs0, const int cs1)
      : _as0(as0), _as1(as1), _bs0(bs0), _bs1(bs1), _cs0(cs0), _cs1(cs1) {}

  // serial rank update
  template <typename OpA, typename OpB, typename ScalarType,
            typename ValueTypeA, typename ValueTypeB, typename ValueTypeC>
  KOKKOS_INLINE_FUNCTION int serial_invoke(OpA opA, OpB opB,
                                           const ScalarType alpha,
                                           const ValueTypeA *KOKKOS_RESTRICT A,
                                           const ValueTypeB *KOKKOS_RESTRICT B,
                                           const int k,
                                           /**/ ValueTypeC *KOKKOS_RESTRICT C);

  // serial rank update for remainder
  template <typename OpA, typename OpB, typename ScalarType,
            typename ValueTypeA, typename ValueTypeB, typename ValueTypeC>
  KOKKOS_INLINE_FUNCTION int serial_invoke(OpA opA, OpB opB,
                                           const ScalarType alpha,
                                           const ValueTypeA *KOKKOS_RESTRICT A,
                                           const ValueTypeB *KOKKOS_RESTRICT B,
                                           const int m, const int k,
                                           /**/ ValueTypeC *KOKKOS_RESTRICT C);

  // serial rank update for remainder
  template <typename OpA, typename OpB, typename ScalarType,
            typename ValueTypeA, typename ValueTypeB, typename ValueTypeC>
  KOKKOS_INLINE_FUNCTION int serial_invoke(OpA opA, OpB opB,
                                           const ScalarType alpha,
                                           const ValueTypeA *KOKKOS_RESTRICT A,
                                           const ValueTypeB *KOKKOS_RESTRICT B,
                                           const int m, const int n,
                                           const int k,
                                           /**/ ValueTypeC *KOKKOS_RESTRICT C);

  template <typename MemberType, typename ScalarType, typename ValueTypeA,
            typename ValueTypeB, typename ValueTypeC>
  KOKKOS_INLINE_FUNCTION int team_invoke(const MemberType &member,
                                         const ScalarType alpha,
                                         const ValueTypeA *KOKKOS_RESTRICT A,
                                         const ValueTypeB *KOKKOS_RESTRICT B,
                                         const int k,
                                         /**/ ValueTypeC *KOKKOS_RESTRICT C);

  // team rank update for remainder
  template <typename MemberType, typename ScalarType, typename ValueTypeA,
            typename ValueTypeB, typename ValueTypeC>
  KOKKOS_INLINE_FUNCTION int team_invoke(const MemberType &member,
                                         const ScalarType alpha,
                                         const ValueTypeA *KOKKOS_RESTRICT A,
                                         const ValueTypeB *KOKKOS_RESTRICT B,
                                         const int m, const int n, const int k,
                                         /**/ ValueTypeC *KOKKOS_RESTRICT C);
};

}  // namespace KokkosBlas

#include "KokkosBlas3_serial_gemm_inner_fixa_impl.hpp"
#include "KokkosBlas3_serial_gemm_inner_fixb_impl.hpp"
#include "KokkosBlas3_serial_gemm_inner_fixc_impl.hpp"
#include "KokkosBlas3_team_gemm_inner_fixc_impl.hpp"

#endif
