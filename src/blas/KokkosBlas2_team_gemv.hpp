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

#ifndef KOKKOSBLAS2_TEAM_GEMV_HPP_
#define KOKKOSBLAS2_TEAM_GEMV_HPP_

#include <KokkosBlas2_team_gemv_spec.hpp>

namespace KokkosBlas {
namespace Experimental {

template <class AlgoTag, class TeamType, class MatrixType, class XVector,
          class YVector, class ScalarType>
void KOKKOS_INLINE_FUNCTION team_gemv(const TeamType& team, const char trans,
                                      const ScalarType& alpha,
                                      const MatrixType& A, const XVector& x,
                                      const ScalarType& beta,
                                      const YVector& y) {
  if (trans == 'N' || trans == 'n')
    TeamGemv<TeamType, Trans::NoTranspose, AlgoTag>::invoke(team, alpha, A, x,
                                                            beta, y);
  if (trans == 'T' || trans == 't')
    TeamGemv<TeamType, Trans::Transpose, AlgoTag>::invoke(team, alpha, A, x,
                                                          beta, y);
  if (trans == 'C' || trans == 'c')
    TeamGemv<TeamType, Trans::ConjTranspose, AlgoTag>::invoke(team, alpha, A, x,
                                                              beta, y);
  //
  // TODO: what letter should be used here ?
  //    * in blas "C" means conjugate-transpose
  //    * in sparse "C" meanse conjugate and "H" conjugate-transpose...
  if (trans == 'X' || trans == 'x')
    TeamGemv<TeamType, Trans::ConjNoTranspose, AlgoTag>::invoke(team, alpha, A,
                                                                x, beta, y);
}

// default AlgoTag
template <class TeamType, class MatrixType, class XVector, class YVector,
          class ScalarType>
void KOKKOS_INLINE_FUNCTION team_gemv(const TeamType& team, const char trans,
                                      const ScalarType& alpha,
                                      const MatrixType& A, const XVector& x,
                                      const ScalarType& beta,
                                      const YVector& y) {
  team_gemv<KokkosBlas::Algo::Gemv::Default>(team, trans, alpha, A, x, beta, y);
}

template <class AlgoTag, class TeamType, class MatrixType, class XVector,
          class YVector, class ScalarType>
void KOKKOS_INLINE_FUNCTION
teamvector_gemv(const TeamType& team, const char trans, const ScalarType& alpha,
                const MatrixType& A, const XVector& x, const ScalarType& beta,
                const YVector& y) {
  if (trans == 'N' || trans == 'n') {
    KokkosBlas::TeamVectorGemv<TeamType, Trans::NoTranspose, AlgoTag>::invoke(
        team, alpha, A, x, beta, y);
  } else if (trans == 'T' || trans == 't') {
    KokkosBlas::TeamVectorGemv<TeamType, Trans::Transpose, AlgoTag>::invoke(
        team, alpha, A, x, beta, y);
  } else if (trans == 'C' || trans == 'c') {
    KokkosBlas::TeamVectorGemv<TeamType, Trans::ConjTranspose, AlgoTag>::invoke(
        team, alpha, A, x, beta, y);
    //
    // TODO: what letter should be used here ?
    //    * in blas "C" means conjugate-transpose
    //    * in sparse "C" meanse conjugate and "H" conjugate-transpose...
  } else if (trans == 'X' || trans == 'x') {
    KokkosBlas::TeamVectorGemv<TeamType, Trans::ConjNoTranspose,
                               AlgoTag>::invoke(team, alpha, A, x, beta, y);
  } else {
    Kokkos::abort("Matrix mode not supported");
  }
}

// default AlgoTag
template <class TeamType, class MatrixType, class XVector, class YVector,
          class ScalarType>
void KOKKOS_INLINE_FUNCTION
team_vector_gemv(const TeamType& team, const char trans,
                 const ScalarType& alpha, const MatrixType& A, const XVector& x,
                 const ScalarType& beta, const YVector& y) {
  teamvector_gemv<KokkosBlas::Algo::Gemv::Default>(team, trans, alpha, A, x,
                                                   beta, y);
}

}  // namespace Experimental
}  // namespace KokkosBlas

#endif
