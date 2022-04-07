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
#ifndef __KOKKOSBATCHED_GESV_HPP__
#define __KOKKOSBATCHED_GESV_HPP__

/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

namespace KokkosBatched {

/// \brief Serial Batched GESV:
///
/// Solve A_l x_l = b_l for all l = 0, ..., N
/// using a batched LU decomposition, 2 batched triangular solves, and a batched
/// static pivoting.
///
/// \tparam MatrixType: Input type for the matrix, needs to be a 3D view
/// \tparam VectorType: Input type for the right-hand side and the solution,
/// needs to be a 2D view
///
/// \param A [in]: batched matrix, a rank 3 view
/// \param X [out]: solution, a rank 2 view
/// \param B [in]: right-hand side, a rank 2 view
/// \param tmp [in]: a rank 3 view used to store temporary variable; dimension
/// must be N x n x (n+4) where N is the batched size and n is the number of
/// rows.
///
/// No nested parallel_for is used inside of the function.
///

struct SerialGesv {
  template <typename MatrixType, typename VectorType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MatrixType A,
                                           const VectorType X,
                                           const VectorType Y,
                                           const MatrixType tmp);
};

/// \brief Team Batched GESV:
///
/// Solve A_l x_l = b_l for all l = 0, ..., N
/// using a batched LU decomposition, 2 batched triangular solves, and a batched
/// static pivoting.
///
/// \tparam MatrixType: Input type for the matrix, needs to be a 3D view
/// \tparam VectorType: Input type for the right-hand side and the solution,
/// needs to be a 2D view
///
/// \param member [in]: TeamPolicy member
/// \param A [in]: batched matrix, a rank 3 view
/// \param X [out]: solution, a rank 2 view
/// \param B [in]: right-hand side, a rank 2 view
///
/// A nested parallel_for with TeamThreadRange is used.
///

template <typename MemberType>
struct TeamGesv {
  template <typename MatrixType, typename VectorType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const MatrixType A,
                                           const VectorType X,
                                           const VectorType Y);
};

/// \brief Team Vector Batched GESV:
///
/// Solve A_l x_l = b_l for all l = 0, ..., N
/// using a batched LU decomposition, 2 batched triangular solves, and a batched
/// static pivoting.
///
/// \tparam MatrixType: Input type for the matrix, needs to be a 3D view
/// \tparam VectorType: Input type for the right-hand side and the solution,
/// needs to be a 2D view
///
/// \param member [in]: TeamPolicy member
/// \param A [in]: batched matrix, a rank 3 view
/// \param X [out]: solution, a rank 2 view
/// \param B [in]: right-hand side, a rank 2 view
///
///   Two nested parallel_for with both TeamVectorRange and ThreadVectorRange
///   (or one with TeamVectorRange) are used inside.
///

template <typename MemberType>
struct TeamVectorGesv {
  template <typename MatrixType, typename VectorType>
  KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member,
                                           const MatrixType A,
                                           const VectorType X,
                                           const VectorType Y);
};

}  // namespace KokkosBatched

#include "KokkosBatched_Gesv_Impl.hpp"

#endif
