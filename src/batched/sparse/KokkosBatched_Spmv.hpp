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
#ifndef __KOKKOSBATCHED_SPMV_HPP__
#define __KOKKOSBATCHED_SPMV_HPP__


/// \author Kim Liegeois (knliege@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

namespace KokkosBatched {

  /// Batched SPMV:
  ///   y_l <- alpha_l * A_l * x_l + beta_l * y_l for all l = 1, ..., N
  /// where:
  ///   * N is the number of matrices, 
  ///   * A_1, ..., A_N are N sparse matrices which share the same sparsity pattern,
  ///   * x_1, ..., x_N are the N input vectors,
  ///   * y_1, ..., y_N are the N output vectors,
  ///   * alpha_1, ..., alpha_N are N scaling factors for x_1, ..., x_N,
  ///   * beta_1, ..., beta_N are N scaling factors for y_1, ..., y_N.
  ///
  /// The matrices are represented using a Compressed Row Storage (CRS) format and
  /// the shared sparsity pattern is reused from one matrix to the others.
  ///
  /// Concretely, instead of providing an array of N matrices to the batched SPMV kernel,
  /// the user provides one row offset array (1D view), one column-index array (1D view),
  /// and one value array (2D view, one dimension for the non-zero indices and one for the
  /// matrix indices).
  ///
  /// 3 implementations are currently provided:
  ///  * SerialSpmv,
  ///  * TeamSpmv,
  ///  * TeamVectorSpmv.
  ///
  /// The naming of those implementations follows the logic of: 
  ///   Kim, K. (2019). Solving Many Small Matrix Problems using Kokkos and 
  ///   KokkosKernels (No. SAND2019-4542PE). Sandia National Lab.(SNL-NM),
  ///   Albuquerque, NM (United States).
  ///

  ///
  /// Serial SPMV
  ///   No nested parallel_for is used inside of the function.
  ///

  template<typename ArgTrans>
  struct SerialSpmv {
    template<typename ValuesViewType,
             typename IntView,
             typename xViewType,
             typename yViewType,
             typename alphaViewType,
             typename betaViewType,
             int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const alphaViewType &alpha,
           const ValuesViewType &values,
           const IntView &row_ptr,
           const IntView &colIndices,
           const xViewType &x,
           const betaViewType &beta,
           const yViewType &y);
  };

  ///
  /// Team SPMV
  ///   A nested parallel_for with TeamThreadRange is used.
  ///

  template<typename MemberType,
           typename ArgTrans>
  struct TeamSpmv {
    template<typename ValuesViewType,
             typename IntView,
             typename xViewType,
             typename yViewType,
             typename alphaViewType,
             typename betaViewType,
             int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const alphaViewType &alpha,
           const ValuesViewType &values,
           const IntView &row_ptr,
           const IntView &colIndices,
           const xViewType &x,
           const betaViewType &beta,
           const yViewType &y);
  };

  ///
  /// TeamVector SPMV
  ///   Two nested parallel_for with both TeamThreadRange and ThreadVectorRange 
  ///   (or one with TeamVectorRange) are used inside.  
  ///

  template<typename MemberType,
           typename ArgTrans>
  struct TeamVectorSpmv {
    template<typename ValuesViewType,
             typename IntView,
             typename xViewType,
             typename yViewType,
             typename alphaViewType,
             typename betaViewType,
             int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const alphaViewType &alpha,
           const ValuesViewType &values,
           const IntView &row_ptr,
           const IntView &colIndices,
           const xViewType &x,
           const betaViewType &beta,
           const yViewType &y);
  };

  ///
  /// Selective Interface
  ///

  template<typename MemberType,
           typename ArgTrans,
           typename ArgMode>
  struct Spmv {
    template<typename ValuesViewType,
             typename IntView,
             typename xViewType,
             typename yViewType,
             typename alphaViewType,
             typename betaViewType,
             int dobeta>
    KOKKOS_INLINE_FUNCTION
    static int
    invoke(const MemberType &member, 
           const alphaViewType &alpha,
           const ValuesViewType &values,
           const IntView &row_ptr,
           const IntView &colIndices,
           const xViewType &x,
           const betaViewType &beta,
           const yViewType &y) {
      int r_val = 0;
      if (std::is_same<ArgMode,Mode::Serial>::value) {
        r_val = SerialSpmv<ArgTrans>::template invoke<ValuesViewType, IntView, xViewType, yViewType, alphaViewType, betaViewType, dobeta>(alpha, values, row_ptr, colIndices, x, beta, y);
      } else if (std::is_same<ArgMode,Mode::Team>::value) {
        r_val = TeamSpmv<MemberType,ArgTrans>::template invoke<ValuesViewType, IntView, xViewType, yViewType, alphaViewType, betaViewType, dobeta>(member, alpha, values, row_ptr, colIndices, x, beta, y);
      } else if (std::is_same<ArgMode,Mode::TeamVector>::value) {
        r_val = TeamVectorSpmv<MemberType,ArgTrans>::template invoke<ValuesViewType, IntView, xViewType, yViewType, alphaViewType, betaViewType, dobeta>(member, alpha, values, row_ptr, colIndices, x, beta, y);
      } 
      return r_val;
    }      
  };

}

#include "KokkosBatched_Spmv_Serial_Impl.hpp"
#include "KokkosBatched_Spmv_Team_Impl.hpp"
#include "KokkosBatched_Spmv_TeamVector_Impl.hpp"
#endif
