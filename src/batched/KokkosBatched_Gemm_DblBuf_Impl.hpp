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
#ifndef __KOKKOSBATCHED_GEMM_DBLBUF_IMPL_HPP__
#define __KOKKOSBATCHED_GEMM_DBLBUF_IMPL_HPP__

#include "KokkosBatched_Util.hpp"
// TODO: #include "KokkosBatched_Gemm_DblBuf_Internal.hpp"

namespace KokkosBatched {
/********************* BEGIN functor-level routines *********************/
///
/// Serial Impl
/// ===========

///
/// Implemented:
/// NT/NT, T/NT, NT/T, T/T
///
/// Not yet immplemented (ConjTranspose):
/// CT/NT, NT/CT, CT/CT
///

/********************* BEGIN functor-level routines *********************/
// template <>
// template <typename ScalarType, typename AViewType, typename BViewType,
//           typename CViewType, typename BoundsCheckType>
// KOKKOS_INLINE_FUNCTION int
// DblBufGemm<Trans::NoTranspose, Trans::NoTranspose>::invoke(
//     const ScalarType alpha, const AViewType &A, const BViewType &B,
//     const ScalarType beta, const CViewType &C) {
//   //  return DblBufGemmInternal<BoundsCheckType>::invoke(
//   //      C.extent(0), C.extent(1), A.extent(0), alpha,
//   //      A.data(), A.stride_0(), A.stride_1(),
//   //      B.data(), B.stride_0(), B.stride_1(), beta,
//   //      C.data(), C.stride_0(), C.stride_1());
//   return 0;
// }
//
// template <>
// template <typename ScalarType, typename AViewType, typename BViewType,
//           typename CViewType, typename BoundsCheckType>
// KOKKOS_INLINE_FUNCTION int
// DblBufGemm<Trans::NoTranspose, Trans::Transpose>::invoke(const ScalarType
// alpha,
//                                                          const AViewType &A,
//                                                          const BViewType &B,
//                                                          const ScalarType
//                                                          beta, const
//                                                          CViewType &C) {
//   //  return DblBufGemmInternal<BoundsCheckType>::invoke(
//   //      C.extent(0), C.extent(1), A.extent(0), alpha,
//   //      A.data(), A.stride_0(), A.stride_1(),
//   //      B.data(), B.stride_1(), B.stride_0(), beta,
//   //      C.data(), C.stride_0(), C.stride_1());
//   return 0;
// }
//
// template <>
// template <typename ScalarType, typename AViewType, typename BViewType,
//           typename CViewType, typename BoundsCheckType>
// KOKKOS_INLINE_FUNCTION int
// DblBufGemm<Trans::Transpose, Trans::NoTranspose>::invoke(const ScalarType
// alpha,
//                                                          const AViewType &A,
//                                                          const BViewType &B,
//                                                          const ScalarType
//                                                          beta, const
//                                                          CViewType &C) {
//   //  return DblBufGemmInternal<BoundsCheckType>::invoke(
//   //      C.extent(0), C.extent(1), A.extent(1), alpha,
//   //      A.data(), A.stride_1(), A.stride_0(),
//   //      B.data(), B.stride_0(), B.stride_1(), beta,
//   //      C.data(), C.stride_0(), C.stride_1());
//   return 0;
// }
//
// template <>
// template <typename ScalarType, typename AViewType, typename BViewType,
//           typename CViewType, typename BoundsCheckType>
// KOKKOS_INLINE_FUNCTION int
// DblBufGemm<Trans::Transpose, Trans::Transpose>::invoke(const ScalarType
// alpha,
//                                                        const AViewType &A,
//                                                        const BViewType &B,
//                                                        const ScalarType beta,
//                                                        const CViewType &C) {
//   //  return DblBufGemmInternal<BoundsCheckType>::invoke(
//   //      C.extent(0), C.extent(1), A.extent(1), alpha,
//   //      A.data(), A.stride_1(), A.stride_0(),
//   //      B.data(), B.stride_1(), B.stride_0(), beta,
//   //      C.data(), C.stride_0(), C.stride_1());
//   return 0;
// }
/********************* END functor-level routines *********************/

/********************* BEGIN non-functor-level routines *********************/
template <class ArgTransA, class ArgTransB, class ArgBatchSzDim,
          class HandleType, class ScalarType, class AViewType, class BViewType,
          class CViewType, class ArgBoundsCheck, int tile_m, int tile_n,
          int tile_k>
class BatchedDblBufGemm {
 private:
  ArgTransA transA_tag;
  ArgTransB transB_tag;
  ArgBatchSzDim batch_layout_tag;
  HandleType *const handle;
  AViewType A;
  BViewType B;
  CViewType C;
  ScalarType alpha, beta;
  ArgBoundsCheck bounds_check_tag;
  size_t league_size, team_size, vector_len, shmem_size;

  void run() {
    using execution_space = typename CViewType::device_type::execution_space;
    using policy_type     = Kokkos::TeamPolicy<execution_space>;
    // TODO: create functor_type

    policy_type team_policy(league_size, team_size, vector_len);
    team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

    Kokkos::parallel_for("BatchedDblBufGemm", *this);
  }

 public:
  int invoke() {
    // constexpr int reg_m = tile_m / tile_k;
    // constexpr int reg_n = tile_n / tile_k + 2*!!(tile_n % tile_k);
    // constexpr int stride_m = tile_m / reg_m;
    // constexpr int stride_n = tile_n / reg_n;
    //  TODO: set league, team, vector, and shmem
    // run();
    return 0;
  }

  BatchedDblBufGemm(HandleType *const _handle, ScalarType _alpha, AViewType _A,
                    BViewType _B, ScalarType _beta, CViewType _C)
      : handle(_handle), A(_A), B(_B), C(_C), alpha(_alpha), beta(_beta) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int &i) const { return; }
};
/********************* END non-functor-level routines *********************/
}  // namespace KokkosBatched

#endif
