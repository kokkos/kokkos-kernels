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
          class CViewType, class ArgBoundsCheck, int TILE_M, int TILE_N,
          int TILE_K>
class BatchedDblBufGemm {
 private:
  HandleType *const __handle;
  AViewType __A;
  BViewType __B;
  CViewType __C;
  ScalarType __alpha, __beta;
  ArgTransA __transA_tag;
  ArgTransB __transB_tag;
  ArgBatchSzDim __batch_layout_tag;
  ArgBoundsCheck __bounds_check_tag;
  int __c_batch_size, __c_m, __c_n;

  using layout_type          = typename CViewType::array_layout;
  using device_type          = typename CViewType::device_type;
  using execution_space_type = typename device_type::execution_space;
  using scratch_space_type =
      typename execution_space_type::scratch_memory_space;
  using view_type_2d_scratch =
      Kokkos::View<ScalarType **, layout_type, scratch_space_type>;

 public:
  BatchedDblBufGemm(HandleType *const handle, ScalarType alpha, AViewType A,
                    BViewType B, ScalarType beta, CViewType C)
      : __handle(handle),
        __A(A),
        __B(B),
        __C(C),
        __alpha(alpha),
        __beta(beta) {}

  int invoke() {
    __run();
    return 0;
  }

 private:
  void __run() {
    using policy_type = Kokkos::TeamPolicy<execution_space_type>;
    using member_type = typename policy_type::member_type;

    // Compile-time expressions required for functor-level register allocations:
    //   Each team uses a shmem buffer and statically allocated register buffer.
    //   Below, we need a 1-1 mapping between GPU threads and register
    //   allocations to ensure that each GPU thread does not step on another
    //   GPU threads' registers. In short, we must map register allocations
    //   to parallel_for loop bounds.
    constexpr int reg_m    = TILE_M / TILE_K;
    constexpr int reg_n    = TILE_N / TILE_K + 2 * !!(TILE_N % TILE_K);
    constexpr int stride_m = TILE_M / reg_m;
    constexpr int stride_n = TILE_N / reg_n;
    using functor_type =
        __Functor<member_type, reg_m, reg_n, stride_m, stride_n>;

    functor_type functor(*this, TILE_M, TILE_N, TILE_K);

    // Each team solves a single tile. Within each tile, the team solves
    // all __n_tile_k_tiles one at a time.
    size_t league_size = __c_batch_size * functor.n_sub_tiles;
    int team_size      = stride_m;
    int vector_len     = stride_n;

    const int max_team_size =
        policy_type(league_size, Kokkos::AUTO, vector_len)
            .team_size_max(functor, Kokkos::ParallelForTag());
    if (team_size > max_team_size) {
      if (KokkosKernels::Impl::kk_is_gpu_exec_space<execution_space_type>()) {
        std::ostringstream os;
        os << "KokkosBatched::BatchedGemm with kernelAlgoType = "
           << std::to_string(__handle->get_kernel_algo_type())
           << " does not support team_size > " << std::to_string(max_team_size)
           << "." << std::endl
           << " The tile dimensions must be adjusted." << std::endl;
        Kokkos::Impl::throw_runtime_exception(os.str());
      } else {
        team_size = max_team_size;
      }
    }

    const int max_vector_len =
        policy_type(league_size, team_size, Kokkos::AUTO).vector_length_max();
    if (vector_len > max_vector_len) {
      if (KokkosKernels::Impl::kk_is_gpu_exec_space<execution_space_type>()) {
        std::ostringstream os;
        os << "KokkosBatched::BatchedGemm with kernelAlgoType = "
           << std::to_string(__handle->get_kernel_algo_type())
           << " does not support vector_len > "
           << std::to_string(max_vector_len) << "." << std::endl
           << " The tile dimensions must be adjusted." << std::endl;
        Kokkos::Impl::throw_runtime_exception(os.str());
      } else {
        vector_len = max_vector_len;
      }
    }

    if (__handle->enableDebug) {
      std::cout << "max_team_size:" << max_team_size
                << " team_size:" << team_size << std::endl
                << "max_vector_len:" << max_vector_len
                << " vector_len:" << vector_len << std::endl
                << "TILE_M:" << TILE_M << std::endl
                << "TILE_N:" << TILE_N << std::endl
                << "TILE_K:" << TILE_K << std::endl;
    }

    // TODO: Use statically allocated shmem
    int shmem_size = view_type_2d_scratch::shmem_size(TILE_M, TILE_K) +
                     view_type_2d_scratch::shmem_size(TILE_K, TILE_N);

    // Each member solves a portion of TILE_K in parallel with other members
    policy_type team_policy(league_size, team_size, vector_len);
    team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

    Kokkos::parallel_for("BatchedDblBufGemm", team_policy, functor);
  }

  template <class MemberType, int REG_M, int REG_N, int STRIDE_M, int STRIDE_N>
  class __Functor {
   private:
    size_t __n_tile_k_tiles;
    unsigned __tile_m, __tile_n, __tile_k;

   public:
    size_t n_sub_tiles;

    __Functor(BatchedDblBufGemm &ei, unsigned tile_m = 1, unsigned tile_n = 1,
              unsigned tile_k = 1)
        : __tile_m(tile_m), __tile_n(tile_n), __tile_k(tile_k) {
      unsigned k;
      if (std::is_same<ArgBatchSzDim, BatchLayout::Left>::value) {
        ei.__c_batch_size = ei.__C.extent_int(0);
        ei.__c_m          = ei.__C.extent_int(1);
        ei.__c_n          = ei.__C.extent_int(2);
        k                 = ei.__A.extent_int(2);
      } else {
        ei.__c_batch_size = ei.__C.extent_int(2);
        ei.__c_m          = ei.__C.extent_int(0);
        ei.__c_n          = ei.__C.extent_int(1);
        k                 = ei.__A.extent_int(1);
      }
      // To handle truncation of tiles per row/col, round up to one extra tile
      // with '!!'. This extra tile will hang off the edge of the 2-rank matrix.
      // For cases where tiles hang off the edge, we over-compute 0s within
      // registers and shmem via a conditional bounds check (selected at
      // compile-time).
      unsigned tiles_per_row =
          ei.__c_m / __tile_m + !!((unsigned)ei.__c_m % __tile_m);
      unsigned tiles_per_col =
          ei.__c_n / __tile_n + !!((unsigned)ei.__c_n % __tile_n);

      // To handle truncation of __n_tile_k_tile, we have logic within the
      // operator for handling a partial __tile_k tile.
      __n_tile_k_tiles = k / __tile_k;
      n_sub_tiles      = tiles_per_row * tiles_per_col;
    }

    void operator()(const MemberType &member) const { return; }
  };
};
/********************* END non-functor-level routines *********************/
}  // namespace KokkosBatched

#endif
