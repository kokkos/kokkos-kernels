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

namespace KokkosBatched {
/********************* BEGIN functor-level routines *********************/
/********************* END   functor-level routines *********************/

namespace Impl {
/********************* BEGIN non-functor-level routines *********************/
///
/// Implemented:
/// NT/NT, T/NT, NT/T, T/T
///
/// Not yet immplemented (ConjTranspose):
/// CT/NT, NT/CT, CT/CT
///

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

  using view_value_type      = typename CViewType::value_type;
  using layout_type          = typename CViewType::array_layout;
  using device_type          = typename CViewType::device_type;
  using execution_space_type = typename device_type::execution_space;
  using scratch_space_type =
      typename execution_space_type::scratch_memory_space;
  using view_type_2d_scratch =
      Kokkos::View<view_value_type **, Kokkos::LayoutRight, scratch_space_type>;

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
    // TODO: check these expressions for all tile_m, tile_n, tile_k in Z+.
    constexpr int reg_m    = TILE_M / TILE_K;
    constexpr int reg_n    = TILE_N / TILE_K + 2 * !!(TILE_N % TILE_K);
    constexpr int stride_m = TILE_K;
    constexpr int stride_n = TILE_N / reg_n;
    using functor_type = Functor<member_type, reg_m, reg_n, stride_m, stride_n>;

    functor_type functor(*this, __A, __B, __C, TILE_M, TILE_N, TILE_K);

    if (__handle->enableDebug) {
      std::cout << "algo_type:" << __handle->get_kernel_algo_type() << std::endl
                << "__c_m:" << __c_m << std::endl
                << "__c_n:" << __c_n << std::endl
                << "reg_m:" << reg_m << std::endl
                << "reg_n:" << reg_n << std::endl
                << "stride_m:" << stride_m << std::endl
                << "stride_n:" << stride_n << std::endl;
    }

    // Each team solves a single tile. Within each tile, the team solves
    // all n_sub_tiles, one at a time.
    size_t league_size = __c_batch_size * functor.get_n_sub_tiles();
    // TODO: determine max_team_size and max_vector_len here instead of using 32
    // and 16
    int team_size =
        std::min(stride_m + functor.n_extra_threads(), (unsigned)32);
    // TODO: why are 2x vector lanes needed rather than just 1 more?
    int vector_len =
        std::min(stride_n * (functor.n_extra_vlanes() + 1), (unsigned)16);

    const int max_team_size =
        policy_type(league_size, Kokkos::AUTO, vector_len)
            .team_size_max(functor, Kokkos::ParallelForTag());
    if (team_size > max_team_size) {
      std::ostringstream os;
      os << "KokkosBatched::BatchedGemm with kernelAlgoType = "
         << std::to_string(__handle->get_kernel_algo_type())
         << " does not support team_size > " << std::to_string(max_team_size)
         << "." << std::endl
         << " The tile dimensions must be adjusted." << std::endl;
      Kokkos::Impl::throw_runtime_exception(os.str());
    }

    const int max_vector_len =
        policy_type(league_size, team_size, Kokkos::AUTO).vector_length_max();
    if (vector_len > max_vector_len) {
      std::ostringstream os;
      os << "KokkosBatched::BatchedGemm with kernelAlgoType = "
         << std::to_string(__handle->get_kernel_algo_type())
         << " does not support vector_len > " << std::to_string(max_vector_len)
         << "." << std::endl
         << " The tile dimensions must be adjusted." << std::endl;
      Kokkos::Impl::throw_runtime_exception(os.str());
    }

    if (__handle->enableDebug) {
      std::cout << "max_team_size:" << max_team_size
                << " team_size:" << team_size << std::endl
                << "max_vector_len:" << max_vector_len
                << " vector_len:" << vector_len << std::endl
                << " league_size:" << league_size << std::endl
                << "TILE_M:" << TILE_M << std::endl
                << "TILE_N:" << TILE_N << std::endl
                << "TILE_K:" << TILE_K << std::endl;
    }

    // NOTE: All but shmem_size args but partial tile sizes can be determined at
    // compile time.
    int shmem_size = view_type_2d_scratch::shmem_size(
                         TILE_M + functor.get_partial_tile_m(), TILE_K) +
                     view_type_2d_scratch::shmem_size(
                         TILE_K, TILE_N + functor.get_partial_tile_n());

    // Each member solves a portion of TILE_K in parallel with other members
    policy_type team_policy(league_size, team_size, vector_len);
    team_policy.set_scratch_size(0, Kokkos::PerTeam(shmem_size));

    Kokkos::parallel_for("BatchedDblBufGemm", team_policy, functor);
  }

 public:
  // Make Functor public for cuda 9.
  // See https://github.com/kokkos/kokkos-kernels/issues/1121.
  template <class MemberType, int REG_M, int REG_N, int STRIDE_M, int STRIDE_N>
  class Functor {
   private:
    BatchedDblBufGemm &__ei;
    AViewType __A;
    BViewType __B;
    CViewType __C;
    ScalarType __alpha, __beta;
    int __k;
    size_t __n_tile_k_tiles, __n_sub_tiles;
    unsigned __tile_m, __tile_n, __tile_k, __tiles_per_col, __tiles_per_row,
        __partial_tile_m, __partial_tile_n, __stride_m, __stride_n, __ts, __vl;

   public:
    size_t get_n_sub_tiles() { return __n_sub_tiles; }
    unsigned get_partial_tile_m() { return __partial_tile_m; }
    unsigned get_partial_tile_n() { return __partial_tile_n; }
    unsigned n_extra_threads() { return __partial_tile_m / REG_M; }
    unsigned n_extra_vlanes() { return __partial_tile_n / REG_N; }

    // NOTE: We cannot use __ei.{__A,__B,__C,__beta,__alpha,__k} in the operator
    // below. If those are used, we  get an invalid memory error from cuda. I
    // suspect this is due the values not being copied to device and then
    // runtime resolution of the host address &__ei.
    Functor(BatchedDblBufGemm &ei, AViewType A, BViewType B, CViewType C,
            unsigned tile_m = 1, unsigned tile_n = 1, unsigned tile_k = 1)
        : __ei(ei),
          __A(A),
          __B(B),
          __C(C),
          __tile_m(tile_m),
          __tile_n(tile_n),
          __tile_k(tile_k) {
      if (std::is_same<ArgBatchSzDim, BatchLayout::Left>::value) {
        ei.__c_batch_size = ei.__C.extent_int(0);
        ei.__c_m          = ei.__C.extent_int(1);
        ei.__c_n          = ei.__C.extent_int(2);
        if (std::is_same<ArgTransA, Trans::Transpose>::value)
          __k = ei.__A.extent_int(1);
        else
          __k = ei.__A.extent_int(2);
      } else {
        ei.__c_batch_size = ei.__C.extent_int(2);
        ei.__c_m          = ei.__C.extent_int(0);
        ei.__c_n          = ei.__C.extent_int(1);
        if (std::is_same<ArgTransA, Trans::Transpose>::value)
          __k = ei.__A.extent_int(0);
        else
          __k = ei.__A.extent_int(1);
      }
      __beta  = ei.__beta;   // Copy to device
      __alpha = ei.__alpha;  // Copy to device
      // To handle cases where ei.__c_{m,n} % __tile_{m,n} != 0 without simply
      // multiplying by zeros, we can adjust the scratch space size in
      // increments of REG_M / REG_M as well as the team / vector sizes up to 2
      // * cur_team_size - 1. For each additional thread, we must increase
      // STRIDE_M by 1. For each additional vlane, we must increase STRIDE_N by
      // 1. Note that each thead can handle REG_M extra rows and each vlane can
      // handle REG_N extra cols.
      __tiles_per_row = ei.__c_m / __tile_m;
      __tiles_per_col = ei.__c_n / __tile_n;

      // Each partial_tile is a multiple of REG_{M,N} since each thread holds
      // register buffers of size REG_{M,N}.
      __partial_tile_m = ei.__c_m % __tile_m && ei.__c_m > __tile_m
                             ? REG_M * ((ei.__c_m - __tile_m) / REG_M + 1)
                             : 0;
      __partial_tile_n = ei.__c_n % __tile_n && ei.__c_n > __tile_n
                             ? REG_N * ((ei.__c_n - __tile_n) / REG_N + 1)
                             : 0;

      __stride_m = STRIDE_M + n_extra_threads();
      __stride_n = STRIDE_N + n_extra_vlanes();

      __ts = __tile_n / REG_N + n_extra_threads();
      __vl = __tile_n / REG_N + n_extra_vlanes();

      // To handle truncation of __n_tile_k_tile, we have logic within the
      // operator for handling a partial __tile_k tile.
      __n_tile_k_tiles = __k / __tile_k;
      __n_sub_tiles    = __tiles_per_row * __tiles_per_col;

      if (ei.__handle->enableDebug) {
        std::cout << "__partial_tile_m:" << __partial_tile_m << std::endl
                  << "__partial_tile_n:" << __partial_tile_n << std::endl
                  << "__stride_m:" << __stride_m << std::endl
                  << "__stride_n:" << __stride_n << std::endl
                  << "__ts:" << __ts << std::endl
                  << "__vl:" << __vl << std::endl;
      }
    }

    KOKKOS_INLINE_FUNCTION
    void __rshmem_and_mult(const MemberType &member, const unsigned &nk,
                           const unsigned &tile_m, const unsigned &tile_n,
                           view_value_type reg_a[REG_M],
                           view_value_type reg_b[REG_N],
                           view_value_type reg_c[REG_M][REG_N],
                           view_type_2d_scratch &svA_scr,
                           view_type_2d_scratch &svB_scr) const {
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, 0, __ts), [&](const int &thread_id) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(member, 0, __vl),
                [&](const int &vlane_id) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                  for (unsigned k = 0; k < nk; ++k) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                    for (int m = 0; m < REG_M; ++m)
                      reg_a[m] = svA_scr(k, thread_id + m * __stride_m);

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                    for (int n = 0; n < REG_N; ++n)
                      reg_b[n] = svB_scr(k, vlane_id + n * __stride_n);

#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                    for (int m = 0; m < REG_M; ++m) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                      for (int n = 0; n < REG_N; ++n)
                        reg_c[m][n] += reg_a[m] * reg_b[n] * __alpha;
                    }
                  }
                });
          });
    }

    KOKKOS_INLINE_FUNCTION
    void operator()(const MemberType &member) const {
      // Allocate registers used for prefetching
      view_value_type prefetch_reg_a[REG_M] = {0}, prefetch_reg_b[REG_N] = {0};

      // Allocate registers used for FMAs
      view_value_type reg_a[REG_M] = {0}, reg_b[REG_N] = {0},
                      reg_c[REG_M][REG_N] = {{0}};

      unsigned batch_idx = member.league_rank() / __n_sub_tiles;

      // Compute starting tile offsets for each team into svA, svB, svC
      unsigned local_team_idx = member.league_rank() % __n_sub_tiles;
      unsigned start_m        = (local_team_idx / __tiles_per_col) * __tile_m;
      unsigned start_n        = (local_team_idx % __tiles_per_col) * __tile_n;

      // Fetch entire 2-rank sub-matrix
      auto svA = subview_wrapper(__A, batch_idx, Kokkos::ALL(), Kokkos::ALL(),
                                 __ei.__batch_layout_tag, __ei.__transA_tag);
      auto svB = subview_wrapper(__B, batch_idx, Kokkos::ALL(), Kokkos::ALL(),
                                 __ei.__batch_layout_tag, __ei.__transB_tag);
      auto svC = subview_wrapper(__C, batch_idx, Kokkos::ALL(), Kokkos::ALL(),
                                 __ei.__batch_layout_tag);

      // Allocate scratch memory buffers used for prefetching
      view_type_2d_scratch svA_scr(member.team_scratch(0), __tile_k,
                                   __tile_m + __partial_tile_m);
      view_type_2d_scratch svB_scr(member.team_scratch(0), __tile_k,
                                   __tile_n + __partial_tile_n);

      // Here we populate scratch memory with one or more "k" tiles for every
      // thread of the team!
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, 0, __ts), [&](const int &thread_id) {
            auto thread_offset = thread_id + start_n;
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(member, 0, __tile_k),
                [&](const int &vlane_id) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                  for (int i = 0; i < REG_N * __stride_n; i += __stride_n)
                    svB_scr(vlane_id, thread_id + i) =
                        access_view_bounds_check<view_value_type>(
                            svB, vlane_id, thread_offset + i,
                            __ei.__bounds_check_tag);
                });
          });
      Kokkos::parallel_for(
          Kokkos::TeamThreadRange(member, 0, __ts), [&](const int &thread_id) {
            auto thread_offset = thread_id + start_m;
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(member, 0, __tile_k),
                [&](const int &vlane_id) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                  for (int i = 0; i < REG_M * __stride_m; i += __stride_m)
                    svA_scr(vlane_id, thread_id + i) =
                        access_view_bounds_check<view_value_type>(
                            svA, thread_offset + i, vlane_id,
                            __ei.__bounds_check_tag);
                });
          });

      // Check whether we have a partial tile
      unsigned partial_tile_k = __k - (__n_tile_k_tiles * __tile_k);
      int partial_tile        = !!(partial_tile_k);

      // Wait for A, B to reside in scratch memory
      member.team_barrier();

      // Each thread calculates a single dot product in chunks of size __tile_k
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
      for (unsigned k_tile_idx = 1;
           k_tile_idx < __n_tile_k_tiles + partial_tile; ++k_tile_idx) {
        auto k_tile_offset = k_tile_idx * __tile_k;

        // Get this threads next __tile_k entries from global memory
        // Each thread has its own copy of prefetch_reg_b. TeamThreadRange runs
        // over all threads in the team.
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, 0, __ts),
            [&](const int &thread_id) {
              auto thread_offset = thread_id + start_n;
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, 0, __tile_k),
                  [&](const int &vlane_id) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                    for (int i = 0; i < REG_N; ++i)
                      prefetch_reg_b[i] =
                          access_view_bounds_check<view_value_type>(
                              svB, vlane_id + k_tile_offset,
                              thread_offset + i * __stride_n,
                              __ei.__bounds_check_tag);
                  });
            });

        // Get this threads next __tile_k entries from global memory
        // Each thread has its own copy of prefetch_reg_b. TeamThreadRange runs
        // over all threads in the team.
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, 0, __ts),
            [&](const int &thread_id) {
              auto thread_offset = thread_id + start_m;
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, 0, __tile_k),
                  [&](const int &vlane_id) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                    for (int i = 0; i < REG_M; ++i)
                      prefetch_reg_a[i] =
                          access_view_bounds_check<view_value_type>(
                              svA, thread_offset + i * __stride_m,
                              vlane_id + k_tile_offset,
                              __ei.__bounds_check_tag);
                  });
            });

        // Multiply
        __rshmem_and_mult(member, __tile_k, __tile_m, __tile_n, reg_a, reg_b,
                          reg_c, svA_scr, svB_scr);
        // Wait for:
        //   1. prefetch_regs to be populated
        //   2. for shmem to no longer be read from
        member.team_barrier();

        // populate shmem from prefetch registers. Each thread has its own copy
        // of prefetch_reg_a.
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, 0, __ts),
            [&](const int &thread_id) {
              auto thread_offset = thread_id;
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, 0, __tile_k),
                  [&](const int &vlane_id) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                    for (int i = 0; i < REG_N; ++i)
                      svB_scr(vlane_id, thread_offset + i * __stride_n) =
                          prefetch_reg_b[i];
                  });
            });

        // populate shmem from prefetch registers. Each thread has its own copy
        // of prefetch_reg_b.
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, 0, __ts),
            [&](const int &thread_id) {
              auto thread_offset = thread_id;
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, 0, __tile_k),
                  [&](const int &vlane_id) {
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                    for (int i = 0; i < REG_M; ++i)
                      svA_scr(vlane_id, thread_offset + i * __stride_m) =
                          prefetch_reg_a[i];
                  });
            });

        // Wait for shmem stores to land before performing next __tile_k
        // multiply
        member.team_barrier();

      }  // end n_tile_k_tiles loop

      // Multiply last tile, may be a partial tile
      partial_tile_k = partial_tile ? partial_tile_k : __tile_k;
      __rshmem_and_mult(member, partial_tile_k, __tile_m, __tile_n, reg_a,
                        reg_b, reg_c, svA_scr, svB_scr);

      // store results back to global memory
      if (__beta == 0.0F) {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, 0, __ts),
            [&](const int &thread_id) {
              auto thread_m_offset = thread_id + start_m;
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, 0, __vl),
                  [&](const int &vlane_id) {
                    auto thread_n_offset = vlane_id + start_n;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                    for (int m = 0; m < REG_M; ++m) {
                      int cm = thread_m_offset + m * __stride_m;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                      for (int n = 0; n < REG_N; ++n) {
                        int cn = thread_n_offset + n * __stride_n;
                        fma_bounds_check(svC, cm, cn, reg_c[m][n],
                                         __ei.__bounds_check_tag);
                      }
                    }
                  });
            });
      } else {
        Kokkos::parallel_for(
            Kokkos::TeamThreadRange(member, 0, __ts),
            [&](const int &thread_id) {
              auto thread_m_offset = thread_id + start_m;
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(member, 0, __vl),
                  [&](const int &vlane_id) {
                    auto thread_n_offset = vlane_id + start_n;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                    for (int m = 0; m < REG_M; ++m) {
                      int cm = thread_m_offset + m * __stride_m;
#if defined(KOKKOS_ENABLE_PRAGMA_UNROLL)
#pragma unroll
#endif  // KOKKOS_ENABLE_PRAGMA_UNROLL
                      for (int n = 0; n < REG_N; ++n) {
                        int cn = thread_n_offset + n * __stride_n;
                        fma_bounds_check(svC, cm, cn, reg_c[m][n], __beta,
                                         __ei.__bounds_check_tag);
                      }
                    }
                  });
            });
      }
    }
  };
};
/********************* END non-functor-level routines *********************/
}  // namespace Impl

}  // namespace KokkosBatched

#endif
