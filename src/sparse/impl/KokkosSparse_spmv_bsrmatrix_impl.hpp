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

#ifndef KOKKOSSPARSE_IMPL_SPMV_BSRMATRIX_IMPL_HPP_
#define KOKKOSSPARSE_IMPL_SPMV_BSRMATRIX_IMPL_HPP_

#if defined(KOKKOS_ENABLE_CUDA) && \
    (defined(KOKKOS_ARCH_VOLTA) || defined(KOKKOS_ARCH_AMPERE))

#include <type_traits>
#include <mma.h>

namespace KokkosSparse {
namespace Experimental {
namespace Impl {

struct BsrMatrixSpMVTensorCoreFunctorParams {
  int teamsPerBlockM;
  int teamsPerBlockN;
  int leagueDim_x;
  int leagueDim_y;
};

/// \brief Functor for the BsrMatrix SpMV multivector implementation utilizing
/// tensor cores.
///
/// \tparam AMatrix The type of the A matrix (a BsrMatrix)
/// \tparam AFragScalar The type of the CUDA wmma fragment that will be loaded
/// from the A matrix. The scalar type of the wmma fragment may be different
/// that that of the A matrix. \tparam FRAG_M (with FRAG_N and FRAG_K), the
/// m-n-k size of the CUDA wmma fragment type. \tparam LEAGUE_DIM_X (with
/// TEAMS_PER_BLOCK_M and TEAMS_PER_BLOCK_N) if non-zero, statically-known
/// launch parameters to reduce the cost of divmod operations on the GPU. If 0,
/// provided runtime values will be used instead.
template <typename AMatrix,
          typename AFragScalar,  // input matrix type and fragment scalar type
          typename XMatrix, typename XFragScalar, typename YMatrix,
          typename YFragScalar, unsigned FRAG_M, unsigned FRAG_N,
          unsigned FRAG_K,  // fragment sizes
          unsigned LEAGUE_DIM_X = 0, unsigned TEAMS_PER_BLOCK_M = 0,
          unsigned TEAMS_PER_BLOCK_N = 0>
struct BsrMatrixSpMVTensorCoreFunctor {
  typedef nvcuda::wmma::accumulator accumulator;
  typedef nvcuda::wmma::row_major row_major;
  typedef nvcuda::wmma::col_major col_major;
  typedef nvcuda::wmma::matrix_a matrix_a;
  typedef nvcuda::wmma::matrix_b matrix_b;
  using FragA = nvcuda::wmma::fragment<matrix_a, FRAG_M, FRAG_N, FRAG_K,
                                       AFragScalar, row_major>;
  using FragX = nvcuda::wmma::fragment<matrix_b, FRAG_M, FRAG_N, FRAG_K,
                                       XFragScalar, row_major>;
  using FragY =
      nvcuda::wmma::fragment<accumulator, FRAG_M, FRAG_N, FRAG_K, YFragScalar>;

  typedef typename AMatrix::device_type Device;
  typedef Kokkos::TeamPolicy<typename Device::execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef typename AMatrix::value_type AScalar;
  typedef typename YMatrix::value_type YScalar;
  typedef typename XMatrix::value_type XScalar;
  typedef typename AMatrix::non_const_ordinal_type AOrdinal;
  typedef typename AMatrix::non_const_size_type AOffset;

  // views of the shared memory used in the functor to cast types to the CUDA
  // wmma types A matrix is MxK X matrix is KxN Y matrix is MxN
  typedef typename Kokkos::View<
      AFragScalar * [FRAG_M][FRAG_K],  // one fragment per warp in the team (2D
                                       // grid of warps in team)
      Kokkos::LayoutRight,
      typename Device::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      AScratchView;
  typedef typename Kokkos::View<
      XFragScalar * [FRAG_K][FRAG_N],
      typename Kokkos::LayoutRight,  // so that [FRAG_K][FRAG_N] part is
                                     // contiguous in memory
      typename Device::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      XScratchView;
  typedef typename Kokkos::View<
      YFragScalar * * [FRAG_M][FRAG_N],
      typename Kokkos::LayoutRight,  // so that [FRAG_M][FRAG_N] part is
                                     // contiguous in memory
      typename Device::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>>
      YScratchView;

  YScalar alpha;
  AMatrix a;
  XMatrix x;
  YScalar beta;
  YMatrix y;

  BsrMatrixSpMVTensorCoreFunctorParams params;

  // a team is a 2D grid of warps
  static constexpr int WARPS_PER_TEAM_X = 2;
  static constexpr int WARPS_PER_TEAM_Y = 2;
  static constexpr int THREADS_PER_WARP = 32;

  BsrMatrixSpMVTensorCoreFunctor() = delete;  // need all runtime parameters

  // the launch parameters should be generated by a call to ::launch_parameters
  BsrMatrixSpMVTensorCoreFunctor(
      YScalar _alpha, AMatrix _a, XMatrix _x, YScalar _beta, YMatrix _y,
      const BsrMatrixSpMVTensorCoreFunctorParams &_params)
      : alpha(_alpha), a(_a), x(_x), beta(_beta), y(_y), params(_params) {}

  size_t league_size() const { return params.leagueDim_x * params.leagueDim_y; }

  size_t team_size() const {
    return THREADS_PER_WARP * WARPS_PER_TEAM_X * WARPS_PER_TEAM_Y;
  }

  // single column of fragments from A
  KOKKOS_INLINE_FUNCTION size_t a_scratch_size() const {
    return WARPS_PER_TEAM_Y * FRAG_M * FRAG_K * sizeof(AFragScalar);
  }
  // single row of fragments from X
  KOKKOS_INLINE_FUNCTION size_t x_scratch_size() const {
    return WARPS_PER_TEAM_X * FRAG_K * FRAG_N * sizeof(XFragScalar);
  }
  // one fragment per warp in the team
  KOKKOS_INLINE_FUNCTION size_t y_scratch_size() const {
    return WARPS_PER_TEAM_X * WARPS_PER_TEAM_Y * FRAG_M * FRAG_N *
           sizeof(YFragScalar);
  }

  size_t team_scratch_size() const {
    return a_scratch_size() + x_scratch_size() + y_scratch_size();
  }

  /// \brief determine the mapping parameters for the 1D Kokkos::parallel_for
  /// space to the hierarchical 2D space of the functor kernel. This should be
  /// called to determine what arguments to pass to the constructor
  static BsrMatrixSpMVTensorCoreFunctorParams launch_parameters(
      const YScalar & /*alpha*/, const AMatrix &a, const XMatrix & /*x*/,
      const YScalar & /*beta*/, const YMatrix &y) {
    BsrMatrixSpMVTensorCoreFunctorParams params;

    // compute how many blocks there are in each dimension of the product MV
    int blocksPerYM = (y.extent(0) + a.blockDim() - 1) / a.blockDim();
    int blocksPerYN = (y.extent(1) + a.blockDim() - 1) / a.blockDim();

    // compute how many fragments are needed to cover each block
    int fragsPerBlockM = (a.blockDim() + FRAG_M - 1) / FRAG_M;
    int fragsPerBlockN = (a.blockDim() + FRAG_N - 1) / FRAG_N;

    // determine how many teams will need to cover each block (Y in M direction,
    // X in N direction)
    params.teamsPerBlockM =
        (fragsPerBlockM + WARPS_PER_TEAM_Y - 1) / WARPS_PER_TEAM_Y;
    params.teamsPerBlockN =
        (fragsPerBlockN + WARPS_PER_TEAM_X - 1) / WARPS_PER_TEAM_X;

    // determine how many teams will be needed co cover the product vector
    int yTeamsM = params.teamsPerBlockM * blocksPerYM;
    int yTeamsN = params.teamsPerBlockN * blocksPerYN;

    // Y dimension to M, X dimension to N
    params.leagueDim_x = yTeamsN;
    params.leagueDim_y = yTeamsM;

    return params;
  }

  // execute the functor with provided launch parameters
  void dispatch() {
    typename BsrMatrixSpMVTensorCoreFunctor::team_policy policy(league_size(),
                                                                team_size());
    policy.set_scratch_size(0, Kokkos::PerTeam(team_scratch_size()));
    Kokkos::parallel_for("KokkosSparse::BsrMatrixSpMVTensorCoreFunctor", policy,
                         *this);
  }

  /*
     Consider the product vector as being made up of blocks that are the
     same size as the blocks in the input sparse matrix.
     teams are tiled across each block
     the size of each team is determined by the 2D grid of warps in the team,
     and the shape of each warp's fragment

     The number of warps per team is static:
     WARPS_PER_TEAM_X * WARPS_PER_TEAM_Y

     Based on its position in the product vector, each team steps through
     corresponding block-sized tiles of A and X. Those tiles are loaded into
     shared memory
     Warps in the team iterate over the shared-memory tile and perform the
     accumuation
     then the fragments write the results back to shared memory, and then
     global memory
  */

  KOKKOS_INLINE_FUNCTION void operator()(const team_member &mbr) const {
    using nvcuda::wmma::fill_fragment;
    using nvcuda::wmma::load_matrix_sync;
    using nvcuda::wmma::mma_sync;
    using nvcuda::wmma::store_matrix_sync;

    FragA fa;
    FragX fx;
    FragY fy;

    // override with template params if given
    const int ld_x = LEAGUE_DIM_X > 0 ? LEAGUE_DIM_X : params.leagueDim_x;
    const int tpbn =
        TEAMS_PER_BLOCK_N > 0 ? TEAMS_PER_BLOCK_N : params.teamsPerBlockN;
    const int tpbm =
        TEAMS_PER_BLOCK_M > 0 ? TEAMS_PER_BLOCK_M : params.teamsPerBlockM;

    // which team I am in the league
    const int teamIdx_x = mbr.league_rank() % ld_x;
    const int teamIdx_y = mbr.league_rank() / ld_x;

    // which block I contribute to in the product vector
    const int blockIdx_i = teamIdx_y / tpbm;
    const int blockIdx_j = teamIdx_x / tpbn;

    // which team am I in the block
    const int teamIdx_i = teamIdx_y % tpbm;
    const int teamIdx_j = teamIdx_x % tpbn;

    // which warp I am in the team
    const int warpIdx_x = (mbr.team_rank() / 32) % WARPS_PER_TEAM_X;
    const int warpIdx_y = (mbr.team_rank() / 32) / WARPS_PER_TEAM_X;

    // which lane I am in the warp
    const int lx = mbr.team_rank() % THREADS_PER_WARP;

    // which row of a/y the fragment this warp contributes to starts at
    const AOrdinal ay_i =
        blockIdx_i * a.blockDim()                // offset due to block
        + teamIdx_i * WARPS_PER_TEAM_Y * FRAG_M  // offset of team within block
        + warpIdx_y * FRAG_M;                    // offset of warp within team

    // which column of x/y the fragments warp will read from/contribute to
    // starts at
    const AOrdinal xy_j = blockIdx_j * a.blockDim() +
                          teamIdx_j * WARPS_PER_TEAM_X * FRAG_N +
                          warpIdx_x * FRAG_N;

    AFragScalar *_sa =
        (AFragScalar *)mbr.team_shmem().get_shmem(a_scratch_size());
    XFragScalar *_sx =
        (XFragScalar *)mbr.team_shmem().get_shmem(x_scratch_size());
    YFragScalar *_sy =
        (YFragScalar *)mbr.team_shmem().get_shmem(y_scratch_size());

    AScratchView sa(_sa, WARPS_PER_TEAM_Y);
    XScratchView sx(_sx, WARPS_PER_TEAM_X);
    YScratchView sy(_sy, WARPS_PER_TEAM_Y, WARPS_PER_TEAM_X);

    // team loads its fragments of Y that make up part or all of the block of Y
    // it's responsible for. each warp loads the part corresponding to its y
    // fragment stage through shared memory to convert to fragment type

    // no need for a team barrier because each warp uses an individual part of
    // shared memory
    for (unsigned i = lx; i < FRAG_M * FRAG_N; i += THREADS_PER_WARP) {
      const unsigned fi = i / FRAG_N;  // position in fragment of Y
      const unsigned fj = i % FRAG_N;
      const AOrdinal bi = teamIdx_i * WARPS_PER_TEAM_Y * FRAG_M +
                          warpIdx_y * FRAG_M + fi;  // position in block of Y
      const AOrdinal bj =
          teamIdx_j * WARPS_PER_TEAM_X * FRAG_N + warpIdx_x * FRAG_N + fj;

      // load 0 outside of the block boundary and y vector boundary
      // load 0 outside of the vector boundary
      if (bi < a.blockDim() && bj < a.blockDim() && xy_j + fj < y.extent(1)) {
        sy(warpIdx_y, warpIdx_x, fi, fj) = beta * y(ay_i + fi, xy_j + fj);
      } else {
        sy(warpIdx_y, warpIdx_x, fi, fj) = 0;
      }
    }
    // no barrier - each warp uses independent shared memory

    // load from the shared memory
#ifdef __CUDA_ARCH__
    load_matrix_sync(fy, &sy(warpIdx_y, warpIdx_x, 0, 0), FRAG_N,
                     nvcuda::wmma::mem_row_major);
#endif

    auto rowView = a.block_row_Const(blockIdx_i);

    // team loops through all blocks in the row
    for (AOffset ci = a.graph.row_map(blockIdx_i);
         ci < a.graph.row_map(blockIdx_i + 1); ++ci) {
      AOrdinal j = a.graph.entries(ci);

      // pointer to the beginning of the block
      const AScalar *ap = nullptr;
      {
        size_t off =
            ci - a.graph.row_map(blockIdx_i);     // which block in this row
        ap = rowView.local_row_in_block(off, 0);  // offset of this block
      }

      // the block may be bigger than a single team,
      // each team is only one fragment long in the K direction
      // so will need to iterate fragments in the K direction across the block
      // the team will collaboratively load the fragments from A and X

      // and require multiple loads and accumulates
      // for mxn grid of fragments in the product vector, we need m rows of
      // fragments from A and n cols of fragments from X. only hold part of a
      // single column of fragments (from A) or part of a single row (from X) at
      // once
      for (AOrdinal bk = 0; bk < a.blockDim(); bk += FRAG_K /*M*/) {
        // team collaborative load of A
        // the footprint is one fragment wide in K direction
        mbr.team_barrier();
        for (unsigned i = mbr.team_rank();
             i < WARPS_PER_TEAM_Y * FRAG_M * FRAG_K; i += mbr.team_size()) {
          const unsigned ti = i / FRAG_K;  // offset inside the fragments
          const unsigned tj = i % FRAG_K;
          // add in offset within block
          const AOrdinal bi = teamIdx_i * WARPS_PER_TEAM_Y * FRAG_M + ti;
          const AOrdinal bj = bk + tj;

          // fill shmem with 0 outside of the block boundary
          if (bi < a.blockDim() && bj < a.blockDim()) {
            sa(ti / FRAG_M, ti % FRAG_M, tj) =
                alpha * ap[bi * a.blockDim() + bj];
          } else {
            sa(ti / FRAG_M, ti % FRAG_M, tj) = 0;
          }
        }

        // collaborative load of X fragments into shared memory
        // entire team loads fragment footprint
        for (unsigned i = mbr.team_rank();
             i < WARPS_PER_TEAM_X * FRAG_N * FRAG_K; i += mbr.team_size()) {
          const unsigned ti =
              i / (WARPS_PER_TEAM_X * FRAG_N);  // position in combined tiles
          const unsigned tj = i % (WARPS_PER_TEAM_X * FRAG_N);

          // add in offset within block
          const AOrdinal bi = bk + ti;
          const AOrdinal bj = teamIdx_j * WARPS_PER_TEAM_X * FRAG_N + tj;

          // load 0 outside of the block boundary
          // x is not necessarily a multiple of block size, so make sure access
          // is in bounds
          if (bi < a.blockDim() && bj < a.blockDim() &&
              unsigned(blockIdx_j * a.blockDim() + bj) < x.extent(1)) {
            // tile is some fragments in the j/n direction that are frag_n wide
            sx(tj / FRAG_N, ti, tj % FRAG_N) = XFragScalar(
                x(j * a.blockDim() + bi, blockIdx_j * a.blockDim() + bj));
          } else {
            sx(tj / FRAG_N, ti, tj % FRAG_N) = 0;
          }
        }
        mbr.team_barrier();

        // load correct fragment from shared memory and accumulate
#ifdef __CUDA_ARCH__

        // only need to do any math if our fragment will write a result back to
        // Y
        if (ay_i < static_cast<AOrdinal>(y.extent(0)) &&
            xy_j < static_cast<AOrdinal>(y.extent(1))) {
          load_matrix_sync(fa, &sa(warpIdx_y, 0, 0), FRAG_K);
          load_matrix_sync(fx, &sx(warpIdx_x, 0, 0), FRAG_N);
          mma_sync(fy, fa, fx, fy);
        }
#endif
      }
    }  // loop through blocks in row of A

    // store Y fragments into shared memory
    store_matrix_sync(&sy(warpIdx_y, warpIdx_x, 0, 0), fy, FRAG_N,
                      nvcuda::wmma::mem_row_major);
    // team loads its fragments of Y that make up part or all of the block of Y
    // it's responsible for. each warp loads the part corresponding to its y
    // fragment
    mbr.team_barrier();
    for (unsigned i = lx; i < FRAG_M * FRAG_N; i += THREADS_PER_WARP) {
      const unsigned fi = i / FRAG_N;  // position in fragment of Y
      const unsigned fj = i % FRAG_N;
      const AOrdinal bi = teamIdx_i * WARPS_PER_TEAM_Y * FRAG_M +
                          warpIdx_y * FRAG_M + fi;  // position in block of Y
      const AOrdinal bj =
          teamIdx_j * WARPS_PER_TEAM_X * FRAG_N + warpIdx_x * FRAG_N + fj;

      // only store inside the block boundary
      // FIXME: what if Y is not wide enough? check y(_, j)
      if (bi < a.blockDim() && bj < a.blockDim() && xy_j + fj < y.extent(1)) {
        y(ay_i + fi, xy_j + fj) = sy(warpIdx_y, warpIdx_x, fi, fj);
      }
    }
    mbr.team_barrier();

    // Suppress unused var warnings
    // TODO (@cwpeason): Should this functor only compile on device?
    (void)fx;
    (void)fa;
    (void)fy;
  }
};

/* Instantiate some common template parameter values
   for BsrMatrixSpMVTensorCoreFunctor.
   This is a struct instead of a function for template...using shorthand
   Discriminates between complex (supported) and non-complex (unsupported)
   scalar types, and throws a runtime error for unsupported types
*/
template <typename AMatrix,
          typename AFragScalar,  // input matrix type and fragment scalar type
          typename XMatrix, typename XFragScalar, typename YMatrix,
          typename YFragScalar, unsigned FRAG_M, unsigned FRAG_N,
          unsigned FRAG_K>
struct BsrMatrixSpMVTensorCoreDispatcher {
  typedef typename AMatrix::value_type AScalar;
  typedef typename YMatrix::value_type YScalar;
  typedef typename XMatrix::value_type XScalar;

  template <unsigned X, unsigned Y, unsigned Z>
  using Dyn = BsrMatrixSpMVTensorCoreFunctor<AMatrix, AFragScalar, XMatrix,
                                             XFragScalar, YMatrix, YFragScalar,
                                             FRAG_M, FRAG_N, FRAG_K, X, Y, Z>;

  // to be used when the various matrix types are supported
  static void tag_dispatch(std::true_type, YScalar alpha, AMatrix a, XMatrix x,
                           YScalar beta, YMatrix y) {
    BsrMatrixSpMVTensorCoreFunctorParams params =
        Dyn<0, 0, 0>::launch_parameters(alpha, a, x, beta, y);

    if (false) {  // consistency of formatting for next sections
    } else if (1 == params.leagueDim_x && 1 == params.teamsPerBlockM &&
               1 == params.teamsPerBlockN) {
      Dyn<1, 1, 1>(alpha, a, x, beta, y, params).dispatch();
    } else if (1 == params.leagueDim_x && 2 == params.teamsPerBlockM &&
               2 == params.teamsPerBlockN) {
      Dyn<1, 2, 2>(alpha, a, x, beta, y, params).dispatch();
    } else if (1 == params.leagueDim_x && 4 == params.teamsPerBlockM &&
               4 == params.teamsPerBlockN) {
      Dyn<1, 4, 4>(alpha, a, x, beta, y, params).dispatch();
    } else if (1 == params.leagueDim_x && 8 == params.teamsPerBlockM &&
               8 == params.teamsPerBlockN) {
      Dyn<1, 8, 8>(alpha, a, x, beta, y, params).dispatch();
    } else if (2 == params.leagueDim_x && 1 == params.teamsPerBlockM &&
               1 == params.teamsPerBlockN) {
      Dyn<2, 1, 1>(alpha, a, x, beta, y, params).dispatch();
    } else if (2 == params.leagueDim_x && 2 == params.teamsPerBlockM &&
               2 == params.teamsPerBlockN) {
      Dyn<2, 2, 2>(alpha, a, x, beta, y, params).dispatch();
    } else if (2 == params.leagueDim_x && 4 == params.teamsPerBlockM &&
               4 == params.teamsPerBlockN) {
      Dyn<2, 4, 4>(alpha, a, x, beta, y, params).dispatch();
    } else if (2 == params.leagueDim_x && 8 == params.teamsPerBlockM &&
               8 == params.teamsPerBlockN) {
      Dyn<2, 8, 8>(alpha, a, x, beta, y, params).dispatch();
    } else {
      Dyn<0, 0, 0>(alpha, a, x, beta, y, params).dispatch();
    }
  }

  // to be used to avoid instantiating on unsupported types
  static void tag_dispatch(std::false_type, YScalar, AMatrix, XMatrix, YScalar,
                           YMatrix) {
    Kokkos::Impl::throw_runtime_exception("unsupported for complex types");
  }

  /*true if T1, T2, or T3 are complex*/
  template <typename T1, typename T2, typename T3>
  struct none_complex {
    const static bool value = !Kokkos::ArithTraits<T1>::is_complex &&
                              !Kokkos::ArithTraits<T2>::is_complex &&
                              !Kokkos::ArithTraits<T3>::is_complex;
  };

  static void dispatch(YScalar alpha, AMatrix a, XMatrix x, YScalar beta,
                       YMatrix y) {
    using tag =
        std::integral_constant<bool,
                               none_complex<AScalar, XScalar, YScalar>::value>;
    tag_dispatch(tag{}, alpha, a, x, beta, y);
  }
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace KokkosSparse

#endif  // #if CUDA && (VOLTA || AMPERE)

//
//
//

#include "KokkosBlas.hpp"
#include "KokkosBatched_Gemv_Serial_Internal.hpp"
#include "KokkosBatched_Gemm_Serial_Internal.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"
#include "KokkosSparse_spmv_impl.hpp"

namespace KokkosSparse {
namespace Experimental {
namespace Impl {
namespace Bsr {

template <class AMatrix, class XVector, class YVector>
struct BSR_GEMV_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef Kokkos::Details::ArithTraits<value_type> ATV;

  //! Nonconst version of the type of column indices in the sparse matrix.
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  //! Nonconst version of the type of row offsets in the sparse matrix.
  typedef typename AMatrix::non_const_size_type size_type;

  const value_type alpha;
  AMatrix m_A;
  XVector m_x;
  YVector m_y;

  const ordinal_type block_dim;
  const ordinal_type blocks_per_team;

  bool conjugate = false;

  BSR_GEMV_Functor(const value_type alpha_, const AMatrix m_A_,
                   const XVector m_x_, const YVector m_y_,
                   const int blocks_per_team_, bool conj_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        m_y(m_y_),
        block_dim(m_A_.blockDim()),
        blocks_per_team(blocks_per_team_),
        conjugate(conj_) {
    static_assert(static_cast<int>(XVector::rank) == 1,
                  "XVector must be a rank 1 View.");
    static_assert(static_cast<int>(YVector::rank) == 1,
                  "YVector must be a rank 1 View.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ordinal_type iBlock) const {
    const auto ystart = iBlock * block_dim;
    const auto start  = m_A.graph.row_map(iBlock);
    const ordinal_type count =
        static_cast<ordinal_type>(m_A.graph.row_map(iBlock + 1) - start);
    const auto row   = m_A.block_row_Const(iBlock);
    const auto beta1 = static_cast<value_type>(1);
    //
    if (conjugate) {
      std::vector<value_type> conjA(block_dim * block_dim);
      for (ordinal_type ic = 0; ic < count; ++ic) {
        const auto Aview = row.block(ic);
        for (ordinal_type ii = 0; ii < block_dim; ++ii) {
          for (ordinal_type jj = 0; jj < block_dim; ++jj)
            conjA[jj + ii * block_dim] =
                Kokkos::ArithTraits<value_type>::conj(Aview(ii, jj));
        }
        //
        const auto xstart = row.block_colidx(ic) * block_dim;
        KokkosBatched::SerialGemvInternal<KokkosBatched::Algo::Gemv::Blocked>::
            invoke<value_type, value_type>(block_dim, block_dim, alpha,
                                           conjA.data(), block_dim, 1,
                                           &m_x(xstart), m_x.stride_0(), beta1,
                                           &m_y(ystart), m_y.stride_0());
      }
    } else {
      for (ordinal_type ic = 0; ic < count; ++ic) {
        const auto Aview  = row.block(ic);
        const auto xstart = row.block_colidx(ic) * block_dim;
        KokkosBatched::SerialGemvInternal<KokkosBatched::Algo::Gemv::Blocked>::
            invoke<value_type, value_type>(block_dim, block_dim, alpha,
                                           Aview.data(), block_dim, 1,
                                           &m_x(xstart), m_x.stride_0(), beta1,
                                           &m_y(ystart), m_y.stride_0());
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &dev) const {
    using y_value_type = typename YVector::non_const_value_type;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(dev, 0, blocks_per_team),
        [&](const ordinal_type &loop) {
          const ordinal_type iBlock =
              static_cast<ordinal_type>(dev.league_rank()) * blocks_per_team +
              loop;
          if (iBlock >= m_A.numRows()) {
            return;
          }
          const auto start = m_A.graph.row_map(iBlock);
          const ordinal_type count =
              static_cast<ordinal_type>(m_A.graph.row_map(iBlock + 1) - start);
          const auto row   = m_A.block_row_Const(iBlock);
          const auto beta1 = static_cast<y_value_type>(1);
          //
          auto yview = Kokkos::subview(
              m_y, Kokkos::make_pair(iBlock * block_dim,
                                     iBlock * block_dim + block_dim));
          //
          for (ordinal_type ic = 0; ic < count; ++ic) {
            const auto Aview  = row.block(ic);
            const auto xstart = row.block_colidx(ic) * block_dim;
            const auto xview  = Kokkos::subview(
                m_x, Kokkos::make_pair(xstart, xstart + block_dim));
            KokkosBlas::gemv("N", alpha, Aview, xview, beta1, yview);
          }
          /*
          //
          for (ordinal_type ir = 0; ir < block_dim; ++ir) {
            y_value_type sum = 0;

            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(dev, count),
                [&](const ordinal_type &iEntry, y_value_type &lsum) {
                  const auto start_col = row.block_colidx(iEntry) * block_dim;
                  for (ordinal_type jr = 0; jr < block_dim; ++jr) {
                    const value_type val =
                        conjugate
                            ? ATV::conj(row.local_block_value(iEntry, ir, jr))
                            : row.local_block_value(iEntry, ir, jr);
                    lsum += val * m_x(start_col + jr);
                  }
                },
                sum);

            Kokkos::single(Kokkos::PerThread(dev), [&]() {
              sum *= alpha;
              m_y(iBlock * block_dim + ir) += sum;
            });
          }
           */
        });
  }
};

/* ******************* */

//
// spMatVec_no_transpose: version for CPU execution spaces
// (RangePolicy or trivial serial impl used)
//
template <class AT, class AO, class AD, class AS, class AlphaType,
          class XVector, class BetaType, class YVector,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_no_transpose(
    const KokkosKernels::Experimental::Controls &controls,
    const AlphaType &alpha,
    const KokkosSparse::Experimental::BsrMatrix<
        AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> &A,
    const XVector &x, const BetaType &beta, YVector &y, bool useConjugate) {
  // This is required to maintain semantics of KokkosKernels native SpMV:
  // if y contains NaN but beta = 0, the result y should be filled with 0.
  // For example, this is useful for passing in uninitialized y and beta=0.
  if (beta == Kokkos::ArithTraits<BetaType>::zero())
    Kokkos::deep_copy(y, Kokkos::ArithTraits<BetaType>::zero());
  else
    KokkosBlas::scal(y, beta, y);

  //
  // Treat the case y <- alpha * A * x + beta * y
  //

  typedef KokkosSparse::Experimental::BsrMatrix<
      AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS>
      AMatrix_Internal;
  typedef typename AMatrix_Internal::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix_Internal::execution_space execution_space;

  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }

  BSR_GEMV_Functor<AMatrix_Internal, XVector, YVector> func(alpha, A, x, y, 1,
                                                            useConjugate);
  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::parallel_for(
        "KokkosSparse::bspmv<NoTranspose,Dynamic>",
        Kokkos::RangePolicy<
            typename AMatrix_Internal::device_type::execution_space,
            Kokkos::Schedule<Kokkos::Dynamic>>(0, A.numRows()),
        func);
  } else {
    Kokkos::parallel_for(
        "KokkosSparse::bspmv<NoTranspose,Static>",
        Kokkos::RangePolicy<
            typename AMatrix_Internal::device_type::execution_space,
            Kokkos::Schedule<Kokkos::Static>>(0, A.numRows()),
        func);
  }
}

/* ******************* */

//
// spMatVec_no_transpose: version for GPU execution spaces (TeamPolicy used)
//
template <class AT, class AO, class AD, class AS, class AlphaType,
          class XVector, class BetaType, class YVector,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_no_transpose(
    const KokkosKernels::Experimental::Controls &controls,
    const AlphaType &alpha,
    const KokkosSparse::Experimental::BsrMatrix<
        AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> &A,
    const XVector &x, const BetaType &beta, YVector &y, bool useConjugate) {
  if (A.numRows() <= static_cast<AO>(0)) {
    return;
  }

  // We need to scale y first ("scaling" by zero just means filling
  // with zeros), since the functor works by atomic-adding into y.
  KokkosBlas::scal(y, beta, y);

  typedef KokkosSparse::Experimental::BsrMatrix<
      AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS>
      AMatrix_Internal;
  typedef typename AMatrix_Internal::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix_Internal::execution_space execution_space;

  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }
  int team_size             = -1;
  int vector_length         = -1;
  int64_t blocks_per_thread = -1;

  //
  // Use the controls to allow the user to pass in some tuning parameters.
  //
  if (controls.isParameter("team size")) {
    team_size = std::stoi(controls.getParameter("team size"));
  }
  if (controls.isParameter("vector length")) {
    vector_length = std::stoi(controls.getParameter("vector length"));
  }
  if (controls.isParameter("rows per thread")) {
    blocks_per_thread = std::stoll(controls.getParameter("rows per thread"));
  }

  //
  // Use the existing launch parameters routine from SPMV
  //
  int64_t blocks_per_team =
      KokkosSparse::Impl::spmv_launch_parameters<execution_space>(
          A.numRows(), A.nnz(), blocks_per_thread, team_size, vector_length);
  int64_t worksets = (A.numRows() + blocks_per_team - 1) / blocks_per_team;

  AMatrix_Internal A_internal = A;

  BSR_GEMV_Functor<AMatrix_Internal, XVector, YVector> func(
      alpha, A_internal, x, y, blocks_per_team, useConjugate);

  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>
        policy(1, 1);
    if (team_size < 0)
      policy = Kokkos::TeamPolicy<execution_space,
                                  Kokkos::Schedule<Kokkos::Dynamic>>(
          worksets, Kokkos::AUTO, vector_length);
    else
      policy = Kokkos::TeamPolicy<execution_space,
                                  Kokkos::Schedule<Kokkos::Dynamic>>(
          worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::bspmv<NoTranspose,Dynamic>", policy,
                         func);
  } else {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>
        policy(1, 1);
    if (team_size < 0)
      policy =
          Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
              worksets, Kokkos::AUTO, vector_length);
    else
      policy =
          Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
              worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::bspmv<NoTranspose, Static>", policy,
                         func);
  }
}

/* ******************* */

template <class AMatrix, class XVector, class YVector>
struct BSR_GEMV_Transpose_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef Kokkos::Details::ArithTraits<value_type> ATV;

  //! Nonconst version of the type of column indices in the sparse matrix.
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  //! Nonconst version of the type of row offsets in the sparse matrix.
  typedef typename AMatrix::non_const_size_type size_type;

  const value_type alpha;

  AMatrix m_A;
  XVector m_x;
  YVector m_y;

  const ordinal_type block_dim;
  const ordinal_type blocks_per_team;

  bool conjugate = false;

  BSR_GEMV_Transpose_Functor(const value_type alpha_, const AMatrix m_A_,
                             const XVector m_x_, const YVector m_y_,
                             const int blocks_per_team_, bool conj_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        m_y(m_y_),
        block_dim(m_A_.blockDim()),
        blocks_per_team(blocks_per_team_),
        conjugate(conj_) {
    static_assert(static_cast<int>(XVector::rank) == 1,
                  "XVector must be a rank 1 View.");
    static_assert(static_cast<int>(YVector::rank) == 1,
                  "YVector must be a rank 1 View.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ordinal_type iBlock) const {
    //
    // Assume that alpha is not zero
    //
    constexpr ordinal_type numBlocks = 1;
    //
    const auto xstart = iBlock * block_dim;
    const auto xview =
        Kokkos::subview(m_x, Kokkos::make_pair(xstart, xstart + block_dim));
    const auto start = m_A.graph.row_map(iBlock);
    const ordinal_type count =
        static_cast<ordinal_type>(m_A.graph.row_map(iBlock + 1) - start);
    const auto row   = m_A.block_row_Const(iBlock);
    const auto beta1 = static_cast<value_type>(1);
    std::vector<value_type> ytmp(block_dim);
    if (conjugate) {
      std::vector<value_type> conjA(block_dim * block_dim);
      for (ordinal_type ic = 0; ic < count; ++ic) {
        const auto Aview = row.block(ic);
        for (ordinal_type ii = 0; ii < block_dim; ++ii) {
          for (ordinal_type jj = 0; jj < block_dim; ++jj)
            conjA[jj + ii * block_dim] =
                Kokkos::ArithTraits<value_type>::conj(Aview(ii, jj));
        }
        //
        for (ordinal_type ii = 0; ii < block_dim; ++ii)
          ytmp[ii] = static_cast<value_type>(0);
        KokkosBatched::SerialGemvInternal<KokkosBatched::Algo::Gemv::Blocked>::
            invoke<value_type, value_type>(
                block_dim, block_dim, alpha, conjA.data(), 1, block_dim,
                xview.data(), xview.stride_0(), beta1, ytmp.data(), 1);
        //
        const auto ystart = row.block_colidx(ic) * block_dim;
        for (ordinal_type ir = 0; ir < block_dim; ++ir)
          Kokkos::atomic_add(&m_y(ystart + ir), ytmp[ir]);
      }
    } else {
      for (ordinal_type ic = 0; ic < count; ++ic) {
        const auto Aview = row.block(ic);
        for (ordinal_type ii = 0; ii < block_dim; ++ii)
          ytmp[ii] = static_cast<value_type>(0);
        KokkosBatched::SerialGemvInternal<KokkosBatched::Algo::Gemv::Blocked>::
            invoke<value_type, value_type>(
                block_dim, block_dim, alpha, Aview.data(), 1, block_dim,
                xview.data(), xview.stride_0(), beta1, ytmp.data(), 1);
        //
        const auto ystart = row.block_colidx(ic) * block_dim;
        for (ordinal_type ir = 0; ir < block_dim; ++ir)
          Kokkos::atomic_add(&m_y(ystart + ir), ytmp[ir]);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &dev) const {
    using y_value_type = typename YVector::non_const_value_type;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(dev, 0, blocks_per_team),
        [&](const ordinal_type &loop) {
          const ordinal_type iBlock =
              static_cast<ordinal_type>(dev.league_rank()) * blocks_per_team +
              loop;
          if (iBlock >= m_A.numRows()) {
            return;
          }
          const auto start = m_A.graph.row_map(iBlock);
          const ordinal_type count =
              static_cast<ordinal_type>(m_A.graph.row_map(iBlock + 1) - start);
          const auto row = m_A.block_row_Const(iBlock);
          //
          for (ordinal_type ir = 0; ir < block_dim; ++ir) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(dev, count),
                [&](const ordinal_type &iEntry) {
                  for (ordinal_type jr = 0; jr < block_dim; ++jr) {
                    const value_type val =
                        conjugate
                            ? ATV::conj(row.local_block_value(iEntry, jr, ir))
                            : row.local_block_value(iEntry, jr, ir);
                    const ordinal_type ind = row.block_colidx(iEntry);
                    Kokkos::atomic_add(
                        &m_y(block_dim * ind + ir),
                        static_cast<y_value_type>(
                            alpha * val * m_x(block_dim * iBlock + jr)));
                  }
                });
          }
        });
  }
};

/* ******************* */

/// \brief  spMatVec_transpose: version for CPU execution spaces (RangePolicy or
/// trivial serial impl used)
template <class AT, class AO, class AD, class AS, class AlphaType,
          class XVector, class BetaType, class YVector,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_transpose(
    const KokkosKernels::Experimental::Controls &controls,
    const AlphaType &alpha,
    const KokkosSparse::Experimental::BsrMatrix<
        AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> &A,
    const XVector &x, const BetaType &beta, YVector &y, bool useConjugate) {
  // This is required to maintain semantics of KokkosKernels native SpMV:
  // if y contains NaN but beta = 0, the result y should be filled with 0.
  // For example, this is useful for passing in uninitialized y and beta=0.
  if (beta == Kokkos::ArithTraits<BetaType>::zero())
    Kokkos::deep_copy(y, Kokkos::ArithTraits<BetaType>::zero());
  else
    KokkosBlas::scal(y, beta, y);

  if (alpha == Kokkos::ArithTraits<AlphaType>::zero()) return;

  //
  // Treat the case y <- alpha * A^T * x + beta * y
  //

  typedef KokkosSparse::Experimental::BsrMatrix<
      AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS>
      AMatrix_Internal;
  typedef typename AMatrix_Internal::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix_Internal::execution_space execution_space;

  AMatrix_Internal A_internal = A;

  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }

  BSR_GEMV_Transpose_Functor<AMatrix_Internal, XVector, YVector> func(
      alpha, A_internal, x, y, 1, useConjugate);
  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::parallel_for(
        "KokkosSparse::bspmv<Transpose,Dynamic>",
        Kokkos::RangePolicy<
            typename AMatrix_Internal::device_type::execution_space,
            Kokkos::Schedule<Kokkos::Dynamic>>(0, A.numRows()),
        func);
  } else {
    Kokkos::parallel_for(
        "KokkosSparse::bspmv<Transpose,Static>",
        Kokkos::RangePolicy<
            typename AMatrix_Internal::device_type::execution_space,
            Kokkos::Schedule<Kokkos::Static>>(0, A.numRows()),
        func);
  }
}

//
// spMatVec_transpose: version for GPU execution spaces (TeamPolicy used)
//
template <class AMatrix, class AlphaType, class XVector, class BetaType,
          class YVector,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_transpose(const KokkosKernels::Experimental::Controls &controls,
                        const AlphaType &alpha, const AMatrix &A,
                        const XVector &x, const BetaType &beta, YVector &y,
                        bool useConjugate) {
  if (A.numRows() <= 0) {
    return;
  }

  // We need to scale y first ("scaling" by zero just means filling
  // with zeros), since the functor works by atomic-adding into y.
  KokkosBlas::scal(y, beta, y);

  typedef typename AMatrix::execution_space execution_space;

  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }
  int team_size             = -1;
  int vector_length         = -1;
  int64_t blocks_per_thread = -1;

  //
  // Use the controls to allow the user to pass in some tuning parameters.
  //
  if (controls.isParameter("team size")) {
    team_size = std::stoi(controls.getParameter("team size"));
  }
  if (controls.isParameter("vector length")) {
    vector_length = std::stoi(controls.getParameter("vector length"));
  }
  if (controls.isParameter("rows per thread")) {
    blocks_per_thread = std::stoll(controls.getParameter("rows per thread"));
  }

  //
  // Use the existing launch parameters routine from SPMV
  //
  int64_t blocks_per_team =
      KokkosSparse::Impl::spmv_launch_parameters<execution_space>(
          A.numRows(), A.nnz(), blocks_per_thread, team_size, vector_length);
  int64_t worksets = (A.numRows() + blocks_per_team - 1) / blocks_per_team;

  BSR_GEMV_Transpose_Functor<AMatrix, XVector, YVector> func(
      alpha, A, x, y, blocks_per_team, useConjugate);

  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>
        policy(1, 1);
    if (team_size < 0)
      policy = Kokkos::TeamPolicy<execution_space,
                                  Kokkos::Schedule<Kokkos::Dynamic>>(
          worksets, Kokkos::AUTO, vector_length);
    else
      policy = Kokkos::TeamPolicy<execution_space,
                                  Kokkos::Schedule<Kokkos::Dynamic>>(
          worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::bspmv<Transpose,Dynamic>", policy,
                         func);
  } else {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>
        policy(1, 1);
    if (team_size < 0)
      policy =
          Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
              worksets, Kokkos::AUTO, vector_length);
    else
      policy =
          Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
              worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::bspmv<Transpose, Static>", policy,
                         func);
  }
}

/* ******************* */

template <class AMatrix, class XVector, class YVector>
struct BSR_GEMM_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef Kokkos::Details::ArithTraits<value_type> ATV;

  //! Nonconst version of the type of column indices in the sparse matrix.
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  //! Nonconst version of the type of row offsets in the sparse matrix.
  typedef typename AMatrix::non_const_size_type size_type;

  const value_type alpha;
  AMatrix m_A;
  XVector m_x;
  YVector m_y;
  const ordinal_type block_dim;
  const ordinal_type num_rhs;

  const ordinal_type blocks_per_team;

  bool conjugate = false;

  BSR_GEMM_Functor(const value_type alpha_, const AMatrix m_A_,
                   const XVector m_x_, const YVector m_y_,
                   const int blocks_per_team_, bool conj_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        m_y(m_y_),
        block_dim(m_A_.blockDim()),
        num_rhs(m_x_.extent(1)),
        blocks_per_team(blocks_per_team_),
        conjugate(conj_) {
    static_assert(static_cast<int>(XVector::rank) == 2,
                  "XVector must be a rank 2 View.");
    static_assert(static_cast<int>(YVector::rank) == 2,
                  "YVector must be a rank 2 View.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ordinal_type iBlock) const {
    //
    const auto ystart = iBlock * block_dim;
    const auto start  = m_A.graph.row_map(iBlock);
    const ordinal_type count =
        static_cast<ordinal_type>(m_A.graph.row_map(iBlock + 1) - start);
    const auto row   = m_A.block_row_Const(iBlock);
    const auto beta1 = static_cast<value_type>(1);
    const auto ldx   = m_x.stride_1();
    const auto ldy   = m_y.stride_1();
    //
    if (conjugate) {
      std::vector<value_type> conjA(block_dim * block_dim);
      for (ordinal_type ic = 0; ic < count; ++ic) {
        const auto Aview = row.block(ic);
        for (ordinal_type ii = 0; ii < block_dim; ++ii) {
          for (ordinal_type jj = 0; jj < block_dim; ++jj)
            conjA[jj + ii * block_dim] =
                Kokkos::ArithTraits<value_type>::conj(Aview(ii, jj));
        }
        //
        const auto xstart = row.block_colidx(ic) * block_dim;
        KokkosBatched::SerialGemmInternal<KokkosBatched::Algo::Gemm::Blocked>::
            invoke<value_type, value_type>(
                block_dim, num_rhs, block_dim, alpha, conjA.data(), block_dim,
                1, &m_x(xstart, 0), m_x.stride_0(), ldx, beta1, &m_y(ystart, 0),
                m_y.stride_0(), ldy);
      }
    } else {
      for (ordinal_type ic = 0; ic < count; ++ic) {
        const auto Aview  = row.block(ic);
        const auto xstart = row.block_colidx(ic) * block_dim;
        KokkosBatched::SerialGemmInternal<KokkosBatched::Algo::Gemm::Blocked>::
            invoke<value_type, value_type>(
                block_dim, num_rhs, block_dim, alpha, Aview.data(), block_dim,
                1, &m_x(xstart, 0), m_x.stride_0(), ldx, beta1, &m_y(ystart, 0),
                m_y.stride_0(), ldy);
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &dev) const {
    using y_value_type = typename YVector::non_const_value_type;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(dev, 0, blocks_per_team),
        [&](const ordinal_type &loop) {
          const ordinal_type iBlock =
              static_cast<ordinal_type>(dev.league_rank()) * blocks_per_team +
              loop;
          if (iBlock >= m_A.numRows()) {
            return;
          }
          //
          //
          const auto start = m_A.graph.row_map(iBlock);
          const ordinal_type count =
              static_cast<ordinal_type>(m_A.graph.row_map(iBlock + 1) - start);
          const KokkosSparse::Experimental::BsrRowViewConst<AMatrix> row(
              m_A.values, m_A.graph.entries, block_dim, count, start);
          const auto nrhs = m_x.extent(1);
          //
          for (ordinal_type ic = 0; ic < nrhs; ++ic) {
            for (ordinal_type ir = 0; ir < block_dim; ++ir) {
              y_value_type sum = 0;

              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(dev, count),
                  [&](const ordinal_type &iEntry, y_value_type &lsum) {
                    const auto start_col = row.block_colidx(iEntry) * block_dim;
                    for (ordinal_type jr = 0; jr < block_dim; ++jr) {
                      const value_type val =
                          conjugate
                              ? ATV::conj(row.local_block_value(iEntry, ir, jr))
                              : row.local_block_value(iEntry, ir, jr);
                      lsum += val * m_x(start_col + jr, ic);
                    }
                  },
                  sum);

              Kokkos::single(Kokkos::PerThread(dev), [&]() {
                sum *= alpha;
                m_y(iBlock * block_dim + ir, ic) += sum;
              });
            }
          }
          //
        });
  }
};

/* ******************* */

//
// spMatMultiVec_no_transpose: version for CPU execution spaces
// (RangePolicy or trivial serial impl used)
//
template <class AT, class AO, class AD, class AS, class AlphaType,
          class XVector, class BetaType, class YVector,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatMultiVec_no_transpose(
    const KokkosKernels::Experimental::Controls &controls,
    const AlphaType &alpha,
    const KokkosSparse::Experimental::BsrMatrix<
        AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> &A,
    const XVector &x, const BetaType &beta, YVector &y, bool useConjugate) {
  // This is required to maintain semantics of KokkosKernels native SpMV:
  // if y contains NaN but beta = 0, the result y should be filled with 0.
  // For example, this is useful for passing in uninitialized y and beta=0.
  if (beta == Kokkos::ArithTraits<BetaType>::zero())
    Kokkos::deep_copy(y, Kokkos::ArithTraits<BetaType>::zero());
  else
    KokkosBlas::scal(y, beta, y);

  //
  // Treat the case y <- alpha * A * x + beta * y
  //

  typedef KokkosSparse::Experimental::BsrMatrix<
      AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS>
      AMatrix_Internal;
  typedef typename AMatrix_Internal::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix_Internal::execution_space execution_space;

  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }

  BSR_GEMM_Functor<AMatrix_Internal, XVector, YVector> func(alpha, A, x, y, 1,
                                                            useConjugate);
  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::parallel_for(
        "KokkosSparse::bsr_spm_mv<NoTranspose,Dynamic>",
        Kokkos::RangePolicy<
            typename AMatrix_Internal::device_type::execution_space,
            Kokkos::Schedule<Kokkos::Dynamic>>(0, A.numRows()),
        func);
  } else {
    Kokkos::parallel_for(
        "KokkosSparse::bsr_spm_mv<NoTranspose,Static>",
        Kokkos::RangePolicy<
            typename AMatrix_Internal::device_type::execution_space,
            Kokkos::Schedule<Kokkos::Static>>(0, A.numRows()),
        func);
  }
}

/* ******************* */

//
// spMatMultiVec_no_transpose: version for GPU execution spaces (TeamPolicy
// used)
//
template <class AT, class AO, class AD, class AS, class AlphaType,
          class XVector, class BetaType, class YVector,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatMultiVec_no_transpose(
    const KokkosKernels::Experimental::Controls &controls,
    const AlphaType &alpha,
    const KokkosSparse::Experimental::BsrMatrix<
        AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> &A,
    const XVector &x, const BetaType &beta, YVector &y, bool useConjugate) {
  if (A.numRows() <= static_cast<AO>(0)) {
    return;
  }

  KokkosBlas::scal(y, beta, y);

  typedef KokkosSparse::Experimental::BsrMatrix<
      AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS>
      AMatrix_Internal;
  typedef typename AMatrix_Internal::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix_Internal::execution_space execution_space;

  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }
  int team_size             = -1;
  int vector_length         = -1;
  int64_t blocks_per_thread = -1;

  //
  // Use the controls to allow the user to pass in some tuning parameters.
  //
  if (controls.isParameter("team size")) {
    team_size = std::stoi(controls.getParameter("team size"));
  }
  if (controls.isParameter("vector length")) {
    vector_length = std::stoi(controls.getParameter("vector length"));
  }
  if (controls.isParameter("rows per thread")) {
    blocks_per_thread = std::stoll(controls.getParameter("rows per thread"));
  }

  //
  // Use the existing launch parameters routine from SPMV
  //
  int64_t blocks_per_team =
      KokkosSparse::Impl::spmv_launch_parameters<execution_space>(
          A.numRows(), A.nnz(), blocks_per_thread, team_size, vector_length);
  int64_t worksets = (A.numRows() + blocks_per_team - 1) / blocks_per_team;

  AMatrix_Internal A_internal = A;

  BSR_GEMM_Functor<AMatrix_Internal, XVector, YVector> func(
      alpha, A_internal, x, y, blocks_per_team, useConjugate);

  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>
        policy(1, 1);
    if (team_size < 0)
      policy = Kokkos::TeamPolicy<execution_space,
                                  Kokkos::Schedule<Kokkos::Dynamic>>(
          worksets, Kokkos::AUTO, vector_length);
    else
      policy = Kokkos::TeamPolicy<execution_space,
                                  Kokkos::Schedule<Kokkos::Dynamic>>(
          worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::bsr_spm_mv<NoTranspose,Dynamic>",
                         policy, func);
  } else {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>
        policy(1, 1);
    if (team_size < 0)
      policy =
          Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
              worksets, Kokkos::AUTO, vector_length);
    else
      policy =
          Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
              worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::bsr_spm_mv<NoTranspose, Static>",
                         policy, func);
  }
}

/* ******************* */
template <class AMatrix, class XVector, class YVector>
struct BSR_GEMM_Transpose_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef Kokkos::Details::ArithTraits<value_type> ATV;

  //! Nonconst version of the type of column indices in the sparse matrix.
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  //! Nonconst version of the type of row offsets in the sparse matrix.
  typedef typename AMatrix::non_const_size_type size_type;

  const value_type alpha;
  AMatrix m_A;
  XVector m_x;
  YVector m_y;
  const ordinal_type block_dim;
  const ordinal_type num_rhs;

  const ordinal_type blocks_per_team;

  bool conjugate = false;

  BSR_GEMM_Transpose_Functor(const value_type alpha_, const AMatrix m_A_,
                             const XVector m_x_, const YVector m_y_,
                             const int blocks_per_team_, bool conj_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        m_y(m_y_),
        block_dim(m_A_.blockDim()),
        num_rhs(m_x_.extent(1)),
        blocks_per_team(blocks_per_team_),
        conjugate(conj_) {
    static_assert(static_cast<int>(XVector::rank) == 2,
                  "XVector must be a rank 2 View.");
    static_assert(static_cast<int>(YVector::rank) == 2,
                  "YVector must be a rank 2 View.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ordinal_type iBlock) const {
    //
    const auto xstart = iBlock * block_dim;
    const auto start  = m_A.graph.row_map(iBlock);
    const ordinal_type count =
        static_cast<ordinal_type>(m_A.graph.row_map(iBlock + 1) - start);
    const auto row   = m_A.block_row_Const(iBlock);
    const auto beta1 = static_cast<value_type>(1);
    const auto ldx   = m_x.stride_1();
    const auto ldy   = m_y.stride_1();
    //
    std::vector<value_type> ytmp(block_dim * num_rhs);
    if (conjugate) {
      std::vector<value_type> conjA(block_dim * block_dim);
      for (ordinal_type ic = 0; ic < count; ++ic) {
        const auto Aview = row.block(ic);
        for (ordinal_type ii = 0; ii < block_dim; ++ii) {
          for (ordinal_type jj = 0; jj < block_dim; ++jj)
            conjA[jj + ii * block_dim] =
                Kokkos::ArithTraits<value_type>::conj(Aview(ii, jj));
        }
        //
        for (size_t ijk = 0; ijk < ytmp.size(); ++ijk)
          ytmp[ijk] = static_cast<value_type>(0);
        //
        KokkosBatched::SerialGemmInternal<KokkosBatched::Algo::Gemm::Blocked>::
            invoke<value_type, value_type>(block_dim, num_rhs, block_dim, alpha,
                                           conjA.data(), 1, block_dim,
                                           &m_x(xstart, 0), m_x.stride_0(), ldx,
                                           beta1, ytmp.data(), 1, block_dim);
        //
        const auto ystart = row.block_colidx(ic) * block_dim;
        for (ordinal_type jr = 0; jr < num_rhs; ++jr) {
          for (ordinal_type ir = 0; ir < block_dim; ++ir)
            Kokkos::atomic_add(&m_y(ystart + ir, jr),
                               ytmp[ir + jr * block_dim]);
        }
      }
    } else {
      for (ordinal_type ic = 0; ic < count; ++ic) {
        const auto Aview = row.block(ic);
        for (size_t ijk = 0; ijk < ytmp.size(); ++ijk)
          ytmp[ijk] = static_cast<value_type>(0);
        KokkosBatched::SerialGemmInternal<KokkosBatched::Algo::Gemm::Blocked>::
            invoke<value_type, value_type>(block_dim, num_rhs, block_dim, alpha,
                                           Aview.data(), 1, block_dim,
                                           &m_x(xstart, 0), m_x.stride_0(), ldx,
                                           beta1, ytmp.data(), 1, block_dim);
        //
        const auto ystart = row.block_colidx(ic) * block_dim;
        for (ordinal_type jr = 0; jr < num_rhs; ++jr) {
          for (ordinal_type ir = 0; ir < block_dim; ++ir)
            Kokkos::atomic_add(&m_y(ystart + ir, jr),
                               ytmp[ir + jr * block_dim]);
        }
      }
    }
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &dev) const {
    using y_value_type = typename YVector::non_const_value_type;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(dev, 0, blocks_per_team),
        [&](const ordinal_type &loop) {
          const ordinal_type iBlock =
              static_cast<ordinal_type>(dev.league_rank()) * blocks_per_team +
              loop;
          if (iBlock >= m_A.numRows()) {
            return;
          }
          //
          const auto start = m_A.graph.row_map(iBlock);
          const ordinal_type count =
              static_cast<ordinal_type>(m_A.graph.row_map(iBlock + 1) - start);
          const KokkosSparse::Experimental::BsrRowViewConst<AMatrix> row(
              m_A.values, m_A.graph.entries, block_dim, count, start);
          const auto nrhs = m_x.extent(1);
          //
          for (ordinal_type ic = 0; ic < nrhs; ++ic) {
            for (ordinal_type ir = 0; ir < block_dim; ++ir) {
              Kokkos::parallel_for(
                  Kokkos::ThreadVectorRange(dev, count),
                  [&](const ordinal_type &iEntry) {
                    for (ordinal_type jr = 0; jr < block_dim; ++jr) {
                      const value_type val =
                          conjugate
                              ? ATV::conj(row.local_block_value(iEntry, jr, ir))
                              : row.local_block_value(iEntry, jr, ir);
                      const ordinal_type ind = row.block_colidx(iEntry);
                      Kokkos::atomic_add(
                          &m_y(block_dim * ind + ir, ic),
                          static_cast<y_value_type>(
                              alpha * val * m_x(block_dim * iBlock + jr, ic)));
                    }
                  });
            }
          }
          //
        });
  }
};

/* ******************* */

/// \brief  spMatMultiVec_transpose: version for CPU execution spaces
/// (RangePolicy or trivial serial impl used)
template <class AT, class AO, class AD, class AS, class AlphaType,
          class XVector, class BetaType, class YVector,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatMultiVec_transpose(
    const KokkosKernels::Experimental::Controls &controls,
    const AlphaType &alpha,
    const KokkosSparse::Experimental::BsrMatrix<
        AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> &A,
    const XVector &x, const BetaType &beta, YVector &y, bool useConjugate) {
  // This is required to maintain semantics of KokkosKernels native SpMV:
  // if y contains NaN but beta = 0, the result y should be filled with 0.
  // For example, this is useful for passing in uninitialized y and beta=0.
  if (beta == Kokkos::ArithTraits<BetaType>::zero())
    Kokkos::deep_copy(y, Kokkos::ArithTraits<BetaType>::zero());
  else
    KokkosBlas::scal(y, beta, y);

  //
  // Treat the case y <- alpha * A^T * x + beta * y
  //

  typedef KokkosSparse::Experimental::BsrMatrix<
      AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS>
      AMatrix_Internal;
  typedef typename AMatrix_Internal::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix_Internal::execution_space execution_space;

  AMatrix_Internal A_internal = A;

  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }

  BSR_GEMM_Transpose_Functor<AMatrix_Internal, XVector, YVector> func(
      alpha, A_internal, x, y, 1, useConjugate);
  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::parallel_for(
        "KokkosSparse::bsr_spm_mv<Transpose,Dynamic>",
        Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>(
            0, A.numRows()),
        func);
  } else {
    Kokkos::parallel_for(
        "KokkosSparse::bsr_spm_mv<Transpose,Static>",
        Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
            0, A.numRows()),
        func);
  }
}

//
// spMatMultiVec_transpose: version for GPU execution spaces (TeamPolicy used)
//
template <class AMatrix, class AlphaType, class XVector, class BetaType,
          class YVector,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatMultiVec_transpose(
    const KokkosKernels::Experimental::Controls &controls,
    const AlphaType &alpha, const AMatrix &A, const XVector &x,
    const BetaType &beta, YVector &y, bool useConjugate) {
  if (A.numRows() <= 0) {
    return;
  }

  KokkosBlas::scal(y, beta, y);

  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::execution_space execution_space;

  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }
  int team_size             = -1;
  int vector_length         = -1;
  int64_t blocks_per_thread = -1;

  //
  // Use the controls to allow the user to pass in some tuning
  // parameters.
  //
  if (controls.isParameter("team size")) {
    team_size = std::stoi(controls.getParameter("team size"));
  }
  if (controls.isParameter("vector length")) {
    vector_length = std::stoi(controls.getParameter("vector length"));
  }
  if (controls.isParameter("rows per thread")) {
    blocks_per_thread = std::stoll(controls.getParameter("rows per thread"));
  }

  //
  // Use the existing launch parameters routine from SPMV
  //
  int64_t blocks_per_team =
      KokkosSparse::Impl::spmv_launch_parameters<execution_space>(
          A.numRows(), A.nnz(), blocks_per_thread, team_size, vector_length);
  int64_t worksets = (A.numRows() + blocks_per_team - 1) / blocks_per_team;

  BSR_GEMM_Transpose_Functor<AMatrix, XVector, YVector> func(
      alpha, A, x, y, blocks_per_team, useConjugate);

  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>
        policy(1, 1);
    if (team_size < 0)
      policy = Kokkos::TeamPolicy<execution_space,
                                  Kokkos::Schedule<Kokkos::Dynamic>>(
          worksets, Kokkos::AUTO, vector_length);
    else
      policy = Kokkos::TeamPolicy<execution_space,
                                  Kokkos::Schedule<Kokkos::Dynamic>>(
          worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::bsr_spm_mv<Transpose,Dynamic>", policy,
                         func);
  } else {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>
        policy(1, 1);
    if (team_size < 0)
      policy =
          Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
              worksets, Kokkos::AUTO, vector_length);
    else
      policy =
          Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
              worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::bsr_spm_mv<Transpose, Static>", policy,
                         func);
  }
}

/* ******************* */

}  // namespace Bsr

}  // namespace Impl
}  // namespace Experimental
}  // namespace KokkosSparse

#endif  // KOKKOSSPARSE_IMPL_SPMV_BSRMATRIX_IMPL_HPP_
