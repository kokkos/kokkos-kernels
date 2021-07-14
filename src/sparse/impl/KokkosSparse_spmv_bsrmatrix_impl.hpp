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

#ifndef KOKKOSSPARSE_IMPL_SPMV_TENSOR_CORE_DEF_HPP_
#define KOKKOSSPARSE_IMPL_SPMV_TENSOR_CORE_DEF_HPP_

#if defined(KOKKOS_ENABLE_CUDA) && \
    (defined(KOKKOS_ARCH_VOLTA) || defined(KOKKOS_ARCH_AMPERE))

#include <mma.h>

namespace KokkosSparse {
namespace Experimental {
namespace Impl {

/*! \brief Functor for multivector SpMV

   Implements y = alpha * A * x + beta * y without transpose or conjugate

   This implementation used a single warp per 16x16x16 fragment of the matrix
   and vectors. Internally, this implementation always uses wmma 16x16x16
   fragments of float += half * half Matrix and vector fragments are cast
   through shared memory, even if they are native half and float types.

   If X/Y are wide enough (>16) to require multiple fragments, groups of 4 of
   those fragments in each row share a single fragment loaded from A. If more
   than four fragments are needed to cover the columns of X/Y, each fragment of
   A will be accessed more than once.

   Different A, X, and Y scalar types are supported by reading tiles at the
   provided precision from global memory,  and converting them to the wmma
   native precision before storing them in shared memory. Then the wmma
   operations occur from there.

   @tparam AMatrix A KokkosSparse::BlockCrsMatrix with blockDim = 16
   @tparam XMatrix A Kokkos::View of rank 2
   @tparam YMatrix A Kokkos::View of rank 2

   @param alpha A YMatrix scalar type
   @param beta An XMatrix scalar type

*/
template <typename AMatrix, typename XMatrix, typename YMatrix>
struct TcFunctor {
  static constexpr int WARPS_PER_TEAM   = 4;
  static constexpr int THREADS_PER_WARP = 32;
  static constexpr int WMMA_TILE_M      = 16;
  static constexpr int WMMA_TILE_N      = 16;
  static constexpr int WMMA_TILE_K      = 16;

  typedef nvcuda::wmma::accumulator accumulator;
  typedef nvcuda::wmma::row_major row_major;
  typedef nvcuda::wmma::col_major col_major;
  typedef nvcuda::wmma::matrix_a matrix_a;
  typedef nvcuda::wmma::matrix_b matrix_b;

  // These fragments are always row-major because we load them from shared
  // memory. if the XMatrix or YMatrix are a different Layout, that is converted
  // to row-major when the entries are moved into shared memory
  using FragA = nvcuda::wmma::fragment<matrix_a, WMMA_TILE_M, WMMA_TILE_N,
                                       WMMA_TILE_K, half, row_major>;
  using FragX = nvcuda::wmma::fragment<matrix_b, WMMA_TILE_M, WMMA_TILE_N,
                                       WMMA_TILE_K, half, row_major>;
  using FragY = nvcuda::wmma::fragment<accumulator, WMMA_TILE_M, WMMA_TILE_N,
                                       WMMA_TILE_K, float>;

  typedef typename AMatrix::device_type Device;
  typedef Kokkos::TeamPolicy<typename Device::execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef typename AMatrix::value_type AScalar;
  typedef typename YMatrix::value_type YScalar;
  typedef typename XMatrix::value_type XScalar;

  // views of the shared memory used in the functor to cast types to the CUDA
  // wmma types
  typedef typename Kokkos::View<
      half[16][16], Kokkos::LayoutRight,
      typename Device::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      AScratchView;
  typedef typename Kokkos::View<
      half *[16][16],
      typename Kokkos::LayoutRight,  // so that [16][16] part is contiguous in
                                     // memory
      typename Device::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      XScratchView;
  typedef typename Kokkos::View<
      float *[16][16],
      typename Kokkos::LayoutRight,  // so that [16][16] part is contiguous in
                                     // memory
      typename Device::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      YScratchView;

  // spmv parameters
  YScalar alpha;
  AMatrix A;
  XMatrix x;
  YScalar beta;
  YMatrix y;

  // the league of teams is 1D, but TcFunctor logically views it as 2D
  int leagueDim_x;
  int leagueDim_y;

  TcFunctor() = delete;  // have to be able to generate the size of the league
  TcFunctor(YScalar _alpha, AMatrix _A, XMatrix _x, YScalar _beta, YMatrix _y)
      : alpha(_alpha), A(_A), x(_x), beta(_beta), y(_y) {
    /*
       Each team will handle a row of fragments of the product multivector.
       Teams are in a 2D space, with one dimension over the columns
       and one dimension over the rows.
       If the team is more than one warp, the warps will each take a fragment of
       the row Each warp in that team can share each fragment of A. If there are
       too many rows, multiple teams may be required for each row
    */
    const int warpsPerTeam = team_size() / THREADS_PER_WARP;
    int yCols              = y.extent(1) / WMMA_TILE_N;  // fragments per row
    leagueDim_x            = (yCols + warpsPerTeam - 1) /
                  warpsPerTeam;  // # teams needed to cover all Y columns
    leagueDim_y = y.extent(0) / WMMA_TILE_M;  // # teams needed to cover Y rows
  }

  int league_size() const { return leagueDim_x * leagueDim_y; }

  /* each team will handle a row of fragments, and each warp a fragment
   */
  int team_size() const {
    if (0 != (y.extent(1) % WMMA_TILE_N)) {
      Kokkos::Impl::throw_runtime_exception(
          "y.extent(0) and A.blockDim() mismatch");
    }
    const int fragsPerRow = y.extent(1) / WMMA_TILE_N;
    return THREADS_PER_WARP * fragsPerRow;
  }
  int team_scratch_size() const {
    // each team needs:
    // 16x16 half for fragments of A
    // each warp needs
    // 16x16 half for converting X fragments
    // 16x16 float for converting Y fragmes
    int warpsPerTeam = team_size() / THREADS_PER_WARP;
    return 16 * 16 * sizeof(half) + warpsPerTeam * 16 * 16 * sizeof(half) +
           warpsPerTeam * 16 * 16 * sizeof(float);
  }

  // equivalent to the wider of T and U, or T if equal
  // used to select the widest floating-point type for explicit casting around
  // half operations
  template <class T, class U>
  struct wider {
    using type = typename std::conditional<sizeof(T) >= sizeof(U), T, U>::type;
  };

  KOKKOS_INLINE_FUNCTION void operator()(const team_member &mbr) const {
    using nvcuda::wmma::fill_fragment;
    using nvcuda::wmma::load_matrix_sync;
    using nvcuda::wmma::mma_sync;
    using nvcuda::wmma::store_matrix_sync;

    typedef typename AMatrix::ordinal_type Ordinal;

    FragA fa;
    FragX fx;
    FragY fy;

    // convert 1D league space into two dimensions
    const Ordinal leagueIdx_x = mbr.league_rank() % (leagueDim_x);
    const Ordinal leagueIdx_y = mbr.league_rank() / (leagueDim_x);
    const Ordinal wx          = mbr.team_rank() / THREADS_PER_WARP;  // warp idx
    const Ordinal lx          = mbr.team_rank() % THREADS_PER_WARP;  // lane idx

    half *_sa = (half *)mbr.team_shmem().get_shmem(16 * 16 * sizeof(half));
    half *_sx = (half *)mbr.team_shmem().get_shmem(
        16 * 16 * sizeof(half) * mbr.team_size() / THREADS_PER_WARP);
    float *_sy = (float *)mbr.team_shmem().get_shmem(
        16 * 16 * sizeof(float) * mbr.team_size() / THREADS_PER_WARP);

    AScratchView sa(_sa);
    XScratchView sx(_sx, mbr.team_size() / THREADS_PER_WARP);
    YScratchView sy(_sy, mbr.team_size() / THREADS_PER_WARP);

    // column of x/y that this warp will contribute to
    // which team * # warps in team + which warp in team
    const Ordinal yj =
        (leagueIdx_x * (mbr.team_size() / THREADS_PER_WARP) + wx) *
        A.blockDim();

    // Y dimension covers rows of A/y
    // each Y dimension team covers WMMA_TILE_M rows (fragment size)
    for (Ordinal i = leagueIdx_y; i < y.extent(0) / WMMA_TILE_M;
         i += leagueDim_y) {
      const Ordinal yi = i * A.blockDim();

      // can't mask off warps entirely, since they must participate in team
      // barriers
      if (yj < y.extent(1)) {
        // load of the Y fragment is staged through shared memory, to convert
        // it from / to the YMatrix scalar type
        // each warp uses its own piece of shared memory

        // fy = y * beta

        if (YScalar(0) == beta) {
          // skip load and just zero Y fragment if beta is 0
#ifdef __CUDA_ARCH__
          fill_fragment(fy, 0.0f);
#endif
        } else if (YScalar(1) != beta ||
                   A.graph.row_map(i) != A.graph.row_map(i + 1)) {
          // need to load & scale Y in any of two cases:
          // beta != 1, meaning Y has to be loaded in order to scale it
          // row is not empty, meaning Y has to be loaded to accumulate into it
          // warp-collaborative load+scale
          for (int fi = lx; fi < 16 * 16; fi += THREADS_PER_WARP) {
            int ti         = fi / 16;
            int tj         = fi % 16;
            sy(wx, ti, tj) = beta * y(yi + ti, yj + tj);
          }
#ifdef __CUDA_ARCH__
          load_matrix_sync(fy, &sy(wx, 0, 0), 16, nvcuda::wmma::mem_row_major);
#endif
        } else {
          // beta was 1 and the matrix row was empty, this piece of Y is
          // unchanged
          ;  // no-op
        }
      }

      auto rowView = A.block_row(i);

      // team loops through all fragments in the row
      for (int ci = A.graph.row_map(i); ci < A.graph.row_map(i + 1); ++ci) {
        // A column (of blocks) for this team tells which X fragment to load
        int j = A.graph.entries(ci);
        if (yj < y.extent(1)) {
          // warp collaborative load of X fragment
          for (int fi = lx; fi < 16 * 16; fi += THREADS_PER_WARP) {
            int ti         = fi / 16;
            int tj         = fi % 16;
            sx(wx, ti, tj) = half(x(j * A.blockDim() + ti, yj + tj));
          }
#ifdef __CUDA_ARCH__
          load_matrix_sync(fx, &sx(wx, 0, 0), 16);
#endif
        }

        int off = ci - A.graph.row_map(i);  // which block in this row
        AScalar *ap =
            rowView.local_row_in_block(off, 0);  // offset of this block
        // BlockCrsMatrix stores it's values in the normal Crs form.
        // therefore, the pitch of each row is different, since it depends
        // on how many blocks are in that row.
        // the pitch between rows within each row of blocks is
        // blockDim * the length of the row (in blocks)
        unsigned lda = A.blockDim() * rowView.length;
        // team loads A fragment
        for (int fi = mbr.team_rank(); fi < 16 * 16; fi += mbr.team_size()) {
          int ti      = fi / 16;
          int tj      = fi % 16;
          using MulTy = typename wider<AScalar, YScalar>::type;
          sa(ti, tj)  = half(MulTy(ap[(ti * lda) + tj]) * MulTy(alpha));
        }

        mbr.team_barrier();  // ensure that warp 0 has loaded A fragment into
                             // scratch memory
        if (yj < y.extent(1)) {
#ifdef __CUDA_ARCH__
          load_matrix_sync(fa, (half *)&sa(0, 0),
                           16);  // load fa from shared memory
#endif
        }
        mbr.team_barrier();  // prevent warp 0 from loading the next A fragment
                             // until everyone has used it

        // fy = fa * fx + fy
        if (yj < y.extent(1)) {
#ifdef __CUDA_ARCH__
          mma_sync(fy, fa, fx, fy);
#endif
        }
      }

      // store y. The store is staged through shared memory to convert back to
      // the YMatrix scalar type
      if (yj < y.extent(1)) {
        // Y was only changed if beta != 1 or the row of A was not empty
        // if Y was not changed, don't store anything since nothing was
        // loaded
        if (YScalar(1) != beta ||
            A.graph.row_map(i) != A.graph.row_map(i + 1)) {
          // store product into shared memory
#ifdef __CUDA_ARCH__
          // store_matrix_sync(yp, fy, ldy, yLayout);
          store_matrix_sync(&sy(wx, 0, 0), fy, 16, nvcuda::wmma::mem_row_major);
#endif
          // cast to product scalar type and store in global memory
          for (int fi = lx; fi < 16 * 16; fi += THREADS_PER_WARP) {
            int ti              = fi / 16;
            int tj              = fi % 16;
            y(yi + ti, yj + tj) = YScalar(sy(wx, ti, tj));
          }
        }
      }
    }  // yj
  }    // operator()
};

/* y = beta * y + alpha * A * x

   General tensor core implementation of y = beta * y + alpha * A * x
   alpha_beta_n_tc: do alpha/beta, mode: N, with tensor cores
*/
template <typename AMatrix, typename XMatrix, typename YMatrix>
void spmv_alpha_beta_n_tc(typename YMatrix::value_type alpha, const AMatrix &A,
                          const XMatrix &x, typename YMatrix::value_type beta,
                          const YMatrix &y) {
  if (0 != y.extent(1) % A.blockDim()) {
    Kokkos::Impl::throw_runtime_exception(
        "y.extent(1) not a multiple of A block dimension");
  }
  if (0 != y.extent(0) % A.blockDim()) {
    Kokkos::Impl::throw_runtime_exception(
        "y.extent(0) not a multiple of A block dimension");
  }

  typedef TcFunctor<AMatrix, XMatrix, YMatrix> Func;

  if (A.blockDim() != Func::WMMA_TILE_M) {
    Kokkos::Impl::throw_runtime_exception("A block dimension must be 16");
  }
  if (A.blockDim() != Func::WMMA_TILE_N) {
    Kokkos::Impl::throw_runtime_exception("A block dimension must be 16");
  }
  if (A.blockDim() != Func::WMMA_TILE_K) {
    Kokkos::Impl::throw_runtime_exception("A block dimension must be 16");
  }

  Func f(alpha, A, x, beta, y);
  typename Func::team_policy policy(f.league_size(), f.team_size());

  policy.set_scratch_size(0, Kokkos::PerTeam(f.team_scratch_size()));
  Kokkos::parallel_for("KokkosSparse::tensor_core_team", policy, f);
}

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

  // views of the shared memory used in the functor to cast types to the CUDA
  // wmma types A matrix is MxK X matrix is KxN Y matrix is MxN
  typedef typename Kokkos::View<
      AFragScalar *[FRAG_M][FRAG_K],  // one fragment per warp in the team (2D
                                      // grid of warps in team)
      Kokkos::LayoutRight,
      typename Device::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      AScratchView;
  typedef typename Kokkos::View<
      XFragScalar *[FRAG_K][FRAG_N],
      typename Kokkos::LayoutRight,  // so that [FRAG_K][FRAG_N] part is
                                     // contiguous in memory
      typename Device::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      XScratchView;
  typedef typename Kokkos::View<
      YFragScalar **[FRAG_M][FRAG_N],
      typename Kokkos::LayoutRight,  // so that [FRAG_M][FRAG_N] part is
                                     // contiguous in memory
      typename Device::execution_space::scratch_memory_space,
      Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      YScratchView;

  YScalar alpha;
  AMatrix a;
  XMatrix x;
  YScalar beta;
  YMatrix y;

  BsrMatrixSpMVTensorCoreFunctorParams params;

 public:
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

    typedef typename AMatrix::ordinal_type Ordinal;

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
    const int ay_i =
        blockIdx_i * a.blockDim()                // offset due to block
        + teamIdx_i * WARPS_PER_TEAM_Y * FRAG_M  // offset of team within block
        + warpIdx_y * FRAG_M;                    // offset of warp within team

    // which column of x/y the fragments warp will read from/contribute to
    // starts at
    const int xy_j = blockIdx_j * a.blockDim() +
                     teamIdx_j * WARPS_PER_TEAM_X * FRAG_N + warpIdx_x * FRAG_N;

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
    for (int i = lx; i < FRAG_M * FRAG_N; i += THREADS_PER_WARP) {
      int fi = i / FRAG_N;  // position in fragment of Y
      int fj = i % FRAG_N;
      int bi = teamIdx_i * WARPS_PER_TEAM_Y * FRAG_M + warpIdx_y * FRAG_M +
               fi;  // position in block of Y
      int bj = teamIdx_j * WARPS_PER_TEAM_X * FRAG_N + warpIdx_x * FRAG_N + fj;

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
    for (int ci = a.graph.row_map(blockIdx_i);
         ci < a.graph.row_map(blockIdx_i + 1); ++ci) {
      int j = a.graph.entries(ci);

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
      for (int bk = 0; bk < a.blockDim(); bk += FRAG_K /*M*/) {
        // team collaborative load of A
        // the footprint is one fragment wide in K direction
        mbr.team_barrier();
        for (int i = mbr.team_rank(); i < WARPS_PER_TEAM_Y * FRAG_M * FRAG_K;
             i += mbr.team_size()) {
          const int ti = i / FRAG_K;  // offset inside the fragments
          const int tj = i % FRAG_K;
          // add in offset within block
          const int bi = teamIdx_i * WARPS_PER_TEAM_Y * FRAG_M + ti;
          const int bj = bk + tj;

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
        for (int i = mbr.team_rank(); i < WARPS_PER_TEAM_X * FRAG_N * FRAG_K;
             i += mbr.team_size()) {
          const int ti =
              i / (WARPS_PER_TEAM_X * FRAG_N);  // position in combined tiles
          const int tj = i % (WARPS_PER_TEAM_X * FRAG_N);

          // add in offset within block
          const int bi = bk + ti;
          const int bj = teamIdx_j * WARPS_PER_TEAM_X * FRAG_N + tj;

          // load 0 outside of the block boundary
          // x is not necessarily a multiple of block size, so make sure access
          // is in bounds
          if (bi < a.blockDim() && bj < a.blockDim() &&
              blockIdx_j * a.blockDim() + bj < x.extent(1)) {
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
        if (ay_i < y.extent(0) && xy_j < y.extent(1)) {
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
    for (int i = lx; i < FRAG_M * FRAG_N; i += THREADS_PER_WARP) {
      int fi = i / FRAG_N;  // position in fragment of Y
      int fj = i % FRAG_N;
      int bi = teamIdx_i * WARPS_PER_TEAM_Y * FRAG_M + warpIdx_y * FRAG_M +
               fi;  // position in block of Y
      int bj = teamIdx_j * WARPS_PER_TEAM_X * FRAG_N + warpIdx_x * FRAG_N + fj;

      // only store inside the block boundary
      // FIXME: what if Y is not wide enough? check y(_, j)
      if (bi < a.blockDim() && bj < a.blockDim() && xy_j + fj < y.extent(1)) {
        y(ay_i + fi, xy_j + fj) = sy(warpIdx_y, warpIdx_x, fi, fj);
      }
    }
    mbr.team_barrier();
  }
};

/* Instantiate some common template parameter values
   for BsrMatrixSpMVTensorCoreFunctor.
   This is a struct instead of a function for template...using shorthand
*/
template <typename AMatrix,
          typename AFragScalar,  // input matrix type and fragment scalar type
          typename XMatrix, typename XFragScalar, typename YMatrix,
          typename YFragScalar, unsigned FRAG_M, unsigned FRAG_N,
          unsigned FRAG_K  // fragment sizes
          >
struct BsrMatrixSpMVTensorCoreDispatcher {
  typedef typename YMatrix::value_type YScalar;

  template <unsigned X, unsigned Y, unsigned Z>
  using Dyn = BsrMatrixSpMVTensorCoreFunctor<AMatrix, AFragScalar, XMatrix,
                                             XFragScalar, YMatrix, YFragScalar,
                                             FRAG_M, FRAG_N, FRAG_K, X, Y, Z>;

  static void dispatch(YScalar alpha, AMatrix a, XMatrix x, YScalar beta,
                       YMatrix y) {
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
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace KokkosSparse

#endif  // #if CUDA && (VOLTA || AMPERE)

#endif  // KOKKOSSPARSE_IMPL_SPMV_TENSOR_CORE_DEF_HPP_
