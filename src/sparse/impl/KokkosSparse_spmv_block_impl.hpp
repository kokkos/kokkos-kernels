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

#ifndef KOKKOSKERNELS_KOKKOSSPARSE_SPMV_BLOCK_IMPL_HPP
#define KOKKOSKERNELS_KOKKOSSPARSE_SPMV_BLOCK_IMPL_HPP

#include "Kokkos_Core.hpp"

#include "KokkosBlas.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_BlockCrsMatrix.hpp"

#include "KokkosSparse_spmv_impl.hpp"

#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Controls.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

namespace KokkosSparse {
namespace Impl {

namespace Utils {

//
// Dispatches a dynamic value call to explicit instantiations
// for small values (compile time range)
// a
// Usage:
//
//  constexpr int MAX_N = 10;                             // explicitly expand for N=1..MAX_N
//  Utils::eti_expand<MAX_N>(n, [&]<int N>(const int n) { // make a call for n
//     ExpandableHelper<N, ...>::call<N, ...>(n, ...);    // dispatch based on n and compile-time N
//  });                                                   // (where N=n for n=1..MAX_N and 0 for n>MAX_N)
//
template<int MAX_N, typename F>
void eti_expand(int n, F f);

} // namespace Utils

/////////////////////////////////////////////////////////

template <class AMatrix, class XVector, class YVector>
bool verifyArguments(const char mode[], const AMatrix &A, const XVector &x,
                     const YVector &y) {
  // Make sure that both x and y have the same rank.
  static_assert(
      static_cast<int>(XVector::rank) == static_cast<int>(YVector::rank),
      "KokkosSparse::spmv: Vector ranks do not match.");
  // Make sure that y is non-const.
  static_assert(std::is_same<typename YVector::value_type,
                             typename YVector::non_const_value_type>::value,
                "KokkosSparse::spmv: Output Vector must be non-const.");

  // Check compatibility of dimensions at run time.
  if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
      (mode[0] == KokkosSparse::Conjugate[0])) {
    if ((x.extent(1) != y.extent(1)) ||
        (static_cast<size_t>(A.numCols()) > static_cast<size_t>(x.extent(0))) ||
        (static_cast<size_t>(A.numRows()) > static_cast<size_t>(y.extent(0)))) {
      std::ostringstream os;
      os << "KokkosSparse::spmv: Dimensions do not match: "
         << ", A: " << A.numRows() << " x " << A.numCols()
         << ", x: " << x.extent(0) << " x " << x.extent(1)
         << ", y: " << y.extent(0) << " x " << y.extent(1);
      Kokkos::Impl::throw_runtime_exception(os.str());
    }
  } else {
    if ((x.extent(1) != y.extent(1)) ||
        (static_cast<size_t>(A.numCols()) > static_cast<size_t>(y.extent(0))) ||
        (static_cast<size_t>(A.numRows()) > static_cast<size_t>(x.extent(0)))) {
      std::ostringstream os;
      os << "KokkosSparse::spmv: Dimensions do not match (transpose): "
         << ", A: " << A.numRows() << " x " << A.numCols()
         << ", x: " << x.extent(0) << " x " << x.extent(1)
         << ", y: " << y.extent(0) << " x " << y.extent(1);
      Kokkos::Impl::throw_runtime_exception(os.str());
    }
  }

  return true;
}

//////////////////////////////////////////////////////////

template <class AMatrix, class XVector, class YVector>
struct BSPMV_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef Kokkos::Details::ArithTraits<value_type> ATV;

  const value_type alpha;
  AMatrix m_A;
  XVector m_x;
  YVector m_y;
  const ordinal_type block_size;

  const ordinal_type blocks_per_team;

  bool conjugate = false;

  BSPMV_Functor(const value_type alpha_, const AMatrix m_A_, const XVector m_x_,
                const YVector m_y_, const int blocks_per_team_, bool conj_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        m_y(m_y_),
        block_size(m_A_.blockDim()),
        blocks_per_team(blocks_per_team_),
        conjugate(conj_)
  {
    static_assert(static_cast<int>(XVector::rank) == 1,
                  "XVector must be a rank 1 View.");
    static_assert(static_cast<int>(YVector::rank) == 1,
                  "YVector must be a rank 1 View.");
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ordinal_type iBlock) const {
    //
    if (iBlock >= m_A.numRows()) {
      return;
    }
    //
    const auto jbeg       = m_A.graph.row_map[iBlock];
    const auto jend       = m_A.graph.row_map[iBlock + 1];
    const auto block_size_2 = block_size * block_size;
    //
    const auto lda = (jend - jbeg) * block_size;
    auto Aval_ptr = &m_A.values[jbeg * block_size_2];
    //
    auto yvec = &m_y[iBlock * block_size];
    //
    ordinal_type j = 0;
    for (auto jb = jbeg; jb < jend; ++jb) {
      const auto col_block = m_A.graph.entries[jb];
      const auto xval_ptr  = &m_x[block_size * col_block];
      for (ordinal_type kr = 0; kr < block_size; ++kr) {
        for (ordinal_type ic = 0; ic < block_size; ++ic) {
          const auto shift = ic + kr * lda + j * block_size;
          const auto aval = conjugate ? ATV::conj(Aval_ptr[shift])
              : Aval_ptr[shifte];
          yvec[kr] += alpha * aval * xval_ptr[ic];
        }
      }
      j += 1;
    }
    //
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const team_member &dev) const
  {
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
          const auto jbeg         = m_A.graph.row_map[iBlock];
          const auto jend         = m_A.graph.row_map[iBlock + 1];
          const auto row_num_blocks = static_cast<ordinal_type>(jend - jbeg);
          const auto block_size_2 = block_size * block_size;
          const auto col_idx      = &m_A.graph.entries[jbeg];
          //
          const auto lda = (jend - jbeg) * block_size;
          const auto *Aval_ptr = &m_A.values[jbeg * block_size_2];
          //
          auto yvec = &m_y[iBlock * block_size];
          y_value_type lsum = 0;
          //
          for (ordinal_type ir = 0; ir < block_size; ++ir) {
            lsum = 0;
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(dev, row_num_blocks),
                [&](const ordinal_type &iEntry, y_value_type &ysum) {
                  const ordinal_type col_block = col_idx[iEntry] * block_size;
                  const auto xval              = &m_x[col_block];
                  const auto *Aval = Aval_ptr + iEntry * block_size;
                                     + ir * lda;
                  for (ordinal_type jc = 0; jc < block_size; ++jc) {
                    const value_type val = conjugate ? ATV::conj(Aval[jc])
                                                     : Aval[jc];
                    lsum += val * xval[jc];
                  }
                },
                lsum);
            //
            Kokkos::single(Kokkos::PerThread(dev), [&]() {
              lsum *= alpha;
              yvec[ir] += lsum;
            });
          }
          //
        }
    );
  }
};

/***********************************/
//
//  This needs to be generalized
//
constexpr size_t bmax = 8;

/***********************************/

#ifdef KOKKOS_ENABLE_SERIAL

template <int M>
inline void spmv_serial_gemv(const Scalar *Aval, const Ordinal lda,
                             const Scalar *x_ptr,
                             std::array<Scalar, Impl::bmax> &y) {
  for (Ordinal ic = 0; ic < M; ++ic) {
    const auto xvalue = x_ptr[ic];
    for (Ordinal kr = 0; kr < M; ++kr) {
      y[kr] += Aval[ic + kr * lda] * xvalue;
    }
  }
}

//
// Explicit blockSize=N case
//
template <class StaticGraph, Ordinal blockSize>
struct SpMV_SerialNoTranspose {

  static inline void spmv(const Scalar alpha, const Scalar *Avalues,
                        const StaticGraph &Agraph, const Scalar *x,
                        Scalar *y, const Ordinal bs) {

    const Ordinal numBlockRows = Agraph.numRows();
    std::array<double, Impl::bmax> tmp{0};
    const Ordinal blockSize2 = blockSize * blockSize;
    for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
      const auto jbeg       = Agraph.row_map[iblock];
      const auto jend       = Agraph.row_map[iblock + 1];
      const auto num_blocks = jend - jbeg;
      tmp.fill(0);
      for (Ordinal jb = 0; jb < num_blocks; ++jb) {
        const auto col_block = Agraph.entries[jb + jbeg];
        const auto xval_ptr  = x + blockSize * col_block;
        auto Aval_ptr        = Avalues + (jb + jbeg) * blockSize2;
        spmv_serial_gemv<blockSize>(Aval_ptr, blockSize, xval_ptr, tmp);
      }
      //
      auto yvec = &y[iblock * blockSize];
      for (Ordinal ii = 0; ii < blockSize; ++ii) {
        yvec[ii] += alpha * tmp[ii];
      }
    }

  }
};

//
// Special blockSize=1 case (optimized)
//
template <class StaticGraph>
struct SpMV_SerialNoTranspose<StaticGraph, 1> {
  static inline void spmv(const Scalar alpha, const Scalar *Avalues,
                          const StaticGraph &Agraph, const Scalar *x,
                          Scalar *y, const Ordinal blockSize) {

    const Ordinal numBlockRows = Agraph.numRows();
    for (Ordinal i = 0; i < numBlockRows; ++i) {
      const auto jbeg = Agraph.row_map[i];
      const auto jend = Agraph.row_map[i + 1];
      double tmp      = 0.0;
      for (Ordinal j = jbeg; j < jend; ++j) {
        const auto alpha_value1 = alpha * Avalues[j];
        const auto col_idx1     = Agraph.entries[j];
        const auto x_val1       = x[col_idx1];
        tmp += alpha_value1 * x_val1;
      }
      y[i] += tmp;
    }
    return;
  }
};

//
// --- Basic approach for large block sizes
//
template <class StaticGraph>
struct SpMV_SerialNoTranspose<StaticGraph, 0> {
  static inline void spmv(const Scalar alpha, const Scalar *Avalues,
                          const StaticGraph &A_graph, const Scalar *x,
                          Scalar *y, const Ordinal blockSize) {

      const Ordinal numBlockRows = A_graph.numRows();
      const Ordinal blockSize_squared = blockSize * blockSize;
      auto yvec = &y[0];
      for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
        const auto jbeg       = A_graph.row_map[iblock];
        const auto jend       = A_graph.row_map[iblock + 1];
        //
        for (Ordinal jb = jbeg; jb < jend; ++jb) {
          const auto col_block = A_graph.entries[jb];
          const auto xval_ptr  = &x[0] + blockSize * col_block;
          const auto Aval_ptr  = Avalues + jb * blockSize_squared;
          for (Ordinal kr = 0; kr < blockSize; ++kr) {
            for (Ordinal ic = 0; ic < blockSize; ++ic) {
              auto q = Aval_ptr[ic + kr * blockSize];
              yvec[kr] += alpha * q * xval_ptr[ic];
            }
          }
        }
        //
        yvec = yvec + blockSize;
      }
  }
};

#endif // defined(KOKKOS_ENABLE_SERIAL)

/* ******************* */


#ifdef KOKKOS_ENABLE_OPENMP
template<typename AMatrix, typename XVector, typename YVector>
void bspmv_raw_openmp_no_transpose(typename YVector::const_value_type& s_a,
                                   AMatrix A, XVector x,
                                   YVector y)
{
  typedef typename YVector::non_const_value_type value_type;
  typedef typename AMatrix::ordinal_type         ordinal_type;
  typedef typename AMatrix::non_const_size_type  size_type;

  typename XVector::const_value_type* KOKKOS_RESTRICT x_ptr = x.data();
  typename YVector::non_const_value_type* KOKKOS_RESTRICT y_ptr = y.data();

  const typename AMatrix::value_type* KOKKOS_RESTRICT matrixCoeffs = A.values.data();
  const ordinal_type* KOKKOS_RESTRICT matrixCols     = A.graph.entries.data();
  const size_type* KOKKOS_RESTRICT matrixRowOffsets  = A.graph.row_map.data();
  const size_type* KOKKOS_RESTRICT threadStarts      = A.graph.row_block_offsets.data();

#if defined(KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::beginParallelFor("KokkosSparse::spmv<RawOpenMP,NoTranspose>", 0, &kpID);
  }
#endif

  typename YVector::const_value_type zero = 0;
  #pragma omp parallel
  {
#if defined(KOKKOS_COMPILER_INTEL) && !defined(__clang__)
    __assume_aligned(x_ptr, 64);
    __assume_aligned(y_ptr, 64);
#endif

    const int myID    = omp_get_thread_num();
    const size_type myStart = threadStarts[myID];
    const size_type myEnd   = threadStarts[myID + 1];
    const auto blockSize = A.blockDim();
    const auto blockSize2 = blockSize * blockSize;

    for(size_type row = myStart; row < myEnd; ++row) {
      const size_type rowStart = matrixRowOffsets[row];
      const size_type rowEnd   = matrixRowOffsets[row + 1];
      //
      const size_type lda = blockSize * (rowEnd - rowStart);
      auto Aval_ptr = &matrixCoeffs[rowStart * blockSize2];
      //
      auto yvec = &y[row * blockSize];
      //
      for (Ordinal jentry = rowStart, j = 0; jentry < rowEnd; ++jentry, ++j) {
        const auto col_block = A.graph.entries[jentry];
        const auto xval_ptr  = &x[blockSize * col_block];
        for (Ordinal ic = 0; ic < blockSize; ++ic) {
          const auto xvalue = xval_ptr[ic];
          for (Ordinal kr = 0; kr < blockSize; ++kr) {
            yvec[kr] += s_a * Aval_ptr[ic + kr * lda + j * blockSize] * xvalue;
          }
        }
      }
      //
    }
  }

#if defined(KOKKOS_ENABLE_PROFILING)
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::endParallelFor(kpID);
  }
#endif

}
#endif


/* ******************* */


//
// spMatVec_no_transpose: version for GPU execution spaces
//
template < class AT, class AO, class AD, class AM, class AS,
          class AlphaType, class XVector, class BetaType, class YVector,
          typename std::enable_if< KokkosKernels::Impl::kk_is_gpu_exec_space<typename YVector::execution_space>()>::type* = nullptr>
void spMatVec_no_transpose(KokkosKernels::Experimental::Controls controls,
                           const AlphaType &alpha,
                           const KokkosSparse::Experimental::BlockCrsMatrix< AT, AO, AD, AM, AS> &A,
                           const XVector &x, const BetaType &beta, YVector &y,
                           bool useFallback, bool useConjugate)
{

  if (A.numRows () <= static_cast<AO> (0)) {
    return;
  }

  typedef KokkosSparse::Experimental::BlockCrsMatrix<AT, AO, AD,
                                                     Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> AMatrix_Internal;
  typedef typename AMatrix_Internal::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix_Internal::execution_space execution_space;

  bool use_dynamic_schedule = false; // Forces the use of a dynamic schedule
  bool use_static_schedule  = false; // Forces the use of a static schedule
  if(controls.isParameter("schedule")) {
    if(controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if(controls.getParameter("schedule") == "static") {
      use_static_schedule  = true;
    }
  }
  int team_size = -1;
  int vector_length = -1;
  int64_t blocks_per_thread = -1;

  //
  // Use the controls to allow the user to pass in some tuning parameters.
  //
  if (controls.isParameter("team size")) {
    team_size       = std::stoi(controls.getParameter("team size"));
  }
  if (controls.isParameter("vector length")) {
    vector_length   = std::stoi(controls.getParameter("vector length"));
  }
  if (controls.isParameter("rows per thread")) {
    blocks_per_thread = std::stoll(controls.getParameter("rows per thread"));
  }

  //
  // Use the existing launch parameters routine from SPMV
  //
  int64_t blocks_per_team = KokkosSparse::Impl::spmv_launch_parameters<execution_space>(A.numRows(),
                                                                     A.nnz(), blocks_per_thread,
                                                                     team_size, vector_length);
  int64_t worksets = (A.numRows() + blocks_per_team - 1) / blocks_per_team;

  AMatrix_Internal A_internal = A;

  BSPMV_Functor<AMatrix_Internal,XVector,YVector> func (alpha, A_internal, x, y, blocks_per_team, useConjugate);

  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic> > policy(1,1);
    if(team_size<0)
      policy = Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets, Kokkos::AUTO, vector_length);
    else
      policy = Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic> >(worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::bspmv<NoTranspose,Dynamic>", policy, func);
  } else {
    Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static> > policy(1,1);
    if(team_size<0)
      policy = Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets, Kokkos::AUTO, vector_length);
    else
      policy = Kokkos::TeamPolicy<execution_space, Kokkos::Schedule<Kokkos::Static> >(worksets, team_size, vector_length);
    Kokkos::parallel_for("KokkosSparse::bspmv<NoTranspose, Static>", policy, func);
  }

}


/* ******************* */


//
// spMatVec_no_transpose: version for CPU execution spaces
//
template < class AT, class AO, class AD, class AM, class AS,
        class AlphaType, class XVector, class BetaType, class YVector,
        typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
        typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_no_transpose(KokkosKernels::Experimental::Controls controls,
                           const AlphaType &alpha,
                           const KokkosSparse::Experimental::BlockCrsMatrix< AT, AO, AD, AM, AS> &A,
                           const XVector &x, const BetaType &beta, YVector &y,
                           bool useFallback, bool useConjugate) {

  if (A.numRows () <= static_cast<AO> (0)) {
    return;
  }

  typedef Kokkos::View<
      typename XVector::const_value_type *,
      typename KokkosKernels::Impl::GetUnifiedLayout<XVector>::array_layout,
      typename XVector::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >
      XVector_Internal;

  typedef Kokkos::View<
      typename YVector::non_const_value_type *,
      typename KokkosKernels::Impl::GetUnifiedLayout<YVector>::array_layout,
      typename YVector::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      YVector_Internal;

  YVector_Internal y_i = y;

  // This is required to maintain semantics of KokkosKernels native SpMV:
  // if y contains NaN but beta = 0, the result y should be filled with 0.
  // For example, this is useful for passing in uninitialized y and beta=0.
  if (beta == Kokkos::ArithTraits<BetaType>::zero())
    Kokkos::deep_copy(y_i, Kokkos::ArithTraits<BetaType>::zero());
  else
    KokkosBlas::scal(y_i, beta, y_i);

  //
  // Treat the case y <- alpha * A * x + beta * y
  //

  XVector_Internal x_i       = x;
  const Ordinal blockSize    = A.blockDim();
  const Ordinal numBlockRows = A.numRows();

  typedef KokkosSparse::Experimental::BlockCrsMatrix<AT, AO, AD,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> AMatrix_Internal;

  AMatrix_Internal A_internal = A;
  const auto &A_graph = A.graph;

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same< typename AMatrix_Internal::device_type::execution_space,
      Kokkos::Serial>::value) {
    Utils::eti_expand<Impl::bmax>(blockSize, [&]<int fixedBlockSize>(const int blockSize) {
      SpMV_SerialNoTranspose<decltype(A_graph), fixedBlockSize>::spmv(alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
    });
    return;
  }
#endif

  typedef typename AMatrix_Internal::execution_space execution_space;

#ifdef KOKKOS_ENABLE_OPENMP
  if ((std::is_same<execution_space, Kokkos::OpenMP>::value) &&
      (std::is_same<typename std::remove_cv<typename AMatrix_Internal::value_type>::type, double>::value) &&
      (std::is_same<typename XVector::non_const_value_type, double>::value) &&
      (std::is_same<typename YVector::non_const_value_type, double>::value) &&
      ((int)A.graph.row_block_offsets.extent(0) == (int)omp_get_max_threads() + 1) &&
      (((uintptr_t)(const void *)(x.data()) % 64) == 0) &&
      (((uintptr_t)(const void *)(y.data()) % 64) == 0))
  {
    bspmv_raw_openmp_no_transpose<AMatrix_Internal, XVector, YVector>(alpha, A, x, y);
    return;
  }
#endif

  bool use_dynamic_schedule = false; // Forces the use of a dynamic schedule
  bool use_static_schedule  = false; // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule  = true;
    }
  }
  BSPMV_Functor<AMatrix_Internal,XVector,YVector> func (alpha, A_internal, x, y, 1, useConjugate);
  if(((A.nnz()>10000000) || use_dynamic_schedule) && !use_static_schedule)
    Kokkos::parallel_for("KokkosSparse::bspmv<NoTranspose,Dynamic>",Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>(0, A.numRows()),func);
  else
    Kokkos::parallel_for("KokkosSparse::bspmv<NoTranspose,Static>",Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(0, A.numRows()),func);

}


/* ******************* */


template <class AMatrix, class XVector, class YVector>
struct BSPMV_Transpose_Functor {
  typedef typename AMatrix::execution_space execution_space;
  typedef typename AMatrix::non_const_ordinal_type ordinal_type;
  typedef typename AMatrix::non_const_value_type value_type;
  typedef typename Kokkos::TeamPolicy<execution_space> team_policy;
  typedef typename team_policy::member_type team_member;
  typedef Kokkos::Details::ArithTraits<value_type> ATV;

  const value_type alpha;
  AMatrix m_A;
  XVector m_x;
  YVector m_y;
  const ordinal_type block_size;

  const ordinal_type blocks_per_team;

  bool conjugate = false;

  BSPMV_Transpose_Functor(const value_type alpha_, const AMatrix m_A_,
                          const XVector m_x_, const YVector m_y_,
                          const int blocks_per_team_, bool conj_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        m_y(m_y_),
        block_size(m_A_.blockDim()),
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
    if (iBlock >= m_A.numRows()) {
      return;
    }
    //
    const auto jbeg         = m_A.graph.row_map[iBlock];
    const auto jend         = m_A.graph.row_map[iBlock + 1];
    const auto block_size_2 = block_size * block_size;
    const auto xvec         = &m_x[iBlock * block_size];
    //
    for (Ordinal jb = jbeg; jb < jend; ++jb) {
      const auto col_block = m_A.graph.entries[jb];
      auto yvec            = &m_y[block_size * col_block];
      const auto Aval_ptr  = &m_A.values[block_size_2 * jb];
      for (Ordinal ic = 0; ic < block_size; ++ic) {
        const auto xvalue = xvec[ic];
        for (Ordinal kr = 0; kr < block_size; ++kr) {
          const auto val = (conjugate)
                               ? ATV::conj(Aval_ptr[kr + ic * block_size])
                               : Aval_ptr[kr + ic * block_size];
          Kokkos::atomic_add(&yvec[kr],
                             static_cast<Scalar>(alpha * val * xvalue));
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
          const auto jbeg           = m_A.graph.row_map[iBlock];
          const auto jend           = m_A.graph.row_map[iBlock + 1];
          const auto row_num_blocks = static_cast<ordinal_type>(jend - jbeg);
          const auto block_size_2   = block_size * block_size;
          const auto col_idx        = &m_A.graph.entries[jbeg];
          const auto *Aval_ptr      = &m_A.values[jbeg * block_size_2];
          //
          const auto xvec = &m_x[iBlock * block_size];
          //
          for (ordinal_type ir = 0; ir < block_size; ++ir) {
            Kokkos::parallel_for(
                Kokkos::ThreadVectorRange(dev, row_num_blocks),
                [&](const ordinal_type &iEntry) {
                  const ordinal_type col_block =
                      col_idx[iEntry + jbeg] * block_size;
                  auto yvec = &m_y[col_block];
                  const auto *Aval =
                      Aval_ptr + iEntry * block_size_2 + ir * block_size;
                  for (ordinal_type jc = 0; jc < block_size; ++jc) {
                    const value_type val =
                        conjugate ? ATV::conj(Aval[jc]) : Aval[jc];
                    Kokkos::atomic_add(&yvec[jc], static_cast<y_value_type>(
                                                      alpha * val * xvec[ir]));
                  }
                });
            //
          }
          //
        });
  }
};


/* ******************* */


#ifdef KOKKOS_ENABLE_OPENMP
template<typename AMatrix, typename XVector, typename YVector>
void bspmv_raw_openmp_transpose(typename YVector::const_value_type& s_a,
                                AMatrix A, XVector x,  YVector y)
{
  typedef typename YVector::non_const_value_type value_type;
  typedef typename AMatrix::ordinal_type         ordinal_type;
  typedef typename AMatrix::non_const_size_type  size_type;

  typename XVector::const_value_type* KOKKOS_RESTRICT x_ptr = x.data();
  typename YVector::non_const_value_type* KOKKOS_RESTRICT y_ptr = y.data();

  const typename AMatrix::value_type* KOKKOS_RESTRICT matrixCoeffs = A.values.data();
  const ordinal_type* KOKKOS_RESTRICT matrixCols     = A.graph.entries.data();
  const size_type* KOKKOS_RESTRICT matrixRowOffsets  = A.graph.row_map.data();
  const size_type* KOKKOS_RESTRICT threadStarts      = A.graph.row_block_offsets.data();

#if defined(KOKKOS_ENABLE_PROFILING)
  uint64_t kpID = 0;
  if(Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::beginParallelFor("KokkosSparse::spmv<RawOpenMP,NoTranspose>", 0, &kpID);
  }
#endif

  typename YVector::const_value_type zero = 0;
  #pragma omp parallel
  {
#if defined(KOKKOS_COMPILER_INTEL) && !defined(__clang__)
    __assume_aligned(x_ptr, 64);
    __assume_aligned(y_ptr, 64);
#endif

    const int myID    = omp_get_thread_num();
    const size_type myStart = threadStarts[myID];
    const size_type myEnd   = threadStarts[myID + 1];
    const auto blockSize = A.blockDim();
    const auto blockSize2 = blockSize * blockSize;

    for (size_type row = myStart; row < myEnd; ++row) {
      const size_type rowStart = matrixRowOffsets[row];
      const size_type rowEnd   = matrixRowOffsets[row + 1];
      //
      const auto xvec = &x[row * blockSize];
      //
      for (Ordinal jblock = rowStart; jblock < rowEnd; ++jblock) {
        const auto col_block = A.graph.entries[jblock];
        auto yvec            = &y[blockSize * col_block];
        const auto Aval_ptr  = &matrixCoeffs[jblock * blockSize2];
        for (Ordinal ic = 0; ic < blockSize; ++ic) {
          const auto xvalue = xvec[ic];
          for (Ordinal kr = 0; kr < blockSize; ++kr) {
            Kokkos::atomic_add( &yvec[kr],
                                static_cast<Scalar>(s_a * Aval_ptr[ic + kr * blockSize] * xvalue) );
          }
        }
      }
    }
  }

#if defined(KOKKOS_ENABLE_PROFILING)
  if (Kokkos::Profiling::profileLibraryLoaded()) {
    Kokkos::Profiling::endParallelFor(kpID);
  }
#endif

}
#endif


/* ******************* */

//
// spMatVec_no_transpose: version for GPU execution spaces (TeamPolicy used)
//
template <class AT, class AO, class AD, class AM, class AS, class AlphaType,
          class XVector, class BetaType, class YVector,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_transpose(
    KokkosKernels::Experimental::Controls controls, const AlphaType &alpha,
    const KokkosSparse::Experimental::BlockCrsMatrix<AT, AO, AD, AM, AS> &A,
    const XVector &x, const BetaType &beta, YVector &y, bool useFallback,
    bool useConjugate) {
  if (A.numRows() <= static_cast<AO>(0)) {
    return;
  }

  typedef KokkosSparse::Experimental::BlockCrsMatrix<
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

  BSPMV_Transpose_Functor<AMatrix_Internal, XVector, YVector> func(
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


template <int M>
inline void spmv_transpose_gemv(const Scalar alpha, const Scalar *Aval,
                                const Ordinal lda, const Ordinal xrow,
                                const Scalar *x_ptr, Scalar *y) {
  for (Ordinal ic = 0; ic < xrow; ++ic) {
    for (Ordinal kr = 0; kr < M; ++kr) {
      const auto alpha_value = alpha * Aval[ic + kr * lda];
      Kokkos::atomic_add(&y[ic], static_cast<Scalar>(alpha_value * x_ptr[kr]));
    }
  }
}

#ifdef KOKKOS_ENABLE_SERIAL
//
// Explicit blockSize=N case
//
template <class StaticGraph, Ordinal blockSize>
struct SpMV_SerialTranspose {

  static inline void spmv(const Scalar alpha, const Scalar *Avalues,
                          const StaticGraph &Agraph, const Scalar *x,
                          Scalar *y, const Ordinal bs) {

    const Ordinal numBlockRows = Agraph.numRows();
    const auto blockSize2 = blockSize * blockSize;
    for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
      const auto jbeg       = Agraph.row_map[iblock];
      const auto jend       = Agraph.row_map[iblock + 1];
      const auto xval_ptr   = &x[iblock * blockSize];
      for (Ordinal jb = jbeg; jb < jend; ++jb) {
        const auto col_block = Agraph.entries[jb];
        auto yvec            = &y[blockSize * col_block];
        const auto Aval_ptr  = &Avalues[jb * blockSize2];
        //
        spmv_transpose_gemv<blockSize>(alpha, Aval_ptr, blockSize, blockSize, xval_ptr, yvec);
      }
    }
  }
};

//
// Special blockSize=1 case (optimized)
//
template <class StaticGraph>
struct SpMV_SerialTranspose<StaticGraph, 1> {

  static inline void spmv(const Scalar alpha, const Scalar *Avalues,
                          const StaticGraph &Agraph, const Scalar *x,
                          Scalar *y, const Ordinal bs) {

    const Ordinal numBlockRows = Agraph.numRows();
    for (Ordinal i = 0; i < numBlockRows; ++i) {
      const auto jbeg = Agraph.row_map[i];
      const auto jend = Agraph.row_map[i + 1];
      for (Ordinal j = jbeg; j < jend; ++j) {
        const auto alpha_value = alpha * Avalues[j];
        const auto col_idx1    = Agraph.entries[j];
        y[col_idx1] += alpha_value * x[i];
      }
    }
  }
};

//
// --- Basic approach for large block sizes
//
template <class StaticGraph>
struct SpMV_SerialTranspose<StaticGraph, 0> {

  static inline void spmv(const Scalar alpha, const Scalar *A_values,
                          const StaticGraph &A_graph, const Scalar *x,
                          Scalar *y, const Ordinal blockSize) {

    const Ordinal numBlockRows = A_graph.numRows();
    const auto blockSize2 = blockSize * blockSize;
    for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
      const auto jbeg       = A_graph.row_map[iblock];
      const auto jend       = A_graph.row_map[iblock + 1];
      const auto xval_ptr   = &x[iblock * blockSize];
      for (Ordinal jb = jbeg; jb < jend; ++jb) {
        auto yvec           = &y[blockSize * A_graph.entries[jb]];
        const auto Aval_ptr = A_values + jb * blockSize2;
        for (Ordinal ic = 0; ic < blockSize; ++ic) {
          for (Ordinal kr = 0; kr < blockSize; ++kr) {
            yvec[ic] += alpha * Aval_ptr[ic + kr * blockSize] * xval_ptr[kr];
          }
        }
      }
    }
  }
};

#endif // defined(KOKKOS_ENABLE_SERIAL)

//
// spMatVec_transpose: version for CPU execution spaces (RangePolicy or
// trivial serial impl used)
//
template < class AT, class AO, class AD, class AM, class AS,
          class AlphaType, class XVector, class BetaType, class YVector,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_transpose(KokkosKernels::Experimental::Controls controls,
                        const AlphaType &alpha,
                        const KokkosSparse::Experimental::BlockCrsMatrix< AT, AO, AD, AM, AS> &A,
                        const XVector &x, const BetaType &beta, YVector &y,
                        bool useFallback, bool useConjugate)
{

  if (A.numCols () <= static_cast<AO> (0)) {
    return;
  }

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
  if ((!useFallback) && (!useConjugate)) {
    // Call the MKL version
    if (useConjugate)
      spmv_block_mkl(controls, KokkosSparse::ConjugateTranspose, alpha, A, x, beta, y);
    else
      spmv_block_mkl(controls, KokkosSparse::Transpose, alpha, A, x, beta, y);
    return;
  }
#endif

  typedef Kokkos::View<
      typename XVector::const_value_type *,
      typename KokkosKernels::Impl::GetUnifiedLayout<XVector>::array_layout,
      typename XVector::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >
      XVector_Internal;

  typedef Kokkos::View<
      typename YVector::non_const_value_type *,
      typename KokkosKernels::Impl::GetUnifiedLayout<YVector>::array_layout,
      typename YVector::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      YVector_Internal;

  YVector_Internal y_i = y;

  // This is required to maintain semantics of KokkosKernels native SpMV:
  // if y contains NaN but beta = 0, the result y should be filled with 0.
  // For example, this is useful for passing in uninitialized y and beta=0.
  if (beta == Kokkos::ArithTraits<BetaType>::zero())
    Kokkos::deep_copy(y_i, Kokkos::ArithTraits<BetaType>::zero());
  else
    KokkosBlas::scal(y_i, beta, y_i);

  typedef KokkosSparse::Experimental::BlockCrsMatrix<AT, AO, AD,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> AMatrix_Internal;

  AMatrix_Internal A_internal = A;

  //
  // Treat the case y <- alpha * A^T * x + beta * y
  //

  XVector_Internal x_i       = x;
  const Ordinal blockSize    = A_internal.blockDim();
  const Ordinal numBlockRows = A_internal.numRows();
  //
  const auto &A_graph = A_internal.graph;
  //
#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same< typename AMatrix_Internal::device_type::execution_space,
                   Kokkos::Serial>::value) {
    Utils::eti_expand<Impl::bmax>(blockSize, [&]<int N>(const int blockSize) {
      SpMV_SerialTranspose<decltype(A_graph), N>::spmv(alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
    });
    return;
  }
#endif

  typedef typename AMatrix_Internal::execution_space execution_space;

#ifdef KOKKOS_ENABLE_OPENMP
  if ((std::is_same<execution_space, Kokkos::OpenMP>::value) &&
  (std::is_same<typename std::remove_cv<typename AMatrix_Internal::value_type>::type, double>::value) &&
  (std::is_same<typename XVector::non_const_value_type, double>::value) &&
  (std::is_same<typename YVector::non_const_value_type, double>::value) &&
  ((int)A.graph.row_block_offsets.extent(0) == (int)omp_get_max_threads() + 1) &&
  (((uintptr_t)(const void *)(x.data()) % 64) == 0) &&
  (((uintptr_t)(const void *)(y.data()) % 64) == 0) &&
  (!useConjugate))
  {
    bspmv_raw_openmp_transpose<AMatrix_Internal, XVector, YVector>(alpha, A, x, y);
    return;
  }
#endif

  BSPMV_Transpose_Functor<AMatrix_Internal,XVector,YVector> func (alpha, A_internal, x, y, 1, useConjugate);
  Kokkos::parallel_for("KokkosSparse::bspmv<Transpose>", Kokkos::RangePolicy<execution_space>(0, A.numRows()),func);

}

namespace Utils {

template<int checkVal, int maxVal, typename F>
struct _eti_expand_n {
  void operator()(int val, F f) {
    if (val == checkVal) {
      f.template operator()<checkVal>(val);
      return;
    }
    constexpr auto jump = (checkVal + maxVal + 1) / 2;
    if (val >= jump)
      _eti_expand_n<jump, maxVal, F>()(val, f);
    else
      _eti_expand_n<checkVal + 1, maxVal, F>()(val, f);
  }
};

template<int maxVal, typename F>
struct _eti_expand_n<maxVal, maxVal, F> { // break the recursion on maxVal
  void operator()(int val, F f) {
    f.template operator()<maxVal>(val);
  }
};

template<int maxVal, typename F>
void eti_expand(int val, F f) {
  if (val > maxVal)
    f.template operator()<0>(val); // call special "dynamic" variant that checks actual param
  else
    _eti_expand_n<1, maxVal, F>()(val, f); // dispatch to explicit variant based on compile-time value
}

} // namespace Utils

}  // namespace Impl

}  // namespace KokkosSparse

#endif  // KOKKOSKERNELS_KOKKOSSPARSE_SPMV_BLOCK_IMPL_HPP
