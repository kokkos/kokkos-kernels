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

#ifndef KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BLOCK_CRS_HPP
#define KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BLOCK_CRS_HPP

#include "Kokkos_Core.hpp"

#include "KokkosBlas.hpp"
#include "KokkosKernels_default_types.hpp"
#include "KokkosSparse_BlockCrsMatrix.hpp"

#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Controls.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

namespace KokkosSparse {
namespace Impl {

//////////////////////////////////////////////////////////

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

constexpr size_t bmax = 12;

///////////////////////////////////
//// This needs to be generalized
using Scalar  = default_scalar;
using Ordinal = default_lno_t;
using Offset  = default_size_type;
using Layout  = default_layout;

using device_type = typename Kokkos::Device<
    Kokkos::DefaultExecutionSpace,
    typename Kokkos::DefaultExecutionSpace::memory_space>;

using crs_matrix_t_ =
    typename KokkosSparse::CrsMatrix<Scalar, Ordinal, device_type, void,
                                     Offset>;

using values_type = typename crs_matrix_t_::values_type;

using bcrs_matrix_t_ = typename KokkosSparse::Experimental::BlockCrsMatrix<
    Scalar, Ordinal, device_type, void, Offset>;

using MultiVector_Internal =
    typename Kokkos::View<Scalar **, Layout, device_type>;

const Scalar SC_ONE  = Kokkos::ArithTraits<Scalar>::one();
const Scalar SC_ZERO = Kokkos::ArithTraits<Scalar>::zero();
//////////////////////////////////

template <int M>
inline void spmv_serial_gemv(Scalar *Aval, const Ordinal lda,
                             const Scalar *x_ptr,
                             std::array<Scalar, Impl::bmax> &y) {
  for (Ordinal ic = 0; ic < M; ++ic) {
    const auto xvalue = x_ptr[ic];
    for (Ordinal kr = 0; kr < M; ++kr) {
      y[kr] += Aval[ic + kr * lda] * xvalue;
    }
  }
}

template <class StaticGraph, int N>
inline void spmv_serial(const double alpha, double *Avalues,
                        const StaticGraph &Agraph, const double *x, double *y) {
  const Ordinal numBlockRows = Agraph.numRows();

  if (N == 1) {
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

  std::array<double, Impl::bmax> tmp;
  auto Aval        = Avalues;
  const Ordinal N2 = N * N;
  for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
    const auto jbeg       = Agraph.row_map[iblock];
    const auto jend       = Agraph.row_map[iblock + 1];
    const auto num_blocks = jend - jbeg;
    const auto lda        = num_blocks * N;
    tmp.fill(0);
    for (Ordinal jb = 0; jb < num_blocks; ++jb) {
      const auto col_block = Agraph.entries[jb + jbeg];
      const auto xval_ptr  = x + N * col_block;
      auto Aval_ptr        = Aval + jb * N;
      //
      spmv_serial_gemv<N>(Aval_ptr, lda, xval_ptr, tmp);
    }
    //
    Aval = Aval + num_blocks * N2;
    //
    auto yvec = &y[iblock * N];
    for (Ordinal ii = 0; ii < N; ++ii) {
      yvec[ii] += alpha * tmp[ii];
    }
  }
}

//
// spMatVec_no_transpose: version for CPU execution spaces (RangePolicy or
// trivial serial impl used)
//
template < class AT, class AO, class AD, class AM, class AS,
        class AlphaType, class XVector, class BetaType, class YVector,
        typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
        typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_no_transpose(const AlphaType &alpha,
                           const KokkosSparse::Experimental::BlockCrsMatrix< AT, AO, AD, AM, AS> &A,
                           const XVector &x, const BetaType &beta, YVector &y,
                           bool useFallback) {

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
  //
  const bool conjugate = false;
  //
  ////////////
  assert(useFallback);
  ////////////

  typedef KokkosSparse::Experimental::BlockCrsMatrix<AT, AO, AD,
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> AMatrix_Internal;

  AMatrix_Internal A_internal = A;
  const auto &A_graph = A.graph;

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same< typename AMatrix_Internal::device_type::execution_space,
      Kokkos::Serial>::value) {
    //
    if (blockSize <= std::min<size_t>(8, Impl::bmax)) {
      switch (blockSize) {
        default:
        case 1:
          spmv_serial<decltype(A_graph), 1>(alpha, &A.values[0], A_graph, &x[0],
                                            &y[0]);
          break;
        case 2:
          spmv_serial<decltype(A_graph), 2>(alpha, &A.values[0], A_graph, &x[0],
                                            &y[0]);
          break;
        case 3:
          spmv_serial<decltype(A_graph), 3>(alpha, &A.values[0], A_graph, &x[0],
                                            &y[0]);
          break;
        case 4:
          spmv_serial<decltype(A_graph), 4>(alpha, &A.values[0], A_graph, &x[0],
                                            &y[0]);
          break;
        case 5:
          spmv_serial<decltype(A_graph), 5>(alpha, &A.values[0], A_graph, &x[0],
                                            &y[0]);
          break;
        case 6:
          spmv_serial<decltype(A_graph), 6>(alpha, &A.values[0], A_graph, &x[0],
                                            &y[0]);
          break;
        case 7:
          spmv_serial<decltype(A_graph), 7>(alpha, &A.values[0], A_graph, &x[0],
                                            &y[0]);
          break;
        case 8:
          spmv_serial<decltype(A_graph), 8>(alpha, &A.values[0], A_graph, &x[0],
                                            &y[0]);
          break;
        case 9:
          spmv_serial<decltype(A_graph), 9>(alpha, &A.values[0], A_graph, &x[0],
                                            &y[0]);
          break;
        case 10:
          spmv_serial<decltype(A_graph), 10>(alpha, &A.values[0], A_graph,
                                             &x[0], &y[0]);
          break;
        case 11:
          spmv_serial<decltype(A_graph), 11>(alpha, &A.values[0], A_graph,
                                             &x[0], &y[0]);
          break;
        case 12:
          spmv_serial<decltype(A_graph), 12>(alpha, &A.values[0], A_graph,
                                             &x[0], &y[0]);
          break;
      }
      return;
    }
    //
    // --- Basic approach for large block sizes
    //
    const Ordinal blockSize_squared = blockSize * blockSize;
    auto Aval = &A.values[0], yvec = &y[0];
    for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
      const auto jbeg       = A_graph.row_map[iblock];
      const auto jend       = A_graph.row_map[iblock + 1];
      const auto num_blocks = jend - jbeg;
      const auto lda        = num_blocks * blockSize;
      //
      for (Ordinal jb = 0, shifta = 0; jb < num_blocks;
           ++jb, shifta += blockSize) {
        const auto col_block = A_graph.entries[jb + jbeg];
        const auto xval_ptr  = &x[0] + blockSize * col_block;
        const auto Aval_ptr  = Aval + shifta;
        for (Ordinal kr = 0; kr < blockSize; ++kr) {
          for (Ordinal ic = 0; ic < blockSize; ++ic) {
            yvec[kr] += alpha * Aval_ptr[ic + kr * lda] * xval_ptr[ic];
          }
        }
      }
      //
      Aval = Aval + lda * blockSize;
      yvec = yvec + blockSize;
    }
    return;
  }
#endif

#ifdef KOKKOS_ENABLE_OPENMP
  //
  // This section terminates for the moment.
  // terminate called recursively
  //
  const Ordinal blockSize2 = blockSize * blockSize;
  //
  KokkosKernels::Experimental::Controls controls;
  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }
  //
  if (blockSize <= std::min<size_t>(8, Impl::bmax)) {
    switch (blockSize) {
      default:
      case 1: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg = A.graph.row_map[iblock];
          const auto jend = A.graph.row_map[iblock + 1];
          for (Ordinal j = jbeg; j < jend; ++j) {
            const auto col_block = A.graph.entries[j];
            y[iblock] += alpha * A.values[j] * x[col_block];
          }
        }
        break;
      }
      case 2: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A.graph.row_map[iblock];
          int j                 = jbeg;
          const auto jend       = A.graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = blockSize * num_blocks;
          const auto Aval       = &A.values[blockSize2 * jbeg];
          std::array<Scalar, Impl::bmax> tmp;
          tmp.fill(0);
          for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
            const auto col_block = A.graph.entries[j];
            const auto xval_ptr  = &x[0] + blockSize * col_block;
            auto Aval_ptr        = Aval + jb * blockSize;
            //
            spmv_serial_gemv<2>(Aval_ptr, lda, xval_ptr, tmp);
          }
          //
          auto yvec = &y[iblock * blockSize];
          for (Ordinal ii = 0; ii < 2; ++ii) {
            yvec[ii] += alpha * tmp[ii];
          }
        }
        break;
      }
      case 3: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A.graph.row_map[iblock];
          int j                 = jbeg;
          const auto jend       = A.graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = blockSize * num_blocks;
          const auto Aval       = &A.values[blockSize2 * jbeg];
          std::array<Scalar, Impl::bmax> tmp;
          tmp.fill(0);
          for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
            const auto col_block = A.graph.entries[j];
            const auto xval_ptr  = &x[0] + blockSize * col_block;
            auto Aval_ptr        = Aval + jb * blockSize;
            //
            spmv_serial_gemv<3>(Aval_ptr, lda, xval_ptr, tmp);
          }
          //
          auto yvec = &y[iblock * blockSize];
          for (Ordinal ii = 0; ii < 3; ++ii) {
            yvec[ii] += alpha * tmp[ii];
          }
        }
        break;
      }
      case 4: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A.graph.row_map[iblock];
          int j                 = jbeg;
          const auto jend       = A.graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = blockSize * num_blocks;
          const auto Aval       = &A.values[blockSize2 * jbeg];
          std::array<Scalar, Impl::bmax> tmp;
          tmp.fill(0);
          for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
            const auto col_block = A.graph.entries[j];
            const auto xval_ptr  = &x[0] + blockSize * col_block;
            auto Aval_ptr        = Aval + jb * blockSize;
            //
            spmv_serial_gemv<4>(Aval_ptr, lda, xval_ptr, tmp);
          }
          //
          auto yvec = &y[iblock * blockSize];
          for (Ordinal ii = 0; ii < 4; ++ii) {
            yvec[ii] += alpha * tmp[ii];
          }
        }
        break;
      }
      case 5: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A.graph.row_map[iblock];
          int j                 = jbeg;
          const auto jend       = A.graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = blockSize * num_blocks;
          const auto Aval       = &A.values[blockSize2 * jbeg];
          std::array<Scalar, Impl::bmax> tmp;
          tmp.fill(0);
          for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
            const auto col_block = A.graph.entries[j];
            const auto xval_ptr  = &x[0] + blockSize * col_block;
            auto Aval_ptr        = Aval + jb * blockSize;
            //
            spmv_serial_gemv<5>(Aval_ptr, lda, xval_ptr, tmp);
          }
          //
          auto yvec = &y[iblock * blockSize];
          for (Ordinal ii = 0; ii < 5; ++ii) {
            yvec[ii] += alpha * tmp[ii];
          }
        }
        break;
      }
      case 6: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A.graph.row_map[iblock];
          int j                 = jbeg;
          const auto jend       = A.graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = blockSize * num_blocks;
          const auto Aval       = &A.values[blockSize2 * jbeg];
          std::array<Scalar, Impl::bmax> tmp;
          tmp.fill(0);
          for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
            const auto col_block = A.graph.entries[j];
            const auto xval_ptr  = &x[0] + blockSize * col_block;
            auto Aval_ptr        = Aval + jb * blockSize;
            //
            spmv_serial_gemv<6>(Aval_ptr, lda, xval_ptr, tmp);
          }
          //
          auto yvec = &y[iblock * blockSize];
          for (Ordinal ii = 0; ii < 6; ++ii) {
            yvec[ii] += alpha * tmp[ii];
          }
        }
        break;
      }
      case 7: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A.graph.row_map[iblock];
          int j                 = jbeg;
          const auto jend       = A.graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = blockSize * num_blocks;
          const auto Aval       = &A.values[blockSize2 * jbeg];
          std::array<Scalar, Impl::bmax> tmp;
          tmp.fill(0);
          for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
            const auto col_block = A.graph.entries[j];
            const auto xval_ptr  = &x[0] + blockSize * col_block;
            auto Aval_ptr        = Aval + jb * blockSize;
            //
            spmv_serial_gemv<7>(Aval_ptr, lda, xval_ptr, tmp);
          }
          //
          auto yvec = &y[iblock * blockSize];
          for (Ordinal ii = 0; ii < 7; ++ii) {
            yvec[ii] += alpha * tmp[ii];
          }
        }
        break;
      }
      case 8: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A.graph.row_map[iblock];
          int j                 = jbeg;
          const auto jend       = A.graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = blockSize * num_blocks;
          const auto Aval       = &A.values[blockSize2 * jbeg];
          std::array<Scalar, Impl::bmax> tmp;
          tmp.fill(0);
          for (Ordinal jb = 0; jb < num_blocks; ++jb, ++j) {
            const auto col_block = A.graph.entries[j];
            const auto xval_ptr  = &x[0] + blockSize * col_block;
            auto Aval_ptr        = Aval + jb * blockSize;
            //
            spmv_serial_gemv<8>(Aval_ptr, lda, xval_ptr, tmp);
          }
          //
          auto yvec = &y[iblock * blockSize];
          for (Ordinal ii = 0; ii < 8; ++ii) {
            yvec[ii] += alpha * tmp[ii];
          }
        }
        break;
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
      const auto jbeg       = A.graph.row_map[iblock];
      const auto jend       = A.graph.row_map[iblock + 1];
      const auto num_blocks = jend - jbeg;
      const auto lda        = blockSize * num_blocks;
      const auto Aval       = &A.values[blockSize2 * jbeg];
      auto yvec             = &y[iblock * blockSize];
      for (Ordinal jb = 0; jb < num_blocks; ++jb) {
        const auto col_block = A.graph.entries[jb + jbeg];
        const auto xval_ptr  = &x[0] + blockSize * col_block;
        const auto Aval_ptr  = Aval + jb * blockSize;
        for (Ordinal ic = 0; ic < blockSize; ++ic) {
          const auto xvalue = xval_ptr[ic];
          for (Ordinal kr = 0; kr < blockSize; ++kr) {
            yvec[kr] += alpha * Aval_ptr[ic + kr * lda] * xvalue;
          }
        }
      }
    }
  }
#endif
  return;
}

template <int M>
inline void spmv_transpose_gemv(const Scalar alpha, Scalar *Aval,
                                const Ordinal lda, const Ordinal xrow,
                                const Scalar *x_ptr, Scalar *y) {
  for (Ordinal ic = 0; ic < xrow; ++ic) {
    for (Ordinal kr = 0; kr < M; ++kr) {
      const auto alpha_value = alpha * Aval[ic + kr * lda];
      Kokkos::atomic_add(&y[ic], static_cast<Scalar>(alpha_value * x_ptr[kr]));
    }
  }
}

template <class StaticGraph, int N>
inline void spmv_transpose_serial(const Scalar alpha, Scalar *Avalues,
                                  const StaticGraph &Agraph, const Scalar *x,
                                  Scalar *y, const Ordinal ldy) {
  const Ordinal numBlockRows = Agraph.numRows();

  if (N == 1) {
    for (Ordinal i = 0; i < numBlockRows; ++i) {
      const auto jbeg = Agraph.row_map[i];
      const auto jend = Agraph.row_map[i + 1];
      for (Ordinal j = jbeg; j < jend; ++j) {
        const auto alpha_value = alpha * Avalues[j];
        const auto col_idx1    = Agraph.entries[j];
        y[col_idx1] += alpha_value * x[i];
      }
    }
    return;
  }

  const auto blockSize2 = N * N;
  for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
    const auto jbeg       = Agraph.row_map[iblock];
    const auto jend       = Agraph.row_map[iblock + 1];
    const auto num_blocks = jend - jbeg;
    const auto lda        = num_blocks * N;
    const auto Aval       = &Avalues[blockSize2 * jbeg];
    const auto xval_ptr   = &x[iblock * N];
    for (Ordinal jb = 0; jb < num_blocks; ++jb) {
      const auto col_block = Agraph.entries[jb + jbeg];
      auto yvec            = &y[N * col_block];
      auto Aval_ptr        = Aval + jb * N;
      //
      spmv_transpose_gemv<N>(alpha, Aval_ptr, lda, N, xval_ptr, yvec);
    }
  }
}

//
// spMatVec_transpose: version for CPU execution spaces (RangePolicy or
// trivial serial impl used)
//
template < class AT, class AO, class AD, class AM, class AS,
          class AlphaType, class XVector, class BetaType, class YVector,
          typename std::enable_if<!KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_transpose(const AlphaType &alpha,
                        const KokkosSparse::Experimental::BlockCrsMatrix< AT, AO, AD, AM, AS> &A,
                        const XVector &x, const BetaType &beta, YVector &y,
                        bool useFallback) {

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

  ////////////
  assert(useFallback);
  ////////////

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
  const bool conjugate = false;
  const auto &A_graph = A_internal.graph;
  //

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same< typename AMatrix_Internal::device_type::execution_space,
                   Kokkos::Serial>::value) {
    //
    if (blockSize <= std::min<size_t>(8, Impl::bmax)) {
      switch (blockSize) {
        default:
        case 1:
          spmv_transpose_serial<decltype(A_graph), 1>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 2:
          spmv_transpose_serial<decltype(A_graph), 2>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 3:
          spmv_transpose_serial<decltype(A_graph), 3>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 4:
          spmv_transpose_serial<decltype(A_graph), 4>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 5:
          spmv_transpose_serial<decltype(A_graph), 5>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 6:
          spmv_transpose_serial<decltype(A_graph), 6>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 7:
          spmv_transpose_serial<decltype(A_graph), 7>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 8:
          spmv_transpose_serial<decltype(A_graph), 8>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 9:
          spmv_transpose_serial<decltype(A_graph), 9>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 10:
          spmv_transpose_serial<decltype(A_graph), 10>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 11:
          spmv_transpose_serial<decltype(A_graph), 11>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
        case 12:
          spmv_transpose_serial<decltype(A_graph), 12>(
              alpha, &A_internal.values[0], A_graph, &x[0], &y[0], blockSize);
          break;
      }
      return;
    }
    //
    // --- Basic approach for large block sizes
    //
    const auto blockSize2 = blockSize * blockSize;
    for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
      const auto jbeg       = A_graph.row_map[iblock];
      const auto jend       = A_graph.row_map[iblock + 1];
      const auto num_blocks = jend - jbeg;
      const auto lda        = num_blocks * blockSize;
      const auto xval_ptr   = &x[iblock * blockSize];
      const auto Aval       = &A_internal.values[jbeg * blockSize2];
      for (Ordinal jb = 0; jb < num_blocks; ++jb) {
        auto yvec           = &y[blockSize * A_graph.entries[jb + jbeg]];
        const auto Aval_ptr = Aval + jb * blockSize;
        for (Ordinal ic = 0; ic < blockSize; ++ic) {
          for (Ordinal kr = 0; kr < blockSize; ++kr) {
            yvec[ic] += alpha * Aval_ptr[ic + kr * lda] * xval_ptr[kr];
          }
        }
      }
    }
    return;
  }
#endif

#ifdef KOKKOS_ENABLE_OPENMP
  //
  // This section terminates for the moment.
  // terminate called recursively
  //
  const Ordinal blockSize2 = blockSize * blockSize;
  //
  KokkosKernels::Experimental::Controls controls;
  bool use_dynamic_schedule = false;  // Forces the use of a dynamic schedule
  bool use_static_schedule  = false;  // Forces the use of a static schedule
  if (controls.isParameter("schedule")) {
    if (controls.getParameter("schedule") == "dynamic") {
      use_dynamic_schedule = true;
    } else if (controls.getParameter("schedule") == "static") {
      use_static_schedule = true;
    }
  }
  //
  if (blockSize <= std::min<size_t>(8, Impl::bmax)) {
    //
    // 2021/06/09 --- Cases for blockSize > 1 need to be modified
    //
    switch (blockSize) {
      default:
      case 1: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg = A.graph.row_map[iblock];
          const auto jend = A.graph.row_map[iblock + 1];
          for (Ordinal j = jbeg; j < jend; ++j) {
            const auto col_block = A.graph.entries[j];
            Kokkos::atomic_add(
                &y[col_block],
                static_cast<Scalar>(alpha * A.values[j] * x[iblock]));
          }
        }
        break;
      }
      case 2: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = num_blocks * blockSize;
          const auto xval_ptr   = &x[iblock * blockSize];
          const auto Aval       = &A.values[blockSize2 * jbeg];
          for (Ordinal jb = 0; jb < num_blocks; ++jb) {
            const auto col_block = A.graph.entries[jb + jbeg];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = Aval + jb * blockSize;
            spmv_transpose_gemv<2>(alpha, Aval_ptr, lda, blockSize, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 3: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = num_blocks * blockSize;
          const auto xval_ptr   = &x[iblock * blockSize];
          const auto Aval       = &A.values[blockSize2 * jbeg];
          for (Ordinal jb = 0; jb < num_blocks; ++jb) {
            const auto col_block = A.graph.entries[jb + jbeg];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = Aval + jb * blockSize;
            spmv_transpose_gemv<3>(alpha, Aval_ptr, lda, blockSize, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 4: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = num_blocks * blockSize;
          const auto xval_ptr   = &x[iblock * blockSize];
          const auto Aval       = &A.values[blockSize2 * jbeg];
          for (Ordinal jb = 0; jb < num_blocks; ++jb) {
            const auto col_block = A.graph.entries[jb + jbeg];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = Aval + jb * blockSize;
            spmv_transpose_gemv<4>(alpha, Aval_ptr, lda, blockSize, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 5: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = num_blocks * blockSize;
          const auto xval_ptr   = &x[iblock * blockSize];
          const auto Aval       = &A.values[blockSize2 * jbeg];
          for (Ordinal jb = 0; jb < num_blocks; ++jb) {
            const auto col_block = A.graph.entries[jb + jbeg];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = Aval + jb * blockSize;
            spmv_transpose_gemv<5>(alpha, Aval_ptr, lda, blockSize, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 6: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = num_blocks * blockSize;
          const auto xval_ptr   = &x[iblock * blockSize];
          const auto Aval       = &A.values[blockSize2 * jbeg];
          for (Ordinal jb = 0; jb < num_blocks; ++jb) {
            const auto col_block = A.graph.entries[jb + jbeg];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = Aval + jb * blockSize;
            spmv_transpose_gemv<6>(alpha, Aval_ptr, lda, blockSize, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 7: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = num_blocks * blockSize;
          const auto xval_ptr   = &x[iblock * blockSize];
          const auto Aval       = &A.values[blockSize2 * jbeg];
          for (Ordinal jb = 0; jb < num_blocks; ++jb) {
            const auto col_block = A.graph.entries[jb + jbeg];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = Aval + jb * blockSize;
            spmv_transpose_gemv<7>(alpha, Aval_ptr, lda, blockSize, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
      case 8: {
#pragma omp parallel for schedule(static)
        for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
          const auto jbeg       = A_graph.row_map[iblock];
          const auto jend       = A_graph.row_map[iblock + 1];
          const auto num_blocks = jend - jbeg;
          const auto lda        = num_blocks * blockSize;
          const auto xval_ptr   = &x[iblock * blockSize];
          const auto Aval       = &A.values[blockSize2 * jbeg];
          for (Ordinal jb = 0; jb < num_blocks; ++jb) {
            const auto col_block = A.graph.entries[jb + jbeg];
            auto yvec            = &y[blockSize * col_block];
            const auto Aval_ptr  = Aval + jb * blockSize;
            spmv_transpose_gemv<8>(alpha, Aval_ptr, lda, blockSize, xval_ptr,
                                   yvec);
          }
        }
        break;
      }
    }
  } else {
#pragma omp parallel for schedule(static)
    for (Ordinal iblock = 0; iblock < numBlockRows; ++iblock) {
      const auto jbeg       = A.graph.row_map[iblock];
      const auto jend       = A.graph.row_map[iblock + 1];
      const auto num_blocks = jend - jbeg;
      const auto xvec       = &x[iblock * blockSize];
      const auto lda        = blockSize * num_blocks;
      const auto Aval       = &A.values[blockSize2 * jbeg];
      for (Ordinal jb = 0; jb < num_blocks; ++jb) {
        const auto col_block = A.graph.entries[jb + jbeg];
        auto yvec            = &y[blockSize * col_block];
        const auto Aval_ptr  = Aval + jb * blockSize;
        for (Ordinal ic = 0; ic < blockSize; ++ic) {
          const auto xvalue = xvec[ic];
          for (Ordinal kr = 0; kr < blockSize; ++kr) {
            Kokkos::atomic_add( &yvec[kr],
                static_cast<Scalar>(alpha * Aval_ptr[kr + ic * lda] * xvalue) );
          }
        }
      }
    }
  }
#endif
  return;
}

}  // namespace Impl

}  // namespace KokkosSparse

#endif  // KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BLOCK_CRS_HPP