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

#ifndef KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BLOCKCRS_HPP
#define KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BLOCKCRS_HPP

#include "Kokkos_Core.hpp"

#include "KokkosSparse_BlockCrsMatrix.hpp"

#include "KokkosSparse_spmv_impl.hpp"
#include "KokkosSparse_spmv_impl_util.hpp"

#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Controls.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

namespace KokkosSparse {

template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
void spmv(KokkosKernels::Experimental::Controls controls, const char mode[],
          const AlphaType &alpha, const AMatrix &A, const XVector &x,
          const BetaType &beta, const YVector &y);

namespace Impl {
namespace BlockCrs {

//
// Explicit blockSize=N case
//
template <int blockSize>
struct BlockCrs_SerialNoTranspose {
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemv(bool useConjugate, const ScalarType alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          const ScalarType *x, ScalarType *y, const int bs) {
    static_assert(((blockSize > 1) && (blockSize <= Impl::bmax)),
                  " Error when specifying the blocksize ");
    std::array<ScalarType, Impl::bmax> tmp;
    const int blockSize2 = blockSize * blockSize;
    if (useConjugate) {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg       = row_map[iblock];
        const SizeType jend       = row_map[iblock + 1];
        const auto Avalues_iblock = Avalues + jbeg * blockSize2;
        const auto lda            = (jend - jbeg) * blockSize;
        tmp.fill(0);
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto xval_ptr  = x + blockSize * col_block;
          const auto Aval_ptr  = Avalues_iblock + (jb - jbeg) * blockSize;
          raw_gemv_c<blockSize>(Aval_ptr, lda, xval_ptr, tmp);
        }
        //
        auto yvec = &y[iblock * blockSize];
        for (int ii = 0; ii < blockSize; ++ii) {
          yvec[ii] = yvec[ii] + alpha * tmp[ii];
        }
      }
    } else {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg       = row_map[iblock];
        const SizeType jend       = row_map[iblock + 1];
        const auto Avalues_iblock = Avalues + jbeg * blockSize2;
        const auto lda            = (jend - jbeg) * blockSize;
        tmp.fill(0);
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto xval_ptr  = x + blockSize * col_block;
          const auto Aval_ptr  = Avalues_iblock + (jb - jbeg) * blockSize;
          raw_gemv_n<blockSize>(Aval_ptr, lda, xval_ptr, tmp);
        }
        //
        auto yvec = &y[iblock * blockSize];
        for (int ii = 0; ii < blockSize; ++ii) {
          yvec[ii] = yvec[ii] + alpha * tmp[ii];
        }
      }
    }
  }
};

//
// Special blockSize=1 case
//
template <>
struct BlockCrs_SerialNoTranspose<1> {
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemv(bool useConjugate, const ScalarType alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          const ScalarType *x, ScalarType *y, const int bs) {
    if (useConjugate) {
      for (DataType i = 0; i < numBlockRows; ++i) {
        const SizeType jbeg = row_map[i];
        const SizeType jend = row_map[i + 1];
        ScalarType tmp      = 0;
        for (SizeType j = jbeg; j < jend; ++j) {
          const auto a_value =
              Kokkos::ArithTraits<ScalarType>::conj(Avalues[j]);
          const auto col_idx1 = entries[j];
          tmp                 = tmp + a_value * x[col_idx1];
        }
        y[i] = y[i] + alpha * tmp;
      }
    } else {
      for (DataType i = 0; i < numBlockRows; ++i) {
        const SizeType jbeg = row_map[i];
        const SizeType jend = row_map[i + 1];
        ScalarType tmp      = 0;
        for (SizeType j = jbeg; j < jend; ++j) {
          const auto a_value  = Avalues[j];
          const auto col_idx1 = entries[j];
          tmp                 = tmp + a_value * x[col_idx1];
        }
        y[i] = y[i] + alpha * tmp;
      }
    }
  }
};

//
// --- Basic approach for large block sizes
//

template <>
struct BlockCrs_SerialNoTranspose<0> {
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemv(bool useConjugate, const ScalarType alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          const ScalarType *x, ScalarType *y, const int bs) {
    const int bs_squared = bs * bs;
    auto yvec            = &y[0];
    //
    const int multiple       = Impl::unroll * (bs / Impl::unroll);
    const bool even_leftover = ((bs - multiple * Impl::unroll) % 2 == 0);
    std::array<ScalarType, Impl::bmax> tmp;
    //
    if (useConjugate) {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        const auto Aval_ptr = Avalues + jbeg * bs_squared;
        const auto lda      = (jend - jbeg) * bs;
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto xval_ptr  = &x[0] + bs * col_block;
          const auto Aentries  = Aval_ptr + (jb - jbeg) * bs;
          for (int kr = 0; kr < multiple; kr += Impl::unroll) {
            tmp.fill(0);
            for (int ic = 0; ic < multiple; ic += Impl::unroll)
              raw_gemv_c<Impl::unroll>(Aentries + ic + kr * lda, lda,
                                       xval_ptr + ic, tmp);
            for (int ic = multiple; ic < bs; ++ic) {
              tmp[0] = tmp[0] + Kokkos::ArithTraits<ScalarType>::conj(
                                    Aentries[ic + kr * lda]) *
                                    xval_ptr[ic];
              tmp[1] = tmp[1] + Kokkos::ArithTraits<ScalarType>::conj(
                                    Aentries[ic + (kr + 1) * lda]) *
                                    xval_ptr[ic];
              tmp[2] = tmp[2] + Kokkos::ArithTraits<ScalarType>::conj(
                                    Aentries[ic + (kr + 2) * lda]) *
                                    xval_ptr[ic];
              tmp[3] = tmp[3] + Kokkos::ArithTraits<ScalarType>::conj(
                                    Aentries[ic + (kr + 3) * lda]) *
                                    xval_ptr[ic];
            }
            //
            yvec[kr]     = yvec[kr] + alpha * tmp[0];
            yvec[kr + 1] = yvec[kr + 1] + alpha * tmp[1];
            yvec[kr + 2] = yvec[kr + 2] + alpha * tmp[2];
            yvec[kr + 3] = yvec[kr + 3] + alpha * tmp[3];
          }
          //
          if (even_leftover) {
            for (int kr = multiple; kr < bs; kr += 2) {
              tmp[0] = 0;
              tmp[1] = 0;
              for (int ic = 0; ic < bs; ic += 2)
                raw_gemv_c<2>(Aentries + ic + kr * lda, lda, xval_ptr + ic,
                              tmp);
              yvec[kr]     = yvec[kr] + alpha * tmp[0];
              yvec[kr + 1] = yvec[kr + 1] + alpha * tmp[1];
            }
          } else {
            for (int kr = multiple; kr < bs; ++kr) {
              tmp[0] = 0;
              for (int ic = 0; ic < bs; ++ic)
                tmp[0] = tmp[0] + Kokkos::ArithTraits<ScalarType>::conj(
                                      Aentries[ic + kr * lda]) *
                                      xval_ptr[ic];
              yvec[kr] = yvec[kr] + alpha * tmp[0];
            }
          }
        }
        //
        yvec = yvec + bs;
      }
    } else {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        const auto lda      = (jend - jbeg) * bs;
        const auto Aval_ptr = Avalues + jbeg * bs_squared;
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto xval_ptr  = &x[0] + bs * col_block;
          const auto Aentries  = Aval_ptr + (jb - jbeg) * bs;
          for (int kr = 0; kr < multiple; kr += Impl::unroll) {
            tmp.fill(0);
            for (int ic = 0; ic < multiple; ic += Impl::unroll)
              raw_gemv_n<Impl::unroll>(Aentries + ic + kr * lda, lda,
                                       xval_ptr + ic, tmp);
            for (int ic = multiple; ic < bs; ++ic) {
              tmp[0] = tmp[0] + Aentries[ic + kr * lda] * xval_ptr[ic];
              tmp[1] = tmp[1] + Aentries[ic + (kr + 1) * lda] * xval_ptr[ic];
              tmp[2] = tmp[2] + Aentries[ic + (kr + 2) * lda] * xval_ptr[ic];
              tmp[3] = tmp[3] + Aentries[ic + (kr + 3) * lda] * xval_ptr[ic];
            }
            //
            yvec[kr]     = yvec[kr] + alpha * tmp[0];
            yvec[kr + 1] = yvec[kr + 1] + alpha * tmp[1];
            yvec[kr + 2] = yvec[kr + 2] + alpha * tmp[2];
            yvec[kr + 3] = yvec[kr + 3] + alpha * tmp[3];
          }
          //
          if (even_leftover) {
            for (int kr = multiple; kr < bs; kr += 2) {
              tmp[0] = 0;
              tmp[1] = 0;
              for (int ic = 0; ic < bs; ic += 2)
                raw_gemv_n<2>(Aentries + ic + kr * lda, lda, xval_ptr + ic,
                              tmp);
              yvec[kr]     = yvec[kr] + alpha * tmp[0];
              yvec[kr + 1] = yvec[kr + 1] + alpha * tmp[1];
            }
          } else {
            for (int kr = multiple; kr < bs; ++kr) {
              tmp[0] = 0;
              for (int ic = 0; ic < bs; ++ic)
                tmp[0] = tmp[0] + Aentries[ic + kr * lda] * xval_ptr[ic];
              yvec[kr] = yvec[kr] + alpha * tmp[0];
            }
          }
        }
        //
        yvec = yvec + bs;
      }
      /////
    }
  }
};

/* ******************* */

template <class AMatrix, class XVector, class YVector>
struct BlockCrs_GEMV_Functor {
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
  const ordinal_type block_size_2;

  const ordinal_type *entries_;

  const ordinal_type blocks_per_team;

  bool conjugate = false;

  BlockCrs_GEMV_Functor(const value_type alpha_, const AMatrix m_A_,
                        const XVector m_x_, const YVector m_y_,
                        const int blocks_per_team_, bool conj_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        m_y(m_y_),
        block_size(m_A_.blockDim()),
        block_size_2(block_size * block_size),
        entries_(m_A_.graph.entries.data()),
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
    const auto jbeg = m_A.graph.row_map[iBlock];
    const auto jend = m_A.graph.row_map[iBlock + 1];
    //
    auto yvec = &m_y[iBlock * block_size];
    //
    const auto Aval_ptr = &m_A.values[jbeg * block_size_2];
    const auto lda      = (jend - jbeg) * block_size;
    //
    if (conjugate) {
      for (auto jb = jbeg; jb < jend; ++jb) {
        const auto col_block = entries_[jb];
        const auto xval_ptr  = &m_x[block_size * col_block];
        const auto Aentries  = Aval_ptr + (jb - jbeg) * block_size;
        for (ordinal_type kr = 0; kr < block_size; ++kr) {
          value_type tmp = 0;
          for (ordinal_type ic = 0; ic < block_size; ++ic) {
            const auto aval = ATV::conj(Aentries[ic + kr * lda]);
            tmp += aval * xval_ptr[ic];
          }
          yvec[kr] += alpha * tmp;
        }
      }
    } else {
      for (auto jb = jbeg; jb < jend; ++jb) {
        const auto col_block = entries_[jb];
        const auto xval_ptr  = &m_x[block_size * col_block];
        const auto Aentries  = Aval_ptr + (jb - jbeg) * block_size;
        for (ordinal_type kr = 0; kr < block_size; ++kr) {
          value_type tmp = 0;
          for (ordinal_type ic = 0; ic < block_size; ++ic) {
            const auto aval = Aentries[ic + kr * lda];
            tmp += aval * xval_ptr[ic];
          }
          yvec[kr] += alpha * tmp;
        }
      }
    }
    //
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
          const auto col_idx        = &entries_[jbeg];
          const auto *Aval_ptr      = &m_A.values[jbeg * block_size_2];
          //
          auto yvec         = &m_y[iBlock * block_size];
          y_value_type lsum = 0;
          //
          for (ordinal_type ir = 0; ir < block_size; ++ir) {
            lsum = 0;
            Kokkos::parallel_reduce(
                Kokkos::ThreadVectorRange(dev, row_num_blocks),
                [&](const ordinal_type &iEntry, y_value_type &ysum) {
                  const ordinal_type col_block = col_idx[iEntry] * block_size;
                  const auto xval              = &m_x[col_block];
                  const auto *Aval =
                      Aval_ptr + iEntry * block_size_2 + ir * block_size;
                  for (ordinal_type jc = 0; jc < block_size; ++jc) {
                    const value_type val =
                        conjugate ? ATV::conj(Aval[jc]) : Aval[jc];
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
    const KokkosSparse::Experimental::BlockCrsMatrix<
        AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> &A,
    const XVector &x, const BetaType &beta, YVector &y, bool useFallback,
    bool useConjugate) {
  typedef Kokkos::View<
      typename XVector::const_value_type *,
      typename KokkosKernels::Impl::GetUnifiedLayout<XVector>::array_layout,
      typename XVector::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>
      XVector_Internal;

  typedef Kokkos::View<
      typename YVector::non_const_value_type *,
      typename KokkosKernels::Impl::GetUnifiedLayout<YVector>::array_layout,
      typename YVector::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>>
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

  XVector_Internal x_i = x;

  typedef KokkosSparse::Experimental::BlockCrsMatrix<
      AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS>
      AMatrix_Internal;

  AMatrix_Internal A_internal = A;
  const auto &A_graph         = A.graph;
  AT alpha_internal           = static_cast<AT>(alpha);

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same<typename AMatrix_Internal::device_type::execution_space,
                   Kokkos::Serial>::value) {
    const auto blockSize    = A.blockDim();
    const auto numBlockRows = A.numRows();
    switch (blockSize) {
      default:
        // Version of arbitrarily large block sizes
        BlockCrs_SerialNoTranspose<0>::gemv(
            useConjugate, alpha_internal, &A_internal.values[0],
            &A_graph.row_map[0], numBlockRows, &A_graph.entries[0], &x[0],
            &y[0], blockSize);
        break;
      case 1:
        BlockCrs_SerialNoTranspose<1>::gemv(
            useConjugate, alpha_internal, &A_internal.values[0],
            &A_graph.row_map[0], numBlockRows, &A_graph.entries[0], &x[0],
            &y[0], blockSize);
        break;
      case 2:
        BlockCrs_SerialNoTranspose<2>::gemv(
            useConjugate, alpha_internal, &A_internal.values[0],
            &A_graph.row_map[0], numBlockRows, &A_graph.entries[0], &x[0],
            &y[0], blockSize);
        break;
      case 3:
        BlockCrs_SerialNoTranspose<3>::gemv(
            useConjugate, alpha_internal, &A_internal.values[0],
            &A_graph.row_map[0], numBlockRows, &A_graph.entries[0], &x[0],
            &y[0], blockSize);
        break;
      case 4:
        BlockCrs_SerialNoTranspose<4>::gemv(
            useConjugate, alpha_internal, &A_internal.values[0],
            &A_graph.row_map[0], numBlockRows, &A_graph.entries[0], &x[0],
            &y[0], blockSize);
        break;
      case 5:
        BlockCrs_SerialNoTranspose<5>::gemv(
            useConjugate, alpha_internal, &A_internal.values[0],
            &A_graph.row_map[0], A_graph.numRows(), &A_graph.entries[0], &x[0],
            &y[0], blockSize);
        break;
      case 6:
        BlockCrs_SerialNoTranspose<6>::gemv(
            useConjugate, alpha_internal, &A_internal.values[0],
            &A_graph.row_map[0], A_graph.numRows(), &A_graph.entries[0], &x[0],
            &y[0], blockSize);
        break;
      case 7:
        BlockCrs_SerialNoTranspose<7>::gemv(
            useConjugate, alpha_internal, &A_internal.values[0],
            &A_graph.row_map[0], A_graph.numRows(), &A_graph.entries[0], &x[0],
            &y[0], blockSize);
        break;
      case 8:
        BlockCrs_SerialNoTranspose<8>::gemv(
            useConjugate, alpha_internal, &A_internal.values[0],
            &A_graph.row_map[0], A_graph.numRows(), &A_graph.entries[0], &x[0],
            &y[0], blockSize);
        break;
      case 9:
        BlockCrs_SerialNoTranspose<9>::gemv(
            useConjugate, alpha_internal, &A_internal.values[0],
            &A_graph.row_map[0], A_graph.numRows(), &A_graph.entries[0], &x[0],
            &y[0], blockSize);
        break;
    }
    return;
  }
#endif

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

  BlockCrs_GEMV_Functor<AMatrix_Internal, XVector, YVector> func(
      alpha, A_internal, x, y, 1, useConjugate);
  if (((A.nnz() > 10000000) || use_dynamic_schedule) && !use_static_schedule) {
    Kokkos::parallel_for(
        "KokkosSparse::bspmv<NoTranspose,Dynamic>",
        Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Dynamic>>(
            0, A.numRows()),
        func);
  } else {
    Kokkos::parallel_for(
        "KokkosSparse::bspmv<NoTranspose,Static>",
        Kokkos::RangePolicy<execution_space, Kokkos::Schedule<Kokkos::Static>>(
            0, A.numRows()),
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
    const KokkosSparse::Experimental::BlockCrsMatrix<
        AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> &A,
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

  BlockCrs_GEMV_Functor<AMatrix_Internal, XVector, YVector> func(
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

///
/* ******************* */

/// \brief Driver structure for single vector right hand side
struct EXTENT_1 {
  //
  template <class ABlockCrsMatrix, class AlphaType, class XVector,
            class BetaType, class YVector>
  static void spmv(KokkosKernels::Experimental::Controls controls,
                   const char mode[], const AlphaType &alpha,
                   const ABlockCrsMatrix &A, const XVector &X,
                   const BetaType &beta, const YVector &Y, bool useFallback) {
    if ((X.stride_0() == 1) || (Y.stride_0() == 1)) {
      if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
          (mode[0] == KokkosSparse::Conjugate[0])) {
        bool useConjugate = (mode[0] == KokkosSparse::Conjugate[0]);
        return spMatVec_no_transpose(controls, alpha, A, X, beta, Y,
                                     useFallback, useConjugate);
      }
    }
    //
    // Fall-back solution where we convert the matrix
    // Expensive as we create a hard copy of the "expanded" graph
    //
    typedef typename ABlockCrsMatrix::ordinal_type OrdinalType;
    typedef typename ABlockCrsMatrix::size_type SizeType;
    //
    const auto blockSize = A.blockDim();
    OrdinalType a_n      = A.numCols() * blockSize;
    SizeType a_nnz       = A.nnz() * (blockSize * blockSize);
    //
    Kokkos::View<OrdinalType *, Kokkos::LayoutLeft,
                 typename ABlockCrsMatrix::device_type,
                 typename XVector::memory_traits>
        my_col_idx("convert col_idx", a_nnz);
    Kokkos::View<SizeType *, Kokkos::LayoutLeft,
                 typename ABlockCrsMatrix::device_type,
                 typename XVector::memory_traits>
        my_row_map("convert row_map", a_n + 1);
    my_row_map(0) = 0;
    size_t icount = 0, jcount = 0;
    for (OrdinalType ib = 0; ib < A.numRows(); ++ib) {
      const auto len =
          blockSize * (A.graph.row_map(ib + 1) - A.graph.row_map(ib));
      for (OrdinalType ii = 0; ii < blockSize; ++ii) {
        for (auto jb = A.graph.row_map(ib); jb < A.graph.row_map(ib + 1);
             ++jb) {
          const auto col_idx = A.graph.entries(jb);
          for (OrdinalType jj = 0; jj < blockSize; ++jj)
            my_col_idx(jcount++) = col_idx * blockSize + jj;
        }
        my_row_map(icount + 1) = my_row_map(icount) + len;
        icount += 1;
      }
    }
    typedef KokkosSparse::CrsMatrix<typename ABlockCrsMatrix::value_type,
                                    typename ABlockCrsMatrix::ordinal_type,
                                    typename ABlockCrsMatrix::device_type,
                                    typename XVector::memory_traits,
                                    typename ABlockCrsMatrix::size_type>
        MyCrsMatrix;
    MyCrsMatrix A1("crs_convert", a_n, a_n, a_nnz, A.values, my_row_map,
                   my_col_idx);
    KokkosSparse::spmv(controls, mode, alpha, A1, X, beta, Y);
  }
};

/// \brief Driver structure for multi-vector right hand side
struct EXTENT_2 {
  template <class ABlockCrsMatrix, class AlphaType, class XVector,
            class BetaType, class YVector>
  static void spmv(KokkosKernels::Experimental::Controls controls,
                   const char mode[], const AlphaType &alpha,
                   const ABlockCrsMatrix &A, const XVector &X,
                   const BetaType &beta, const YVector &Y, bool useFallback) {
    //
    // Fall-back solution where we convert the matrix
    // Expensive as we create a hard copy of the "expanded" graph
    //
    typedef typename ABlockCrsMatrix::ordinal_type OrdinalType;
    typedef typename ABlockCrsMatrix::size_type SizeType;
    //
    const auto blockSize = A.blockDim();
    OrdinalType a_n      = A.numCols() * blockSize;
    SizeType a_nnz       = A.nnz() * (blockSize * blockSize);
    //
    Kokkos::View<OrdinalType *, Kokkos::LayoutLeft,
                 typename ABlockCrsMatrix::device_type,
                 typename XVector::memory_traits>
        my_col_idx("convert col_idx", a_nnz);
    Kokkos::View<SizeType *, Kokkos::LayoutLeft,
                 typename ABlockCrsMatrix::device_type,
                 typename XVector::memory_traits>
        my_row_map("convert row_map", a_n + 1);
    my_row_map(0) = 0;
    size_t icount = 0, jcount = 0;
    for (OrdinalType ib = 0; ib < A.numRows(); ++ib) {
      const auto len =
          blockSize * (A.graph.row_map(ib + 1) - A.graph.row_map(ib));
      for (OrdinalType ii = 0; ii < blockSize; ++ii) {
        for (auto jb = A.graph.row_map(ib); jb < A.graph.row_map(ib + 1);
             ++jb) {
          const auto col_idx = A.graph.entries(jb);
          for (OrdinalType jj = 0; jj < blockSize; ++jj)
            my_col_idx(jcount++) = col_idx * blockSize + jj;
        }
        my_row_map(icount + 1) = my_row_map(icount) + len;
        icount += 1;
      }
    }
    typedef KokkosSparse::CrsMatrix<typename ABlockCrsMatrix::value_type,
                                    typename ABlockCrsMatrix::ordinal_type,
                                    typename ABlockCrsMatrix::device_type,
                                    typename XVector::memory_traits,
                                    typename ABlockCrsMatrix::size_type>
        MyCrsMatrix;
    MyCrsMatrix A1("crs_convert", a_n, a_n, a_nnz, A.values, my_row_map,
                   my_col_idx);
    KokkosSparse::spmv(controls, mode, alpha, A1, X, beta, Y);
  }
};

///

template <class AlphaType, class ABlockCrsMatrix, class XVector, class BetaType,
          class YVector>
void spmv(KokkosKernels::Experimental::Controls controls, const char mode[],
          const AlphaType &alpha, const ABlockCrsMatrix &A, const XVector &X,
          const BetaType &beta, const YVector &Y) {
  //
  Impl::verifyArguments(mode, A, X, Y);
  //
  if (alpha == Kokkos::ArithTraits<AlphaType>::zero() || A.numRows() == 0 ||
      A.numCols() == 0 || A.nnz() == 0) {
    // This is required to maintain semantics of KokkosKernels native SpMV:
    // if y contains NaN but beta = 0, the result y should be filled with 0.
    // For example, this is useful for passing in uninitialized y and beta=0.
    if (beta == Kokkos::ArithTraits<BetaType>::zero())
      Kokkos::deep_copy(Y, Kokkos::ArithTraits<BetaType>::zero());
    else
      KokkosBlas::scal(Y, beta, Y);
    return;
  }
  //
  typedef KokkosSparse::Experimental::BlockCrsMatrix<
      typename ABlockCrsMatrix::value_type,
      typename ABlockCrsMatrix::ordinal_type,
      typename ABlockCrsMatrix::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      typename ABlockCrsMatrix::size_type>
      AMatrix_Internal;
  AMatrix_Internal Ainternal = A;
  //
  // Whether to call KokkosKernel's native implementation, even if a TPL impl is
  // available
  bool useFallback = controls.isParameter("algorithm") &&
                     controls.getParameter("algorithm") == "native";
  //
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
  // cuSPARSE does not support the conjugate mode (C), and cuSPARSE 9 only
  // supports the normal (N) mode.
  if (std::is_same<typename AMatrix_Internal::device_type::memory_space,
                   Kokkos::CudaSpace>::value ||
      std::is_same<typename AMatrix_Internal::device_type::memory_space,
                   Kokkos::CudaUVMSpace>::value) {
#if (9000 <= CUDA_VERSION)
    useFallback = useFallback || (mode[0] != NoTranspose[0]);
#endif
#if defined(CUSPARSE_VERSION) && (10300 <= CUSPARSE_VERSION)
    useFallback = useFallback || (mode[0] == Conjugate[0]);
#endif
  }
#endif
//
#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
  if (std::is_same<typename AMatrix_Internal::device_type::memory_space,
                   Kokkos::HostSpace>::value) {
    useFallback = useFallback || (mode[0] == Conjugate[0]);
  }
#endif
  //
  using X_Tensor_Rank =
      typename std::conditional<static_cast<int>(XVector::rank) == 2, EXTENT_2,
                                EXTENT_1>::type;
  X_Tensor_Rank::spmv(controls, mode, alpha, Ainternal, X, beta, Y,
                      useFallback);
  //
}

}  // namespace BlockCrs

}  // namespace Impl

}  // namespace KokkosSparse

#endif  // KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BLOCKCRS_HPP
