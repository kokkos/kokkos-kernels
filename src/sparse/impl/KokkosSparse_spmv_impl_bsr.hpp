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

#ifndef KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BSR_HPP
#define KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BSR_HPP

#include "Kokkos_Core.hpp"

#include "KokkosSparse_BsrMatrix.hpp"

#include "KokkosSparse_spmv_impl.hpp"
#include "KokkosSparse_spmv_impl_util.hpp"

#include "KokkosKernels_helpers.hpp"
#include "KokkosKernels_Controls.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_ExecSpaceUtils.hpp"

namespace KokkosSparse {

namespace Impl {
namespace Bsr {

//
// Explicit case
// These functions will be instantiated when the block size is
// smaller than Impl::bmax
//
template <int blockSize>
struct Bsr_SerialNoTranspose {
  static constexpr int rhsUnroll = 16;
  static constexpr int tmp_size  = Impl::bmax * rhsUnroll;

  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemv(bool useConjugate, const ScalarType &alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          const ScalarType *x, ScalarType *y, const int bs) {
    static_assert(((blockSize > 1) && (blockSize <= Impl::bmax)),
                  " Error when specifying the blocksize ");
    constexpr int blockSize2 = blockSize * blockSize;
    std::array<ScalarType, Impl::bmax> tmp;
    if (useConjugate) {
      const auto lda = blockSize;
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        tmp.fill(0);
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto xval_ptr  = x + blockSize * col_block;
          auto Aval_ptr        = Avalues + jb * blockSize2;
          raw_gemv_c<blockSize, ScalarType>(Aval_ptr, lda, xval_ptr, tmp);
        }
        //
        auto yvec = &y[iblock * blockSize];
        raw_axpy<blockSize, ScalarType>(alpha, tmp, yvec);
      }
    } else {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        tmp.fill(0);
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto xval_ptr  = x + blockSize * col_block;
          auto Aval_ptr        = Avalues + jb * blockSize2;
          raw_gemv_n<blockSize, ScalarType>(Aval_ptr, blockSize, xval_ptr, tmp);
        }
        //
        auto yvec = &y[iblock * blockSize];
        raw_axpy<blockSize, ScalarType>(alpha, tmp, yvec);
      }
    }
  }

  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemm(bool useConjugate, const ScalarType &alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          int xrhs, const ScalarType *x, int ldx, ScalarType *y,
                          int ldy, const int bs = blockSize) {
    std::vector<ScalarType> tmp(bs * xrhs);
    const int blockSize2 = blockSize * blockSize;
    if (useConjugate) {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        tmp.assign(blockSize * xrhs, 0);
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          auto Aval_ptr        = Avalues + jb * blockSize2;
          const auto xval_ptr  = &x[col_block * blockSize];
          raw_gemm_c<blockSize, ScalarType>(Aval_ptr, blockSize, xval_ptr, ldx,
                                            xrhs, tmp);
        }
        //
        for (int ic = 0; ic < xrhs; ++ic) {
          auto yvec = &y[iblock * blockSize + ic * ldy];
          for (int jr = 0; jr < blockSize; ++jr)
            yvec[jr] = yvec[jr] + alpha * tmp[jr + blockSize * ic];
        }
      }
    } else {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        tmp.assign(blockSize * xrhs, 0);
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          auto Aval_ptr        = Avalues + jb * blockSize2;
          const auto xval_ptr  = &x[col_block * blockSize];
          raw_gemm_n<blockSize, ScalarType>(Aval_ptr, blockSize, xval_ptr, ldx,
                                            xrhs, tmp);
        }
        //
        for (int ic = 0; ic < xrhs; ++ic) {
          auto yvec = &y[iblock * blockSize + ic * ldy];
          for (int jr = 0; jr < blockSize; ++jr)
            yvec[jr] = yvec[jr] + alpha * tmp[jr + blockSize * ic];
        }
      }
    }
  }
};

//
// Special case: blockSize = 1
//
template <>
struct Bsr_SerialNoTranspose<1> {
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemv(bool useConjugate, const ScalarType &alpha,
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

  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemm(bool useConjugate, const ScalarType &alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          int xrhs, const ScalarType *x, int ldx, ScalarType *y,
                          int ldy, const int bs = 1) {
    std::vector<ScalarType> tmp(bs * xrhs);
    if (useConjugate) {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto acoeff =
              Kokkos::ArithTraits<ScalarType>::conj(Avalues[jb]);
          for (int ic = 0; ic < xrhs; ++ic) {
            y[iblock + ic * ldy] += alpha * acoeff * x[col_block + ic * ldy];
          }
        }
      }
    } else {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto acoeff    = Avalues[jb];
          for (int ic = 0; ic < xrhs; ++ic) {
            y[iblock + ic * ldy] += alpha * acoeff * x[col_block + ic * ldy];
          }
        }
      }
    }
  }
};

//
// Basic approach for large block sizes
//
template <>
struct Bsr_SerialNoTranspose<0> {
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemv(bool useConjugate, const ScalarType &alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          const ScalarType *x, ScalarType *y, const int bs) {
    const int bs_squared = bs * bs;
    auto yvec            = &y[0];
    //
    const int multiple       = Impl::unroll * (bs / Impl::unroll);
    const bool even_leftover = ((bs - multiple * Impl::unroll) % 2 == 0);
    std::array<ScalarType, Impl::bmax> tmp;
    const auto lda = bs;
    //
    if (useConjugate) {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto xval_ptr  = &x[0] + bs * col_block;
          const auto Aentries  = Avalues + jb * bs_squared;
          for (int kr = 0; kr < multiple; kr += Impl::unroll) {
            tmp.fill(0);
            for (int ic = 0; ic < multiple; ic += Impl::unroll)
              raw_gemv_c<Impl::unroll, ScalarType, Impl::unroll>(
                  Aentries + ic + kr * lda, lda, xval_ptr + ic, tmp);
            for (int ic = multiple; ic < bs; ++ic)
              raw_gemv_c<Impl::unroll, ScalarType, 1>(Aentries + ic + kr * lda,
                                                      lda, xval_ptr + ic, tmp);
            //
            raw_axpy<Impl::unroll, ScalarType>(alpha, tmp, yvec + kr);
          }
          //
          if (even_leftover) {
            for (int kr = multiple; kr < bs; kr += 2) {
              tmp[0] = 0;
              tmp[1] = 0;
              for (int ic = 0; ic < bs; ic += 2)
                raw_gemv_c<2, ScalarType, 2>(Aentries + ic + kr * lda, lda,
                                             xval_ptr + ic, tmp);
              raw_axpy<2, ScalarType>(alpha, tmp, yvec + kr);
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
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto xval_ptr  = &x[0] + bs * col_block;
          const auto Aentries  = Avalues + jb * bs_squared;
          for (int kr = 0; kr < multiple; kr += Impl::unroll) {
            tmp.fill(0);
            for (int ic = 0; ic < multiple; ic += Impl::unroll)
              raw_gemv_n<Impl::unroll, ScalarType, Impl::unroll>(
                  Aentries + ic + kr * lda, lda, xval_ptr + ic, tmp);
            for (int ic = multiple; ic < bs; ++ic)
              raw_gemv_n<Impl::unroll, ScalarType, 1>(Aentries + ic + kr * lda,
                                                      lda, xval_ptr + ic, tmp);
            //
            raw_axpy<Impl::unroll, ScalarType>(alpha, tmp, yvec + kr);
          }
          //
          if (even_leftover) {
            for (int kr = multiple; kr < bs; kr += 2) {
              tmp[0] = 0;
              tmp[1] = 0;
              for (int ic = 0; ic < bs; ic += 2)
                raw_gemv_n<2, ScalarType, 2>(Aentries + ic + kr * lda, lda,
                                             xval_ptr + ic, tmp);
              raw_axpy<2, ScalarType>(alpha, tmp, yvec + kr);
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
    }
  }
  //
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemm(bool useConjugate, const ScalarType &alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          int xrhs, const ScalarType *x, int ldx, ScalarType *y,
                          int ldy, const int bs = 0) {
    const int blockSize2 = bs * bs;
    if (useConjugate) {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          auto Aval_ptr        = Avalues + jb * blockSize2;
          for (int ic = 0; ic < xrhs; ++ic) {
            for (DataType ir = 0; ir < bs; ++ir) {
              for (DataType jr = 0; jr < bs; ++jr) {
                const auto avalue = Kokkos::ArithTraits<ScalarType>::conj(
                    Aval_ptr[jr + ir * bs]);
                y[ir + iblock * bs + ic * ldy] +=
                    alpha * avalue * x[jr + col_block * bs + ic * ldx];
              }
            }
          }
        }
      }
    } else {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          auto Aval_ptr        = Avalues + jb * blockSize2;
          for (int ic = 0; ic < xrhs; ++ic) {
            for (DataType ir = 0; ir < bs; ++ir) {
              for (DataType jr = 0; jr < bs; ++jr) {
                y[ir + iblock * bs + ic * ldy] +=
                    alpha * Aval_ptr[jr + ir * bs] *
                    x[jr + col_block * bs + ic * ldx];
              }
            }
          }
        }
      }
    }
  }
};

/* ------------------- */

template <class AMatrix, class XVector, class YVector>
struct BSR_GEMV_Functor {
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

  const ordinal_type *entries;
  const value_type *matrixCoeffs;

  const ordinal_type blocks_per_team;

  bool conjugate = false;

  BSR_GEMV_Functor(const value_type alpha_, const AMatrix m_A_,
                   const XVector m_x_, const YVector m_y_,
                   const int blocks_per_team_, bool conj_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        m_y(m_y_),
        block_size(m_A_.blockDim()),
        block_size_2(block_size * block_size),
        entries(m_A_.graph.entries.data()),
        matrixCoeffs(&m_A.values[0]),
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
    auto yvec                        = &m_y[iBlock * block_size];
    const auto jbeg = m_A.graph.row_map(iBlock);
    const auto jend = m_A.graph.row_map(iBlock + 1);
    if (conjugate) {
      for (auto jb = jbeg; jb < jend; ++jb) {
        const auto col_block = m_A.graph.entries(jb);
        const auto shift = jb * block_size * block_size;
        for (ordinal_type ir = 0; ir < block_size; ++ir) {
          value_type tmp = 0;
          for (ordinal_type jc = 0; jc < block_size; ++jc) {
            const auto avalue = 
                  Kokkos::ArithTraits< value_type >::conj(m_A.values(shift + ir * block_size + jc));
            tmp += avalue * m_x[col_block * block_size + jc];
          }
          yvec[ir] += alpha * tmp;
        }
      }
    }
    else {
      for (auto jb = jbeg; jb < jend; ++jb) {
        const auto col_block = m_A.graph.entries(jb);
        const auto shift = jb * block_size * block_size;
        for (ordinal_type ir = 0; ir < block_size; ++ir) {
          value_type tmp = 0;
          for (ordinal_type jc = 0; jc < block_size; ++jc) {
            tmp += m_A.values(shift + ir * block_size + jc) * m_x[col_block * block_size + jc];
          }
          yvec[ir] += alpha * tmp;
        }
      }
    }
/*
    constexpr ordinal_type numBlocks = 1;
    const auto *local_row_map        = &m_A.graph.row_map[iBlock];
    switch (block_size) {
      default:
        Bsr_SerialNoTranspose<0>::gemv(conjugate, alpha, matrixCoeffs,
                                       local_row_map, numBlocks, entries,
                                       &m_x[0], yvec, block_size);
        break;
      case 1:
        Bsr_SerialNoTranspose<1>::gemv(conjugate, alpha, matrixCoeffs,
                                       local_row_map, numBlocks, entries,
                                       &m_x[0], yvec, block_size);
        break;
      case 2:
        Bsr_SerialNoTranspose<2>::gemv(conjugate, alpha, matrixCoeffs,
                                       local_row_map, numBlocks, entries,
                                       &m_x[0], yvec, block_size);
        break;
      case 3:
        Bsr_SerialNoTranspose<3>::gemv(conjugate, alpha, matrixCoeffs,
                                       local_row_map, numBlocks, entries,
                                       &m_x[0], yvec, block_size);
        break;
      case 4:
        Bsr_SerialNoTranspose<4>::gemv(conjugate, alpha, matrixCoeffs,
                                       local_row_map, 1, entries, &m_x[0], yvec,
                                       block_size);
        break;
    }
*/
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
          const auto col_idx        = &entries[jbeg];
          const auto *Aval_ptr      = &matrixCoeffs[jbeg * block_size_2];
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
    const KokkosSparse::Experimental::BsrMatrix<
        AT, AO, AD, Kokkos::MemoryTraits<Kokkos::Unmanaged>, AS> &A,
    const XVector &x, const BetaType &beta, YVector &y, bool useFallback,
    bool useConjugate) {
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

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same<typename AMatrix_Internal::device_type::execution_space,
                   Kokkos::Serial>::value) {
    const auto blockSize    = A.blockDim();
    const auto numBlockRows = A.numRows();
    const auto &A_graph = A.graph;
    const auto alpha_internal = static_cast<AT>(alpha);
    switch (blockSize) {
      default:
        // Version for arbitrarily large block size
        Bsr_SerialNoTranspose<0>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 1:
        Bsr_SerialNoTranspose<1>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 2:
        Bsr_SerialNoTranspose<2>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 3:
        Bsr_SerialNoTranspose<3>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 4:
        Bsr_SerialNoTranspose<4>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 5:
        Bsr_SerialNoTranspose<5>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 6:
        Bsr_SerialNoTranspose<6>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 7:
        Bsr_SerialNoTranspose<7>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 8:
        Bsr_SerialNoTranspose<8>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 9:
        Bsr_SerialNoTranspose<9>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
    }
    return;
  }
#endif

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
    const XVector &x, const BetaType &beta, YVector &y, bool useFallback,
    bool useConjugate) {
  if (A.numRows() <= static_cast<AO>(0)) {
    return;
  }

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

//
// Explicit blockSize=N case
//
template <int blockSize>
struct Bsr_SerialTranspose {
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemv(bool useConjugate, const ScalarType &alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          const ScalarType *x, ScalarType *y, const int bs) {
    static_assert(((blockSize > 1) && (blockSize <= Impl::bmax)),
                  " Error when specifying the blocksize ");
    std::array<ScalarType, Impl::bmax> tmp;
    const int blockSize2 = blockSize * blockSize;
    if (useConjugate) {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          tmp.fill(0);
          const auto col_block = entries[jb];
          const auto xval_ptr  = x + iblock * blockSize;
          auto Aval_ptr        = Avalues + jb * blockSize2;
          raw_gemv_h<blockSize, ScalarType>(Aval_ptr, blockSize, xval_ptr, tmp);
          auto yvec = &y[col_block * blockSize];
          raw_axpy<blockSize, ScalarType>(alpha, tmp, yvec);
        }
      }
    } else {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          tmp.fill(0);
          const auto col_block = entries[jb];
          const auto xval_ptr  = x + iblock * blockSize;
          auto Aval_ptr        = Avalues + jb * blockSize2;
          raw_gemv_t<blockSize, ScalarType>(Aval_ptr, blockSize, xval_ptr, tmp);
          auto yvec = &y[col_block * blockSize];
          raw_axpy<blockSize, ScalarType>(alpha, tmp, yvec);
        }
      }
    }
  }
  //
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemm(bool useConjugate, const ScalarType &alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          int xrhs, const ScalarType *x, int ldx, ScalarType *y,
                          int ldy, const int bs = blockSize) {
    std::vector<ScalarType> tmp(bs * xrhs);
    const int blockSize2 = blockSize * blockSize;
    if (useConjugate) {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          tmp.assign(blockSize * xrhs, 0);
          const auto col_block = entries[jb];
          auto Aval_ptr        = Avalues + jb * blockSize2;
          const auto xval_ptr  = &x[iblock * blockSize];
          raw_gemm_h<blockSize, ScalarType>(Aval_ptr, blockSize, xval_ptr, ldx,
                                            xrhs, tmp);
          for (int ic = 0; ic < xrhs; ++ic) {
            auto yvec = &y[col_block * blockSize + ic * ldy];
            for (int jr = 0; jr < blockSize; ++jr)
              yvec[jr] = yvec[jr] + alpha * tmp[jr + blockSize * ic];
          }
        }
      }
    } else {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          tmp.assign(blockSize * xrhs, 0);
          const auto col_block = entries[jb];
          auto Aval_ptr        = Avalues + jb * blockSize2;
          const auto xval_ptr  = &x[iblock * blockSize];
          raw_gemm_t<blockSize, ScalarType>(Aval_ptr, blockSize, xval_ptr, ldx,
                                            xrhs, tmp);
          for (int ic = 0; ic < xrhs; ++ic) {
            auto yvec = &y[col_block * blockSize + ic * ldy];
            for (int jr = 0; jr < blockSize; ++jr)
              yvec[jr] = yvec[jr] + alpha * tmp[jr + blockSize * ic];
          }
        }
      }
    }
  }
};

//
// Special blockSize=1 case
//
template <>
struct Bsr_SerialTranspose<1> {
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemv(bool useConjugate, const ScalarType &alpha,
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
          y[col_idx1] += alpha * a_value * x[i];
        }
      }
    } else {
      for (DataType i = 0; i < numBlockRows; ++i) {
        const SizeType jbeg = row_map[i];
        const SizeType jend = row_map[i + 1];
        ScalarType tmp      = 0;
        for (SizeType j = jbeg; j < jend; ++j) {
          const auto a_value  = Avalues[j];
          const auto col_idx1 = entries[j];
          y[col_idx1] += alpha * a_value * x[i];
        }
      }
    }
  }
  //
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemm(bool useConjugate, const ScalarType &alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          int xrhs, const ScalarType *x, int ldx, ScalarType *y,
                          int ldy, const int bs = 1) {
    std::vector<ScalarType> tmp(bs * xrhs);
    if (useConjugate) {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto acoeff =
              Kokkos::ArithTraits<ScalarType>::conj(Avalues[jb]);
          for (int ic = 0; ic < xrhs; ++ic) {
            y[col_block + ic * ldy] += alpha * acoeff * x[iblock + ic * ldy];
          }
        }
      }
    } else {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto acoeff    = Avalues[jb];
          for (int ic = 0; ic < xrhs; ++ic) {
            y[col_block + ic * ldy] += alpha * acoeff * x[iblock + ic * ldy];
          }
        }
      }
    }
  }
};

//
// --- Basic approach for large block sizes
//

template <>
struct Bsr_SerialTranspose<0> {
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemv(bool useConjugate, const ScalarType &alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          const ScalarType *x, ScalarType *y, const int bs) {
    const int bs_squared = bs * bs;
    //
    const int multiple       = Impl::unroll * (bs / Impl::unroll);
    const bool even_leftover = ((bs - multiple * Impl::unroll) % 2 == 0);
    std::array<ScalarType, Impl::bmax> tmp;
    const auto lda = bs;
    //
    if (useConjugate) {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        const auto xval_ptr = &x[0] + iblock * bs;
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto Aval_ptr  = Avalues + jb * bs_squared;
          auto yvec            = &y[col_block * bs];
          for (int kr = 0; kr < multiple; kr += Impl::unroll) {
            tmp.fill(0);
            for (int ic = 0; ic < multiple; ic += Impl::unroll)
              raw_gemv_h<Impl::unroll, ScalarType, Impl::unroll>(
                  Aval_ptr + kr + ic * lda, lda, xval_ptr + ic, tmp);
            for (int ic = multiple; ic < bs; ++ic)
              raw_gemv_h<Impl::unroll, ScalarType, 1>(Aval_ptr + kr + ic * lda,
                                                      lda, xval_ptr + ic, tmp);
            //
            raw_axpy<Impl::unroll, ScalarType>(alpha, tmp, yvec + kr);
          }
          //
          if (even_leftover) {
            for (int kr = multiple; kr < bs; kr += 2) {
              tmp[0] = 0;
              tmp[1] = 0;
              for (int ic = 0; ic < bs; ic += 2)
                raw_gemv_h<2, ScalarType, 2>(Aval_ptr + kr + ic * lda, lda,
                                             xval_ptr + ic, tmp);
              raw_axpy<2, ScalarType>(alpha, tmp, yvec + kr);
            }
          } else {
            for (int kr = multiple; kr < bs; ++kr) {
              tmp[0] = 0;
              for (int ic = 0; ic < bs; ++ic) {
                const auto a_value = Kokkos::ArithTraits<ScalarType>::conj(
                    Aval_ptr[kr + ic * lda]);
                tmp[0] = tmp[0] + a_value * xval_ptr[ic];
              }
              yvec[kr] = yvec[kr] + alpha * tmp[0];
            }
          }
        }
      }
    } else {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        const auto xval_ptr = &x[0] + iblock * bs;
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          const auto Aval_ptr  = Avalues + jb * bs_squared;
          auto yvec            = &y[col_block * bs];
          for (int kr = 0; kr < multiple; kr += Impl::unroll) {
            tmp.fill(0);
            for (int ic = 0; ic < multiple; ic += Impl::unroll)
              raw_gemv_t<Impl::unroll, ScalarType, Impl::unroll>(
                  Aval_ptr + kr + ic * lda, lda, xval_ptr + ic, tmp);
            for (int ic = multiple; ic < bs; ++ic)
              raw_gemv_t<Impl::unroll, ScalarType, 1>(Aval_ptr + kr + ic * lda,
                                                      lda, xval_ptr + ic, tmp);
            //
            raw_axpy<Impl::unroll, ScalarType>(alpha, tmp, yvec + kr);
          }
          //
          if (even_leftover) {
            for (int kr = multiple; kr < bs; kr += 2) {
              tmp[0] = 0;
              tmp[1] = 0;
              for (int ic = 0; ic < bs; ic += 2)
                raw_gemv_t<2, ScalarType, 2>(Aval_ptr + kr + ic * lda, lda,
                                             xval_ptr + ic, tmp);
              raw_axpy<2, ScalarType>(alpha, tmp, yvec + kr);
            }
          } else {
            for (int kr = multiple; kr < bs; ++kr) {
              tmp[0] = 0;
              for (int ic = 0; ic < bs; ++ic)
                tmp[0] = tmp[0] + Aval_ptr[kr + ic * lda] * xval_ptr[ic];
              yvec[kr] = yvec[kr] + alpha * tmp[0];
            }
          }
        }
      }
    }
  }
  //
  template <typename ScalarType, typename DataType, typename SizeType>
  static inline void gemm(bool useConjugate, const ScalarType &alpha,
                          const ScalarType *Avalues, const SizeType *row_map,
                          DataType numBlockRows, const DataType *entries,
                          int xrhs, const ScalarType *x, int ldx, ScalarType *y,
                          int ldy, const int bs = 0) {
    const int blockSize2 = bs * bs;
    if (useConjugate) {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          auto Aval_ptr        = Avalues + jb * blockSize2;
          for (int ic = 0; ic < xrhs; ++ic) {
            for (DataType ir = 0; ir < bs; ++ir) {
              for (DataType jr = 0; jr < bs; ++jr) {
                const auto avalue = Kokkos::ArithTraits<ScalarType>::conj(
                    Aval_ptr[jr + ir * bs]);
                y[jr + col_block * bs + ic * ldy] +=
                    alpha * avalue * x[ir + iblock * bs + ic * ldx];
              }
            }
          }
        }
      }
    } else {
      for (DataType iblock = 0; iblock < numBlockRows; ++iblock) {
        const SizeType jbeg = row_map[iblock];
        const SizeType jend = row_map[iblock + 1];
        for (SizeType jb = jbeg; jb < jend; ++jb) {
          const auto col_block = entries[jb];
          auto Aval_ptr        = Avalues + jb * blockSize2;
          for (int ic = 0; ic < xrhs; ++ic) {
            for (DataType ir = 0; ir < bs; ++ir) {
              for (DataType jr = 0; jr < bs; ++jr) {
                y[jr + col_block * bs + ic * ldy] +=
                    alpha * Aval_ptr[jr + ir * bs] *
                    x[ir + iblock * bs + ic * ldx];
              }
            }
          }
        }
      }
    }
  }
};

/* ******************* */

template <class AMatrix, class XVector, class YVector>
struct BSR_GEMV_Transpose_Functor {
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

  BSR_GEMV_Transpose_Functor(const value_type alpha_, const AMatrix m_A_,
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
    const auto xvec = &m_x[iBlock * block_size];
    //
    if (conjugate) {
      for (auto jb = jbeg; jb < jend; ++jb) {
        const auto col_block = m_A.graph.entries[jb];
        auto yvec            = &m_y[block_size * col_block];
        const auto Aval_ptr  = &m_A.values[block_size_2 * jb];
        for (int ic = 0; ic < block_size; ++ic) {
          const auto xvalue = xvec[ic];
          for (int kr = 0; kr < block_size; ++kr) {
            const auto val = ATV::conj(Aval_ptr[kr + ic * block_size]);
            Kokkos::atomic_add(&yvec[kr],
                               static_cast<value_type>(alpha * val * xvalue));
          }
        }
      }
    } else {
      for (auto jb = jbeg; jb < jend; ++jb) {
        const auto col_block = m_A.graph.entries[jb];
        auto yvec            = &m_y[block_size * col_block];
        const auto Aval_ptr  = &m_A.values[block_size_2 * jb];
        for (int ic = 0; ic < block_size; ++ic) {
          const auto xvalue = xvec[ic];
          for (int kr = 0; kr < block_size; ++kr) {
            const auto val = Aval_ptr[kr + ic * block_size];
            Kokkos::atomic_add(&yvec[kr],
                               static_cast<value_type>(alpha * val * xvalue));
          }
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
    const XVector &x, const BetaType &beta, YVector &y, bool useFallback,
    bool useConjugate) {
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

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same<execution_space, Kokkos::Serial>::value) {
    const auto blockSize    = A.blockDim();
    const auto numBlockRows = A.numRows();
    const auto &A_graph = A.graph;
    AT alpha_internal   = static_cast<AT>(alpha);
    switch (blockSize) {
      default:
        Bsr_SerialTranspose<0>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 1:
        Bsr_SerialTranspose<1>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 2:
        Bsr_SerialTranspose<2>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 3:
        Bsr_SerialTranspose<3>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 4:
        Bsr_SerialTranspose<4>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 5:
        Bsr_SerialTranspose<5>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 6:
        Bsr_SerialTranspose<6>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 7:
        Bsr_SerialTranspose<7>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 8:
        Bsr_SerialTranspose<8>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
      case 9:
        Bsr_SerialTranspose<9>::gemv(
            useConjugate, alpha_internal, &A.values[0], &A_graph.row_map[0],
            numBlockRows, &A_graph.entries[0], &x[0], &y[0], blockSize);
        break;
    }
    return;
  }
#endif

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
            typename AMatrix_Internal::device_type::execution_space, Kokkos::Schedule<Kokkos::Dynamic>>(
            0, A.numRows()),
        func);
  } else {
    Kokkos::parallel_for(
        "KokkosSparse::bspmv<Transpose,Static>",
        Kokkos::RangePolicy<
            typename AMatrix_Internal::device_type::execution_space, Kokkos::Schedule<Kokkos::Static>>(
            0, A.numRows()),
        func);
  }
}

//
// spMatVec_transpose: version for GPU execution spaces (TeamPolicy used)
//
template <class AMatrix, class AlphaType,
          class XVector, class BetaType, class YVector,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatVec_transpose(
    const KokkosKernels::Experimental::Controls &controls,
    const AlphaType &alpha,
    const AMatrix &A,
    const XVector &x, const BetaType &beta, YVector &y, bool useFallback,
    bool useConjugate) {

  if (A.numRows() <= 0) {
    return;
  }

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
  const ordinal_type num_rhs;

  const ordinal_type *entries;
  const value_type *matrixCoeffs;

  const ordinal_type blocks_per_team;

  bool conjugate = false;

  BSR_GEMM_Functor(const value_type alpha_, const AMatrix m_A_,
                   const XVector m_x_, const YVector m_y_,
                   const int blocks_per_team_, bool conj_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        m_y(m_y_),
        block_size(m_A_.blockDim()),
        block_size_2(block_size * block_size),
        num_rhs(m_x_.extent(1)),
        entries(m_A_.graph.entries.data()),
        matrixCoeffs(&m_A.values[0]),
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
    if (iBlock >= m_A.numRows()) {
      return;
    }
    //
    const auto jbeg = m_A.graph.row_map(iBlock);
    const auto jend = m_A.graph.row_map(iBlock + 1);
    if (conjugate) {
      for (auto jb = jbeg; jb < jend; ++jb) {
        const auto col_block = m_A.graph.entries(jb);
        for (ordinal_type jc = 0; jc < num_rhs; ++jc) {
          for (ordinal_type ir = 0; ir < block_size; ++ir) {
            for (ordinal_type jr = 0; jr < block_size; ++jr) {
              const auto a_value = Kokkos::ArithTraits< value_type >::conj(
                            m_A.values(jr + ir * block_size + jb * block_size_2));
              m_y(ir + iBlock * block_size, jc) += alpha * a_value *
                  m_x(jr + col_block * block_size, jc);
            }
          }
        }
      }
    }
    else {
      for (auto jb = jbeg; jb < jend; ++jb) {
        const auto col_block = m_A.graph.entries(jb);
        for (ordinal_type jc = 0; jc < num_rhs; ++jc) {
          for (ordinal_type ir = 0; ir < block_size; ++ir) {
            for (ordinal_type jr = 0; jr < block_size; ++jr) {
              m_y(ir + iBlock * block_size, jc) += alpha *
                  m_A.values(jr + ir * block_size + jb * block_size_2) *
                  m_x(jr + col_block * block_size, jc);
            }
          }
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
          const auto jbeg           = m_A.graph.row_map(iBlock);
          const auto jend           = m_A.graph.row_map(iBlock + 1);
          const auto row_num_blocks = static_cast<ordinal_type>(jend - jbeg);
          const auto col_idx        = &m_A.graph.entries[jbeg];
          const auto *Aval_ptr      = &matrixCoeffs[jbeg * block_size_2];
          //
          y_value_type lsum = 0;
          //
          for (ordinal_type ic = 0; ic < num_rhs; ++ic) {
            auto yvec = &m_y(iBlock * block_size, ic);
            for (ordinal_type ir = 0; ir < block_size; ++ir) {
              lsum = 0;
              Kokkos::parallel_reduce(
                  Kokkos::ThreadVectorRange(dev, row_num_blocks),
                  [&](const ordinal_type &iEntry, y_value_type &ysum) {
                    const ordinal_type col_block = col_idx[iEntry] * block_size;
                    const auto xval              = &m_x(col_block, ic);
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
          }
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
    const XVector &x, const BetaType &beta, YVector &y, bool useFallback,
    bool useConjugate) {
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

  const auto ldx   = x.stride_1();
  const auto xrhs  = x.extent(1);
  const auto x_ptr = &x(0, 0);

  const auto ldy = y.stride_1();
  auto y_ptr     = &y(0, 0);

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same<typename AMatrix_Internal::device_type::execution_space,
                   Kokkos::Serial>::value) {
    const auto blockSize    = A.blockDim();
    const auto numBlockRows = A.numRows();
    const auto &A_graph = A.graph;
    AT alpha_internal   = static_cast<AT>(alpha);
    switch (blockSize) {
      default:
        // Version for arbitrarily large block size
        Bsr_SerialNoTranspose<0>::gemm(useConjugate, alpha_internal,
                                       &A.values[0], &A_graph.row_map[0],
                                       numBlockRows, &A_graph.entries[0], xrhs,
                                       x_ptr, ldx, y_ptr, ldy, blockSize);
        break;
      case 1:
        Bsr_SerialNoTranspose<1>::gemm(useConjugate, alpha_internal,
                                       &A.values[0], &A_graph.row_map[0],
                                       numBlockRows, &A_graph.entries[0], xrhs,
                                       x_ptr, ldx, y_ptr, ldy, blockSize);
        break;
      case 2:
        Bsr_SerialNoTranspose<2>::gemm(useConjugate, alpha_internal,
                                       &A.values[0], &A_graph.row_map[0],
                                       numBlockRows, &A_graph.entries[0], xrhs,
                                       x_ptr, ldx, y_ptr, ldy, blockSize);
        break;
      case 3:
        Bsr_SerialNoTranspose<3>::gemm(useConjugate, alpha_internal,
                                       &A.values[0], &A_graph.row_map[0],
                                       numBlockRows, &A_graph.entries[0], xrhs,
                                       x_ptr, ldx, y_ptr, ldy, blockSize);
        break;
      case 4:
        Bsr_SerialNoTranspose<4>::gemm(useConjugate, alpha_internal,
                                       &A.values[0], &A_graph.row_map[0],
                                       numBlockRows, &A_graph.entries[0], xrhs,
                                       x_ptr, ldx, y_ptr, ldy, blockSize);
        break;
      case 5:
        Bsr_SerialNoTranspose<5>::gemm(useConjugate, alpha_internal,
                                       &A.values[0], &A_graph.row_map[0],
                                       numBlockRows, &A_graph.entries[0], xrhs,
                                       x_ptr, ldx, y_ptr, ldy, blockSize);
        break;
      case 6:
        Bsr_SerialNoTranspose<6>::gemm(useConjugate, alpha_internal,
                                       &A.values[0], &A_graph.row_map[0],
                                       numBlockRows, &A_graph.entries[0], xrhs,
                                       x_ptr, ldx, y_ptr, ldy, blockSize);
        break;
      case 7:
        Bsr_SerialNoTranspose<7>::gemm(useConjugate, alpha_internal,
                                       &A.values[0], &A_graph.row_map[0],
                                       numBlockRows, &A_graph.entries[0], xrhs,
                                       x_ptr, ldx, y_ptr, ldy, blockSize);
        break;
      case 8:
        Bsr_SerialNoTranspose<8>::gemm(useConjugate, alpha_internal,
                                       &A.values[0], &A_graph.row_map[0],
                                       numBlockRows, &A_graph.entries[0], xrhs,
                                       x_ptr, ldx, y_ptr, ldy, blockSize);
        break;
      case 9:
        Bsr_SerialNoTranspose<9>::gemm(useConjugate, alpha_internal,
                                       &A.values[0], &A_graph.row_map[0],
                                       numBlockRows, &A_graph.entries[0], xrhs,
                                       x_ptr, ldx, y_ptr, ldy, blockSize);
        break;
    }
    return;
  }
#endif

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
    const XVector &x, const BetaType &beta, YVector &y, bool useFallback,
    bool useConjugate) {
  if (A.numRows() <= static_cast<AO>(0)) {
    return;
  }

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
  const ordinal_type num_rhs;

  const ordinal_type *entries;
  const value_type *matrixCoeffs;

  const ordinal_type blocks_per_team;

  bool conjugate = false;

  BSR_GEMM_Transpose_Functor(const value_type alpha_, const AMatrix m_A_,
                             const XVector m_x_, const YVector m_y_,
                             const int blocks_per_team_, bool conj_)
      : alpha(alpha_),
        m_A(m_A_),
        m_x(m_x_),
        m_y(m_y_),
        block_size(m_A_.blockDim()),
        block_size_2(block_size * block_size),
        num_rhs(m_x_.extent(1)),
        entries(m_A_.graph.entries.data()),
        matrixCoeffs(&m_A.values[0]),
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
    if (iBlock >= m_A.numRows()) {
      return;
    }
    //
    const auto jbeg = m_A.graph.row_map(iBlock);
    const auto jend = m_A.graph.row_map(iBlock + 1);
    for (auto jb = jbeg; jb < jend; ++jb) {
      const auto col_block = m_A.graph.entries(jb);
      for (ordinal_type jc = 0; jc < num_rhs; ++jc) {
        for (ordinal_type ir = 0; ir < block_size; ++ir) {
          for (ordinal_type jr = 0; jr < block_size; ++jr) {
            Kokkos::atomic_add(
                &m_y(ir + col_block * block_size, jc),
                static_cast<value_type>(
                    alpha *
                    m_A.values(ir + jr * block_size + jb * block_size_2) *
                    m_x(jr + iBlock * block_size, jc)));
          }
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
          const auto jbeg = m_A.graph.row_map(iBlock);
          const auto jend = m_A.graph.row_map(iBlock + 1);
          for (auto jb = jbeg; jb < jend; ++jb) {
            const auto col_block = m_A.graph.entries(jb);
            for (ordinal_type jc = 0; jc < num_rhs; ++jc) {
              for (ordinal_type ir = 0; ir < block_size; ++ir) {
                for (ordinal_type jr = 0; jr < block_size; ++jr) {
                  Kokkos::single(Kokkos::PerThread(dev), [&]() {
                    m_y(ir + col_block * block_size, jc) +=
                        alpha *
                        m_A.values(ir + jr * block_size + jb * block_size_2) *
                        m_x(jr + iBlock * block_size, jc);
                  });
                }
              }
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
    const XVector &x, const BetaType &beta, YVector &y, bool useFallback,
    bool useConjugate) {
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

  const auto ldx   = x.stride_1();
  const auto xrhs  = x.extent(1);
  const auto x_ptr = &x(0, 0);

  const auto ldy = y.stride_1();
  auto y_ptr     = &y(0, 0);

#if defined(KOKKOS_ENABLE_SERIAL)
  if (std::is_same<execution_space, Kokkos::Serial>::value) {
    const auto blockSize    = A.blockDim();
    const auto numBlockRows = A.numRows();
    const auto &A_graph = A.graph;
    AT alpha_internal   = static_cast<AT>(alpha);
    switch (blockSize) {
      default:
        Bsr_SerialTranspose<0>::gemm(useConjugate, alpha_internal, &A.values[0],
                                     &A_graph.row_map[0], numBlockRows,
                                     &A_graph.entries[0], xrhs, x_ptr, ldx,
                                     y_ptr, ldy, blockSize);
        break;
      case 1:
        Bsr_SerialTranspose<1>::gemm(useConjugate, alpha_internal, &A.values[0],
                                     &A_graph.row_map[0], numBlockRows,
                                     &A_graph.entries[0], xrhs, x_ptr, ldx,
                                     y_ptr, ldy, blockSize);
        break;
      case 2:
        Bsr_SerialTranspose<2>::gemm(useConjugate, alpha_internal, &A.values[0],
                                     &A_graph.row_map[0], numBlockRows,
                                     &A_graph.entries[0], xrhs, x_ptr, ldx,
                                     y_ptr, ldy, blockSize);
        break;
      case 3:
        Bsr_SerialTranspose<3>::gemm(useConjugate, alpha_internal, &A.values[0],
                                     &A_graph.row_map[0], numBlockRows,
                                     &A_graph.entries[0], xrhs, x_ptr, ldx,
                                     y_ptr, ldy, blockSize);
        break;
      case 4:
        Bsr_SerialTranspose<4>::gemm(useConjugate, alpha_internal, &A.values[0],
                                     &A_graph.row_map[0], numBlockRows,
                                     &A_graph.entries[0], xrhs, x_ptr, ldx,
                                     y_ptr, ldy, blockSize);
        break;
      case 5:
        Bsr_SerialTranspose<5>::gemm(useConjugate, alpha_internal, &A.values[0],
                                     &A_graph.row_map[0], A_graph.numRows(),
                                     &A_graph.entries[0], xrhs, x_ptr, ldx,
                                     y_ptr, ldy, blockSize);
        break;
      case 6:
        Bsr_SerialTranspose<6>::gemm(useConjugate, alpha_internal, &A.values[0],
                                     &A_graph.row_map[0], A_graph.numRows(),
                                     &A_graph.entries[0], xrhs, x_ptr, ldx,
                                     y_ptr, ldy, blockSize);
        break;
      case 7:
        Bsr_SerialTranspose<7>::gemm(useConjugate, alpha_internal, &A.values[0],
                                     &A_graph.row_map[0], A_graph.numRows(),
                                     &A_graph.entries[0], xrhs, x_ptr, ldx,
                                     y_ptr, ldy, blockSize);
        break;
      case 8:
        Bsr_SerialTranspose<8>::gemm(useConjugate, alpha_internal, &A.values[0],
                                     &A_graph.row_map[0], A_graph.numRows(),
                                     &A_graph.entries[0], xrhs, x_ptr, ldx,
                                     y_ptr, ldy, blockSize);
        break;
      case 9:
        Bsr_SerialTranspose<9>::gemm(useConjugate, alpha_internal, &A.values[0],
                                     &A_graph.row_map[0], A_graph.numRows(),
                                     &A_graph.entries[0], xrhs, x_ptr, ldx,
                                     y_ptr, ldy, blockSize);
        break;
    }
    return;
  }
#endif

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
template <class AMatrix, class AlphaType,
          class XVector, class BetaType, class YVector,
          typename std::enable_if<KokkosKernels::Impl::kk_is_gpu_exec_space<
              typename YVector::execution_space>()>::type * = nullptr>
void spMatMultiVec_transpose(
    const KokkosKernels::Experimental::Controls &controls,
    const AlphaType &alpha,
    const AMatrix &A,
    const XVector &x, const BetaType &beta, YVector &y, bool useFallback,
    bool useConjugate) {

  if (A.numRows() <= 0) {
    return;
  }

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

/// \brief Driver structure for single vector right hand side
struct EXTENT_1 {
  template <class ABsrMatrix, class AlphaType, class XVector, class BetaType,
            class YVector>
  static void spmv(KokkosKernels::Experimental::Controls controls,
                   const char mode[], const AlphaType &alpha,
                   const ABsrMatrix &A, const XVector &X, const BetaType &beta,
                   const YVector &Y, bool useFallback) {
    if ((X.stride_0() == 1) && (Y.stride_0() == 1)) {
      if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
          (mode[0] == KokkosSparse::Conjugate[0])) {
        bool useConjugate = (mode[0] == KokkosSparse::Conjugate[0]);
        return spMatVec_no_transpose(controls, alpha, A, X, beta, Y,
                                     useFallback, useConjugate);
      } else if ((mode[0] == KokkosSparse::Transpose[0]) ||
                 (mode[0] == KokkosSparse::ConjugateTranspose[0])) {
        bool useConjugate = (mode[0] == KokkosSparse::ConjugateTranspose[0]);
        return spMatVec_transpose(controls, alpha, A, X, beta, Y, useFallback,
                                  useConjugate);
      }
    }
    //
    // Fall-back un-optimized implementation
    //
    const auto numBlockRows = A.numRows();
    const auto blockSize    = A.blockDim();
    const auto blockSize2   = blockSize * blockSize;
    using ordinal_type      = typename ABsrMatrix::non_const_ordinal_type;
    using ScalarType        = typename ABsrMatrix::non_const_value_type;
    for (ordinal_type ii = 0; ii < numBlockRows * blockSize; ++ii)
      Y(ii) = beta * Y(ii);
    //
    if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
        (mode[0] == KokkosSparse::Conjugate[0])) {
      bool useConjugate = (mode[0] == KokkosSparse::Conjugate[0]);
      for (ordinal_type iblock = 0; iblock < numBlockRows; ++iblock) {
        const auto jbeg = A.graph.row_map(iblock);
        const auto jend = A.graph.row_map(iblock + 1);
        for (auto jb = jbeg; jb < jend; ++jb) {
          const auto col_block = A.graph.entries(jb);
          for (ordinal_type ir = 0; ir < blockSize; ++ir) {
            for (ordinal_type jr = 0; jr < blockSize; ++jr) {
              const auto avalue =
                  (useConjugate)
                      ? Kokkos::ArithTraits<ScalarType>::conj(
                            A.values(jr + ir * blockSize + jb * blockSize2))
                      : A.values(jr + ir * blockSize + jb * blockSize2);
              Y(ir + iblock * blockSize) +=
                  alpha * avalue * X(jr + col_block * blockSize);
            }
          }
        }
      }
      return;
    } else if ((mode[0] == KokkosSparse::Transpose[0]) ||
               (mode[0] == KokkosSparse::ConjugateTranspose[0])) {
      bool useConjugate = (mode[0] == KokkosSparse::ConjugateTranspose[0]);
      for (ordinal_type iblock = 0; iblock < numBlockRows; ++iblock) {
        const auto jbeg = A.graph.row_map(iblock);
        const auto jend = A.graph.row_map(iblock + 1);
        for (auto jb = jbeg; jb < jend; ++jb) {
          const auto col_block = A.graph.entries(jb);
          for (ordinal_type ir = 0; ir < blockSize; ++ir) {
            for (ordinal_type jr = 0; jr < blockSize; ++jr) {
              const auto avalue =
                  (useConjugate)
                      ? Kokkos::ArithTraits<ScalarType>::conj(
                            A.values(ir + jr * blockSize + jb * blockSize2))
                      : A.values(ir + jr * blockSize + jb * blockSize2);
              Y(ir + col_block * blockSize) +=
                  alpha * avalue * X(jr + iblock * blockSize);
            }
          }
        }
      }
      return;
    }
  }
};

/// \brief Driver structure for multi-vector right hand side
struct EXTENT_2 {
  template <class ABsrMatrix, class AlphaType, class XVector, class BetaType,
            class YVector>
  static void spmv(KokkosKernels::Experimental::Controls controls,
                   const char mode[], const AlphaType &alpha,
                   const ABsrMatrix &A, const XVector &X, const BetaType &beta,
                   const YVector &Y, bool useFallback) {
    //
    if ((X.stride_0() == 1) && (Y.stride_0() == 1)) {
      if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
          (mode[0] == KokkosSparse::Conjugate[0])) {
        bool useConjugate = (mode[0] == KokkosSparse::Conjugate[0]);
        if (X.extent(1) == 1) {
          const auto x0 = Kokkos::subview(X, Kokkos::ALL(), 0);
          auto y0       = Kokkos::subview(Y, Kokkos::ALL(), 0);
          return spMatVec_no_transpose(controls, alpha, A, x0, beta, y0,
                                       useFallback, useConjugate);
        } else {
          return spMatMultiVec_no_transpose(controls, alpha, A, X, beta, Y,
                                            useFallback, useConjugate);
        }
      } else if ((mode[0] == KokkosSparse::Transpose[0]) ||
                 (mode[0] == KokkosSparse::ConjugateTranspose[0])) {
        bool useConjugate = (mode[0] == KokkosSparse::ConjugateTranspose[0]);
        if (X.extent(1) == 1) {
          const auto x0 = Kokkos::subview(X, Kokkos::ALL(), 0);
          auto y0       = Kokkos::subview(Y, Kokkos::ALL(), 0);
          return spMatVec_transpose(controls, alpha, A, x0, beta, y0,
                                    useFallback, useConjugate);
        } else {
          return spMatMultiVec_transpose(controls, alpha, A, X, beta, Y,
                                         useFallback, useConjugate);
        }
      }
    }
    //
    // Fall-back un-optimized implementation
    //
    const auto numBlockRows = A.numRows();
    const auto blockSize    = A.blockDim();
    const auto blockSize2   = blockSize * blockSize;
    using ordinal_type      = typename ABsrMatrix::non_const_ordinal_type;
    using ScalarType        = typename ABsrMatrix::non_const_value_type;
    for (ordinal_type jc = 0; jc < X.extent(1); ++jc)
      for (ordinal_type ii = 0; ii < numBlockRows * blockSize; ++ii)
        Y(ii, jc) = beta * Y(ii, jc);
    //
    if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
        (mode[0] == KokkosSparse::Conjugate[0])) {
      bool useConjugate = (mode[0] == KokkosSparse::Conjugate[0]);
      for (ordinal_type iblock = 0; iblock < numBlockRows; ++iblock) {
        const auto jbeg = A.graph.row_map(iblock);
        const auto jend = A.graph.row_map(iblock + 1);
        for (auto jb = jbeg; jb < jend; ++jb) {
          const auto col_block = A.graph.entries(jb);
          for (ordinal_type ir = 0; ir < blockSize; ++ir) {
            for (ordinal_type jr = 0; jr < blockSize; ++jr) {
              const auto avalue =
                  (useConjugate)
                      ? Kokkos::ArithTraits<ScalarType>::conj(
                            A.values(jr + ir * blockSize + jb * blockSize2))
                      : A.values(jr + ir * blockSize + jb * blockSize2);
              for (ordinal_type jc = 0; jc < X.extent(1); ++jc)
                Y(ir + iblock * blockSize, jc) +=
                    alpha * avalue * X(jr + col_block * blockSize, jc);
            }
          }
        }
      }
      return;
    } else if ((mode[0] == KokkosSparse::Transpose[0]) ||
               (mode[0] == KokkosSparse::ConjugateTranspose[0])) {
      bool useConjugate = (mode[0] == KokkosSparse::ConjugateTranspose[0]);
      for (ordinal_type iblock = 0; iblock < numBlockRows; ++iblock) {
        const auto jbeg = A.graph.row_map(iblock);
        const auto jend = A.graph.row_map(iblock + 1);
        for (auto jb = jbeg; jb < jend; ++jb) {
          const auto col_block = A.graph.entries(jb);
          for (ordinal_type ir = 0; ir < blockSize; ++ir) {
            for (ordinal_type jr = 0; jr < blockSize; ++jr) {
              const auto avalue =
                  (useConjugate)
                      ? Kokkos::ArithTraits<ScalarType>::conj(
                            A.values(ir + jr * blockSize + jb * blockSize2))
                      : A.values(ir + jr * blockSize + jb * blockSize2);
              for (ordinal_type jc = 0; jc < X.extent(1); ++jc)
                Y(ir + col_block * blockSize, jc) +=
                    alpha * avalue * X(jr + iblock * blockSize, jc);
            }
          }
        }
      }
      return;
    }
  }
};

/// \brief Entry point for Matrix-"Vector" product with BsrMatrix
template <class ABsrMatrix, class AlphaType, class XVector, class BetaType,
          class YVector>
void spmv(KokkosKernels::Experimental::Controls controls, const char mode[],
          const AlphaType &alpha, const ABsrMatrix &A, const XVector &X,
          const BetaType &beta, const YVector &Y) {
  //
  Impl::verifyArguments(mode, A, X, Y);
  //
  typedef KokkosSparse::Experimental::BsrMatrix<
      typename ABsrMatrix::value_type, typename ABsrMatrix::ordinal_type,
      typename ABsrMatrix::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      typename ABsrMatrix::size_type>
      AMatrix_Internal;
  AMatrix_Internal Ainternal = A;
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
}

}  // namespace Bsr

}  // namespace Impl

}  // namespace KokkosSparse

#endif  // KOKKOSKERNELS_KOKKOSSPARSE_SPMV_IMPL_BSR_HPP

