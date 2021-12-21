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
#ifndef KOKKOSSPARSE_IMPL_SPMV_BLOCKCRSMATRIX_SPEC_HPP_
#define KOKKOSSPARSE_IMPL_SPMV_BLOCKCRSMATRIX_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>

#include "KokkosSparse_BlockCrsMatrix.hpp"
#include "KokkosKernels_Controls.hpp"
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosSparse_spmv_blockcrsmatrix_impl.hpp>
#endif

namespace KokkosSparse {
namespace Experimental {
namespace Impl {

// default is no eti available
template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM>
struct spmv_blockcrsmatrix_eti_spec_avail {
  enum : bool { value = false };
};

// default is no eti available
template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM>
struct spmv_mv_blockcrsmatrix_eti_spec_avail {
  enum : bool { value = false };
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace KokkosSparse

#define KOKKOSSPARSE_SPMV_BLOCKCRSMATRIX_ETI_SPEC_AVAIL(                  \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  template <>                                                             \
  struct spmv_blockcrsmatrix_eti_spec_avail<                              \
      const SCALAR_TYPE, const ORDINAL_TYPE,                              \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,         \
      SCALAR_TYPE const *, LAYOUT_TYPE,                                   \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>,     \
      SCALAR_TYPE *, LAYOUT_TYPE,                                         \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > {                         \
    enum : bool { value = true };                                         \
  };

#define KOKKOSSPARSE_SPMV_MV_BLOCKCRSMATRIX_ETI_SPEC_AVAIL(               \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  template <>                                                             \
  struct spmv_mv_blockcrsmatrix_eti_spec_avail<                           \
      const SCALAR_TYPE, const ORDINAL_TYPE,                              \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,         \
      SCALAR_TYPE const *, LAYOUT_TYPE,                                   \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>,     \
      SCALAR_TYPE *, LAYOUT_TYPE,                                         \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > {                         \
    enum : bool { value = true };                                         \
  };

// Include which ETIs are available
#include <KokkosSparse_spmv_blockcrsmatrix_tpl_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosSparse_spmv_blockcrsmatrix_eti_spec_avail.hpp>
#include <generated_specializations_hpp/KokkosSparse_spmv_mv_blockcrsmatrix_eti_spec_avail.hpp>

namespace KokkosSparse {
namespace Experimental {
namespace Impl {

// declaration
template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM,
          bool eti_spec_avail = spmv_blockcrsmatrix_eti_spec_avail<
              AT, AO, AD, AM, AS, XT, XL, XD, XM, YT, YL, YD, YM>::value>
struct SPMV_BLOCKCRSMATRIX {
  typedef BlockCrsMatrix<AT, AO, AD, AM, AS> AMatrix;
  typedef Kokkos::View<XT, XL, XD, XM> XVector;
  typedef Kokkos::View<YT, YL, YD, YM> YVector;
  typedef typename YVector::non_const_value_type YScalar;

  static void spmv_blockcrsmatrix(
      const KokkosKernels::Experimental::Controls &controls, const char mode[],
      const YScalar &alpha, const AMatrix &A, const XVector &x,
      const YScalar &beta, const YVector &y);
};

// declaration
template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM,
          bool eti_spec_avail = spmv_mv_blockcrsmatrix_eti_spec_avail<
              AT, AO, AD, AM, AS, XT, XL, XD, XM, YT, YL, YD, YM>::value>
struct SPMV_MV_BLOCKCRSMATRIX {
  typedef BlockCrsMatrix<AT, AO, AD, AM, AS> AMatrix;
  typedef Kokkos::View<XT, XL, XD, XM> XVector;
  typedef Kokkos::View<YT, YL, YD, YM> YVector;
  typedef typename YVector::non_const_value_type YScalar;

  static void spmv_mv_blockcrsmatrix(
      const KokkosKernels::Experimental::Controls &controls, const char mode[],
      const YScalar &alpha, const AMatrix &A, const XVector &x,
      const YScalar &beta, const YVector &y);
};

// actual implementations to be compiled
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM>
struct SPMV_BLOCKCRSMATRIX<AT, AO, AD, AM, AS, XT, XL, XD, XM, YT, YL, YD, YM,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  typedef BlockCrsMatrix<AT, AO, AD, AM, AS> AMatrix;
  typedef Kokkos::View<XT, XL, XD, XM> XVector;
  typedef Kokkos::View<YT, YL, YD, YM> YVector;
  typedef typename YVector::non_const_value_type YScalar;

  static void spmv_blockcrsmatrix(
      const KokkosKernels::Experimental::Controls &controls, const char mode[],
      const YScalar &alpha, const AMatrix &A, const XVector &X,
      const YScalar &beta, const YVector &Y) {
    //
    if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
        (mode[0] == KokkosSparse::Conjugate[0])) {
      bool useConjugate = (mode[0] == KokkosSparse::Conjugate[0]);
      return BCRS::spMatVec_no_transpose(controls, alpha, A, X, beta, Y,
                                         useConjugate);
    }
    //
    // Naive implementation
    //
    auto h_a_row_map = Kokkos::create_mirror_view(A.graph.row_map);
    Kokkos::deep_copy(h_a_row_map, A.graph.row_map);
    //
    auto h_a_entries = Kokkos::create_mirror_view(A.graph.entries);
    Kokkos::deep_copy(h_a_entries, A.graph.entries);
    //
    auto h_a_values = Kokkos::create_mirror_view(A.values);
    Kokkos::deep_copy(h_a_values, A.values);
    //
    auto h_x = Kokkos::create_mirror_view(X);
    Kokkos::deep_copy(h_x, X);
    //
    auto h_y = Kokkos::create_mirror_view(Y);
    Kokkos::deep_copy(h_y, Y);
    //
    const auto numBlockRows = A.numRows();
    const auto blockSize    = A.blockDim();
    const auto blockSize2   = blockSize * blockSize;
    using ordinal_type      = typename AMatrix::non_const_ordinal_type;
    using size_type         = typename AMatrix::non_const_size_type;
    using value_type        = typename AMatrix::non_const_value_type;
    //
    for (ordinal_type ii = 0; ii < numBlockRows * blockSize; ++ii)
      h_y(ii) = beta * h_y(ii);
    //
    bool useConjugate = (mode[0] == KokkosSparse::ConjugateTranspose[0]);
    for (ordinal_type iblock = 0; iblock < numBlockRows; ++iblock) {
      const auto jbeg = h_a_row_map(iblock);
      const auto jend = h_a_row_map(iblock + 1);
      for (ordinal_type ir = 0; ir < blockSize; ++ir) {
        for (auto jb = jbeg; jb < jend; ++jb) {
          const auto col_block = h_a_entries(jb);
          for (ordinal_type jr = 0; jr < blockSize; ++jr) {
            const size_type index = jbeg * blockSize2 + ir +
                                    (jb - jbeg) * blockSize +
                                    jr * blockSize * (jend - jbeg);
            const auto avalue =
                (useConjugate)
                    ? Kokkos::ArithTraits<value_type>::conj(h_a_values(index))
                    : h_a_values(index);
            h_y(ir + col_block * blockSize) +=
                alpha * avalue * h_x(jr + iblock * blockSize);
          }
        }
      }
    }
    //
    Kokkos::deep_copy(Y, h_y);
    //
  }
};

template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM>
struct SPMV_MV_BLOCKCRSMATRIX<AT, AO, AD, AM, AS, XT, XL, XD, XM, YT, YL, YD,
                              YM, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  typedef BlockCrsMatrix<AT, AO, AD, AM, AS> AMatrix;
  typedef Kokkos::View<XT, XL, XD, XM> XVector;
  typedef Kokkos::View<YT, YL, YD, YM> YVector;
  typedef typename YVector::non_const_value_type YScalar;

  static void spmv_mv_blockcrsmatrix(
      const KokkosKernels::Experimental::Controls &controls, const char mode[],
      const YScalar &alpha, const AMatrix &A, const XVector &X,
      const YScalar &beta, const YVector &Y) {
    //
    auto h_a_row_map = Kokkos::create_mirror_view(A.graph.row_map);
    Kokkos::deep_copy(h_a_row_map, A.graph.row_map);
    //
    auto h_a_entries = Kokkos::create_mirror_view(A.graph.entries);
    Kokkos::deep_copy(h_a_entries, A.graph.entries);
    //
    auto h_a_values = Kokkos::create_mirror_view(A.values);
    Kokkos::deep_copy(h_a_values, A.values);
    //
    auto h_x = Kokkos::create_mirror_view(X);
    Kokkos::deep_copy(h_x, X);
    //
    auto h_y = Kokkos::create_mirror_view(Y);
    Kokkos::deep_copy(h_y, Y);
    //
    const auto numRhs       = X.extent(1);
    const auto numBlockRows = A.numRows();
    const auto blockSize    = A.blockDim();
    const auto blockSize2   = blockSize * blockSize;
    using ordinal_type      = typename AMatrix::non_const_ordinal_type;
    using size_type         = typename AMatrix::non_const_size_type;
    using value_type        = typename AMatrix::non_const_value_type;
    //
    for (ordinal_type jj = 0; jj < numRhs; ++jj) {
      for (ordinal_type ii = 0; ii < numBlockRows * blockSize; ++ii)
        h_y(ii, jj) = beta * h_y(ii, jj);
    }
    //
    if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
        (mode[0] == KokkosSparse::Conjugate[0])) {
      bool useConjugate = (mode[0] == KokkosSparse::Conjugate[0]);
      for (ordinal_type iblock = 0; iblock < numBlockRows; ++iblock) {
        const auto jbeg = h_a_row_map(iblock);
        const auto jend = h_a_row_map(iblock + 1);
        for (auto jb = jbeg; jb < jend; ++jb) {
          const auto col_block = h_a_entries(jb);
          for (ordinal_type jj = 0; jj < numRhs; ++jj) {
            for (ordinal_type jr = 0; jr < blockSize; ++jr) {
              const auto alpha_x = alpha * h_x(jr + col_block * blockSize, jj);
              for (ordinal_type ir = 0; ir < blockSize; ++ir) {
                const size_type index = jbeg * blockSize2 + jr +
                                        (jb - jbeg) * blockSize +
                                        ir * (jend - jbeg) * blockSize;
                const auto avalue = (useConjugate)
                                        ? Kokkos::ArithTraits<value_type>::conj(
                                              h_a_values(index))
                                        : h_a_values(index);
                h_y(ir + iblock * blockSize, jj) += avalue * alpha_x;
              }
            }
          }
        }
      }
    } else if ((mode[0] == KokkosSparse::Transpose[0]) ||
               (mode[0] == KokkosSparse::ConjugateTranspose[0])) {
      bool useConjugate = (mode[0] == KokkosSparse::Conjugate[0]);
      for (ordinal_type iblock = 0; iblock < numBlockRows; ++iblock) {
        const auto jbeg = h_a_row_map(iblock);
        const auto jend = h_a_row_map(iblock + 1);
        for (auto jb = jbeg; jb < jend; ++jb) {
          const auto col_block = h_a_entries(jb);
          for (ordinal_type jj = 0; jj < numRhs; ++jj) {
            for (ordinal_type ir = 0; ir < blockSize; ++ir) {
              for (ordinal_type jr = 0; jr < blockSize; ++jr) {
                const size_type index = jbeg * blockSize2 + ir +
                                        (jb - jbeg) * blockSize +
                                        jr * blockSize * (jend - jbeg);
                const auto avalue = (useConjugate)
                                        ? Kokkos::ArithTraits<value_type>::conj(
                                              h_a_values(index))
                                        : h_a_values(index);
                h_y(ir + col_block * blockSize, jj) +=
                    alpha * avalue * h_x(jr + iblock * blockSize, jj);
              }
            }
          }
        }
      }
    }
    //
    Kokkos::deep_copy(Y, h_y);
    //
  }
};

#endif  // !defined(KOKKOSKERNELS_ETI_ONLY) ||
// KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
}  // namespace Impl
}  // namespace Experimental
}  // namespace KokkosSparse

// declare / instantiate the vector version
// Instantiate with A,x,y are all the requested Scalar type (no instantiation of
// mixed-precision operands)
#define KOKKOSSPARSE_SPMV_BLOCKCRSMATRIX_ETI_SPEC_DECL(                   \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  extern template struct SPMV_BLOCKCRSMATRIX<                             \
      const SCALAR_TYPE, const ORDINAL_TYPE,                              \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,         \
      SCALAR_TYPE const *, LAYOUT_TYPE,                                   \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>,     \
      SCALAR_TYPE *, LAYOUT_TYPE,                                         \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true>;

#define KOKKOSSPARSE_SPMV_BLOCKCRSMATRIX_ETI_SPEC_INST(                   \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  template struct SPMV_BLOCKCRSMATRIX<                                    \
      const SCALAR_TYPE, const ORDINAL_TYPE,                              \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,         \
      SCALAR_TYPE const *, LAYOUT_TYPE,                                   \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>,     \
      SCALAR_TYPE *, LAYOUT_TYPE,                                         \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true>;

// declare / instantiate the 2D MV version
// Instantiate with A,x,y are all the requested Scalar type (no instantiation of
// mixed-precision operands)
#define KOKKOSSPARSE_SPMV_MV_BLOCKCRSMATRIX_ETI_SPEC_DECL(                \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  extern template struct SPMV_MV_BLOCKCRSMATRIX<                          \
      const SCALAR_TYPE, const ORDINAL_TYPE,                              \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,         \
      SCALAR_TYPE const **, LAYOUT_TYPE,                                  \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>,     \
      SCALAR_TYPE **, LAYOUT_TYPE,                                        \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true>;

#define KOKKOSSPARSE_SPMV_MV_BLOCKCRSMATRIX_ETI_SPEC_INST(                \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  template struct SPMV_MV_BLOCKCRSMATRIX<                                 \
      const SCALAR_TYPE, const ORDINAL_TYPE,                              \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,         \
      SCALAR_TYPE const **, LAYOUT_TYPE,                                  \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>,     \
      SCALAR_TYPE **, LAYOUT_TYPE,                                        \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true>;

#include <KokkosSparse_spmv_blockcrsmatrix_tpl_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosSparse_spmv_blockcrsmatrix_eti_spec_decl.hpp>
#include <generated_specializations_hpp/KokkosSparse_spmv_mv_blockcrsmatrix_eti_spec_decl.hpp>

#endif  // KOKKOSSPARSE_IMPL_SPMV_BLOCKCRSMATRIX_SPEC_HPP_
