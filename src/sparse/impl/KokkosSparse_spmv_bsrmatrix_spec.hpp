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
#ifndef KOKKOSSPARSE_IMPL_SPMV_BSRMATRIX_SPEC_HPP_
#define KOKKOSSPARSE_IMPL_SPMV_BSRMATRIX_SPEC_HPP_

#include <KokkosKernels_config.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>

#include "KokkosSparse_BsrMatrix.hpp"
#include "KokkosKernels_Controls.hpp"
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
#include <KokkosSparse_spmv_bsrmatrix_impl.hpp>
#endif

namespace KokkosSparse {
namespace Experimental {
namespace Impl {

// default is no eti available
template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM>
struct spmv_bsrmatrix_eti_spec_avail {
  enum : bool { value = false };
};

template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM>
struct spmv_mv_bsrmatrix_eti_spec_avail {
  enum : bool { value = false };
};

}  // namespace Impl
}  // namespace Experimental
}  // namespace KokkosSparse

#define KOKKOSSPARSE_SPMV_BSRMATRIX_ETI_SPEC_AVAIL(                       \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  template <>                                                             \
  struct spmv_bsrmatrix_eti_spec_avail<                                   \
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

#define KOKKOSSPARSE_SPMV_MV_BSRMATRIX_ETI_SPEC_AVAIL(                    \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  template <>                                                             \
  struct spmv_mv_bsrmatrix_eti_spec_avail<                                \
      const SCALAR_TYPE, const ORDINAL_TYPE,                              \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,         \
      SCALAR_TYPE const **, LAYOUT_TYPE,                                  \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>,     \
      SCALAR_TYPE **, LAYOUT_TYPE,                                        \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged> > {                         \
    enum : bool { value = true };                                         \
  };

// Include which ETIs are available
#include <generated_specializations_hpp/KokkosSparse_spmv_bsrmatrix_eti_spec_avail.hpp>

namespace KokkosSparse {
namespace Experimental {
namespace Impl {

// declaration
template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM,
          bool eti_spec_avail = spmv_bsrmatrix_eti_spec_avail<
              AT, AO, AD, AM, AS, XT, XL, XD, XM, YT, YL, YD, YM>::value>
struct SPMV_BSRMATRIX {
  typedef BsrMatrix<AT, AO, AD, AM, AS> AMatrix;
  typedef Kokkos::View<XT, XL, XD, XM> XVector;
  typedef Kokkos::View<YT, YL, YD, YM> YVector;
  typedef typename YVector::non_const_value_type YScalar;

  static void spmv_bsrmatrix(
      const KokkosKernels::Experimental::Controls &controls, const char mode[],
      const YScalar &alpha, const AMatrix &A, const XVector &x,
      const YScalar &beta, const YVector &y);
};

// declaration
template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM,
          bool eti_spec_avail = spmv_mv_bsrmatrix_eti_spec_avail<
              AT, AO, AD, AM, AS, XT, XL, XD, XM, YT, YL, YD, YM>::value>
struct SPMV_MV_BSRMATRIX {
  typedef BsrMatrix<AT, AO, AD, AM, AS> AMatrix;
  typedef Kokkos::View<XT, XL, XD, XM> XVector;
  typedef Kokkos::View<YT, YL, YD, YM> YVector;
  typedef typename YVector::non_const_value_type YScalar;

  static void spmv_mv_bsrmatrix(
      const KokkosKernels::Experimental::Controls &controls, const char mode[],
      const YScalar &alpha, const AMatrix &A, const XVector &x,
      const YScalar &beta, const YVector &y);
};

// actual implementations to be compiled
#if !defined(KOKKOSKERNELS_ETI_ONLY) || KOKKOSKERNELS_IMPL_COMPILE_LIBRARY

template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM>
struct SPMV_BSRMATRIX<AT, AO, AD, AM, AS, XT, XL, XD, XM, YT, YL, YD, YM,
                      KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  typedef BsrMatrix<AT, AO, AD, AM, AS> AMatrix;
  typedef Kokkos::View<XT, XL, XD, XM> XVector;
  typedef Kokkos::View<YT, YL, YD, YM> YVector;
  typedef typename YVector::non_const_value_type YScalar;

  static void spmv_bsrmatrix(
      const KokkosKernels::Experimental::Controls &controls, const char mode[],
      const YScalar &alpha, const AMatrix &A, const XVector &X,
      const YScalar &beta, const YVector &Y) {
    //
    // Whether to call KokkosKernel's native implementation, even if a TPL impl
    // is available
    //
    bool useFallback = controls.isParameter("algorithm") &&
                       controls.getParameter("algorithm") == "native";
    //
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
    // cuSPARSE does not support the conjugate mode (C), and cuSPARSE 9 only
    // supports the normal (N) mode.
    if (std::is_same<typename AMatrix::device_type::memory_space,
                     Kokkos::CudaSpace>::value ||
        std::is_same<typename AMatrix::device_type::memory_space,
                     Kokkos::CudaUVMSpace>::value) {
#if (9000 <= CUDA_VERSION)
      useFallback = useFallback || (mode[0] != NoTranspose[0]);
#endif
#if defined(CUSPARSE_VERSION) && (10300 <= CUSPARSE_VERSION)
      useFallback = useFallback || (mode[0] == Conjugate[0]);
#endif
    }
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
    if (std::is_same<typename AMatrix::device_type::memory_space,
                     Kokkos::HostSpace>::value) {
      useFallback = useFallback || (mode[0] == Conjugate[0]);
    }
#endif

    if ((X.stride_0() == 1) && (Y.stride_0() == 1)) {
      if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
          (mode[0] == KokkosSparse::Conjugate[0])) {
        bool useConjugate = (mode[0] == KokkosSparse::Conjugate[0]);
        return Bsr::spMatVec_no_transpose(controls, alpha, A, X, beta, Y,
                                          useFallback, useConjugate);
      } else if ((mode[0] == KokkosSparse::Transpose[0]) ||
                 (mode[0] == KokkosSparse::ConjugateTranspose[0])) {
        bool useConjugate = (mode[0] == KokkosSparse::ConjugateTranspose[0]);
        return Bsr::spMatVec_transpose(controls, alpha, A, X, beta, Y,
                                       useFallback, useConjugate);
      }
    }

    //
    // Fall-back un-optimized implementation
    // This implementation is independent of the layout for the vectors X and Y
    //
    const auto numBlockRows = A.numRows();
    const auto blockSize    = A.blockDim();
    const auto blockSize2   = blockSize * blockSize;
    using ordinal_type      = typename AMatrix::non_const_ordinal_type;
    using ScalarType        = typename AMatrix::non_const_value_type;
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

template <class AT, class AO, class AD, class AM, class AS, class XT, class XL,
          class XD, class XM, class YT, class YL, class YD, class YM>
struct SPMV_MV_BSRMATRIX<AT, AO, AD, AM, AS, XT, XL, XD, XM, YT, YL, YD, YM,
                         KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {
  typedef BsrMatrix<AT, AO, AD, AM, AS> AMatrix;
  typedef Kokkos::View<XT, XL, XD, XM> XVector;
  typedef Kokkos::View<YT, YL, YD, YM> YVector;
  typedef typename YVector::non_const_value_type YScalar;

  static void spmv_mv_bsrmatrix(
      const KokkosKernels::Experimental::Controls &controls, const char mode[],
      const YScalar &alpha, const AMatrix &A, const XVector &X,
      const YScalar &beta, const YVector &Y) {
    // user explicitly requests a particular precision
    bool requestMixed  = false;
    bool requestDouble = false;
    if (controls.isParameter("tc_precision")) {
      if (controls.getParameter("tc_precision") == "mixed") {
        requestMixed = true;
      } else if (controls.getParameter("tc_precision") == "double") {
        requestDouble = true;
      }
    }

    /*
  #if defined(KOKKOS_ENABLE_CUDA) && \
    (defined(KOKKOS_ARCH_VOLTA) || defined(KOKKOS_ARCH_AMPERE))
    if ((mode[0] == NoTranspose[0]) &&
  (KokkosKernels::Impl::kk_is_gpu_mem_space< typename AMatrix::memory_space>())
  && (KokkosKernels::Impl::kk_is_gpu_mem_space< typename
  XVector::memory_space>()) && (KokkosKernels::Impl::kk_is_gpu_mem_space<
    typename yVector::memory_space>()))
  #endif
  */

#if defined(KOKKOS_ARCH_AMPERE)
    typedef typename XVector::non_const_value_type XScalar;
    typedef typename AMatrix::non_const_value_type AScalar;
    typedef Kokkos::Experimental::half_t Half;

    /* Ampere has double += double * double and float += half * half

    use whichever is requested.
    If none requested, used mixed precision if the inputs are mixed, otherwise
    use double
    */

    // input precision matches a tensor core fragment type
    constexpr bool operandsHalfHalfFloat = std::is_same<AScalar, Half>::value &&
                                           std::is_same<XScalar, Half>::value &&
                                           std::is_same<YScalar, float>::value;

    if (requestMixed) {
      BsrMatrixSpMVTensorCoreDispatcher<AMatrix, half, XVector, half, YVector,
                                        float, 16, 16, 16>::dispatch(alpha, A,
                                                                     X, beta,
                                                                     Y);
    } else if (requestDouble) {
      BsrMatrixSpMVTensorCoreDispatcher<AMatrix, double, XVector, double,
                                        YVector, double, 8, 8,
                                        4>::dispatch(alpha, A, X, beta, Y);
    } else if (operandsHalfHalfFloat) {
      BsrMatrixSpMVTensorCoreDispatcher<AMatrix, half, XVector, half, YVector,
                                        float, 16, 16, 16>::dispatch(alpha, A,
                                                                     X, beta,
                                                                     Y);
    } else {
      BsrMatrixSpMVTensorCoreDispatcher<AMatrix, double, XVector, double,
                                        YVector, double, 8, 8,
                                        4>::dispatch(alpha, A, x, beta, y);
    }

#elif defined(KOKKOS_ARCH_VOLTA)
    /* Volta has float += half * half
       use it for all matrices
    */
    if (requestDouble) {
      Kokkos::Impl::throw_runtime_exception(
          "KokkosSparse::spmv[algorithm=experimental_bsr_tc] "
          "tc_precision=double unsupported KOKKOS_ARCH_VOLTA");
    }
    BsrMatrixSpMVTensorCoreDispatcher<AMatrix, half, XVector, half, YVector,
                                      float, 16, 16, 16>::dispatch(alpha, A, X,
                                                                   beta, Y);
    (void)requestMixed;  // unused
#endif  // KOKKOS_ARCH

    //
    // Whether to call KokkosKernel's native implementation, even if a TPL impl
    // is available
    //
    bool useFallback = controls.isParameter("algorithm") &&
                       controls.getParameter("algorithm") == "native";
    //
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
    // cuSPARSE does not support the conjugate mode (C), and cuSPARSE 9 only
    // supports the normal (N) mode.
    if (std::is_same<typename AMatrix::device_type::memory_space,
                     Kokkos::CudaSpace>::value ||
        std::is_same<typename AMatrix::device_type::memory_space,
                     Kokkos::CudaUVMSpace>::value) {
#if (9000 <= CUDA_VERSION)
      useFallback = useFallback || (mode[0] != NoTranspose[0]);
#endif
#if defined(CUSPARSE_VERSION) && (10300 <= CUSPARSE_VERSION)
      useFallback = useFallback || (mode[0] == Conjugate[0]);
#endif
    }
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
    if (std::is_same<typename AMatrix::device_type::memory_space,
                     Kokkos::HostSpace>::value) {
      useFallback = useFallback || (mode[0] == Conjugate[0]);
    }
#endif

    if ((X.stride_0() == 1) && (Y.stride_0() == 1)) {
      if ((mode[0] == KokkosSparse::NoTranspose[0]) ||
          (mode[0] == KokkosSparse::Conjugate[0])) {
        bool useConjugate = (mode[0] == KokkosSparse::Conjugate[0]);
        if (X.extent(1) == 1) {
          const auto x0 = Kokkos::subview(X, Kokkos::ALL(), 0);
          auto y0       = Kokkos::subview(Y, Kokkos::ALL(), 0);
          return Bsr::spMatVec_no_transpose(controls, alpha, A, x0, beta, y0,
                                            useFallback, useConjugate);
        } else {
          return Bsr::spMatMultiVec_no_transpose(controls, alpha, A, X, beta, Y,
                                                 useFallback, useConjugate);
        }
      } else if ((mode[0] == KokkosSparse::Transpose[0]) ||
                 (mode[0] == KokkosSparse::ConjugateTranspose[0])) {
        bool useConjugate = (mode[0] == KokkosSparse::ConjugateTranspose[0]);
        if (X.extent(1) == 1) {
          const auto x0 = Kokkos::subview(X, Kokkos::ALL(), 0);
          auto y0       = Kokkos::subview(Y, Kokkos::ALL(), 0);
          return Bsr::spMatVec_transpose(controls, alpha, A, x0, beta, y0,
                                         useFallback, useConjugate);
        } else {
          return Bsr::spMatMultiVec_transpose(controls, alpha, A, X, beta, Y,
                                              useFallback, useConjugate);
        }
      }
    }

    //
    // Fall-back un-optimized implementation
    // This implementation is independent of the layout for the vectors X and Y
    //
    const auto numBlockRows = A.numRows();
    const auto blockSize    = A.blockDim();
    const auto blockSize2   = blockSize * blockSize;
    using ordinal_type      = typename AMatrix::non_const_ordinal_type;
    using ScalarType        = typename AMatrix::non_const_value_type;
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

#endif  // !defined(KOKKOSKERNELS_ETI_ONLY) ||
        // KOKKOSKERNELS_IMPL_COMPILE_LIBRARY
}  // namespace Impl
}  // namespace Experimental
}  // namespace KokkosSparse

// declare / instantiate the vector version
// Instantiate with A,x,y are all the requested Scalar type (no instantiation of
// mixed-precision operands)
#define KOKKOSSPARSE_SPMV_BSRMATRIX_ETI_SPEC_DECL(                        \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  extern template struct SPMV_BSRMATRIX<                                  \
      const SCALAR_TYPE, const ORDINAL_TYPE,                              \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,         \
      SCALAR_TYPE const *, LAYOUT_TYPE,                                   \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>,     \
      SCALAR_TYPE *, LAYOUT_TYPE,                                         \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true>;

#define KOKKOSSPARSE_SPMV_BSRMATRIX_ETI_SPEC_INST(                        \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  template struct SPMV_BSRMATRIX<                                         \
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
#define KOKKOSSPARSE_SPMV_MV_BSRMATRIX_ETI_SPEC_DECL(                     \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  extern template struct SPMV_MV_BSRMATRIX<                               \
      const SCALAR_TYPE, const ORDINAL_TYPE,                              \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,         \
      SCALAR_TYPE const **, LAYOUT_TYPE,                                  \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>,     \
      SCALAR_TYPE **, LAYOUT_TYPE,                                        \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true>;

#define KOKKOSSPARSE_SPMV_MV_BSRMATRIX_ETI_SPEC_INST(                     \
    SCALAR_TYPE, ORDINAL_TYPE, OFFSET_TYPE, LAYOUT_TYPE, EXEC_SPACE_TYPE, \
    MEM_SPACE_TYPE)                                                       \
  template struct SPMV_MV_BSRMATRIX<                                      \
      const SCALAR_TYPE, const ORDINAL_TYPE,                              \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, const OFFSET_TYPE,         \
      SCALAR_TYPE const **, LAYOUT_TYPE,                                  \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>,     \
      SCALAR_TYPE **, LAYOUT_TYPE,                                        \
      Kokkos::Device<EXEC_SPACE_TYPE, MEM_SPACE_TYPE>,                    \
      Kokkos::MemoryTraits<Kokkos::Unmanaged>, true>;

#include <generated_specializations_hpp/KokkosSparse_spmv_bsrmatrix_eti_spec_decl.hpp>

#endif  // KOKKOSSPARSE_IMPL_SPMV_BSRMATRIX_SPEC_HPP_
