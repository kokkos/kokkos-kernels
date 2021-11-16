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

#ifndef KOKKOSKERNELS_KOKKOSSPARSE_SPMV_BSR_TPL_SPEC_DECL_HPP
#define KOKKOSKERNELS_KOKKOSSPARSE_SPMV_BSR_TPL_SPEC_DECL_HPP

#include "KokkosKernels_Controls.hpp"
#include "KokkosSparse_spmv_tpl_spec_decl.hpp"

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
#include <mkl.h>

namespace KokkosSparse {
namespace Impl {

#if (__INTEL_MKL__ > 2017)
// MKL 2018 and above: use new interface: sparse_matrix_t and mkl_sparse_?_mv()

inline matrix_descr getDescription() {
  matrix_descr A_descr;
  A_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  A_descr.mode = SPARSE_FILL_MODE_FULL;
  A_descr.diag = SPARSE_DIAG_NON_UNIT;
  return A_descr;
}

inline void spmv_block_impl_mkl(sparse_operation_t op, float alpha, float beta,
                                int m, int n, int b, const int* Arowptrs,
                                const int* Aentries, const float* Avalues,
                                const float* x, float* y) {
  sparse_matrix_t A_mkl;
  mkl_safe_call(mkl_sparse_s_create_bsr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, m, n, b,
      const_cast<int*>(Arowptrs), const_cast<int*>(Arowptrs + 1),
      const_cast<int*>(Aentries), const_cast<float*>(Avalues)));

  matrix_descr A_descr = getDescription();
  mkl_safe_call(mkl_sparse_s_mv(op, alpha, A_mkl, A_descr, x, beta, y));
}

inline void spmv_block_impl_mkl(sparse_operation_t op, double alpha,
                                double beta, int m, int n, int b,
                                const int* Arowptrs, const int* Aentries,
                                const double* Avalues, const double* x,
                                double* y) {
  sparse_matrix_t A_mkl;
  mkl_safe_call(mkl_sparse_d_create_bsr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, m, n, b,
      const_cast<int*>(Arowptrs), const_cast<int*>(Arowptrs + 1),
      const_cast<int*>(Aentries), const_cast<double*>(Avalues)));

  matrix_descr A_descr = getDescription();
  mkl_safe_call(mkl_sparse_d_mv(op, alpha, A_mkl, A_descr, x, beta, y));
}

inline void spmv_block_impl_mkl(sparse_operation_t op,
                                Kokkos::complex<float> alpha,
                                Kokkos::complex<float> beta, int m, int n,
                                int b, const int* Arowptrs, const int* Aentries,
                                const Kokkos::complex<float>* Avalues,
                                const Kokkos::complex<float>* x,
                                Kokkos::complex<float>* y) {
  sparse_matrix_t A_mkl;
  mkl_safe_call(mkl_sparse_c_create_bsr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, m, n, b,
      const_cast<int*>(Arowptrs), const_cast<int*>(Arowptrs + 1),
      const_cast<int*>(Aentries), (MKL_Complex8*)Avalues));

  MKL_Complex8& alpha_mkl = reinterpret_cast<MKL_Complex8&>(alpha);
  MKL_Complex8& beta_mkl  = reinterpret_cast<MKL_Complex8&>(beta);
  matrix_descr A_descr    = getDescription();
  mkl_safe_call(mkl_sparse_c_mv(op, alpha_mkl, A_mkl, A_descr,
                                reinterpret_cast<const MKL_Complex8*>(x),
                                beta_mkl, reinterpret_cast<MKL_Complex8*>(y)));
}

inline void spmv_block_impl_mkl(sparse_operation_t op,
                                Kokkos::complex<double> alpha,
                                Kokkos::complex<double> beta, int m, int n,
                                int b, const int* Arowptrs, const int* Aentries,
                                const Kokkos::complex<double>* Avalues,
                                const Kokkos::complex<double>* x,
                                Kokkos::complex<double>* y) {
  sparse_matrix_t A_mkl;
  mkl_safe_call(mkl_sparse_z_create_bsr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, SPARSE_LAYOUT_ROW_MAJOR, m, n, b,
      const_cast<int*>(Arowptrs), const_cast<int*>(Arowptrs + 1),
      const_cast<int*>(Aentries), (MKL_Complex16*)Avalues));

  matrix_descr A_descr     = getDescription();
  MKL_Complex16& alpha_mkl = reinterpret_cast<MKL_Complex16&>(alpha);
  MKL_Complex16& beta_mkl  = reinterpret_cast<MKL_Complex16&>(beta);
  mkl_safe_call(mkl_sparse_z_mv(op, alpha_mkl, A_mkl, A_descr,
                                reinterpret_cast<const MKL_Complex16*>(x),
                                beta_mkl, reinterpret_cast<MKL_Complex16*>(y)));
}

#endif

#if (__INTEL_MKL__ == 2017)

inline void spmv_block_impl_mkl(char mode, float alpha, float beta, int m,
                                int n, int b, const int* Arowptrs,
                                const int* Aentries, const float* Avalues,
                                const float* x, float* y) {
  mkl_sbsrmv(&mode, &m, &n, &b, &alpha, "G**C", Avalues, Aentries, Arowptrs,
             Arowptrs + 1, x, &beta, y);
}

inline void spmv_block_impl_mkl(char mode, double alpha, double beta, int m,
                                int n, int b, const int* Arowptrs,
                                const int* Aentries, const double* Avalues,
                                const double* x, double* y) {
  mkl_dbsrmv(&mode, &m, &n, &b, &alpha, "G**C", Avalues, Aentries, Arowptrs,
             Arowptrs + 1, x, &beta, y);
}

inline void spmv_block_impl_mkl(char mode, Kokkos::complex<float> alpha,
                                Kokkos::complex<float> beta, int m, int n,
                                int b, const int* Arowptrs, const int* Aentries,
                                const Kokkos::complex<float>* Avalues,
                                const Kokkos::complex<float>* x,
                                Kokkos::complex<float>* y) {
  const MKL_Complex8* alpha_mkl = reinterpret_cast<const MKL_Complex8*>(&alpha);
  const MKL_Complex8* beta_mkl  = reinterpret_cast<const MKL_Complex8*>(&beta);
  const MKL_Complex8* Avalues_mkl =
      reinterpret_cast<const MKL_Complex8*>(Avalues);
  const MKL_Complex8* x_mkl = reinterpret_cast<const MKL_Complex8*>(x);
  MKL_Complex8* y_mkl       = reinterpret_cast<MKL_Complex8*>(y);
  mkl_cbsrmv(&mode, &m, &n, &b, alpha_mkl, "G**C", Avalues_mkl, Aentries,
             Arowptrs, Arowptrs + 1, x_mkl, beta_mkl, y_mkl);
}

inline void spmv_block_impl_mkl(char mode, Kokkos::complex<double> alpha,
                                Kokkos::complex<double> beta, int m, int n,
                                int b, const int* Arowptrs, const int* Aentries,
                                const Kokkos::complex<double>* Avalues,
                                const Kokkos::complex<double>* x,
                                Kokkos::complex<double>* y) {
  const MKL_Complex16* alpha_mkl =
      reinterpret_cast<const MKL_Complex16*>(&alpha);
  const MKL_Complex16* beta_mkl = reinterpret_cast<const MKL_Complex16*>(&beta);
  const MKL_Complex16* Avalues_mkl =
      reinterpret_cast<const MKL_Complex16*>(Avalues);
  const MKL_Complex16* x_mkl = reinterpret_cast<const MKL_Complex16*>(x);
  MKL_Complex16* y_mkl       = reinterpret_cast<MKL_Complex16*>(y);
  mkl_zbsrmv(&mode, &m, &n, &b, alpha_mkl, "G**C", Avalues_mkl, Aentries,
             Arowptrs, Arowptrs + 1, x_mkl, beta_mkl, y_mkl);
}
#endif

/// \brief Driver for call to MKL routines
///
template <typename ScalarType, class AMatrix, class XVector, class YVector>
    void spmv_block_mkl(
        const KokkosKernels::Experimental::Controls &controls, const char mode[],
        const ScalarType& alpha, const AMatrix &A,
        const XVector& x, const ScalarType& beta, const YVector& y) {
  std::string label = "KokkosSparse::spmv[BLOCK_TPL_MKL," +
      Kokkos::ArithTraits<ScalarType>::name() + "]";
  Kokkos::Profiling::pushRegion(label);
  spmv_block_impl_mkl(mode_kk_to_mkl(mode[0]), alpha, beta, A.numRows(),
                      A.numCols(), A.blockDim(), A.graph.row_map.data(),
                      A.graph.entries.data(), A.values.data(), x.data(),
                      y.data());
  Kokkos::Profiling::popRegion();
}

}  // namespace Impl

}  // namespace KokkosSparse

#endif

// cuSPARSE
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include "cusparse.h"
#include "KokkosKernels_SparseUtils_cusparse.hpp"

//
// From  https://docs.nvidia.com/cuda/cusparse/index.html#bsrmv
// Several comments on bsrmv():
// - Only blockDim > 1 is supported
// - Only CUSPARSE_OPERATION_NON_TRANSPOSE is supported
// - Only CUSPARSE_MATRIX_TYPE_GENERAL is supported.
//
namespace KokkosSparse {
namespace Impl {

template <class AMatrix, class XVector, class YVector>
    void spmv_block_impl_cusparse(
        const KokkosKernels::Experimental::Controls& controls, const char mode[],
        typename YVector::non_const_value_type const& alpha, const AMatrix& A,
        const XVector& x, typename YVector::non_const_value_type const& beta,
        const YVector& y) {
      using offset_type = typename AMatrix::non_const_size_type;
      using entry_type  = typename AMatrix::non_const_ordinal_type;
      using value_type  = typename AMatrix::non_const_value_type;

      /* initialize cusparse library */
      cusparseHandle_t cusparseHandle = controls.getCusparseHandle();

      /* Set the operation mode */
      cusparseOperation_t myCusparseOperation;
      switch (toupper(mode[0])) {
        case 'N': myCusparseOperation = CUSPARSE_OPERATION_NON_TRANSPOSE; break;
        default:
        {
          std::cerr << "Mode " << mode << " invalid for cuSPARSE SpMV.\n";
          throw std::invalid_argument("Invalid mode");
        }
          break;
      }

#if (9000 <= CUDA_VERSION)

/* create and set the matrix descriptor */
cusparseMatDescr_t descrA = 0;
KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&descrA));
KOKKOS_CUSPARSE_SAFE_CALL(
    cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL));
KOKKOS_CUSPARSE_SAFE_CALL(
    cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO));
cusparseDirection_t dirA = CUSPARSE_DIRECTION_ROW;

/* perform the actual SpMV operation */
if (std::is_same<int, offset_type>::value) {
  if (std::is_same<value_type, float>::value) {
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseSbsrmv(
        cusparseHandle, dirA, myCusparseOperation, A.numRows(), A.numCols(),
        A.nnz(), reinterpret_cast<float const*>(&alpha), descrA,
        reinterpret_cast<float const*>(A.values.data()),
        A.graph.row_map.data(), A.graph.entries.data(), A.blockDim(),
        reinterpret_cast<float const*>(x.data()),
        reinterpret_cast<float const*>(&beta),
        reinterpret_cast<float*>(y.data())));
  } else if (std::is_same<value_type, double>::value) {
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseDbsrmv(
        cusparseHandle, dirA, myCusparseOperation, A.numRows(), A.numCols(),
        A.nnz(), reinterpret_cast<double const*>(&alpha), descrA,
        reinterpret_cast<double const*>(A.values.data()),
        A.graph.row_map.data(), A.graph.entries.data(), A.blockDim(),
        reinterpret_cast<double const*>(x.data()),
        reinterpret_cast<double const*>(&beta),
        reinterpret_cast<double*>(y.data())));
  } else if (std::is_same<value_type, Kokkos::complex<float> >::value) {
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseCbsrmv(
        cusparseHandle, dirA, myCusparseOperation, A.numRows(), A.numCols(),
        A.nnz(), reinterpret_cast<cuComplex const*>(&alpha), descrA,
        reinterpret_cast<cuComplex const*>(A.values.data()),
        A.graph.row_map.data(), A.graph.entries.data(), A.blockDim(),
        reinterpret_cast<cuComplex const*>(x.data()),
        reinterpret_cast<cuComplex const*>(&beta),
        reinterpret_cast<cuComplex*>(y.data())));
  } else if (std::is_same<value_type, Kokkos::complex<double> >::value) {
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseZbsrmv(
        cusparseHandle, dirA, myCusparseOperation, A.numRows(), A.numCols(),
        A.nnz(), reinterpret_cast<cuDoubleComplex const*>(&alpha), descrA,
        reinterpret_cast<cuDoubleComplex const*>(A.values.data()),
        A.graph.row_map.data(), A.graph.entries.data(), A.blockDim(),
        reinterpret_cast<cuDoubleComplex const*>(x.data()),
        reinterpret_cast<cuDoubleComplex const*>(&beta),
        reinterpret_cast<cuDoubleComplex*>(y.data())));
  } else {
    throw std::logic_error(
        "Trying to call cusparse SpMV with a scalar type not float/double, "
        "nor complex of either!");
  }
} else {
  throw std::logic_error(
      "With cuSPARSE pre-10.0, offset type must be int. Something wrong with "
      "TPL avail logic.");
}

KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyMatDescr(descrA));
#endif  // CUDA_VERSION
    }

    /// \brief Driver for call to cuSparse routines
    ///
    template < class AMatrix, class XVector, class YVector,
        typename AlphaType, typename BetaType >
        void spmv_block_cusparse(
            const KokkosKernels::Experimental::Controls &controls, const char mode[],
            const AlphaType& alpha, const AMatrix &A,
            const XVector& x, const BetaType& beta, const YVector& y) {
          using ScalarType = typename YVector::non_const_value_type;
          std::string label = "KokkosSparse::spmv[BLOCK_TPL_CUSPARSE," +
              Kokkos::ArithTraits<ScalarType>::name() + "]";
          Kokkos::Profiling::pushRegion(label);
          spmv_block_impl_cusparse(controls, mode, alpha, A, x, beta, y);
          Kokkos::Profiling::popRegion();
        }

}  // namespace Impl
}  // namespace KokkosSparse

#endif

#endif  // KOKKOSKERNELS_KOKKOSSPARSE_SPMV_BSR_TPL_SPEC_DECL_HPP
