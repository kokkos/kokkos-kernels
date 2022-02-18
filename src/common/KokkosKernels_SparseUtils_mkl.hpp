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

#ifndef _KOKKOSKERNELS_SPARSEUTILS_MKL_HPP
#define _KOKKOSKERNELS_SPARSEUTILS_MKL_HPP

#include "KokkosKernels_config.h"

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL

#include <mkl.h>

namespace KokkosSparse {
namespace Impl {

inline void mkl_internal_safe_call(sparse_status_t mkl_status, const char *name,
                                   const char *file = nullptr,
                                   const int line   = 0) {
  if (SPARSE_STATUS_SUCCESS != mkl_status) {
    std::ostringstream oss;
    oss << "MKL call \"" << name << "\" encountered error at " << file << ":"
        << line << '\n';
    Kokkos::abort(oss.str().c_str());
  }
}

#define MKL_SAFE_CALL(call) \
  KokkosSparse::Impl::mkl_internal_safe_call(call, #call, __FILE__, __LINE__)

inline sparse_operation_t mode_kk_to_mkl(char mode_kk) {
  switch (toupper(mode_kk)) {
    case 'N': return SPARSE_OPERATION_NON_TRANSPOSE;
    case 'T': return SPARSE_OPERATION_TRANSPOSE;
    case 'H': return SPARSE_OPERATION_CONJUGATE_TRANSPOSE;
    default:;
  }
  throw std::invalid_argument(
      "Invalid mode for MKL (should be one of N, T, H)");
}

template <typename value_type>
struct mkl_is_supported_value_type : std::false_type {};

template <>
struct mkl_is_supported_value_type<float> : std::true_type {};
template <>
struct mkl_is_supported_value_type<double> : std::true_type {};

// MKLSparseMatrix provides thin wrapper around MKL matrix handle
// (sparse_matrix_t) and encapsulates MKL call dispatches related to details
// like value_type, allowing simple client code in kernels.
template <typename value_type>
class MKLSparseMatrix {
  sparse_matrix_t mtx;

  static_assert(mkl_is_supported_value_type<value_type>::value,
                "Scalar type used in MKLSparseMatrix<value_type> is NOT "
                "supported by MKL");

 public:
  inline MKLSparseMatrix(sparse_matrix_t mtx_) : mtx(mtx_) {}

  // Constructs MKL sparse matrix from KK sparse views (m rows x n cols)
  inline MKLSparseMatrix(const MKL_INT num_rows, const MKL_INT num_cols,
                         MKL_INT *xadj, MKL_INT *adj, value_type *values);

  // Allows using MKLSparseMatrix directly in MKL calls
  inline operator sparse_matrix_t() const { return mtx; }

  // Exports MKL sparse matrix contents into KK views
  inline void export_data(MKL_INT &num_rows, MKL_INT &num_cols,
                          MKL_INT *&rows_start, MKL_INT *&columns,
                          value_type *&values);

  inline void destroy() { MKL_SAFE_CALL(mkl_sparse_destroy(mtx)); }
};

template <>
inline MKLSparseMatrix<float>::MKLSparseMatrix(const MKL_INT rows,
                                               const MKL_INT cols,
                                               MKL_INT *xadj, MKL_INT *adj,
                                               float *values) {
  MKL_SAFE_CALL(mkl_sparse_s_create_csr(&mtx, SPARSE_INDEX_BASE_ZERO, rows,
                                        cols, xadj, xadj + 1, adj, values));
}

template <>
inline MKLSparseMatrix<double>::MKLSparseMatrix(const MKL_INT rows,
                                                const MKL_INT cols,
                                                MKL_INT *xadj, MKL_INT *adj,
                                                double *values) {
  MKL_SAFE_CALL(mkl_sparse_d_create_csr(&mtx, SPARSE_INDEX_BASE_ZERO, rows,
                                        cols, xadj, xadj + 1, adj, values));
}

template <>
inline void MKLSparseMatrix<float>::export_data(MKL_INT &num_rows,
                                                MKL_INT &num_cols,
                                                MKL_INT *&rows_start,
                                                MKL_INT *&columns,
                                                float *&values) {
  sparse_index_base_t indexing;
  MKL_INT *rows_end;
  MKL_SAFE_CALL(mkl_sparse_s_export_csr(mtx, &indexing, &num_rows, &num_cols,
                                        &rows_start, &rows_end, &columns,
                                        &values));
  if (SPARSE_INDEX_BASE_ZERO != indexing) {
    throw std::runtime_error(
        "Expected zero based indexing in exported MKL sparse matrix\n");
    return;
  }
}

template <>
inline void MKLSparseMatrix<double>::export_data(MKL_INT &num_rows,
                                                 MKL_INT &num_cols,
                                                 MKL_INT *&rows_start,
                                                 MKL_INT *&columns,
                                                 double *&values) {
  sparse_index_base_t indexing;
  MKL_INT *rows_end;
  MKL_SAFE_CALL(mkl_sparse_d_export_csr(mtx, &indexing, &num_rows, &num_cols,
                                        &rows_start, &rows_end, &columns,
                                        &values));
  if (SPARSE_INDEX_BASE_ZERO != indexing) {
    throw std::runtime_error(
        "Expected zero based indexing in exported MKL sparse matrix\n");
    return;
  }
}

}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSKERNELS_ENABLE_TPL_MKL

#endif  // _KOKKOSKERNELS_SPARSEUTILS_MKL_HPP