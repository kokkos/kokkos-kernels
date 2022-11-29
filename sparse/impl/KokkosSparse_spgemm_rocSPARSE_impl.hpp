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

#ifndef _KOKKOSSPGEMMROCSPARSE_HPP
#define _KOKKOSSPGEMMROCSPARSE_HPP

#include <KokkosKernels_config.h>

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE

#include "KokkosKernels_Controls.hpp"
#include "rocsparse.h"

namespace KokkosSparse {

namespace Impl {

//=============================================================================
// Overload rocsparse_Xcsrgemm_buffer_size() over scalar types
inline rocsparse_status rocsparse_Xcsrgemm_buffer_size(
    rocsparse_handle handle, rocsparse_operation trans_A,
    rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n,
    rocsparse_int k, const float *alpha, const rocsparse_mat_descr descr_A,
    rocsparse_int nnz_A, const rocsparse_int *csr_row_ptr_A,
    const rocsparse_int *csr_col_ind_A, const rocsparse_mat_descr descr_B,
    rocsparse_int nnz_B, const rocsparse_int *csr_row_ptr_B,
    const rocsparse_int *csr_col_ind_B, const float *beta,
    const rocsparse_mat_descr descr_D, rocsparse_int nnz_D,
    const rocsparse_int *csr_row_ptr_D, const rocsparse_int *csr_col_ind_D,
    rocsparse_mat_info info_C, size_t *buffer_size) {
  return rocsparse_scsrgemm_buffer_size(
      handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_row_ptr_A,
      csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B, csr_col_ind_B, beta,
      descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, buffer_size);
}

inline rocsparse_status rocsparse_Xcsrgemm_buffer_size(
    rocsparse_handle handle, rocsparse_operation trans_A,
    rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n,
    rocsparse_int k, const double *alpha, const rocsparse_mat_descr descr_A,
    rocsparse_int nnz_A, const rocsparse_int *csr_row_ptr_A,
    const rocsparse_int *csr_col_ind_A, const rocsparse_mat_descr descr_B,
    rocsparse_int nnz_B, const rocsparse_int *csr_row_ptr_B,
    const rocsparse_int *csr_col_ind_B, const double *beta,
    const rocsparse_mat_descr descr_D, rocsparse_int nnz_D,
    const rocsparse_int *csr_row_ptr_D, const rocsparse_int *csr_col_ind_D,
    rocsparse_mat_info info_C, size_t *buffer_size) {
  return rocsparse_dcsrgemm_buffer_size(
      handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_row_ptr_A,
      csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B, csr_col_ind_B, beta,
      descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, buffer_size);
}

inline rocsparse_status rocsparse_Xcsrgemm_buffer_size(
    rocsparse_handle handle, rocsparse_operation trans_A,
    rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n,
    rocsparse_int k, const Kokkos::complex<float> *alpha,
    const rocsparse_mat_descr descr_A, rocsparse_int nnz_A,
    const rocsparse_int *csr_row_ptr_A, const rocsparse_int *csr_col_ind_A,
    const rocsparse_mat_descr descr_B, rocsparse_int nnz_B,
    const rocsparse_int *csr_row_ptr_B, const rocsparse_int *csr_col_ind_B,
    const Kokkos::complex<float> *beta, const rocsparse_mat_descr descr_D,
    rocsparse_int nnz_D, const rocsparse_int *csr_row_ptr_D,
    const rocsparse_int *csr_col_ind_D, rocsparse_mat_info info_C,
    size_t *buffer_size) {
  return rocsparse_ccsrgemm_buffer_size(
      handle, trans_A, trans_B, m, n, k,
      reinterpret_cast<const rocsparse_float_complex *>(alpha), descr_A, nnz_A,
      csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B,
      csr_col_ind_B, reinterpret_cast<const rocsparse_float_complex *>(beta),
      descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, buffer_size);
}

inline rocsparse_status rocsparse_Xcsrgemm_buffer_size(
    rocsparse_handle handle, rocsparse_operation trans_A,
    rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n,
    rocsparse_int k, const Kokkos::complex<double> *alpha,
    const rocsparse_mat_descr descr_A, rocsparse_int nnz_A,
    const rocsparse_int *csr_row_ptr_A, const rocsparse_int *csr_col_ind_A,
    const rocsparse_mat_descr descr_B, rocsparse_int nnz_B,
    const rocsparse_int *csr_row_ptr_B, const rocsparse_int *csr_col_ind_B,
    const Kokkos::complex<double> *beta, const rocsparse_mat_descr descr_D,
    rocsparse_int nnz_D, const rocsparse_int *csr_row_ptr_D,
    const rocsparse_int *csr_col_ind_D, rocsparse_mat_info info_C,
    size_t *buffer_size) {
  return rocsparse_zcsrgemm_buffer_size(
      handle, trans_A, trans_B, m, n, k,
      reinterpret_cast<const rocsparse_double_complex *>(alpha), descr_A, nnz_A,
      csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_row_ptr_B,
      csr_col_ind_B, reinterpret_cast<const rocsparse_double_complex *>(beta),
      descr_D, nnz_D, csr_row_ptr_D, csr_col_ind_D, info_C, buffer_size);
}

//=============================================================================
// Overload rocsparse_Xcsrgemm_numeric() over scalar types
inline rocsparse_status rocsparse_Xcsrgemm_numeric(
    rocsparse_handle handle, rocsparse_operation trans_A,
    rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n,
    rocsparse_int k, const float *alpha, const rocsparse_mat_descr descr_A,
    rocsparse_int nnz_A, const float *csr_val_A,
    const rocsparse_int *csr_row_ptr_A, const rocsparse_int *csr_col_ind_A,
    const rocsparse_mat_descr descr_B, rocsparse_int nnz_B,
    const float *csr_val_B, const rocsparse_int *csr_row_ptr_B,
    const rocsparse_int *csr_col_ind_B, const float *beta,
    const rocsparse_mat_descr descr_D, rocsparse_int nnz_D,
    const float *csr_val_D, const rocsparse_int *csr_row_ptr_D,
    const rocsparse_int *csr_col_ind_D, const rocsparse_mat_descr descr_C,
    rocsparse_int nnz_C, float *csr_val_C, const rocsparse_int *csr_row_ptr_C,
    const rocsparse_int *csr_col_ind_C, const rocsparse_mat_info info_C,
    void *buffer) {
  return rocsparse_scsrgemm_numeric(
      handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_val_A,
      csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_val_B, csr_row_ptr_B,
      csr_col_ind_B, beta, descr_D, nnz_D, csr_val_D, csr_row_ptr_D,
      csr_col_ind_D, descr_C, nnz_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C,
      info_C, buffer);
}

inline rocsparse_status rocsparse_Xcsrgemm_numeric(
    rocsparse_handle handle, rocsparse_operation trans_A,
    rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n,
    rocsparse_int k, const double *alpha, const rocsparse_mat_descr descr_A,
    rocsparse_int nnz_A, const double *csr_val_A,
    const rocsparse_int *csr_row_ptr_A, const rocsparse_int *csr_col_ind_A,
    const rocsparse_mat_descr descr_B, rocsparse_int nnz_B,
    const double *csr_val_B, const rocsparse_int *csr_row_ptr_B,
    const rocsparse_int *csr_col_ind_B, const double *beta,
    const rocsparse_mat_descr descr_D, rocsparse_int nnz_D,
    const double *csr_val_D, const rocsparse_int *csr_row_ptr_D,
    const rocsparse_int *csr_col_ind_D, const rocsparse_mat_descr descr_C,
    rocsparse_int nnz_C, double *csr_val_C, const rocsparse_int *csr_row_ptr_C,
    const rocsparse_int *csr_col_ind_C, const rocsparse_mat_info info_C,
    void *buffer) {
  return rocsparse_dcsrgemm_numeric(
      handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_val_A,
      csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_val_B, csr_row_ptr_B,
      csr_col_ind_B, beta, descr_D, nnz_D, csr_val_D, csr_row_ptr_D,
      csr_col_ind_D, descr_C, nnz_C, csr_val_C, csr_row_ptr_C, csr_col_ind_C,
      info_C, buffer);
}

inline rocsparse_status rocsparse_Xcsrgemm_numeric(
    rocsparse_handle handle, rocsparse_operation trans_A,
    rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n,
    rocsparse_int k, const Kokkos::complex<float> *alpha,
    const rocsparse_mat_descr descr_A, rocsparse_int nnz_A,
    const Kokkos::complex<float> *csr_val_A, const rocsparse_int *csr_row_ptr_A,
    const rocsparse_int *csr_col_ind_A, const rocsparse_mat_descr descr_B,
    rocsparse_int nnz_B, const Kokkos::complex<float> *csr_val_B,
    const rocsparse_int *csr_row_ptr_B, const rocsparse_int *csr_col_ind_B,
    const Kokkos::complex<float> *beta, const rocsparse_mat_descr descr_D,
    rocsparse_int nnz_D, const Kokkos::complex<float> *csr_val_D,
    const rocsparse_int *csr_row_ptr_D, const rocsparse_int *csr_col_ind_D,
    const rocsparse_mat_descr descr_C, rocsparse_int nnz_C,
    Kokkos::complex<float> *csr_val_C, const rocsparse_int *csr_row_ptr_C,
    const rocsparse_int *csr_col_ind_C, const rocsparse_mat_info info_C,
    void *buffer) {
  return rocsparse_ccsrgemm_numeric(
      handle, trans_A, trans_B, m, n, k,
      reinterpret_cast<const rocsparse_float_complex *>(alpha), descr_A, nnz_A,
      reinterpret_cast<const rocsparse_float_complex *>(csr_val_A),
      csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B,
      reinterpret_cast<const rocsparse_float_complex *>(csr_val_B),
      csr_row_ptr_B, csr_col_ind_B,
      reinterpret_cast<const rocsparse_float_complex *>(beta), descr_D, nnz_D,
      reinterpret_cast<const rocsparse_float_complex *>(csr_val_D),
      csr_row_ptr_D, csr_col_ind_D, descr_C, nnz_C,
      reinterpret_cast<rocsparse_float_complex *>(csr_val_C), csr_row_ptr_C,
      csr_col_ind_C, info_C, buffer);
}

inline rocsparse_status rocsparse_Xcsrgemm_numeric(
    rocsparse_handle handle, rocsparse_operation trans_A,
    rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n,
    rocsparse_int k, const Kokkos::complex<double> *alpha,
    const rocsparse_mat_descr descr_A, rocsparse_int nnz_A,
    const Kokkos::complex<double> *csr_val_A,
    const rocsparse_int *csr_row_ptr_A, const rocsparse_int *csr_col_ind_A,
    const rocsparse_mat_descr descr_B, rocsparse_int nnz_B,
    const Kokkos::complex<double> *csr_val_B,
    const rocsparse_int *csr_row_ptr_B, const rocsparse_int *csr_col_ind_B,
    const Kokkos::complex<double> *beta, const rocsparse_mat_descr descr_D,
    rocsparse_int nnz_D, const Kokkos::complex<double> *csr_val_D,
    const rocsparse_int *csr_row_ptr_D, const rocsparse_int *csr_col_ind_D,
    const rocsparse_mat_descr descr_C, rocsparse_int nnz_C,
    Kokkos::complex<double> *csr_val_C, const rocsparse_int *csr_row_ptr_C,
    const rocsparse_int *csr_col_ind_C, const rocsparse_mat_info info_C,
    void *buffer) {
  return rocsparse_zcsrgemm_numeric(
      handle, trans_A, trans_B, m, n, k,
      reinterpret_cast<const rocsparse_double_complex *>(alpha), descr_A, nnz_A,
      reinterpret_cast<const rocsparse_double_complex *>(csr_val_A),
      csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B,
      reinterpret_cast<const rocsparse_double_complex *>(csr_val_B),
      csr_row_ptr_B, csr_col_ind_B,
      reinterpret_cast<const rocsparse_double_complex *>(beta), descr_D, nnz_D,
      reinterpret_cast<const rocsparse_double_complex *>(csr_val_D),
      csr_row_ptr_D, csr_col_ind_D, descr_C, nnz_C,
      reinterpret_cast<rocsparse_double_complex *>(csr_val_C), csr_row_ptr_C,
      csr_col_ind_C, info_C, buffer);
}

//=============================================================================
template <typename KernelHandle, typename index_type, typename size_type,
          typename scalar_type>
inline typename std::enable_if<
    (not std::is_same<index_type, rocsparse_int>::value) or
        (not std::is_same<size_type, rocsparse_int>::value),
    void>::type
rocsparse_spgemm_symbolic_internal(KernelHandle *handle, index_type m,
                                   index_type n, index_type k, size_type nnz_A,
                                   const size_type *rowptrA,
                                   const index_type *colidxA, bool transposeA,
                                   size_type nnz_B, const size_type *rowptrB,
                                   const index_type *colidxB, bool transposeB,
                                   size_type *rowptrC) {
  // normal code should use the specializations and not go here
  throw std::runtime_error(
      "The installed rocsparse does not support the index type and size type");
}

template <typename KernelHandle, typename index_type, typename size_type,
          typename scalar_type>
inline
    typename std::enable_if<std::is_same<index_type, rocsparse_int>::value and
                                std::is_same<size_type, rocsparse_int>::value,
                            void>::type
    rocsparse_spgemm_symbolic_internal(
        KernelHandle *handle, index_type m, index_type n, index_type k,
        size_type nnz_A, const size_type *rowptrA, const index_type *colidxA,
        bool transposeA, size_type nnz_B, const size_type *rowptrB,
        const index_type *colidxB, bool transposeB, size_type *rowptrC) {
  handle->create_rocsparse_spgemm_handle(transposeA, transposeB);
  typename KernelHandle::rocSparseSpgemmHandleType *h =
      handle->get_rocsparse_spgemm_handle();

  // alpha, beta are on host, but since we use singleton on the rocsparse
  // handle, we save/restore the pointer mode to not interference with
  // others' use
  const auto alpha = Kokkos::ArithTraits<scalar_type>::one();
  const auto beta  = Kokkos::ArithTraits<scalar_type>::zero();
  rocsparse_pointer_mode oldPtrMode;

  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
      rocsparse_get_pointer_mode(h->rocsparseHandle, &oldPtrMode));
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_set_pointer_mode(
      h->rocsparseHandle, rocsparse_pointer_mode_host));

  // C = alpha * OpA(A) * OpB(B) + beta * D
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_Xcsrgemm_buffer_size(
      h->rocsparseHandle, h->opA, h->opB, m, n, k, &alpha, h->descr_A, nnz_A,
      rowptrA, colidxA, h->descr_B, nnz_B, rowptrB, colidxB, &beta, h->descr_D,
      0, NULL, NULL, h->info_C, &h->bufferSize));

  KOKKOS_IMPL_HIP_SAFE_CALL(hipMalloc(&h->buffer, h->bufferSize));

  rocsparse_int C_nnz = 0;
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_csrgemm_nnz(
      h->rocsparseHandle, h->opA, h->opB, m, n, k, h->descr_A, nnz_A, rowptrA,
      colidxA, h->descr_B, nnz_B, rowptrB, colidxB, h->descr_D, 0, NULL, NULL,
      h->descr_C, rowptrC, &C_nnz, h->info_C, h->buffer));

  handle->set_c_nnz(C_nnz);
  h->C_populated = false;  // sparsity pattern of C is not set yet, so this is a
                           // fake symbolic
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
      rocsparse_set_pointer_mode(h->rocsparseHandle, oldPtrMode));
}

template <
    typename KernelHandle, typename ain_row_index_view_type,
    typename ain_nonzero_index_view_type, typename bin_row_index_view_type,
    typename bin_nonzero_index_view_type, typename cin_row_index_view_type>
void rocsparse_spgemm_symbolic(
    KernelHandle *handle, typename KernelHandle::nnz_lno_t m,
    typename KernelHandle::nnz_lno_t n, typename KernelHandle::nnz_lno_t k,
    ain_row_index_view_type rowptrA, ain_nonzero_index_view_type colidxA,
    bool transposeA, bin_row_index_view_type rowptrB,
    bin_nonzero_index_view_type colidxB, bool transposeB,
    cin_row_index_view_type rowptrC) {
  using index_type  = typename KernelHandle::nnz_lno_t;
  using size_type   = typename KernelHandle::size_type;
  using scalar_type = typename KernelHandle::nnz_scalar_t;

  // In case the KernelHandle uses const types!
  using non_const_index_type  = typename std::remove_cv<index_type>::type;
  using non_const_size_type   = typename std::remove_cv<size_type>::type;
  using non_const_scalar_type = typename std::remove_cv<scalar_type>::type;

  handle->set_sort_option(1);  // tells users the output matrix is sorted
  rocsparse_spgemm_symbolic_internal<KernelHandle, non_const_index_type,
                                     non_const_size_type,
                                     non_const_scalar_type>(
      handle, m, n, k, colidxA.extent(0), rowptrA.data(), colidxA.data(),
      transposeA, colidxB.extent(0), rowptrB.data(), colidxB.data(), transposeB,
      rowptrC.data());
}

//=============================================================================
template <typename KernelHandle, typename index_type, typename size_type,
          typename scalar_type>
inline typename std::enable_if<
    (not std::is_same<index_type, rocsparse_int>::value) or
        (not std::is_same<size_type, rocsparse_int>::value),
    void>::type
rocsparse_spgemm_numeric_internal(
    KernelHandle *handle, index_type m, index_type n, index_type k,
    size_type nnz_A, const size_type *rowptrA, const index_type *colidxA,
    const scalar_type *valuesA, size_type nnz_B, const size_type *rowptrB,
    const index_type *colidxB, const scalar_type *valuesB, size_type nnz_C,
    const size_type *rowptrC, index_type *colidxC, scalar_type *valuesC) {
  // normal code should use the specializations and not go here
  throw std::runtime_error(
      "The installed rocsparse does not support the index type and size type");
}

template <typename KernelHandle, typename index_type, typename size_type,
          typename scalar_type>
inline
    typename std::enable_if<std::is_same<index_type, rocsparse_int>::value and
                                std::is_same<size_type, rocsparse_int>::value,
                            void>::type
    rocsparse_spgemm_numeric_internal(
        KernelHandle *handle, index_type m, index_type n, index_type k,
        size_type nnz_A, const size_type *rowptrA, const index_type *colidxA,
        const scalar_type *valuesA, size_type nnz_B, const size_type *rowptrB,
        const index_type *colidxB, const scalar_type *valuesB, size_type nnz_C,
        const size_type *rowptrC, index_type *colidxC, scalar_type *valuesC) {
  typename KernelHandle::rocSparseSpgemmHandleType *h =
      handle->get_rocsparse_spgemm_handle();

  const auto alpha = Kokkos::ArithTraits<scalar_type>::one();
  const auto beta  = Kokkos::ArithTraits<scalar_type>::zero();
  rocsparse_pointer_mode oldPtrMode;

  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
      rocsparse_get_pointer_mode(h->rocsparseHandle, &oldPtrMode));
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_set_pointer_mode(
      h->rocsparseHandle, rocsparse_pointer_mode_host));

  if (!h->C_populated) {
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_csrgemm_symbolic(
        h->rocsparseHandle, h->opA, h->opB, m, n, k, h->descr_A, nnz_A, rowptrA,
        colidxA, h->descr_B, nnz_B, rowptrB, colidxB, h->descr_D, 0, NULL, NULL,
        h->descr_C, nnz_C, rowptrC, colidxC, h->info_C, h->buffer));
    h->C_populated = true;
  }

  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_Xcsrgemm_numeric(
      h->rocsparseHandle, h->opA, h->opB, m, n, k, &alpha, h->descr_A, nnz_A,
      valuesA, rowptrA, colidxA, h->descr_B, nnz_B, valuesB, rowptrB, colidxB,
      &beta, h->descr_D, 0, NULL, NULL, NULL, h->descr_C, nnz_C, valuesC,
      rowptrC, colidxC, h->info_C, h->buffer));
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
      rocsparse_set_pointer_mode(h->rocsparseHandle, oldPtrMode));
}

template <
    typename KernelHandle, typename ain_row_index_view_type,
    typename ain_nonzero_index_view_type, typename ain_nonzero_value_view_type,
    typename bin_row_index_view_type, typename bin_nonzero_index_view_type,
    typename bin_nonzero_value_view_type, typename cin_row_index_view_type,
    typename cin_nonzero_index_view_type, typename cin_nonzero_value_view_type>
void rocsparse_spgemm_numeric(
    KernelHandle *handle, typename KernelHandle::nnz_lno_t m,
    typename KernelHandle::nnz_lno_t n, typename KernelHandle::nnz_lno_t k,
    ain_row_index_view_type rowptrA, ain_nonzero_index_view_type colidxA,
    ain_nonzero_value_view_type valuesA, bool /* transposeA */,
    bin_row_index_view_type rowptrB, bin_nonzero_index_view_type colidxB,
    bin_nonzero_value_view_type valuesB, bool /* transposeB */,
    cin_row_index_view_type rowptrC, cin_nonzero_index_view_type colidxC,
    cin_nonzero_value_view_type valuesC) {
  using index_type  = typename KernelHandle::nnz_lno_t;
  using size_type   = typename KernelHandle::size_type;
  using scalar_type = typename KernelHandle::nnz_scalar_t;

  // In case the KernelHandle uses const types!
  using non_const_index_type  = typename std::remove_cv<index_type>::type;
  using non_const_size_type   = typename std::remove_cv<size_type>::type;
  using non_const_scalar_type = typename std::remove_cv<scalar_type>::type;

  rocsparse_spgemm_numeric_internal<KernelHandle, non_const_index_type,
                                    non_const_size_type, non_const_scalar_type>(
      handle, m, n, k, colidxA.extent(0), rowptrA.data(), colidxA.data(),
      valuesA.data(), colidxB.extent(0), rowptrB.data(), colidxB.data(),
      valuesB.data(), colidxC.extent(0), rowptrC.data(), colidxC.data(),
      valuesC.data());
}

}  // namespace Impl
}  // namespace KokkosSparse

#endif
#endif
