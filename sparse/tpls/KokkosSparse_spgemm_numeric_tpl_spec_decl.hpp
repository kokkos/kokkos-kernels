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

#ifndef KOKKOSPARSE_SPGEMM_NUMERIC_TPL_SPEC_DECL_HPP_
#define KOKKOSPARSE_SPGEMM_NUMERIC_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include "cusparse.h"
#include "KokkosSparse_Utils_cusparse.hpp"
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#include "rocsparse/rocsparse.h"
#include "KokkosSparse_Utils_rocsparse.hpp"
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
#include "KokkosSparse_Utils_mkl.hpp"
#include "mkl_spblas.h"
#endif

namespace KokkosSparse {
namespace Impl {

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#if (CUDA_VERSION >= 11040)

// 11.4+ supports generic API with reuse (full symbolic/numeric separation)
template <typename KernelHandle, typename lno_t, typename ConstRowMapType,
          typename ConstEntriesType, typename ConstValuesType,
          typename EntriesType, typename ValuesType>
void spgemm_numeric_cusparse(
    KernelHandle *handle, lno_t m, lno_t n, lno_t k,
    const ConstRowMapType &row_mapA, const ConstEntriesType &entriesA,
    const ConstValuesType &valuesA, const ConstRowMapType &row_mapB,
    const ConstEntriesType &entriesB, const ConstValuesType &valuesB,
    const ConstRowMapType &row_mapC, const EntriesType &entriesC,
    const ValuesType &valuesC) {
  using scalar_type = typename KernelHandle::nnz_scalar_t;
  auto sh           = handle->get_spgemm_handle();
  auto h            = sh->get_cusparse_spgemm_handle();

  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_A, (void *)row_mapA.data(),
                             (void *)entriesA.data(), (void *)valuesA.data()));

  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_B, (void *)row_mapB.data(),
                             (void *)entriesB.data(), (void *)valuesB.data()));

  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_C, (void *)row_mapC.data(),
                             (void *)entriesC.data(), (void *)valuesC.data()));

  if (!sh->are_entries_computed()) {
    cusparseSpGEMMreuse_copy(h->cusparseHandle, h->opA, h->opB, h->descr_A,
                             h->descr_B, h->descr_C, h->alg, h->spgemmDescr,
                             &h->bufferSize5, NULL);
    cudaMalloc((void **)&h->buffer5, h->bufferSize5);
    cusparseSpGEMMreuse_copy(h->cusparseHandle, h->opA, h->opB, h->descr_A,
                             h->descr_B, h->descr_C, h->alg, h->spgemmDescr,
                             &h->bufferSize5, h->buffer5);
    if (!sh->get_c_nnz()) {
      // cuSPARSE does not populate C rowptrs if C has no entries
      cudaStream_t stream;
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseGetStream(h->cusparseHandle, &stream));
      cudaMemsetAsync(
          (void *)row_mapC.data(), 0,
          row_mapC.extent(0) * sizeof(typename ConstRowMapType::value_type));
    }
    cudaFree(h->buffer3);
    h->buffer3 = NULL;
    sh->set_computed_rowptrs();
    sh->set_computed_entries();
  }

  // C' = alpha * opA(A) * opB(B) + beta * C
  const auto alpha = Kokkos::ArithTraits<scalar_type>::one();
  const auto beta  = Kokkos::ArithTraits<scalar_type>::zero();

  // alpha, beta are on host, but since we use singleton on the cusparse
  // handle, we save/restore the pointer mode to not interference with
  // others' use
  cusparsePointerMode_t oldPtrMode;
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseGetPointerMode(h->cusparseHandle, &oldPtrMode));
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseSetPointerMode(h->cusparseHandle, CUSPARSE_POINTER_MODE_HOST));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMMreuse_compute(
      h->cusparseHandle, h->opA, h->opB, &alpha, h->descr_A, h->descr_B, &beta,
      h->descr_C, h->scalarType, h->alg, h->spgemmDescr));
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseSetPointerMode(h->cusparseHandle, oldPtrMode));
}

#elif (CUDA_VERSION >= 11000)
// 11.0-11.3 supports only the generic API, but not reuse.
template <typename KernelHandle, typename lno_t, typename ConstRowMapType,
          typename ConstEntriesType, typename ConstValuesType,
          typename EntriesType, typename ValuesType>
void spgemm_numeric_cusparse(
    KernelHandle *handle, lno_t m, lno_t n, lno_t k,
    const ConstRowMapType &row_mapA, const ConstEntriesType &entriesA,
    const ConstValuesType &valuesA, const ConstRowMapType &row_mapB,
    const ConstEntriesType &entriesB, const ConstValuesType &valuesB,
    const ConstRowMapType &row_mapC, const EntriesType &entriesC,
    const ValuesType &valuesC) {
  auto sh = handle->get_spgemm_handle();
  auto h  = sh->get_cusparse_spgemm_handle();
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_A, (void *)row_mapA.data(),
                             (void *)entriesA.data(), (void *)valuesA.data()));
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_B, (void *)row_mapB.data(),
                             (void *)entriesB.data(), (void *)valuesB.data()));
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_C, (void *)row_mapC.data(),
                             (void *)entriesC.data(), (void *)valuesC.data()));
  const auto alpha = Kokkos::ArithTraits<scalar_type>::one();
  const auto beta  = Kokkos::ArithTraits<scalar_type>::zero();
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_compute(
      h->cusparseHandle, h->opA, h->opB, &alpha, h->descr_A, h->descr_B, &beta,
      h->descr_C, h->scalarType, CUSPARSE_SPGEMM_DEFAULT, h->spgemmDescr,
      &h->bufferSize4, h->buffer4));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_copy(
      h->cusparseHandle, h->opA, h->opB, &alpha, h->descr_A, h->descr_B, &beta,
      h->descr_C, h->scalarType, CUSPARSE_SPGEMM_DEFAULT, h->spgemmDescr));
}

#else
// 10.x supports the pre-generic interface.
template <typename KernelHandle, typename lno_t, typename ConstRowMapType,
          typename ConstEntriesType, typename ConstValuesType,
          typename EntriesType, typename ValuesType>
void spgemm_numeric_cusparse(
    KernelHandle *handle, lno_t m, lno_t n, lno_t k,
    const ConstRowMapType &row_mapA, const ConstEntriesType &entriesA,
    const ConstValuesType &valuesA, const ConstRowMapType &row_mapB,
    const ConstEntriesType &entriesB, const ConstValuesType &valuesB,
    const ConstRowMapType &row_mapC, const EntriesType &entriesC,
    const ValuesType &valuesC) {
  using scalar_type = typename KernelHandle::nnz_scalar_t;
  auto sh           = handle->get_spgemm_handle();
  auto h            = handle->get_cusparse_spgemm_handle();

  int nnzA = entriesA.extent(0);
  int nnzB = entriesB.extent(0);

  if constexpr (std::is_same<scalar_type, float>::value) {
    cusparseScsrgemm(h->handle, h->transA, h->transB, m, n, k, h->a_descr, nnzA,
                     valuesA.data(), row_mapA.data(), entriesA.data(),
                     h->b_descr, nnzB, valuesB.data(), row_mapB.data(),
                     entriesB.data(), h->c_descr, valuesC.data(),
                     row_mapC.data(), entriesC.data());
  } else if (std::is_same<scalar_type, double>::value) {
    cusparseDcsrgemm(h->handle, h->transA, h->transB, m, n, k, h->a_descr, nnzA,
                     valuesA.data(), row_mapA.data(), entriesA.data(),
                     h->b_descr, nnzB, valuesB.data(), row_mapB.data(),
                     entriesB.data(), h->c_descr, valuesC.data(),
                     row_mapC.data(), entriesC.data());
  } else if (std::is_same<scalar_type, Kokkos::complex<float> >::value) {
    cusparseCcsrgemm(h->handle, h->transA, h->transB, m, n, k, h->a_descr, nnzA,
                     (const cuComplex *)valuesA.data(), row_mapA.data(),
                     entriesA.data(), h->b_descr, nnzB,
                     (const cuComplex *)valuesB.data(), row_mapB.data(),
                     entriesB.data(), h->c_descr, (cuComplex *)valuesC.data(),
                     row_mapC.data(), entriesC.data());
  } else if (std::is_same<scalar_type, Kokkos::complex<double> >::value) {
    cusparseZcsrgemm(h->handle, h->transA, h->transB, m, n, k, h->a_descr, nnzA,
                     (const cuDoubleComplex *)valuesA.data(), row_mapA.data(),
                     entriesA.data(), h->b_descr, nnzB,
                     (const cuDoubleComplex *)valuesB.data(), row_mapB.data(),
                     entriesB.data(), h->c_descr,
                     (cuDoubleComplex *)valuesC.data(), row_mapC.data(),
                     entriesC.data());
  }
}

#endif

#define SPGEMM_NUMERIC_DECL_CUSPARSE(SCALAR, MEMSPACE, TPL_AVAIL)        \
  template <>                                                                  \
  struct SPGEMM_NUMERIC<                                                       \
      KokkosKernels::Experimental::KokkosKernelsHandle<                        \
          const int, const int, const SCALAR, Kokkos::Cuda, MEMSPACE,          \
          MEMSPACE>,                                                           \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const SCALAR *, default_layout,                             \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const SCALAR *, default_layout,                             \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<int *, default_layout,                                      \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<SCALAR *, default_layout,                                   \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, TPL_AVAIL> {                                                 \
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<     \
        const int, const int, const SCALAR, Kokkos::Cuda, MEMSPACE, MEMSPACE>; \
    using c_int_view_t =                                                       \
        Kokkos::View<const int *, default_layout,                              \
                     Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                   \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;                \
    using int_view_t = Kokkos::View<int *, default_layout,                     \
                                    Kokkos::Device<Kokkos::Cuda, MEMSPACE>,    \
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >; \
    using c_scalar_view_t =                                                    \
        Kokkos::View<const SCALAR *, default_layout,                           \
                     Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                   \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;                \
    using scalar_view_t =                                                      \
        Kokkos::View<SCALAR *, default_layout,                                 \
                     Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                   \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;                \
    static void spgemm_numeric(KernelHandle *handle,                           \
                               typename KernelHandle::nnz_lno_t m,             \
                               typename KernelHandle::nnz_lno_t n,             \
                               typename KernelHandle::nnz_lno_t k,             \
                               c_int_view_t row_mapA, c_int_view_t entriesA,   \
                               c_scalar_view_t valuesA, bool,                  \
                               c_int_view_t row_mapB, c_int_view_t entriesB,   \
                               c_scalar_view_t valuesB, bool,                  \
                               c_int_view_t row_mapC, int_view_t entriesC,     \
                               scalar_view_t valuesC) {                        \
      std::string label = "KokkosSparse::spgemm_numeric[TPL_CUSPARSE," +               \
                          Kokkos::ArithTraits<SCALAR>::name() + "]";           \
      Kokkos::Profiling::pushRegion(label);                                    \
      spgemm_numeric_cusparse(handle, m, n, k, row_mapA, entriesA, valuesA,    \
                              row_mapB, entriesB, valuesB, row_mapC, entriesC, \
                              valuesC);                                        \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define SPGEMM_NUMERIC_DECL_CUSPARSE_S(SCALAR, TPL_AVAIL)            \
  SPGEMM_NUMERIC_DECL_CUSPARSE(SCALAR, Kokkos::CudaSpace, TPL_AVAIL) \
  SPGEMM_NUMERIC_DECL_CUSPARSE(SCALAR, Kokkos::CudaUVMSpace, TPL_AVAIL)

SPGEMM_NUMERIC_DECL_CUSPARSE_S(float, true)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(double, true)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(Kokkos::complex<float>, true)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(Kokkos::complex<double>, true)

SPGEMM_NUMERIC_DECL_CUSPARSE_S(float, false)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(double, false)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(Kokkos::complex<float>, false)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(Kokkos::complex<double>, false)
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
//=============================================================================
// Overload rocsparse_Xcsrgemm_numeric() over scalar types
#define ROCSPARSE_XCSRGEMM_NUMERIC_SPEC(scalar_type, TOKEN)                   \
  inline rocsparse_status rocsparse_Xcsrgemm_numeric(                         \
      rocsparse_handle handle, rocsparse_operation trans_A,                   \
      rocsparse_operation trans_B, rocsparse_int m, rocsparse_int n,          \
      rocsparse_int k, const scalar_type *alpha,                              \
      const rocsparse_mat_descr descr_A, rocsparse_int nnz_A,                 \
      const scalar_type *csr_val_A, const rocsparse_int *csr_row_ptr_A,       \
      const rocsparse_int *csr_col_ind_A, const rocsparse_mat_descr descr_B,  \
      rocsparse_int nnz_B, const scalar_type *csr_val_B,                      \
      const rocsparse_int *csr_row_ptr_B, const rocsparse_int *csr_col_ind_B, \
      const scalar_type *beta, const rocsparse_mat_descr descr_D,             \
      rocsparse_int nnz_D, const scalar_type *csr_val_D,                      \
      const rocsparse_int *csr_row_ptr_D, const rocsparse_int *csr_col_ind_D, \
      const rocsparse_mat_descr descr_C, rocsparse_int nnz_C,                 \
      scalar_type *csr_val_C, const rocsparse_int *csr_row_ptr_C,             \
      const rocsparse_int *csr_col_ind_C, const rocsparse_mat_info info_C,    \
      void *buffer) {                                                         \
    return rocsparse_##TOKEN##csrgemm_numeric(                                \
        handle, trans_A, trans_B, m, n, k, alpha, descr_A, nnz_A, csr_val_A,  \
        csr_row_ptr_A, csr_col_ind_A, descr_B, nnz_B, csr_val_B,              \
        csr_row_ptr_B, csr_col_ind_B, beta, descr_D, nnz_D, csr_val_D,        \
        csr_row_ptr_D, csr_col_ind_D, descr_C, nnz_C, csr_val_C,              \
        csr_row_ptr_C, csr_col_ind_C, info_C, buffer);                        \
  }

ROCSPARSE_XCSRGEMM_NUMERIC_SPEC(float, s)
ROCSPARSE_XCSRGEMM_NUMERIC_SPEC(double, d)
ROCSPARSE_XCSRGEMM_NUMERIC_SPEC(rocsparse_float_complex, c)
ROCSPARSE_XCSRGEMM_NUMERIC_SPEC(rocsparse_double_complex, z)

template <
    typename KernelHandle, typename ain_row_index_view_type,
    typename ain_nonzero_index_view_type, typename ain_nonzero_value_view_type,
    typename bin_row_index_view_type, typename bin_nonzero_index_view_type,
    typename bin_nonzero_value_view_type, typename cin_row_index_view_type,
    typename cin_nonzero_index_view_type, typename cin_nonzero_value_view_type>
void spgemm_numeric_rocsparse(
    KernelHandle *handle, typename KernelHandle::nnz_lno_t m,
    typename KernelHandle::nnz_lno_t n, typename KernelHandle::nnz_lno_t k,
    ain_row_index_view_type rowptrA, ain_nonzero_index_view_type colidxA,
    ain_nonzero_value_view_type valuesA, bin_row_index_view_type rowptrB,
    bin_nonzero_index_view_type colidxB, bin_nonzero_value_view_type valuesB,
    cin_row_index_view_type rowptrC, cin_nonzero_index_view_type colidxC,
    cin_nonzero_value_view_type valuesC) {
  using index_type  = typename KernelHandle::nnz_lno_t;
  using size_type   = typename KernelHandle::size_type;
  using scalar_type = typename KernelHandle::nnz_scalar_t;
  using rocsparse_scalar_type =
      typename kokkos_to_rocsparse_type<scalar_type>::type;

  typename KernelHandle::rocSparseSpgemmHandleType *h =
      handle->get_rocsparse_spgemm_handle();

  const auto alpha = Kokkos::ArithTraits<scalar_type>::one();
  const auto beta  = Kokkos::ArithTraits<scalar_type>::zero();
  rocsparse_pointer_mode oldPtrMode;

  auto nnz_A = colidxA.extent(0);
  auto nnz_B = colidxB.extent(0);
  auto nnz_C = colidxC.extent(0);

  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
      rocsparse_get_pointer_mode(h->rocsparseHandle, &oldPtrMode));
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_set_pointer_mode(
      h->rocsparseHandle, rocsparse_pointer_mode_host));

  if (!handle->are_entries_computed()) {
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_csrgemm_symbolic(
        h->rocsparseHandle, h->opA, h->opB, m, k, n, h->descr_A, nnz_A,
        rowptrA.data(), colidxA.data(), h->descr_B, nnz_B, rowptrB.data(),
        colidxB.data(), h->descr_D, 0, NULL, NULL, h->descr_C, nnz_C,
        rowptrC.data(), colidxC.data(), h->info_C, h->buffer));
    handle->set_computed_entries();
  }

  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_Xcsrgemm_numeric(
      h->rocsparseHandle, h->opA, h->opB, m, k, n,
      reinterpret_cast<const rocsparse_scalar_type *>(&alpha), h->descr_A,
      nnz_A, reinterpret_cast<const rocsparse_scalar_type *>(valuesA.data()),
      rowptrA.data(), colidxA.data(), h->descr_B, nnz_B,
      reinterpret_cast<const rocsparse_scalar_type *>(valuesB.data()),
      rowptrB.data(), colidxB.data(),
      reinterpret_cast<const rocsparse_scalar_type *>(&beta), h->descr_D, 0,
      NULL, NULL, NULL, h->descr_C, nnz_C,
      reinterpret_cast<rocsparse_scalar_type *>(valuesC.data()), rowptrC.data(),
      colidxC.data(), h->info_C, h->buffer));
  // Restore old pointer mode
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
      rocsparse_set_pointer_mode(h->rocsparseHandle, oldPtrMode));
  handle->set_call_numeric();
}

#define SPGEMM_NUMERIC_DECL_ROCSPARSE(SCALAR, TPL_AVAIL)                 \
  template <>                                                                  \
  struct SPGEMM_NUMERIC<                                                       \
      KokkosKernels::Experimental::KokkosKernelsHandle<                        \
          const int, const int, const SCALAR, Kokkos::HIP, Kokkos::HIPSpace,   \
          Kokkos::HIPSpace>,                                                   \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const SCALAR *, default_layout,                             \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const SCALAR *, default_layout,                             \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<int *, default_layout,                                      \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<SCALAR *, default_layout,                                   \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, TPL_AVAIL> {                                                 \
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<     \
        const int, const int, const SCALAR, Kokkos::HIP, Kokkos::HIPSpace,     \
        Kokkos::HIPSpace>;                                                     \
    using c_int_view_t =                                                       \
        Kokkos::View<const int *, default_layout,                              \
                     Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;                \
    using int_view_t =                                                         \
        Kokkos::View<int *, default_layout,                                    \
                     Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;                \
    using c_scalar_view_t =                                                    \
        Kokkos::View<const SCALAR *, default_layout,                           \
                     Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;                \
    using scalar_view_t =                                                      \
        Kokkos::View<SCALAR *, default_layout,                                 \
                     Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;                \
    static void spgemm_numeric(KernelHandle *handle,                           \
                               typename KernelHandle::nnz_lno_t m,             \
                               typename KernelHandle::nnz_lno_t n,             \
                               typename KernelHandle::nnz_lno_t k,             \
                               c_int_view_t row_mapA, c_int_view_t entriesA,   \
                               c_scalar_view_t valuesA, bool,                  \
                               c_int_view_t row_mapB, c_int_view_t entriesB,   \
                               c_scalar_view_t valuesB, bool,                  \
                               c_int_view_t row_mapC, int_view_t entriesC,     \
                               scalar_view_t valuesC) {                        \
      std::string label = "KokkosSparse::spgemm_numeric[TPL_ROCSPARSE," +              \
                          Kokkos::ArithTraits<SCALAR>::name() + "]";           \
      Kokkos::Profiling::pushRegion(label);                                    \
      spgemm_numeric_rocsparse(handle->get_spgemm_handle(), m, n, k, row_mapA, \
                               entriesA, valuesA, row_mapB, entriesB, valuesB, \
                               row_mapC, entriesC, valuesC);                   \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

SPGEMM_NUMERIC_DECL_ROCSPARSE(float, true)
SPGEMM_NUMERIC_DECL_ROCSPARSE(double, true)
SPGEMM_NUMERIC_DECL_ROCSPARSE(Kokkos::complex<float>, true)
SPGEMM_NUMERIC_DECL_ROCSPARSE(Kokkos::complex<double>, true)

SPGEMM_NUMERIC_DECL_ROCSPARSE(float, false)
SPGEMM_NUMERIC_DECL_ROCSPARSE(double, false)
SPGEMM_NUMERIC_DECL_ROCSPARSE(Kokkos::complex<float>, false)
SPGEMM_NUMERIC_DECL_ROCSPARSE(Kokkos::complex<double>, false)
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
template <
    typename KernelHandle, typename ain_row_index_view_type,
    typename ain_nonzero_index_view_type, typename ain_nonzero_value_view_type,
    typename bin_row_index_view_type, typename bin_nonzero_index_view_type,
    typename bin_nonzero_value_view_type, typename cin_row_index_view_type,
    typename cin_nonzero_index_view_type, typename cin_nonzero_value_view_type>
void spgemm_numeric_mkl(
    KernelHandle *handle, typename KernelHandle::nnz_lno_t m,
    typename KernelHandle::nnz_lno_t n, typename KernelHandle::nnz_lno_t k,
    ain_row_index_view_type rowptrA, ain_nonzero_index_view_type colidxA, ain_nonzero_value_view_type valuesA,
    bin_row_index_view_type rowptrB, bin_nonzero_index_view_type colidxB, bin_nonzero_value_view_type valuesB,
    cin_row_index_view_type rowptrC, cin_nonzero_index_view_type colidxC, cin_nonzero_value_view_type valuesC)
    {
  using ExecSpace = typename KernelHandle::HandleExecSpace;
  using index_type  = typename KernelHandle::nnz_lno_t;
  using size_type   = typename KernelHandle::size_type;
  using scalar_type = typename KernelHandle::nnz_scalar_t;
  using MKLMatrix = MKLSparseMatrix<scalar_type>;
  size_type c_nnz = handle->get_c_nnz();
  if(c_nnz == size_type(0))
  {
    handle->set_computed_entries();
    handle->set_call_numeric();
    return;
  }
  MKLMatrix A(m, n, const_cast<size_type*>(rowptrA.data()), const_cast<index_type*>(colidxA.data()), const_cast<scalar_type*>(valuesA.data()));
  MKLMatrix B(n, k, const_cast<size_type*>(rowptrB.data()), const_cast<index_type*>(colidxB.data()), const_cast<scalar_type*>(valuesB.data()));
  auto mklSpgemmHandle = handle->get_mkl_spgemm_handle();
  ;
  bool computedEntries = false;
  matrix_descr generalDescr;
  generalDescr.type = SPARSE_MATRIX_TYPE_GENERAL;
  generalDescr.mode = SPARSE_FILL_MODE_FULL;
  generalDescr.diag = SPARSE_DIAG_NON_UNIT;
  if(!handle->are_entries_computed()) {
    KOKKOSKERNELS_MKL_SAFE_CALL(
      mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, A, SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, B, SPARSE_STAGE_FINALIZE_MULT_NO_VAL, &mklSpgemmHandle->C));
    handle->set_computed_entries();
    computedEntries = true;
  }
  KOKKOSKERNELS_MKL_SAFE_CALL(
      mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, A, SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, B, SPARSE_STAGE_FINALIZE_MULT, &mklSpgemmHandle->C));
  MKLMatrix wrappedC(mklSpgemmHandle->C);
  MKL_INT nrows = 0, ncols = 0;
  MKL_INT* rowptrRaw = NULL;
  MKL_INT* colidxRaw = NULL;
  scalar_type* valuesRaw = NULL;
  wrappedC.export_data(nrows, ncols, rowptrRaw, colidxRaw, valuesRaw);
  Kokkos::View<index_type*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> colidxRawView(colidxRaw, c_nnz);
  Kokkos::View<scalar_type*, Kokkos::HostSpace, Kokkos::MemoryTraits<Kokkos::Unmanaged>> valuesRawView(valuesRaw, c_nnz);
  if(computedEntries)
    Kokkos::deep_copy(ExecSpace(), colidxC, colidxRawView);
  Kokkos::deep_copy(ExecSpace(), valuesC, valuesRawView);
  handle->set_call_numeric();
}

#define SPGEMM_NUMERIC_DECL_MKL(SCALAR, EXEC, TPL_AVAIL)               \
  template <>                                                                 \
  struct SPGEMM_NUMERIC<                                                     \
      KokkosKernels::Experimental::KokkosKernelsHandle<                       \
          const int, const int, const SCALAR, EXEC, Kokkos::HostSpace,  \
          Kokkos::HostSpace>,                                                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const SCALAR *, default_layout,                             \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const SCALAR *, default_layout,                             \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<int *, default_layout,                                      \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<SCALAR *, default_layout,                                   \
                   Kokkos::Device<EXEC, Kokkos::HostSpace>,              \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, TPL_AVAIL> {                                                \
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<    \
        const int, const int, const SCALAR, EXEC, Kokkos::HostSpace, Kokkos::HostSpace>;                                                    \
    using c_int_view_t =                                                      \
        Kokkos::View<const int *, default_layout,                             \
                     Kokkos::Device<EXEC, Kokkos::HostSpace>,           \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;               \
    using int_view_t =                                                        \
        Kokkos::View<int *, default_layout,                                   \
                     Kokkos::Device<EXEC, Kokkos::HostSpace>,           \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;               \
    using c_scalar_view_t =                                                    \
        Kokkos::View<const SCALAR *, default_layout,                           \
                     Kokkos::Device<EXEC, Kokkos::HostSpace>,            \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;                \
    using scalar_view_t =                                                      \
        Kokkos::View<SCALAR *, default_layout,                                 \
                     Kokkos::Device<EXEC, Kokkos::HostSpace>,            \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;                \
    static void spgemm_numeric(KernelHandle *handle,                           \
                               typename KernelHandle::nnz_lno_t m,             \
                               typename KernelHandle::nnz_lno_t n,             \
                               typename KernelHandle::nnz_lno_t k,             \
                               c_int_view_t row_mapA, c_int_view_t entriesA,   \
                               c_scalar_view_t valuesA, bool,                  \
                               c_int_view_t row_mapB, c_int_view_t entriesB,   \
                               c_scalar_view_t valuesB, bool,                  \
                               c_int_view_t row_mapC, int_view_t entriesC,     \
                               scalar_view_t valuesC) {                        \
      std::string label = "KokkosSparse::spgemm_numeric[TPL_MKL," +             \
                          Kokkos::ArithTraits<SCALAR>::name() + "]";          \
      Kokkos::Profiling::pushRegion(label);                                   \
      spgemm_numeric_mkl(handle->get_spgemm_handle(), m, n, k,         \
                                row_mapA, entriesA, valuesA, row_mapB, entriesB, valuesB, \
                                row_mapC, entriesC, valuesC);                 \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define SPGEMM_NUMERIC_DECL_MKL_SE(SCALAR, EXEC) \
SPGEMM_NUMERIC_DECL_MKL(SCALAR, EXEC, true) \
SPGEMM_NUMERIC_DECL_MKL(SCALAR, EXEC, false)

#define SPGEMM_NUMERIC_DECL_MKL_E(EXEC) \
SPGEMM_NUMERIC_DECL_MKL_SE(float, EXEC) \
SPGEMM_NUMERIC_DECL_MKL_SE(double, EXEC) \
SPGEMM_NUMERIC_DECL_MKL_SE(Kokkos::complex<float>, EXEC) \
SPGEMM_NUMERIC_DECL_MKL_SE(Kokkos::complex<double>, EXEC)

#ifdef KOKKOS_ENABLE_SERIAL
SPGEMM_NUMERIC_DECL_MKL_E(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
SPGEMM_NUMERIC_DECL_MKL_E(Kokkos::OpenMP)
#endif
#endif

}  // namespace Impl
}  // namespace KokkosSparse

#endif
