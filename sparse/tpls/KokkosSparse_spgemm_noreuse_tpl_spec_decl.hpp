/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 4.0
//       Copyright (2022) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Part of Kokkos, under the Apache License v2.0 with LLVM Exceptions.
// See https://kokkos.org/LICENSE for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//@HEADER
*/

#ifndef KOKKOSPARSE_SPGEMM_NOREUSE_TPL_SPEC_DECL_HPP_
#define KOKKOSPARSE_SPGEMM_NOREUSE_TPL_SPEC_DECL_HPP_

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

#if (CUDA_VERSION >= 11000)

template <typename Matrix, typename MatrixConst>
Matrix spgemm_noreuse_cusparse(const MatrixConst &A, const MatrixConst &B) {
  using Scalar                = typename Matrix::value_type;
  cudaDataType cudaScalarType = Impl::cuda_data_type_from<Scalar>();
  KokkosKernels::Experimental::Controls kkControls;
  cusparseHandle_t cusparseHandle = kkControls.getCusparseHandle();
  cusparseOperation_t op          = CUSPARSE_OPERATION_NON_TRANSPOSE;
  cusparseSpMatDescr_t descr_A, descr_B, descr_C;
  cusparseSpGEMMDescr_t spgemmDescr;
  cusparseSpGEMMAlg_t alg = CUSPARSE_SPGEMM_DEFAULT;
  size_t bufferSize1 = 0, bufferSize2 = 0;
  void *buffer1 = nullptr, *buffer2 = nullptr;
  // A is m*n, B is n*k, C is m*k
  int m            = A.numRows();
  int n            = B.numRows();
  int k            = B.numCols();
  const auto alpha = Kokkos::ArithTraits<Scalar>::one();
  const auto beta  = Kokkos::ArithTraits<Scalar>::zero();
  typename Matrix::row_map_type::non_const_type row_mapC(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "C rowmap"), m + 1);

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_createDescr(&spgemmDescr));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
      &descr_A, m, n, A.graph.entries.extent(0), (void *)A.graph.row_map.data(),
      (void *)A.graph.entries.data(), (void *)A.values.data(),
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      cudaScalarType));

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
      &descr_B, n, k, B.graph.entries.extent(0), (void *)B.graph.row_map.data(),
      (void *)B.graph.entries.data(), (void *)B.values.data(),
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      cudaScalarType));

  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCreateCsr(&descr_C, m, k, 0, (void *)row_mapC.data(), nullptr,
                        nullptr, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                        CUSPARSE_INDEX_BASE_ZERO, cudaScalarType));

  //----------------------------------------------------------------------
  // query workEstimation buffer size, allocate, then call again with buffer.
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_workEstimation(
      cusparseHandle, op, op, &alpha, descr_A, descr_B, &beta, descr_C,
      cudaScalarType, alg, spgemmDescr, &bufferSize1, nullptr));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&buffer1, bufferSize1));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_workEstimation(
      cusparseHandle, op, op, &alpha, descr_A, descr_B, &beta, descr_C,
      cudaScalarType, alg, spgemmDescr, &bufferSize1, buffer1));

  //----------------------------------------------------------------------
  // query compute buffer size, allocate, then call again with buffer.

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_compute(
      cusparseHandle, op, op, &alpha, descr_A, descr_B, &beta, descr_C,
      cudaScalarType, alg, spgemmDescr, &bufferSize2, nullptr));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&buffer2, bufferSize2));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_compute(
      cusparseHandle, op, op, &alpha, descr_A, descr_B, &beta, descr_C,
      cudaScalarType, alg, spgemmDescr, &bufferSize2, buffer2));
  int64_t unused1, unused2, c_nnz;
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseSpMatGetSize(descr_C, &unused1, &unused2, &c_nnz));

  typename Matrix::index_type entriesC(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "C entries"), c_nnz);
  typename Matrix::values_type valuesC(
      Kokkos::view_alloc(Kokkos::WithoutInitializing, "C values"), c_nnz);

  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(descr_C, (void *)row_mapC.data(),
                             (void *)entriesC.data(), (void *)valuesC.data()));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_compute(
      cusparseHandle, op, op, &alpha, descr_A, descr_B, &beta, descr_C,
      cudaScalarType, alg, spgemmDescr, &bufferSize2, buffer2));
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseSpGEMM_copy(cusparseHandle, op, op, &alpha, descr_A, descr_B,
                          &beta, descr_C, cudaScalarType, alg, spgemmDescr));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroySpMat(descr_A));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroySpMat(descr_B));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroySpMat(descr_C));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_destroyDescr(spgemmDescr));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(buffer1));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(buffer2));
  return Matrix("C", m, k, c_nnz, valuesC, row_mapC, entriesC);
}

#else

/*
// Generic (using overloads) wrapper for cusparseXcsrgemm (where X is S, D, C,
// or Z). Accepts Kokkos types (e.g. Kokkos::complex<float>) for Scalar and
// handles casting to cuSparse types internally.

#define CUSPARSE_XCSRGEMM_SPEC(KokkosType, CusparseType, Abbreviation)     \
  inline cusparseStatus_t cusparseXcsrgemm(                                \
      cusparseHandle_t handle, cusparseOperation_t transA,                 \
      cusparseOperation_t transB, int m, int n, int k,                     \
      const cusparseMatDescr_t descrA, const int nnzA,                     \
      const KokkosType *csrSortedValA, const int *csrSortedRowPtrA,        \
      const int *csrSortedColIndA, const cusparseMatDescr_t descrB,        \
      const int nnzB, const KokkosType *csrSortedValB,                     \
      const int *csrSortedRowPtrB, const int *csrSortedColIndB,            \
      const cusparseMatDescr_t descrC, KokkosType *csrSortedValC,          \
      const int *csrSortedRowPtrC, int *csrSortedColIndC) {                \
    return cusparse##Abbreviation##csrgemm(                                \
        handle, transA, transB, m, n, k, descrA, nnzA,                     \
        reinterpret_cast<const CusparseType *>(csrSortedValA),             \
        csrSortedRowPtrA, csrSortedColIndA, descrB, nnzB,                  \
        reinterpret_cast<const CusparseType *>(csrSortedValB),             \
        csrSortedRowPtrB, csrSortedColIndB, descrC,                        \
        reinterpret_cast<CusparseType *>(csrSortedValC), csrSortedRowPtrC, \
        csrSortedColIndC);                                                 \
  }

CUSPARSE_XCSRGEMM_SPEC(float, float, S)
CUSPARSE_XCSRGEMM_SPEC(double, double, D)
CUSPARSE_XCSRGEMM_SPEC(Kokkos::complex<float>, cuComplex, C)
CUSPARSE_XCSRGEMM_SPEC(Kokkos::complex<double>, cuDoubleComplex, Z)

#undef CUSPARSE_XCSRGEMM_SPEC

// 10.x supports the pre-generic interface.
template <typename KernelHandle, typename lno_t, typename ConstRowMapType,
          typename ConstEntriesType, typename ConstValuesType,
          typename EntriesType, typename ValuesType>
void spgemm_noreuse_cusparse(
    lno_t m, lno_t n, lno_t k,
    const ConstRowMapType &row_mapA, const ConstEntriesType &entriesA,
    const ConstValuesType &valuesA, const ConstRowMapType &row_mapB,
    const ConstEntriesType &entriesB, const ConstValuesType &valuesB,
    const RowMapType &row_mapC, EntriesType &entriesC, ValuesType &valuesC) {
  auto h = handle->get_cusparse_spgemm_handle();

  int nnzA = entriesA.extent(0);
  int nnzB = entriesB.extent(0);

  // Only call numeric if C actually has entries
  if (handle->get_c_nnz()) {
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseXcsrgemm(
        h->cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
        CUSPARSE_OPERATION_NON_TRANSPOSE, m, k, n, h->generalDescr, nnzA,
        valuesA.data(), row_mapA.data(), entriesA.data(), h->generalDescr, nnzB,
        valuesB.data(), row_mapB.data(), entriesB.data(), h->generalDescr,
        valuesC.data(), row_mapC.data(), entriesC.data()));
  }
  handle->set_computed_entries();
  handle->set_call_numeric();
}
*/

#endif

#define SPGEMM_NOREUSE_DECL_CUSPARSE(SCALAR, MEMSPACE, TPL_AVAIL)            \
  template <>                                                                \
  struct SPGEMM_NOREUSE<                                                     \
      KokkosSparse::CrsMatrix<                                               \
          SCALAR, int, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, void, int>,   \
      KokkosSparse::CrsMatrix<                                               \
          const SCALAR, const int, Kokkos::Device<Kokkos::Cuda, MEMSPACE>,   \
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, const int>,               \
      KokkosSparse::CrsMatrix<                                               \
          const SCALAR, const int, Kokkos::Device<Kokkos::Cuda, MEMSPACE>,   \
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, const int>,               \
      true, TPL_AVAIL> {                                                     \
    using Matrix = KokkosSparse::CrsMatrix<                                  \
        SCALAR, int, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, void, int>;     \
    using ConstMatrix = KokkosSparse::CrsMatrix<                             \
        const SCALAR, const int, Kokkos::Device<Kokkos::Cuda, MEMSPACE>,     \
        Kokkos::MemoryTraits<Kokkos::Unmanaged>, const int>;                 \
    static KokkosSparse::CrsMatrix<                                          \
        SCALAR, int, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, void, int>      \
    spgemm_noreuse(const ConstMatrix &A, bool, const ConstMatrix &B, bool) { \
      std::string label = "KokkosSparse::spgemm_noreuse[TPL_CUSPARSE," +     \
                          Kokkos::ArithTraits<SCALAR>::name() + "]";         \
      Kokkos::Profiling::pushRegion(label);                                  \
      Matrix C = spgemm_noreuse_cusparse<Matrix>(A, B);                      \
      Kokkos::Profiling::popRegion();                                        \
      return C;                                                              \
    }                                                                        \
  };

#define SPGEMM_NOREUSE_DECL_CUSPARSE_S(SCALAR, TPL_AVAIL)            \
  SPGEMM_NOREUSE_DECL_CUSPARSE(SCALAR, Kokkos::CudaSpace, TPL_AVAIL) \
  SPGEMM_NOREUSE_DECL_CUSPARSE(SCALAR, Kokkos::CudaUVMSpace, TPL_AVAIL)

SPGEMM_NOREUSE_DECL_CUSPARSE_S(float, true)
SPGEMM_NOREUSE_DECL_CUSPARSE_S(double, true)
SPGEMM_NOREUSE_DECL_CUSPARSE_S(Kokkos::complex<float>, true)
SPGEMM_NOREUSE_DECL_CUSPARSE_S(Kokkos::complex<double>, true)

SPGEMM_NOREUSE_DECL_CUSPARSE_S(float, false)
SPGEMM_NOREUSE_DECL_CUSPARSE_S(double, false)
SPGEMM_NOREUSE_DECL_CUSPARSE_S(Kokkos::complex<float>, false)
SPGEMM_NOREUSE_DECL_CUSPARSE_S(Kokkos::complex<double>, false)
#endif

/*
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
//=============================================================================
// Overload rocsparse_Xcsrgemm_numeric() over scalar types
#define ROCSPARSE_XCSRGEMM_NOREUSE_SPEC(scalar_type, TOKEN)                   \
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

ROCSPARSE_XCSRGEMM_NOREUSE_SPEC(float, s)
ROCSPARSE_XCSRGEMM_NOREUSE_SPEC(double, d)
ROCSPARSE_XCSRGEMM_NOREUSE_SPEC(rocsparse_float_complex, c)
ROCSPARSE_XCSRGEMM_NOREUSE_SPEC(rocsparse_double_complex, z)

template <
  typename KernelHandle, typename ain_row_index_view_type,
  typename ain_nonzero_index_view_type, typename ain_nonzero_value_view_type,
  typename bin_row_index_view_type, typename bin_nonzero_index_view_type,
  typename bin_nonzero_value_view_type, typename cin_row_index_view_type,
  typename cin_nonzero_index_view_type, typename cin_nonzero_value_view_type>
void spgemm_noreuse_rocsparse(
  KernelHandle *handle, typename KernelHandle::nnz_lno_t m,
  typename KernelHandle::nnz_lno_t n, typename KernelHandle::nnz_lno_t k,
  ain_row_index_view_type rowptrA, ain_nonzero_index_view_type colidxA,
  ain_nonzero_value_view_type valuesA, bin_row_index_view_type rowptrB,
  bin_nonzero_index_view_type colidxB, bin_nonzero_value_view_type valuesB,
  cin_row_index_view_type rowptrC, cin_nonzero_index_view_type colidxC,
  cin_nonzero_value_view_type valuesC) {
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
      colidxB.data(), h->descr_D, 0, nullptr, nullptr, h->descr_C, nnz_C,
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
    nullptr, nullptr, nullptr, h->descr_C, nnz_C,
    reinterpret_cast<rocsparse_scalar_type *>(valuesC.data()), rowptrC.data(),
    colidxC.data(), h->info_C, h->buffer));
// Restore old pointer mode
KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
    rocsparse_set_pointer_mode(h->rocsparseHandle, oldPtrMode));
handle->set_call_numeric();
}

#define SPGEMM_NOREUSE_DECL_ROCSPARSE(SCALAR, TPL_AVAIL)                       \
template <>                                                                  \
struct SPGEMM_NOREUSE<                                                       \
    KokkosKernels::Experimental::KokkosKernelsHandle<                        \
        const int, const int, const SCALAR, Kokkos::HIP, Kokkos::HIPSpace,   \
        Kokkos::HIPSpace>,                                                   \
    Kokkos::View<const int *, default_layout,                                \
                 Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
    Kokkos::View<const int *, default_layout,                                \
                 Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
    Kokkos::View<const SCALAR *, default_layout,                             \
                 Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
    Kokkos::View<const int *, default_layout,                                \
                 Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
    Kokkos::View<const int *, default_layout,                                \
                 Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
    Kokkos::View<const SCALAR *, default_layout,                             \
                 Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
    Kokkos::View<const int *, default_layout,                                \
                 Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
    Kokkos::View<int *, default_layout,                                      \
                 Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
    Kokkos::View<SCALAR *, default_layout,                                   \
                 Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,              \
                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                   \
    true, TPL_AVAIL> {                                                       \
  using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<     \
      const int, const int, const SCALAR, Kokkos::HIP, Kokkos::HIPSpace,     \
      Kokkos::HIPSpace>;                                                     \
  using c_int_view_t =                                                       \
      Kokkos::View<const int *, default_layout,                              \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
  using int_view_t =                                                         \
      Kokkos::View<int *, default_layout,                                    \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
  using c_scalar_view_t =                                                    \
      Kokkos::View<const SCALAR *, default_layout,                           \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
  using scalar_view_t =                                                      \
      Kokkos::View<SCALAR *, default_layout,                                 \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,            \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;                 \
  static void spgemm_noreuse(KernelHandle *handle,                           \
                             typename KernelHandle::nnz_lno_t m,             \
                             typename KernelHandle::nnz_lno_t n,             \
                             typename KernelHandle::nnz_lno_t k,             \
                             c_int_view_t row_mapA, c_int_view_t entriesA,   \
                             c_scalar_view_t valuesA, bool,                  \
                             c_int_view_t row_mapB, c_int_view_t entriesB,   \
                             c_scalar_view_t valuesB, bool,                  \
                             c_int_view_t row_mapC, int_view_t entriesC,     \
                             scalar_view_t valuesC) {                        \
    std::string label = "KokkosSparse::spgemm_noreuse[TPL_ROCSPARSE," +      \
                        Kokkos::ArithTraits<SCALAR>::name() + "]";           \
    Kokkos::Profiling::pushRegion(label);                                    \
    spgemm_noreuse_rocsparse(handle->get_spgemm_handle(), m, n, k, row_mapA, \
                             entriesA, valuesA, row_mapB, entriesB, valuesB, \
                             row_mapC, entriesC, valuesC);                   \
    Kokkos::Profiling::popRegion();                                          \
  }                                                                          \
};

SPGEMM_NOREUSE_DECL_ROCSPARSE(float, true)
SPGEMM_NOREUSE_DECL_ROCSPARSE(double, true)
SPGEMM_NOREUSE_DECL_ROCSPARSE(Kokkos::complex<float>, true)
SPGEMM_NOREUSE_DECL_ROCSPARSE(Kokkos::complex<double>, true)

SPGEMM_NOREUSE_DECL_ROCSPARSE(float, false)
SPGEMM_NOREUSE_DECL_ROCSPARSE(double, false)
SPGEMM_NOREUSE_DECL_ROCSPARSE(Kokkos::complex<float>, false)
SPGEMM_NOREUSE_DECL_ROCSPARSE(Kokkos::complex<double>, false)
#endif
*/

/*
#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
template <
  typename KernelHandle, typename ain_row_index_view_type,
  typename ain_nonzero_index_view_type, typename ain_nonzero_value_view_type,
  typename bin_row_index_view_type, typename bin_nonzero_index_view_type,
  typename bin_nonzero_value_view_type, typename cin_row_index_view_type,
  typename cin_nonzero_index_view_type, typename cin_nonzero_value_view_type>
void spgemm_noreuse_mkl(
  KernelHandle *handle, typename KernelHandle::nnz_lno_t m,
  typename KernelHandle::nnz_lno_t n, typename KernelHandle::nnz_lno_t k,
  ain_row_index_view_type rowptrA, ain_nonzero_index_view_type colidxA,
  ain_nonzero_value_view_type valuesA, bin_row_index_view_type rowptrB,
  bin_nonzero_index_view_type colidxB, bin_nonzero_value_view_type valuesB,
  cin_row_index_view_type rowptrC, cin_nonzero_index_view_type colidxC,
  cin_nonzero_value_view_type valuesC) {
using ExecSpace   = typename KernelHandle::HandleExecSpace;
using index_type  = typename KernelHandle::nnz_lno_t;
using size_type   = typename KernelHandle::size_type;
using scalar_type = typename KernelHandle::nnz_scalar_t;
using MKLMatrix   = MKLSparseMatrix<scalar_type>;
size_type c_nnz   = handle->get_c_nnz();
if (c_nnz == size_type(0)) {
  handle->set_computed_entries();
  handle->set_call_numeric();
  return;
}
MKLMatrix A(m, n, const_cast<size_type *>(rowptrA.data()),
            const_cast<index_type *>(colidxA.data()),
            const_cast<scalar_type *>(valuesA.data()));
MKLMatrix B(n, k, const_cast<size_type *>(rowptrB.data()),
            const_cast<index_type *>(colidxB.data()),
            const_cast<scalar_type *>(valuesB.data()));
auto mklSpgemmHandle = handle->get_mkl_spgemm_handle();
bool computedEntries = false;
matrix_descr generalDescr;
generalDescr.type = SPARSE_MATRIX_TYPE_GENERAL;
generalDescr.mode = SPARSE_FILL_MODE_FULL;
generalDescr.diag = SPARSE_DIAG_NON_UNIT;
KOKKOSKERNELS_MKL_SAFE_CALL(
    mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, A,
                    SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, B,
                    SPARSE_STAGE_FINALIZE_MULT_NO_VAL, &mklSpgemmHandle->C));
KOKKOSKERNELS_MKL_SAFE_CALL(
    mkl_sparse_sp2m(SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, A,
                    SPARSE_OPERATION_NON_TRANSPOSE, generalDescr, B,
                    SPARSE_STAGE_FINALIZE_MULT, &mklSpgemmHandle->C));
KOKKOSKERNELS_MKL_SAFE_CALL(mkl_sparse_order(mklSpgemmHandle->C));
MKLMatrix wrappedC(mklSpgemmHandle->C);
MKL_INT nrows = 0, ncols = 0;
MKL_INT *rowptrRaw     = nullptr;
MKL_INT *colidxRaw     = nullptr;
scalar_type *valuesRaw = nullptr;
wrappedC.export_data(nrows, ncols, rowptrRaw, colidxRaw, valuesRaw);
Kokkos::View<index_type *, Kokkos::HostSpace,
             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    colidxRawView(colidxRaw, c_nnz);
Kokkos::View<scalar_type *, Kokkos::HostSpace,
             Kokkos::MemoryTraits<Kokkos::Unmanaged>>
    valuesRawView(valuesRaw, c_nnz);
Kokkos::deep_copy(ExecSpace(), colidxC, colidxRawView);
Kokkos::deep_copy(ExecSpace(), valuesC, valuesRawView);
handle->set_call_numeric();
handle->set_computed_entries();
}

#define SPGEMM_NOREUSE_DECL_MKL(SCALAR, EXEC, TPL_AVAIL)                       \
template <>                                                                  \
struct SPGEMM_NOREUSE<KokkosKernels::Experimental::KokkosKernelsHandle<      \
                          const int, const int, const SCALAR, EXEC,          \
                          Kokkos::HostSpace, Kokkos::HostSpace>,             \
                      Kokkos::View<const int *, default_layout,              \
                                   Kokkos::Device<EXEC, Kokkos::HostSpace>,  \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
                      Kokkos::View<const int *, default_layout,              \
                                   Kokkos::Device<EXEC, Kokkos::HostSpace>,  \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
                      Kokkos::View<const SCALAR *, default_layout,           \
                                   Kokkos::Device<EXEC, Kokkos::HostSpace>,  \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
                      Kokkos::View<const int *, default_layout,              \
                                   Kokkos::Device<EXEC, Kokkos::HostSpace>,  \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
                      Kokkos::View<const int *, default_layout,              \
                                   Kokkos::Device<EXEC, Kokkos::HostSpace>,  \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
                      Kokkos::View<const SCALAR *, default_layout,           \
                                   Kokkos::Device<EXEC, Kokkos::HostSpace>,  \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
                      Kokkos::View<const int *, default_layout,              \
                                   Kokkos::Device<EXEC, Kokkos::HostSpace>,  \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
                      Kokkos::View<int *, default_layout,                    \
                                   Kokkos::Device<EXEC, Kokkos::HostSpace>,  \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
                      Kokkos::View<SCALAR *, default_layout,                 \
                                   Kokkos::Device<EXEC, Kokkos::HostSpace>,  \
                                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
                      true, TPL_AVAIL> {                                     \
  static void spgemm_noreuse(KernelHandle *handle,                           \
                             typename KernelHandle::nnz_lno_t m,             \
                             typename KernelHandle::nnz_lno_t n,             \
                             typename KernelHandle::nnz_lno_t k,             \
                             c_int_view_t row_mapA, c_int_view_t entriesA,   \
                             c_scalar_view_t valuesA, bool,                  \
                             c_int_view_t row_mapB, c_int_view_t entriesB,   \
                             c_scalar_view_t valuesB, bool,                  \
                             c_int_view_t row_mapC, int_view_t entriesC,     \
                             scalar_view_t valuesC) {                        \
    std::string label = "KokkosSparse::spgemm_noreuse[TPL_MKL," +            \
                        Kokkos::ArithTraits<SCALAR>::name() + "]";           \
    Kokkos::Profiling::pushRegion(label);                                    \
    spgemm_noreuse_mkl(handle->get_spgemm_handle(), m, n, k, row_mapA,       \
                       entriesA, valuesA, row_mapB, entriesB, valuesB,       \
                       row_mapC, entriesC, valuesC);                         \
    Kokkos::Profiling::popRegion();                                          \
  }                                                                          \
};

#define SPGEMM_NOREUSE_DECL_MKL_SE(SCALAR, EXEC) \
SPGEMM_NOREUSE_DECL_MKL(SCALAR, EXEC, true)    \
SPGEMM_NOREUSE_DECL_MKL(SCALAR, EXEC, false)

#define SPGEMM_NOREUSE_DECL_MKL_E(EXEC)                    \
SPGEMM_NOREUSE_DECL_MKL_SE(float, EXEC)                  \
SPGEMM_NOREUSE_DECL_MKL_SE(double, EXEC)                 \
SPGEMM_NOREUSE_DECL_MKL_SE(Kokkos::complex<float>, EXEC) \
SPGEMM_NOREUSE_DECL_MKL_SE(Kokkos::complex<double>, EXEC)

#ifdef KOKKOS_ENABLE_SERIAL
SPGEMM_NOREUSE_DECL_MKL_E(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
SPGEMM_NOREUSE_DECL_MKL_E(Kokkos::OpenMP)
#endif
#endif
*/

}  // namespace Impl
}  // namespace KokkosSparse

#endif
