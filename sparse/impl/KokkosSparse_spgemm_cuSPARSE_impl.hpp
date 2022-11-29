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

#ifndef _KOKKOSSPGEMMCUSPARSE_HPP
#define _KOKKOSSPGEMMCUSPARSE_HPP

#include "KokkosKernels_Controls.hpp"
#include "KokkosSparse_Utils_cusparse.hpp"

namespace KokkosSparse {

namespace Impl {

template <
    typename KernelHandle, typename ain_row_index_view_type,
    typename ain_nonzero_index_view_type, typename bin_row_index_view_type,
    typename bin_nonzero_index_view_type, typename cin_row_index_view_type>
void cuSPARSE_symbolic(KernelHandle *handle, typename KernelHandle::nnz_lno_t m,
                       typename KernelHandle::nnz_lno_t n,
                       typename KernelHandle::nnz_lno_t k,
                       ain_row_index_view_type row_mapA,
                       ain_nonzero_index_view_type entriesA,

                       bool transposeA, bin_row_index_view_type row_mapB,
                       bin_nonzero_index_view_type entriesB, bool transposeB,
                       cin_row_index_view_type row_mapC) {
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
  using device1 = typename ain_row_index_view_type::device_type;
  using device2 = typename ain_nonzero_index_view_type::device_type;
  using idx     = typename KernelHandle::nnz_lno_t;

  // TODO this is not correct, check memory space.
  if (std::is_same<Kokkos::Cuda, device1>::value) {
    throw std::runtime_error(
        "MEMORY IS NOT ALLOCATED IN GPU DEVICE for CUSPARSE\n");
    // return;
  }
  if (std::is_same<Kokkos::Cuda, device2>::value) {
    throw std::runtime_error(
        "MEMORY IS NOT ALLOCATED IN GPU DEVICE for CUSPARSE\n");
    // return;
  }

  // CUDA_VERSION coming along with CUDAToolkit is easier to find than
  // CUSPARSE_VERSION
#if (CUDA_VERSION >= 11040)
  // Newest versions of cuSPARSE have the generic SpGEMM interface, with "reuse"
  // functions.
  if (!std::is_same<typename std::remove_cv<idx>::type, int>::value ||
      !std::is_same<
          typename std::remove_cv<typename KernelHandle::size_type>::type,
          int>::value) {
    throw std::runtime_error(
        "cusparseSpGEMMreuse requires local ordinals to be 32-bit integer.");
  }

  handle->set_sort_option(1);  // tells users the output is sorted
  handle->create_cusparse_spgemm_handle(transposeA, transposeB);
  typename KernelHandle::cuSparseSpgemmHandleType *h =
      handle->get_cusparse_spgemm_handle();

  // Follow
  // https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spgemm_reuse
  void *buffer1      = NULL;
  void *buffer2      = NULL;
  size_t bufferSize1 = 0;
  size_t bufferSize2 = 0;

  // When nnz is not zero, cusparseCreateCsr insists non-null a value pointer,
  // which however is not available in this function. So we fake it with the
  // entries instead. Fortunately, it seems cupsarse does not access that in the
  // symbolic phase.
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
      &h->descr_A, m, n, entriesA.extent(0), (void *)row_mapA.data(),
      (void *)entriesA.data(), (void *)entriesA.data() /*fake*/,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      h->scalarType));

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
      &h->descr_B, n, k, entriesB.extent(0), (void *)row_mapB.data(),
      (void *)entriesB.data(), (void *)entriesB.data() /*fake*/,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      h->scalarType));

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
      &h->descr_C, m, k, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, h->scalarType));

  //----------------------------------------------------------------------
  // ask bufferSize1 bytes for external memory
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMMreuse_workEstimation(
      h->cusparseHandle, h->opA, h->opB, h->descr_A, h->descr_B, h->descr_C,
      h->alg, h->spgemmDescr, &bufferSize1, NULL));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&buffer1, bufferSize1));
  // inspect matrices A and B to understand the memory requirement for the next
  // step
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMMreuse_workEstimation(
      h->cusparseHandle, h->opA, h->opB, h->descr_A, h->descr_B, h->descr_C,
      h->alg, h->spgemmDescr, &bufferSize1, buffer1));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(buffer1));

  //----------------------------------------------------------------------
  // Compute nnz of C
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMMreuse_nnz(
      h->cusparseHandle, h->opA, h->opB, h->descr_A, h->descr_B, h->descr_C,
      h->alg, h->spgemmDescr, &bufferSize2, NULL, &h->bufferSize3, NULL,
      &h->bufferSize4, NULL));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&buffer2, bufferSize2));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&h->buffer3, h->bufferSize3));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&h->buffer4, h->bufferSize4));

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMMreuse_nnz(
      h->cusparseHandle, h->opA, h->opB, h->descr_A, h->descr_B, h->descr_C,
      h->alg, h->spgemmDescr, &bufferSize2, buffer2, &h->bufferSize3,
      h->buffer3, &h->bufferSize4, h->buffer4));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(buffer2));

  int64_t C_nrow, C_ncol, C_nnz;
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseSpMatGetSize(h->descr_C, &C_nrow, &C_ncol, &C_nnz));
  if (C_nnz > std::numeric_limits<int>::max()) {
    throw std::runtime_error("nnz of C overflowed over 32-bit int\n");
  }
  handle->set_c_nnz(C_nnz);
  h->C_populated = false;  // sparsity pattern of C is not set yet
  (void)row_mapC;

#elif defined(CUSPARSE_VERSION) && (11000 <= CUSPARSE_VERSION)
  using scalar_type = typename KernelHandle::nnz_scalar_t;
  // cuSPARSE from CUDA 11.0-11.3 (inclusive) supports the new "generic" SpGEMM
  // interface, just not the "reuse" set of functions. This means compute must
  // be called in both symbolic and numeric (otherwise, the NNZ of C can't be
  // known by symbolic)
  if (!std::is_same<typename std::remove_cv<idx>::type, int>::value ||
      !std::is_same<
          typename std::remove_cv<typename KernelHandle::size_type>::type,
          int>::value) {
    throw std::runtime_error(
        "cusparseSpGEMM requires local ordinals to be 32-bit integer.");
  }

  handle->set_sort_option(1);  // tells users the output is sorted
  handle->create_cusparse_spgemm_handle(transposeA, transposeB);
  typename KernelHandle::cuSparseSpgemmHandleType *h =
      handle->get_cusparse_spgemm_handle();

  // Follow
  // https://github.com/NVIDIA/CUDALibrarySamples/blob/master/cuSPARSE/spgemm

  const auto alpha = Kokkos::ArithTraits<scalar_type>::one();
  const auto beta  = Kokkos::ArithTraits<scalar_type>::zero();

  // In non-reuse interface, forced to give A,B dummy values to
  // cusparseSpGEMM_compute. And it actually reads them, so they must be
  // allocated and of the correct type. This compute will be called again in
  // numeric with the real values.
  //
  // The dummy values can be uninitialized. cusparseSpGEMM_compute does
  // not remove numerical zeros from the sparsity pattern.
  void *dummyValues;
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc(
      &dummyValues,
      sizeof(scalar_type) * std::max(entriesA.extent(0), entriesB.extent(0))));

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
      &h->descr_A, m, n, entriesA.extent(0), (void *)row_mapA.data(),
      (void *)entriesA.data(), dummyValues, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, h->scalarType));

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
      &h->descr_B, n, k, entriesB.extent(0), (void *)row_mapB.data(),
      (void *)entriesB.data(), dummyValues, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, h->scalarType));

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
      &h->descr_C, m, k, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, h->scalarType));

  //----------------------------------------------------------------------
  // query workEstimation buffer size, allocate, then call again with buffer.
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_workEstimation(
      h->cusparseHandle, h->opA, h->opB, &alpha, h->descr_A, h->descr_B, &beta,
      h->descr_C, h->scalarType, h->alg, h->spgemmDescr, &h->bufferSize3,
      NULL));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&h->buffer3, h->bufferSize3));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_workEstimation(
      h->cusparseHandle, h->opA, h->opB, &alpha, h->descr_A, h->descr_B, &beta,
      h->descr_C, h->scalarType, h->alg, h->spgemmDescr, &h->bufferSize3,
      h->buffer3));

  //----------------------------------------------------------------------
  // query compute buffer size, allocate, then call again with buffer.

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_compute(
      h->cusparseHandle, h->opA, h->opB, &alpha, h->descr_A, h->descr_B, &beta,
      h->descr_C, h->scalarType, CUSPARSE_SPGEMM_DEFAULT, h->spgemmDescr,
      &h->bufferSize4, NULL));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&h->buffer4, h->bufferSize4));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_compute(
      h->cusparseHandle, h->opA, h->opB, &alpha, h->descr_A, h->descr_B, &beta,
      h->descr_C, h->scalarType, CUSPARSE_SPGEMM_DEFAULT, h->spgemmDescr,
      &h->bufferSize4, h->buffer4));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(dummyValues));

  int64_t C_nrow, C_ncol, C_nnz;
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseSpMatGetSize(h->descr_C, &C_nrow, &C_ncol, &C_nnz));
  if (C_nnz > std::numeric_limits<int>::max()) {
    throw std::runtime_error("nnz of C overflowed over 32-bit int\n");
  }
  handle->set_c_nnz(C_nnz);
  h->C_populated = false;  // sparsity pattern of C is not set yet
  (void)row_mapC;

#else

  if (std::is_same<idx, int>::value &&
      std::is_same<typename KernelHandle::size_type, int>::value) {
    const idx *a_xadj = (const idx *)row_mapA.data();
    const idx *b_xadj = (const idx *)row_mapB.data();
    idx *c_xadj       = (idx *)row_mapC.data();

    const idx *a_adj = entriesA.data();
    const idx *b_adj = entriesB.data();
    handle->create_cusparse_spgemm_handle(transposeA, transposeB);
    typename KernelHandle::cuSparseSpgemmHandleType *h =
        handle->get_cusparse_spgemm_handle();

    int nnzA = entriesA.extent(0);
    int nnzB = entriesB.extent(0);

    int baseC, nnzC;
    int *nnzTotalDevHostPtr = &nnzC;

    handle->set_sort_option(1);  // tells users the output is sorted
    cusparseXcsrgemmNnz(h->handle, h->transA, h->transB, (int)m, (int)n, (int)k,
                        h->a_descr, nnzA, (int *)a_xadj, (int *)a_adj,
                        h->b_descr, nnzB, (int *)b_xadj, (int *)b_adj,
                        h->c_descr, (int *)c_xadj, nnzTotalDevHostPtr);

    if (NULL != nnzTotalDevHostPtr) {
      nnzC = *nnzTotalDevHostPtr;
    } else {
      cudaMemcpy(&nnzC, c_xadj + m, sizeof(int), cudaMemcpyDeviceToHost);
      cudaMemcpy(&baseC, c_xadj, sizeof(int), cudaMemcpyDeviceToHost);
      nnzC -= baseC;
    }
    handle->set_c_nnz(nnzC);
    // entriesC =
    // cin_nonzero_index_view_type(Kokkos::view_alloc(Kokkos::WithoutInitializing,
    // "entriesC"), nnzC);
  } else {
    throw std::runtime_error(
        "CUSPARSE requires local ordinals to be integer.\n");
    // return;
  }
#endif
#else
  (void)handle;
  (void)m;
  (void)n;
  (void)k;
  (void)row_mapA;
  (void)row_mapB;
  (void)row_mapC;
  (void)entriesA;
  (void)entriesB;
  (void)transposeA;
  (void)transposeB;
  throw std::runtime_error("CUSPARSE IS NOT DEFINED\n");
  // return;
#endif
}

template <
    typename KernelHandle, typename ain_row_index_view_type,
    typename ain_nonzero_index_view_type, typename ain_nonzero_value_view_type,
    typename bin_row_index_view_type, typename bin_nonzero_index_view_type,
    typename bin_nonzero_value_view_type, typename cin_row_index_view_type,
    typename cin_nonzero_index_view_type, typename cin_nonzero_value_view_type>
void cuSPARSE_apply(
    KernelHandle *handle, typename KernelHandle::nnz_lno_t m,
    typename KernelHandle::nnz_lno_t n, typename KernelHandle::nnz_lno_t k,
    ain_row_index_view_type row_mapA, ain_nonzero_index_view_type entriesA,
    ain_nonzero_value_view_type valuesA,

    bool /* transposeA */, bin_row_index_view_type row_mapB,
    bin_nonzero_index_view_type entriesB, bin_nonzero_value_view_type valuesB,
    bool /* transposeB */, cin_row_index_view_type row_mapC,
    cin_nonzero_index_view_type entriesC, cin_nonzero_value_view_type valuesC) {
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
  typedef typename KernelHandle::nnz_lno_t idx;

  typedef typename KernelHandle::nnz_scalar_t scalar_type;

  typedef typename ain_row_index_view_type::device_type device1;
  typedef typename ain_nonzero_index_view_type::device_type device2;
  typedef typename ain_nonzero_value_view_type::device_type device3;

  if (std::is_same<Kokkos::Cuda, device1>::value) {
    throw std::runtime_error(
        "MEMORY IS NOT ALLOCATED IN GPU DEVICE for CUSPARSE\n");
    // return;
  }
  if (std::is_same<Kokkos::Cuda, device2>::value) {
    throw std::runtime_error(
        "MEMORY IS NOT ALLOCATED IN GPU DEVICE for CUSPARSE\n");
    // return;
  }
  if (std::is_same<Kokkos::Cuda, device3>::value) {
    throw std::runtime_error(
        "MEMORY IS NOT ALLOCATED IN GPU DEVICE for CUSPARSE\n");
    // return;
  }
#if (CUDA_VERSION >= 11040)
  typename KernelHandle::cuSparseSpgemmHandleType *h =
      handle->get_cusparse_spgemm_handle();

  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_A, (void *)row_mapA.data(),
                             (void *)entriesA.data(), (void *)valuesA.data()));

  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_B, (void *)row_mapB.data(),
                             (void *)entriesB.data(), (void *)valuesB.data()));

  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_C, (void *)row_mapC.data(),
                             (void *)entriesC.data(), (void *)valuesC.data()));

  if (!h->C_populated) {
    cusparseSpGEMMreuse_copy(h->cusparseHandle, h->opA, h->opB, h->descr_A,
                             h->descr_B, h->descr_C, h->alg, h->spgemmDescr,
                             &h->bufferSize5, NULL);
    cudaMalloc((void **)&h->buffer5, h->bufferSize5);
    cusparseSpGEMMreuse_copy(h->cusparseHandle, h->opA, h->opB, h->descr_A,
                             h->descr_B, h->descr_C, h->alg, h->spgemmDescr,
                             &h->bufferSize5, h->buffer5);
    cudaFree(h->buffer3);
    h->buffer3     = NULL;
    h->C_populated = true;
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

  (void)m;
  (void)n;
  (void)k;

#elif (CUSPARSE_VERSION >= 11000)
  using scalar_type = typename KernelHandle::nnz_scalar_t;
  const auto alpha  = Kokkos::ArithTraits<scalar_type>::one();
  const auto beta   = Kokkos::ArithTraits<scalar_type>::zero();
  typename KernelHandle::cuSparseSpgemmHandleType *h =
      handle->get_cusparse_spgemm_handle();
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_A, (void *)row_mapA.data(),
                             (void *)entriesA.data(), (void *)valuesA.data()));
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_B, (void *)row_mapB.data(),
                             (void *)entriesB.data(), (void *)valuesB.data()));
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->descr_C, (void *)row_mapC.data(),
                             (void *)entriesC.data(), (void *)valuesC.data()));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_compute(
      h->cusparseHandle, h->opA, h->opB, &alpha, h->descr_A, h->descr_B, &beta,
      h->descr_C, h->scalarType, CUSPARSE_SPGEMM_DEFAULT, h->spgemmDescr,
      &h->bufferSize4, h->buffer4));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMM_copy(
      h->cusparseHandle, h->opA, h->opB, &alpha, h->descr_A, h->descr_B, &beta,
      h->descr_C, h->scalarType, CUSPARSE_SPGEMM_DEFAULT, h->spgemmDescr));
  (void)m;
  (void)n;
  (void)k;

#else

  if (std::is_same<idx, int>::value) {
    int *a_xadj = (int *)row_mapA.data();
    int *b_xadj = (int *)row_mapB.data();
    int *c_xadj = (int *)row_mapC.data();

    int *a_adj = (int *)entriesA.data();
    int *b_adj = (int *)entriesB.data();
    int *c_adj = (int *)entriesC.data();

    typename KernelHandle::cuSparseSpgemmHandleType *h =
        handle->get_cusparse_spgemm_handle();

    int nnzA = entriesA.extent(0);
    int nnzB = entriesB.extent(0);

    scalar_type *a_ew = (scalar_type *)valuesA.data();
    scalar_type *b_ew = (scalar_type *)valuesB.data();
    scalar_type *c_ew = (scalar_type *)valuesC.data();

    if (std::is_same<scalar_type, float>::value) {
      cusparseScsrgemm(h->handle, h->transA, h->transB, m, n, k, h->a_descr,
                       nnzA, (float *)a_ew, a_xadj, a_adj, h->b_descr, nnzB,
                       (float *)b_ew, b_xadj, b_adj, h->c_descr, (float *)c_ew,
                       c_xadj, c_adj);
    } else if (std::is_same<scalar_type, double>::value) {
      cusparseDcsrgemm(h->handle, h->transA, h->transB, m, n, k, h->a_descr,
                       nnzA, (double *)a_ew, a_xadj, a_adj, h->b_descr, nnzB,
                       (double *)b_ew, b_xadj, b_adj, h->c_descr,
                       (double *)c_ew, c_xadj, c_adj);
    } else {
      throw std::runtime_error(
          "CUSPARSE requires float or double values. cuComplex and "
          "cuDoubleComplex are not implemented yet.\n");
      // return;
    }

  } else {
    throw std::runtime_error(
        "CUSPARSE requires local ordinals to be integer.\n");
    // return;
  }
#endif
#else
  (void)handle;
  (void)m;
  (void)n;
  (void)k;
  (void)row_mapA;
  (void)row_mapB;
  (void)row_mapC;
  (void)entriesA;
  (void)entriesB;
  (void)entriesC;
  (void)valuesA;
  (void)valuesB;
  (void)valuesC;
  throw std::runtime_error("CUSPARSE IS NOT DEFINED\n");
  // return;
#endif
}
}  // namespace Impl
}  // namespace KokkosSparse

#endif
