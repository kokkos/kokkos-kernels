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

#ifndef _KOKKOSSPGEMMCUSPARSE_HPP
#define _KOKKOSSPGEMMCUSPARSE_HPP

//#define KOKKOSKERNELS_ENABLE_TPL_CUSPARSE

#include "KokkosKernels_Controls.hpp"
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include "cusparse.h"
#endif
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
  using device1     = typename ain_row_index_view_type::device_type;
  using device2     = typename ain_nonzero_index_view_type::device_type;
  using idx         = typename KernelHandle::nnz_lno_t;
  using size_type   = typename KernelHandle::size_type;
  using scalar_type = typename KernelHandle::nnz_scalar_t;

  // In case the KernelHandle uses const types!
  using non_const_idx       = typename std::remove_cv<idx>::type;
  using non_const_size_type = typename std::remove_cv<size_type>::type;

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

#if (CUSPARSE_VERSION >= 11400)
  if (!std::is_same<non_const_idx, int>::value ||
      !std::is_same<non_const_size_type, int>::value) {
    throw std::runtime_error(
        "cusparseSpGEMMreuse requires local ordinals to be 32-bit integer.");
  }

  handle->create_cuSPARSE_Handle(transposeA, transposeB);
  typename KernelHandle::SPGEMMcuSparseHandleType *h =
      handle->get_cuSparseHandle();

  // Follow
  // https://github.com/NVIDIA/CUDALibrarySamples/tree/master/cuSPARSE/spgemm_reuse
  void *dBuffer1     = NULL;
  void *dBuffer2     = NULL;
  size_t bufferSize1 = 0;
  size_t bufferSize2 = 0;

  // When nnz is not zero, cusparseCreateCsr insists non-null a value pointer,
  // which however is not available in this function. So we fake it with the
  // entries instead. Fortunately, it seems cupsarse does not access that in the
  // symbolic phase.
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
      &h->A_descr, m, n, entriesA.extent(0), (void *)row_mapA.data(),
      (void *)entriesA.data(), (void *)entriesA.data() /*fake*/,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      h->scalarType));

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
      &h->B_descr, n, k, entriesB.extent(0), (void *)row_mapB.data(),
      (void *)entriesB.data(), (void *)entriesB.data() /*fake*/,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
      h->scalarType));

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
      &h->C_descr, m, k, 0, NULL, NULL, NULL, CUSPARSE_INDEX_32I,
      CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO, h->scalarType));

  //----------------------------------------------------------------------
  // ask bufferSize1 bytes for external memory
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMMreuse_workEstimation(
      h->handle, h->opA, h->opB, h->A_descr, h->B_descr, h->C_descr, h->alg,
      h->spgemmDescr, &bufferSize1, NULL));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&dBuffer1, bufferSize1));
  // inspect matrices A and B to understand the memory requirement for the next
  // step
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMMreuse_workEstimation(
      h->handle, h->opA, h->opB, h->A_descr, h->B_descr, h->C_descr, h->alg,
      h->spgemmDescr, &bufferSize1, dBuffer1));

  //----------------------------------------------------------------------
  // Compute nnz of C
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMMreuse_nnz(
      h->handle, h->opA, h->opB, h->A_descr, h->B_descr, h->C_descr, h->alg,
      h->spgemmDescr, &bufferSize2, NULL, &h->bufferSize3, NULL,
      &h->bufferSize4, NULL));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&dBuffer2, bufferSize2));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&h->dBuffer3, h->bufferSize3));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&h->dBuffer4, h->bufferSize4));

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMMreuse_nnz(
      h->handle, h->opA, h->opB, h->A_descr, h->B_descr, h->C_descr, h->alg,
      h->spgemmDescr, &bufferSize2, dBuffer2, &h->bufferSize3, h->dBuffer3,
      &h->bufferSize4, h->dBuffer4));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(dBuffer1));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(dBuffer2));

  int64_t C_nrow, C_ncol, C_nnz;
  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseSpMatGetSize(h->C_descr, &C_nrow, &C_ncol, &C_nnz));
  if (C_nnz > std::numeric_limits<int>::max()) {
    throw std::runtime_error("nnz of C overflowed over 32-bit int\n");
  }
  handle->set_c_nnz(C_nnz);
  h->C_populated = false;  // sparsity pattern of C is not set yet

#elif defined(CUSPARSE_VERSION) && (11000 <= CUSPARSE_VERSION)
  throw std::runtime_error(
      "SpGEMM cuSPARSE backend is not yet supported for this CUDA version\n");
#else

  if (std::is_same<idx, int>::value && std::is_same<size_type, int>::value) {
    const idx *a_xadj = (const idx *)row_mapA.data();
    const idx *b_xadj = (const idx *)row_mapB.data();
    idx *c_xadj       = (idx *)row_mapC.data();

    const idx *a_adj = entriesA.data();
    const idx *b_adj = entriesB.data();
    handle->create_cuSPARSE_Handle(transposeA, transposeB);
    typename KernelHandle::SPGEMMcuSparseHandleType *h =
        handle->get_cuSparseHandle();

    int nnzA = entriesA.extent(0);
    int nnzB = entriesB.extent(0);

    int baseC, nnzC;
    int *nnzTotalDevHostPtr = &nnzC;

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

  typedef typename KernelHandle::nnz_scalar_t value_type;

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
#if (CUSPARSE_VERSION >= 11400)
  typename KernelHandle::SPGEMMcuSparseHandleType *h =
      handle->get_cuSparseHandle();

  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->A_descr, (void *)row_mapA.data(),
                             (void *)entriesA.data(), (void *)valuesA.data()));

  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->B_descr, (void *)row_mapB.data(),
                             (void *)entriesB.data(), (void *)valuesB.data()));

  KOKKOS_CUSPARSE_SAFE_CALL(
      cusparseCsrSetPointers(h->C_descr, (void *)row_mapC.data(),
                             (void *)entriesC.data(), (void *)valuesC.data()));

  if (!h->C_populated) {
    cusparseSpGEMMreuse_copy(h->handle, h->opA, h->opB, h->A_descr, h->B_descr,
                             h->C_descr, h->alg, h->spgemmDescr,
                             &h->bufferSize5, NULL);
    cudaMalloc((void **)&h->dBuffer5, h->bufferSize5);
    cusparseSpGEMMreuse_copy(h->handle, h->opA, h->opB, h->A_descr, h->B_descr,
                             h->C_descr, h->alg, h->spgemmDescr,
                             &h->bufferSize5, h->dBuffer5);
    cudaFree(h->dBuffer3);
    h->dBuffer3    = NULL;
    h->C_populated = true;
  }

  // C' = alpha * opA(A) * opB(B) + beta * C
  const auto alpha = Kokkos::ArithTraits<value_type>::one();
  const auto beta  = Kokkos::ArithTraits<value_type>::zero();
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpGEMMreuse_compute(
      h->handle, h->opA, h->opB, &alpha, h->A_descr, h->B_descr, &beta,
      h->C_descr, h->scalarType, h->alg, h->spgemmDescr));

#elif (CUSPARSE_VERSION >= 11000)
  throw std::runtime_error(
      "SpGEMM cuSPARSE backend is not yet supported for this CUDA version\n");
#else
  if (std::is_same<idx, int>::value) {
    int *a_xadj = (int *)row_mapA.data();
    int *b_xadj = (int *)row_mapB.data();
    int *c_xadj = (int *)row_mapC.data();

    int *a_adj = (int *)entriesA.data();
    int *b_adj = (int *)entriesB.data();
    int *c_adj = (int *)entriesC.data();

    typename KernelHandle::SPGEMMcuSparseHandleType *h =
        handle->get_cuSparseHandle();

    int nnzA = entriesA.extent(0);
    int nnzB = entriesB.extent(0);

    value_type *a_ew = (value_type *)valuesA.data();
    value_type *b_ew = (value_type *)valuesB.data();
    value_type *c_ew = (value_type *)valuesC.data();

    if (std::is_same<value_type, float>::value) {
      cusparseScsrgemm(h->handle, h->transA, h->transB, m, n, k, h->a_descr,
                       nnzA, (float *)a_ew, a_xadj, a_adj, h->b_descr, nnzB,
                       (float *)b_ew, b_xadj, b_adj, h->c_descr, (float *)c_ew,
                       c_xadj, c_adj);
    } else if (std::is_same<value_type, double>::value) {
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
