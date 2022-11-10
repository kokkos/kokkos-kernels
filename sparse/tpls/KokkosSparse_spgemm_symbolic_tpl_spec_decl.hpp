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

#ifndef KOKKOSPARSE_SPGEMM_SYMBOLIC_TPL_SPEC_DECL_HPP_
#define KOKKOSPARSE_SPGEMM_SYMBOLIC_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include "cusparse.h"
#include "KokkosSparse_Utils_cusparse.hpp"
#endif

namespace KokkosSparse {
namespace Impl {

//=====================
//  SpGEMM Symbolic
//=====================

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
// NOTE: all versions of cuSPARSE 10.x and 11.x support exactly the same matrix
// types, so there is no ifdef'ing on versions needed in avail. Offset and
// Ordinal must both be 32-bit. Even though the "generic" API lets you specify
// offsets and ordinals independently as either 16, 32 or 64-bit integers,
// cusparse will just fail at runtime if you don't use 32 for both.

#if (CUDA_VERSION >= 11040)
// 11.4+ supports generic API with reuse (full symbolic/numeric separation)
template <typename KernelHandle, typename lno_t, typename ConstRowMapType,
          typename ConstEntriesType, typename RowMapType>
void spgemm_symbolic_cusparse(KernelHandle *handle, lno_t m, lno_t n, lno_t k,
                              const ConstRowMapType &row_mapA,
                              const ConstEntriesType &entriesA,
                              const ConstRowMapType &row_mapB,
                              const ConstEntriesType &entriesB,
                              const RowMapType &row_mapC) {
  auto sh = handle->get_spgemm_handle();
  sh->set_sort_option(1);  // tells users the output is sorted
  sh->create_cusparse_spgemm_handle(false, false);
  auto h = sh->get_cusparse_spgemm_handle();

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
  sh->set_c_nnz(C_nnz);
  h->C_populated = false;  // sparsity pattern of C is not set yet
  sh->set_call_symbolic();
}

#elif (CUDA_VERSION >= 11000)
// 11.0-11.3 supports only the generic API, but not reuse.
template <typename KernelHandle, typename lno_t, typename ConstRowMapType,
          typename ConstEntriesType, typename RowMapType>
void spgemm_symbolic_cusparse(KernelHandle *handle, lno_t m, lno_t n, lno_t k,
                              const ConstRowMapType &row_mapA,
                              const ConstEntriesType &entriesA,
                              const ConstRowMapType &row_mapB,
                              const ConstEntriesType &entriesB,
                              const RowMapType &row_mapC) {
  auto sh = handle->get_spgemm_handle();
  sh->set_sort_option(1);  // tells users the output is sorted
  sh->create_cusparse_spgemm_handle(false, false);
  auto h = sh->get_cusparse_spgemm_handle();

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
  sh->set_c_nnz(C_nnz);
  h->C_populated = false;  // sparsity pattern of C is not set yet
  sh->set_call_symbolic();
}

#else
// 10.x supports the pre-generic interface.
template <typename KernelHandle, typename lno_t, typename ConstRowMapType,
          typename ConstEntriesType, typename RowMapType>
void spgemm_symbolic_cusparse(KernelHandle* handle, lno_t m, lno_t n, lno_t k,
                              const ConstRowMapType& row_mapA,
                              const ConstEntriesType& entriesA,
                              const ConstRowMapType& row_mapB,
                              const ConstEntriesType& entriesB,
                              const RowMapType& row_mapC) {
  auto sh = handle->get_spgemm_handle();
  sh->create_cusparse_spgemm_handle(false, false);
  auto h = sh->get_cusparse_spgemm_handle();

  int nnzA = entriesA.extent(0);
  int nnzB = entriesB.extent(0);

  int baseC, nnzC;
  int* nnzTotalDevHostPtr = &nnzC;

  handle->set_sort_option(1);  // tells users the output is sorted
  cusparseXcsrgemmNnz(h->handle, h->transA, h->transB, (int)m, (int)n, (int)k,
                      h->a_descr, nnzA, row_mapA.data(), entriesA.data(),
                      h->b_descr, nnzB, row_mapB.data(), entriesB.data(),
                      h->c_descr, row_mapC.data(), nnzTotalDevHostPtr);

  if (NULL != nnzTotalDevHostPtr) {
    nnzC = *nnzTotalDevHostPtr;
  } else {
    cudaMemcpy(&nnzC, c_xadj + m, sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&baseC, c_xadj, sizeof(int), cudaMemcpyDeviceToHost);
    nnzC -= baseC;
  }
  sh->set_c_nnz(nnzC);
  sh->set_call_symbolic();
}

#endif

#define SPGEMM_SYMBOLIC_DECL_CUSPARSE(SCALAR, MEMSPACE)                        \
  template <>                                                                  \
  struct SPGEMM_SYMBOLIC<                                                      \
      KokkosKernels::Experimental::KokkosKernelsHandle<                        \
          const int, const int, const SCALAR, Kokkos::Cuda, MEMSPACE,          \
          MEMSPACE>,                                                           \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const int *, default_layout,                                \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<int *, default_layout,                                      \
                   Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY> {                              \
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<     \
        const int, const int, const SCALAR, Kokkos::Cuda, MEMSPACE, MEMSPACE>; \
    using c_int_view_t =                                                       \
        Kokkos::View<const int *, default_layout,                              \
                     Kokkos::Device<Kokkos::Cuda, MEMSPACE>,                   \
                     Kokkos::MemoryTraits<Kokkos::Unmanaged> >;                \
    using int_view_t = Kokkos::View<int *, default_layout,                     \
                                    Kokkos::Device<Kokkos::Cuda, MEMSPACE>,    \
                                    Kokkos::MemoryTraits<Kokkos::Unmanaged> >; \
    static void spgemm_symbolic(KernelHandle *handle,                          \
                                typename KernelHandle::nnz_lno_t m,            \
                                typename KernelHandle::nnz_lno_t n,            \
                                typename KernelHandle::nnz_lno_t k,            \
                                c_int_view_t row_mapA, c_int_view_t entriesA,  \
                                bool, c_int_view_t row_mapB,                   \
                                c_int_view_t entriesB, bool,                   \
                                int_view_t row_mapC) {                         \
      std::string label = "KokkosSparse::spgemm[TPL_CUSPARSE," +               \
                          Kokkos::ArithTraits<SCALAR>::name() + "]";           \
      Kokkos::Profiling::pushRegion(label);                                    \
      spgemm_symbolic_cusparse(handle, m, n, k, row_mapA, entriesA, row_mapB,  \
                               entriesB, row_mapC);                            \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define SPGEMM_SYMBOLIC_DECL_CUSPARSE_S(SCALAR)            \
  SPGEMM_SYMBOLIC_DECL_CUSPARSE(SCALAR, Kokkos::CudaSpace) \
  SPGEMM_SYMBOLIC_DECL_CUSPARSE(SCALAR, Kokkos::CudaUVMSpace)

SPGEMM_SYMBOLIC_DECL_CUSPARSE_S(float)
SPGEMM_SYMBOLIC_DECL_CUSPARSE_S(double)
SPGEMM_SYMBOLIC_DECL_CUSPARSE_S(Kokkos::complex<float>)
SPGEMM_SYMBOLIC_DECL_CUSPARSE_S(Kokkos::complex<double>)

#endif
}  // namespace Impl
}  // namespace KokkosSparse

#endif
