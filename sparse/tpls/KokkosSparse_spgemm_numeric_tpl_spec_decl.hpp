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

#define SPGEMM_NUMERIC_DECL_CUSPARSE(SCALAR, MEMSPACE, COMPILE_LIBRARY)        \
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
      true, COMPILE_LIBRARY> {                                                 \
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
      std::string label = "KokkosSparse::spgemm[TPL_CUSPARSE," +               \
                          Kokkos::ArithTraits<SCALAR>::name() + "]";           \
      Kokkos::Profiling::pushRegion(label);                                    \
      spgemm_numeric_cusparse(handle, m, n, k, row_mapA, entriesA, valuesA,    \
                              row_mapB, entriesB, valuesB, row_mapC, entriesC, \
                              valuesC);                                        \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define SPGEMM_NUMERIC_DECL_CUSPARSE_S(SCALAR, COMPILE_LIBRARY)            \
  SPGEMM_NUMERIC_DECL_CUSPARSE(SCALAR, Kokkos::CudaSpace, COMPILE_LIBRARY) \
  SPGEMM_NUMERIC_DECL_CUSPARSE(SCALAR, Kokkos::CudaUVMSpace, COMPILE_LIBRARY)

SPGEMM_NUMERIC_DECL_CUSPARSE_S(float, true)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(double, true)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(Kokkos::complex<float>, true)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(Kokkos::complex<double>, true)

SPGEMM_NUMERIC_DECL_CUSPARSE_S(float, false)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(double, false)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(Kokkos::complex<float>, false)
SPGEMM_NUMERIC_DECL_CUSPARSE_S(Kokkos::complex<double>, false)
#endif

}  // namespace Impl
}  // namespace KokkosSparse

#endif
