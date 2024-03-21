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

#ifndef KOKKOSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_DECL_HPP_
#define KOKKOSPARSE_SPTRSV_SYMBOLIC_TPL_SPEC_DECL_HPP_


// cuSPARSE
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include "cusparse.h"
#include "KokkosSparse_Utils_cusparse.hpp"
#include "KokkosKernels_Handle.hpp"

namespace KokkosSparse {
namespace Impl {

template <class ExecutionSpace, class KernelHandle, class ain_row_index_view_type,
	  class ain_nonzero_index_view_type, class ain_values_scalar_view_type>
void sptrsv_analysis_cusparse(ExecutionSpace &space, KernelHandle *sptrsv_handle,
		     ain_row_index_view_type row_map,
		     ain_nonzero_index_view_type entries,
		     ain_values_scalar_view_type values, const bool trans) {
  using idx_type     = typename KernelHandle::nnz_lno_t;
  using size_type    = typename KernelHandle::size_type;
  using scalar_type  = typename KernelHandle::scalar_t;
  using memory_space = typename KernelHandle::memory_space;

#if (CUDA_VERSION >= 11030)
  using nnz_scalar_view_t = typename KernelHandle::nnz_scalar_view_t;
  using KAT               = Kokkos::ArithTraits<scalar_type>;

  const bool is_lower = sptrsv_handle->is_lower_tri();
  sptrsv_handle->create_cuSPARSE_Handle(trans, is_lower);

  const idx_type nrows = sptrsv_handle->get_nrows();
  typename KernelHandle::SPTRSVcuSparseHandleType *h =
    sptrsv_handle->get_cuSparseHandle();

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSetStream(h->handle, space.cuda_stream()));

  int64_t nnz = static_cast<int64_t>(entries.extent(0));
  size_t pBufferSize;
  const scalar_type alpha = KAT::one();

  cusparseIndexType_t cudaCsrRowMapType =
    cusparse_index_type_t_from<idx_type>();
  cusparseIndexType_t cudaCsrColIndType =
    cusparse_index_type_t_from<idx_type>();
  cudaDataType cudaValueType = cuda_data_type_from<scalar_type>();

  // Create sparse matrix in CSR format
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
					      &(h->matDescr), static_cast<int64_t>(nrows),
					      static_cast<int64_t>(nrows), nnz, (void *)row_map.data(), (void *)entries.data(),
					      (void *)values.data(), cudaCsrRowMapType, cudaCsrColIndType,
					      CUSPARSE_INDEX_BASE_ZERO, cudaValueType));

  // Create dummy dense vector B (RHS)
  nnz_scalar_view_t b_dummy(Kokkos::view_alloc(space, "b_dummy"), nrows);
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateDnVec(&(h->vecBDescr_dummy), static_cast<int64_t>(nrows),
						b_dummy.data(), cudaValueType));

  // Create dummy dense vector X (LHS)
  nnz_scalar_view_t x_dummy(Kokkos::view_alloc(space, "x_dummy"), nrows);
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateDnVec(&(h->vecXDescr_dummy), static_cast<int64_t>(nrows),
						x_dummy.data(), cudaValueType));

  // Specify Lower|Upper fill mode
  if (is_lower) {
    cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_LOWER;
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpMatSetAttribute(h->matDescr, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode)));
  } else {
    cusparseFillMode_t fillmode = CUSPARSE_FILL_MODE_UPPER;
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpMatSetAttribute(h->matDescr, CUSPARSE_SPMAT_FILL_MODE, &fillmode, sizeof(fillmode)));
  }

  // Specify Unit|Non-Unit diagonal type.
  cusparseDiagType_t diagtype = CUSPARSE_DIAG_TYPE_NON_UNIT;
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpMatSetAttribute(h->matDescr, CUSPARSE_SPMAT_DIAG_TYPE, &diagtype, sizeof(diagtype)));

  // Allocate an external buffer for analysis
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpSV_bufferSize(h->handle, h->transpose, &alpha, h->matDescr, h->vecBDescr_dummy,
						    h->vecXDescr_dummy, cudaValueType, CUSPARSE_SPSV_ALG_DEFAULT,
						    h->spsvDescr, &pBufferSize));

  // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMalloc((void **)&(h->pBuffer), pBufferSize));

  // Run analysis
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpSV_analysis(h->handle, h->transpose, &alpha, h->matDescr, h->vecBDescr_dummy,
						  h->vecXDescr_dummy, cudaValueType, CUSPARSE_SPSV_ALG_DEFAULT,
						  h->spsvDescr, h->pBuffer));

  // Destroy dummy dense vector descriptors
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyDnVec(h->vecBDescr_dummy));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyDnVec(h->vecXDescr_dummy));
#else  // CUDA_VERSION < 11030

  bool is_lower = sptrsv_handle->is_lower_tri();
  sptrsv_handle->create_cuSPARSE_Handle(trans, is_lower);

  typename KernelHandle::SPTRSVcuSparseHandleType *h =
    sptrsv_handle->get_cuSparseHandle();

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSetStream(h->handle, space.cuda_stream()));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsrsv2Info(&(h->info)));

  // query how much memory used in csrsv2, and allocate the buffer
  int nnz = entries.extent_int(0);
  int pBufferSize;

  const scalar_type *vals = values.data();

  if constexpr (std::is_same<scalar_type, double>::value) {
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseDcsrsv2_bufferSize(h->handle, h->transpose, nrows, nnz, h->descr,
							 values.data(), row_map.data(), entries.data(),
							 h->info, &pBufferSize));

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaError_t my_error;
    my_error = cudaMalloc((void **)&(h->pBuffer), pBufferSize);

    if (cudaSuccess != my_error)
      std::cout << "cudmalloc pBuffer error_t error name "
		<< cudaGetErrorString(my_error) << std::endl;

    KOKKOS_CUSPARSE_SAFE_CALL(cusparseDcsrsv2_analysis(
						       h->handle, h->transpose, nrows, nnz, h->descr, values.data(),
						       row_map.data(), entries.data(), h->info, h->policy, h->pBuffer));
  } else if (std::is_same<scalar_type, float>::value) {
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseScsrsv2_bufferSize(h->handle, h->transpose, nrows, nnz, h->descr,
							 values.data(), row_map.data(), entries.data(), h->info,
							 &pBufferSize));

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaError_t my_error;
    my_error = cudaMalloc((void **)&(h->pBuffer), pBufferSize);

    if (cudaSuccess != my_error)
      std::cout << "cudmalloc pBuffer error_t error name "
		<< cudaGetErrorString(my_error) << std::endl;

    KOKKOS_CUSPARSE_SAFE_CALL(cusparseScsrsv2_analysis(h->handle, h->transpose, nrows, nnz, h->descr, values.data(),
						       row_map.data(), entries.data(), h->info, h->policy, h->pBuffer));
  } else if (std::is_same<scalar_type, Kokkos::complex<double> >::value) {
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseZcsrsv2_bufferSize(h->handle, h->transpose, nrows, nnz, h->descr,
							 reinterpret_cast<cuDoubleComplex *>(values.data()), row_map.data(),
							 entries.data(), h->info, &pBufferSize));

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaError_t my_error;
    my_error = cudaMalloc((void **)&(h->pBuffer), pBufferSize);

    if (cudaSuccess != my_error)
      std::cout << "cudmalloc pBuffer error_t error name "
		<< cudaGetErrorString(my_error) << std::endl;

    KOKKOS_CUSPARSE_SAFE_CALL(cusparseZcsrsv2_analysis(h->handle, h->transpose, nrows, nnz,
						       h->descr, reinterpret_cast<cuDoubleComplex *>(values.data()),
						       row_map.data(), entries.data(), h->info, h->policy, h->pBuffer));
  } else if (std::is_same<scalar_type, Kokkos::complex<float> >::value) {
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseCcsrsv2_bufferSize(h->handle, h->transpose, nrows, nnz, h->descr,
							 reinterpret_cast<cuComplex *>(values.data()),
							 row_map.data(), entries.data(), h->info, &pBufferSize));

    // pBuffer returned by cudaMalloc is automatically aligned to 128 bytes.
    cudaError_t my_error;
    my_error = cudaMalloc((void **)&(h->pBuffer), pBufferSize);

    if (cudaSuccess != my_error)
      std::cout << "cudmalloc pBuffer error_t error name "
		<< cudaGetErrorString(my_error) << std::endl;

    KOKKOS_CUSPARSE_SAFE_CALL(cusparseCcsrsv2_analysis(
						       h->handle, h->transpose, nrows, nnz, h->descr,
						       reinterpret_cast<cuComplex *>(values.data()),
						       row_map.data(), entries.data(), h->info, h->policy, h->pBuffer));
  }
#endif  // CUDA_VERSION >= 11030
}  // sptrsv_cusparse()


#define KOKKOSSPARSE_SPTRSV_SYMBOLIC_CUSPARSE(SCALAR, LAYOUT, MEMSPACE)	\
  template<>								\
  struct SPTRSV_SYMBOLIC<Kokkos::Cuda,					\
      KokkosKernels::Experimental::KokkosKernelsHandle<const int, const int, const SCALAR, Kokkos::Cuda, MEMSPACE, MEMSPACE>, \
      Kokkos::View<const int *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >,    \
      Kokkos::View<const int *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >,    \
      Kokkos::View<const SCALAR *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >,true, true> {					\
									\
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<const int,  const int, const SCALAR, Kokkos::Cuda, MEMSPACE, MEMSPACE>; \
    using RowMapType  = Kokkos::View<const int *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >; \
    using EntriesType = Kokkos::View<const int *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >; \
    using ValuesType  = Kokkos::View<const SCALAR *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >; \
									\
    static void sptrsv_symbolic(const Kokkos::Cuda& space,		\
				KernelHandle *handle,			\
				const RowMapType row_map,		\
				const EntriesType entries,              \
				const ValuesType values) {		\
      bool trans = false;						\
      typename KernelHandle::SPTRSVHandleType *sptrsv_handle = handle->get_sptrsv_handle(); \
      std::string label = "KokkosSparse::sptrsv_symbolic[TPL_CUSPARSE," \
	+ Kokkos::ArithTraits<SCALAR>::name() + "]";                    \
      Kokkos::Profiling::pushRegion(label);                             \
      sptrsv_analysis_cusparse(space, sptrsv_handle, row_map, entries,  \
		      values, trans);					\
      Kokkos::Profiling::popRegion();                                   \
    }									\
  };

KOKKOSSPARSE_SPTRSV_SYMBOLIC_CUSPARSE(float, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_CUSPARSE(double, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_CUSPARSE(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_CUSPARSE(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::CudaSpace)

KOKKOSSPARSE_SPTRSV_SYMBOLIC_CUSPARSE(float, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_CUSPARSE(double, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_CUSPARSE(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPTRSV_SYMBOLIC_CUSPARSE(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)

}  // namespace Impl
}  // namespace KokkosSparse

#endif  // KOKKOSKERNELS_ENABLE_TPL_CUSPARSE


#endif
