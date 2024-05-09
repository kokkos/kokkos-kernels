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

#ifndef KOKKOSPARSE_SPTRSV_SOLVE_TPL_SPEC_DECL_HPP_
#define KOKKOSPARSE_SPTRSV_SOLVE_TPL_SPEC_DECL_HPP_


// cuSPARSE
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include "cusparse.h"
#include "KokkosSparse_Utils_cusparse.hpp"
#include "KokkosKernels_Handle.hpp"

namespace KokkosSparse {
namespace Impl {

template<class ExecutionSpace, class KernelHandle, class RowMapType,
         class EntriesType, class ValuesType, class BType, class XType>
void sptrsv_solve_cusparse(ExecutionSpace& space,
			   KernelHandle *sptrsv_handle,
			   RowMapType row_map,
			   EntriesType entries,
			   ValuesType values,
			   BType rhs, XType lhs,
			   const bool /*trans*/) {
  using idx_type     = typename KernelHandle::nnz_lno_t;
  using size_type    = typename KernelHandle::size_type;
  using scalar_type  = typename KernelHandle::scalar_t;

#if (CUDA_VERSION >= 11030)
  using memory_space = typename KernelHandle::memory_space;

  (void)row_map;
  (void)entries;
  (void)values;

  // cusparseDnVecDescr_t vecBDescr, vecXDescr;

  const idx_type nrows = sptrsv_handle->get_nrows();
  typename KernelHandle::SPTRSVcuSparseHandleType *h =
    sptrsv_handle->get_cuSparseHandle();

  KOKKOS_CUSPARSE_SAFE_CALL(
  			    cusparseSetStream(h->handle, space.cuda_stream()));

  const scalar_type alpha = scalar_type(1.0);

  const cudaDataType cudaValueType = cuda_data_type_from<scalar_type>();

  // Create dense vector B (RHS)
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateDnVec(&(h->vecBDescr), static_cast<int64_t>(nrows),
						(void *)rhs.data(), cudaValueType));

  // Create dense vector X (LHS)
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateDnVec(&(h->vecXDescr), static_cast<int64_t>(nrows),
						(void *)lhs.data(), cudaValueType));

  // Solve
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpSV_solve(h->handle, h->transpose, &alpha, h->matDescr, h->vecBDescr,
					       h->vecXDescr, cudaValueType, CUSPARSE_SPSV_ALG_DEFAULT, h->spsvDescr));

  // Destroy dense vector descriptors
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyDnVec(h->vecBDescr));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyDnVec(h->vecXDescr));

#else  // CUDA_VERSION < 11030

    cusparseStatus_t status;

    const idx_type nrows = sptrsv_handle->get_nrows();
    typename KernelHandle::SPTRSVcuSparseHandleType *h =
        sptrsv_handle->get_cuSparseHandle();

    KOKKOS_CUSPARSE_SAFE_CALL(cusparseSetStream(h->handle, space.cuda_stream()));

    int nnz = entries.extent_int(0);
    const int *rm = !std::is_same<size_type, int>::value
                        ? sptrsv_handle->get_int_rowmap_ptr()
                        : (const int *)row_map.data();
    const int *ent          = (const int *)entries.data();
    const scalar_type *vals = values.data();
    const scalar_type *bv   = rhs.data();
    scalar_type *xv         = lhs.data();

    if constexpr (std::is_same_v<scalar_type, double>) {
      if (h->pBuffer == nullptr) {
        std::cout << "  pBuffer invalid" << std::endl;
      }

      const scalar_type alpha = Kokkos::ArithTraits<scalar_type>::one();
      status = cusparseDcsrsv2_solve(
          h->handle, h->transpose, nrows, nnz, &alpha, h->descr, (double *)vals,
          (int *)rm, (int *)ent, h->info, (double *)bv, (double *)xv, h->policy,
          h->pBuffer);

      if (CUSPARSE_STATUS_SUCCESS != status)
        std::cout << "solve status error name " << (status) << std::endl;
    } else if (std::is_same<scalar_type, float>::value) {
      if (h->pBuffer == nullptr) {
        std::cout << "  pBuffer invalid" << std::endl;
      }

      const scalar_type alpha = Kokkos::ArithTraits<scalar_type>::one();
      status = cusparseScsrsv2_solve(h->handle, h->transpose, nrows, nnz,
                                     &alpha, h->descr, (float *)vals, (int *)rm,
                                     (int *)ent, h->info, (float *)bv,
                                     (float *)xv, h->policy, h->pBuffer);

      if (CUSPARSE_STATUS_SUCCESS != status)
        std::cout << "solve status error name " << (status) << std::endl;
    } else if (std::is_same<scalar_type, Kokkos::complex<double> >::value) {
      cuDoubleComplex cualpha;
      cualpha.x = 1.0;
      cualpha.y = 0.0;
      status    = cusparseZcsrsv2_solve(
          h->handle, h->transpose, nrows, nnz, &cualpha, h->descr,
          (cuDoubleComplex *)vals, (int *)rm, (int *)ent, h->info,
          (cuDoubleComplex *)bv, (cuDoubleComplex *)xv, h->policy, h->pBuffer);

      if (CUSPARSE_STATUS_SUCCESS != status)
        std::cout << "solve status error name " << (status) << std::endl;
    } else if (std::is_same<scalar_type, Kokkos::complex<float> >::value) {
      cuComplex cualpha;
      cualpha.x = 1.0;
      cualpha.y = 0.0;
      status    = cusparseCcsrsv2_solve(
          h->handle, h->transpose, nrows, nnz, &cualpha, h->descr,
          (cuComplex *)vals, (int *)rm, (int *)ent, h->info, (cuComplex *)bv,
          (cuComplex *)xv, h->policy, h->pBuffer);

      if (CUSPARSE_STATUS_SUCCESS != status)
        std::cout << "solve status error name " << (status) << std::endl;
    } else {
      throw std::runtime_error("CUSPARSE wrapper error: unsupported type.\n");
    }
#endif
}

template <class ExecutionSpace, class KernelHandle,
          class ain_row_index_view_type, class ain_nonzero_index_view_type,
          class ain_values_scalar_view_type, class b_values_scalar_view_type,
          class x_values_scalar_view_type>
void sptrsv_solve_streams_cusparse(
    const std::vector<ExecutionSpace> &execspace_v,
    std::vector<KernelHandle> &handle_v,
    const std::vector<ain_row_index_view_type> &row_map_v,
    const std::vector<ain_nonzero_index_view_type> &entries_v,
    const std::vector<ain_values_scalar_view_type> &values_v,
    const std::vector<b_values_scalar_view_type> &rhs_v,
    std::vector<x_values_scalar_view_type> &lhs_v, bool /*trans*/
) {
  using idx_type         = typename KernelHandle::nnz_lno_t;
  using size_type        = typename KernelHandle::size_type;
  using scalar_type      = typename KernelHandle::nnz_scalar_t;
  using memory_space     = typename KernelHandle::HandlePersistentMemorySpace;
  using sptrsvHandleType = typename KernelHandle::SPTRSVHandleType;
  using sptrsvCuSparseHandleType =
      typename sptrsvHandleType::SPTRSVcuSparseHandleType;

  int nstreams = execspace_v.size();
#if (CUDA_VERSION >= 11030)
  (void)row_map_v;
  (void)entries_v;
  (void)values_v;

  const bool is_cuda_space =
      std::is_same<memory_space, Kokkos::CudaSpace>::value ||
      std::is_same<memory_space, Kokkos::CudaUVMSpace>::value ||
      std::is_same<memory_space, Kokkos::CudaHostPinnedSpace>::value;

  const bool is_idx_type_supported = std::is_same<idx_type, int>::value ||
                                     std::is_same<idx_type, int64_t>::value;

  if constexpr (!is_cuda_space) {
    throw std::runtime_error(
        "KokkosKernels sptrsvcuSPARSE_solve_streams: MEMORY IS NOT ALLOCATED "
        "IN GPU DEVICE for CUSPARSE\n");
  } else if constexpr (!is_idx_type_supported) {
    throw std::runtime_error(
        "CUSPARSE requires local ordinals to be integer (32 bits or 64 "
        "bits).\n");
  } else {
    const scalar_type alpha = scalar_type(1.0);

    cudaDataType cudaValueType = cuda_data_type_from<scalar_type>();

    std::vector<sptrsvCuSparseHandleType *> h_v(nstreams);

    for (int i = 0; i < nstreams; i++) {
      sptrsvHandleType *sptrsv_handle = handle_v[i].get_sptrsv_handle();
      h_v[i]                          = sptrsv_handle->get_cuSparseHandle();

      // Bind cuspare handle to a stream
      KOKKOS_CUSPARSE_SAFE_CALL(
          cusparseSetStream(h_v[i]->handle, execspace_v[i].cuda_stream()));

      int64_t nrows = static_cast<int64_t>(sptrsv_handle->get_nrows());

      // Create dense vector B (RHS)
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateDnVec(
          &(h_v[i]->vecBDescr), nrows, (void *)rhs_v[i].data(), cudaValueType));

      // Create dense vector X (LHS)
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateDnVec(
          &(h_v[i]->vecXDescr), nrows, (void *)lhs_v[i].data(), cudaValueType));
    }

    // Solve
    for (int i = 0; i < nstreams; i++) {
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpSV_solve(
          h_v[i]->handle, h_v[i]->transpose, &alpha, h_v[i]->matDescr,
          h_v[i]->vecBDescr, h_v[i]->vecXDescr, cudaValueType,
          CUSPARSE_SPSV_ALG_DEFAULT, h_v[i]->spsvDescr));
    }

    // Destroy dense vector descriptors
    for (int i = 0; i < nstreams; i++) {
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyDnVec(h_v[i]->vecBDescr));
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyDnVec(h_v[i]->vecXDescr));
    }
  }
#else  // CUDA_VERSION < 11030
  const bool is_cuda_space =
      std::is_same<memory_space, Kokkos::CudaSpace>::value ||
      std::is_same<memory_space, Kokkos::CudaUVMSpace>::value ||
      std::is_same<memory_space, Kokkos::CudaHostPinnedSpace>::value;

  if constexpr (!is_cuda_space) {
    throw std::runtime_error(
        "KokkosKernels sptrsvcuSPARSE_solve_streams: MEMORY IS NOT ALLOCATED "
        "IN GPU DEVICE for CUSPARSE\n");
  } else if constexpr (!std::is_same<idx_type, int>::value) {
    throw std::runtime_error(
        "CUSPARSE requires local ordinals to be integer.\n");
  } else {
    const scalar_type alpha = scalar_type(1.0);
    std::vector<sptrsvHandleType *> sptrsv_handle_v(nstreams);
    std::vector<sptrsvCuSparseHandleType *> h_v(nstreams);
    std::vector<const int *> rm_v(nstreams);
    std::vector<const int *> ent_v(nstreams);
    std::vector<const scalar_type *> vals_v(nstreams);
    std::vector<const scalar_type *> bv_v(nstreams);
    std::vector<scalar_type *> xv_v(nstreams);

    for (int i = 0; i < nstreams; i++) {
      sptrsv_handle_v[i] = handle_v[i].get_sptrsv_handle();
      h_v[i]             = sptrsv_handle_v[i]->get_cuSparseHandle();

      // Bind cuspare handle to a stream
      KOKKOS_CUSPARSE_SAFE_CALL(
          cusparseSetStream(h_v[i]->handle, execspace_v[i].cuda_stream()));

      if (h_v[i]->pBuffer == nullptr) {
        std::cout << "  pBuffer invalid on stream " << i << std::endl;
      }
      rm_v[i] = !std::is_same<size_type, int>::value
                    ? sptrsv_handle_v[i]->get_int_rowmap_ptr()
                    : reinterpret_cast<const int *>(row_map_v[i].data());
      ent_v[i]  = reinterpret_cast<const int *>(entries_v[i].data());
      vals_v[i] = values_v[i].data();
      bv_v[i]   = rhs_v[i].data();
      xv_v[i]   = lhs_v[i].data();
    }

    for (int i = 0; i < nstreams; i++) {
      int nnz   = entries_v[i].extent_int(0);
      int nrows = static_cast<int>(sptrsv_handle_v[i]->get_nrows());
      if (std::is_same<scalar_type, double>::value) {
        KOKKOS_CUSPARSE_SAFE_CALL(cusparseDcsrsv2_solve(
            h_v[i]->handle, h_v[i]->transpose, nrows, nnz,
            reinterpret_cast<const double *>(&alpha), h_v[i]->descr,
            reinterpret_cast<const double *>(vals_v[i]),
            reinterpret_cast<const int *>(rm_v[i]),
            reinterpret_cast<const int *>(ent_v[i]), h_v[i]->info,
            reinterpret_cast<const double *>(bv_v[i]),
            reinterpret_cast<double *>(xv_v[i]), h_v[i]->policy,
            h_v[i]->pBuffer));
      } else if (std::is_same<scalar_type, float>::value) {
        KOKKOS_CUSPARSE_SAFE_CALL(cusparseScsrsv2_solve(
            h_v[i]->handle, h_v[i]->transpose, nrows, nnz,
            reinterpret_cast<const float *>(&alpha), h_v[i]->descr,
            reinterpret_cast<const float *>(vals_v[i]),
            reinterpret_cast<const int *>(rm_v[i]),
            reinterpret_cast<const int *>(ent_v[i]), h_v[i]->info,
            reinterpret_cast<const float *>(bv_v[i]),
            reinterpret_cast<float *>(xv_v[i]), h_v[i]->policy,
            h_v[i]->pBuffer));
      } else if (std::is_same<scalar_type, Kokkos::complex<double> >::value) {
        KOKKOS_CUSPARSE_SAFE_CALL(cusparseZcsrsv2_solve(
            h_v[i]->handle, h_v[i]->transpose, nrows, nnz,
            reinterpret_cast<const cuDoubleComplex *>(&alpha), h_v[i]->descr,
            reinterpret_cast<const cuDoubleComplex *>(vals_v[i]),
            reinterpret_cast<const int *>(rm_v[i]),
            reinterpret_cast<const int *>(ent_v[i]), h_v[i]->info,
            reinterpret_cast<const cuDoubleComplex *>(bv_v[i]),
            reinterpret_cast<cuDoubleComplex *>(xv_v[i]), h_v[i]->policy,
            h_v[i]->pBuffer));
      } else if (std::is_same<scalar_type, Kokkos::complex<float> >::value) {
        KOKKOS_CUSPARSE_SAFE_CALL(cusparseCcsrsv2_solve(
            h_v[i]->handle, h_v[i]->transpose, nrows, nnz,
            reinterpret_cast<const cuComplex *>(&alpha), h_v[i]->descr,
            reinterpret_cast<const cuComplex *>(vals_v[i]),
            reinterpret_cast<const int *>(rm_v[i]),
            reinterpret_cast<const int *>(ent_v[i]), h_v[i]->info,
            reinterpret_cast<const cuComplex *>(bv_v[i]),
            reinterpret_cast<cuComplex *>(xv_v[i]), h_v[i]->policy,
            h_v[i]->pBuffer));
      } else {
        throw std::runtime_error("CUSPARSE wrapper error: unsupported type.\n");
      }
    }
  }
#endif
}

#define KOKKOSSPARSE_SPTRSV_SOLVE_CUSPARSE(SCALAR, LAYOUT, MEMSPACE)	\
  template<>								\
  struct SPTRSV_SOLVE<Kokkos::Cuda,					\
      KokkosKernels::Experimental::KokkosKernelsHandle<const int, const int, const SCALAR, Kokkos::Cuda, MEMSPACE, MEMSPACE>, \
      Kokkos::View<const int *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >,    \
      Kokkos::View<const int *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >,    \
      Kokkos::View<const SCALAR *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      Kokkos::View<const SCALAR *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >, \
      Kokkos::View<SCALAR *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged> >, true> { \
									\
    using KernelHandle = KokkosKernels::Experimental::KokkosKernelsHandle<const int,  const int, const SCALAR, Kokkos::Cuda, MEMSPACE, MEMSPACE>; \
    using RowMapType  = Kokkos::View<const int *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >; \
    using EntriesType = Kokkos::View<const int *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >; \
    using ValuesType  = Kokkos::View<const SCALAR *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >; \
    using BType = Kokkos::View<const SCALAR *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >; \
    using XType = Kokkos::View<SCALAR *, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged> >; \
									\
    static void sptrsv_solve(const Kokkos::Cuda& space,   		\
			     KernelHandle *handle,   			\
			     const RowMapType row_map,  		\
			     const EntriesType entries,                 \
			     const ValuesType values,                   \
			     const BType b, XType x) {	        	\
      bool trans = false;						\
      typename KernelHandle::SPTRSVHandleType *sptrsv_handle = handle->get_sptrsv_handle(); \
      std::string label = "KokkosSparse::sptrsv_solve[TPL_CUSPARSE,"    \
	+ Kokkos::ArithTraits<SCALAR>::name() + "]";                    \
      Kokkos::Profiling::pushRegion(label);                             \
      sptrsv_solve_cusparse(space, sptrsv_handle, row_map, entries,     \
		            values, b, x, trans);			\
      Kokkos::Profiling::popRegion();                                   \
    }									\
									\
    static void sptrsv_solve_streams(const std::vector<Kokkos::Cuda>& space_v, std::vector<KernelHandle> &handle_v, const std::vector<RowMapType> &row_map_v, const std::vector<EntriesType> &entries_v, const std::vector<ValuesType> &values_v, const std::vector<BType> &b_v, std::vector<XType> &x_v) { \
									\
      std::string label = "KokkosSparse::sptrsv_solve_streams[TPL_CUSPARSE," \
	+ Kokkos::ArithTraits<SCALAR>::name() + "]";                    \
      Kokkos::Profiling::pushRegion(label);                             \
      sptrsv_solve_streams_cusparse(space_v, handle_v, row_map_v,	\
			            entries_v, values_v, b_v, x_v,      \
			            false);				\
      Kokkos::Profiling::popRegion();                                   \
    }									\
  };

KOKKOSSPARSE_SPTRSV_SOLVE_CUSPARSE(float, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPTRSV_SOLVE_CUSPARSE(double, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPTRSV_SOLVE_CUSPARSE(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPTRSV_SOLVE_CUSPARSE(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::CudaSpace)

KOKKOSSPARSE_SPTRSV_SOLVE_CUSPARSE(float, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPTRSV_SOLVE_CUSPARSE(double, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPTRSV_SOLVE_CUSPARSE(Kokkos::complex<float>, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPTRSV_SOLVE_CUSPARSE(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)

}  // namespace Impl
}  // namespace KokkosSparse
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUSPARSE

#endif
