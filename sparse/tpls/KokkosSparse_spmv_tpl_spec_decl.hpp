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

#ifndef KOKKOSPARSE_SPMV_TPL_SPEC_DECL_HPP_
#define KOKKOSPARSE_SPMV_TPL_SPEC_DECL_HPP_

#include <sstream>

#include "KokkosKernels_Controls.hpp"

// cuSPARSE
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include "cusparse.h"
#include "KokkosSparse_Utils_cusparse.hpp"

namespace KokkosSparse {
namespace Impl {

template <class Handle, class AMatrix, class XVector, class YVector>
void spmv_cusparse(const Kokkos::Cuda& exec,
                   Handle* handle,
                   const char mode[],
                   typename YVector::non_const_value_type const& alpha,
                   const AMatrix& A, const XVector& x,
                   typename YVector::non_const_value_type const& beta,
                   const YVector& y) {
  using offset_type = typename AMatrix::non_const_size_type;
  using value_type  = typename AMatrix::non_const_value_type;

  /* initialize cusparse library */
  cusparseHandle_t cusparseHandle =
      KokkosKernels::Impl::CusparseSingleton::singleton().cusparseHandle;
  /* Set cuSPARSE to use the given stream until this function exits */
  TemporarySetCusparseStream(cusparseHandle, exec);

  /* Set the operation mode */
  cusparseOperation_t myCusparseOperation;
  switch (toupper(mode[0])) {
    case 'N': myCusparseOperation = CUSPARSE_OPERATION_NON_TRANSPOSE; break;
    case 'T': myCusparseOperation = CUSPARSE_OPERATION_TRANSPOSE; break;
    case 'H':
      myCusparseOperation = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE;
      break;
    default: {
      std::ostringstream out;
      out << "Mode " << mode << " invalid for cuSPARSE SpMV.\n";
      throw std::invalid_argument(out.str());
    }
  }
  // cuSPARSE doesn't directly support mode H with real values, but this is
  // equivalent to mode T
  if (myCusparseOperation == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE &&
      !Kokkos::ArithTraits<value_type>::isComplex)
    myCusparseOperation = CUSPARSE_OPERATION_TRANSPOSE;

// Hopefully this corresponds to CUDA reelase 10.1, which is the first to
// include the "generic" API
#if defined(CUSPARSE_VERSION) && (10300 <= CUSPARSE_VERSION)

  using entry_type = typename AMatrix::non_const_ordinal_type;

  cudaDataType myCudaDataType;
  if (std::is_same<value_type, float>::value)
    myCudaDataType = CUDA_R_32F;
  else if (std::is_same<value_type, double>::value)
    myCudaDataType = CUDA_R_64F;
  else if (std::is_same<value_type, Kokkos::complex<float>>::value)
    myCudaDataType = CUDA_C_32F;
  else if (std::is_same<value_type, Kokkos::complex<double>>::value)
    myCudaDataType = CUDA_C_64F;
  else
    throw std::logic_error(
        "Scalar (data) type of CrsMatrix isn't supported by cuSPARSE, yet TPL "
        "layer says it is");

  /* Check that cusparse can handle the types of the input Kokkos::CrsMatrix */
  const cusparseIndexType_t myCusparseOffsetType =
      cusparse_index_type_t_from<offset_type>();
  const cusparseIndexType_t myCusparseEntryType =
      cusparse_index_type_t_from<entry_type>();

  /* create lhs and rhs */
  cusparseDnVecDescr_t vecX, vecY;
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateDnVec(
      &vecX, x.extent_int(0), (void*)x.data(), myCudaDataType));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateDnVec(
      &vecY, y.extent_int(0), (void*)y.data(), myCudaDataType));

  KokkosSparse::Impl::CuSparse10_SpMV_Data* subhandle;

  if (handle->is_set_up) {
    subhandle =
        dynamic_cast<KokkosSparse::Impl::CuSparse10_SpMV_Data*>(handle->tpl);
    if (!subhandle)
      throw std::runtime_error(
          "KokkosSparse::spmv: subhandle is not set up for cusparse");
  } else {
    subhandle   = new KokkosSparse::Impl::CuSparse10_SpMV_Data(exec);
    handle->tpl = subhandle;

    // select cusparse algo
#if CUSPARSE_VERSION >= 11400
    subhandle->algo = (handle->algo == KokkosSparse::SPMV_MERGE_PATH)
                          ? CUSPARSE_SPMV_CSR_ALG2
                          : CUSPARSE_SPMV_ALG_DEFAULT;
#else
    subhandle->algo = (handle->algo == KokkosSparse::SPMV_MERGE_PATH)
                          ? CUSPARSE_CSRMV_ALG2
                          : CUSPARSE_MV_ALG_DEFAULT;
#endif

    /* create matrix */
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateCsr(
        &subhandle->mat, A.numRows(), A.numCols(), A.nnz(),
        (void*)A.graph.row_map.data(), (void*)A.graph.entries.data(),
        (void*)A.values.data(), myCusparseOffsetType, myCusparseEntryType,
        CUSPARSE_INDEX_BASE_ZERO, myCudaDataType));

    /* size and allocate buffer */
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpMV_bufferSize(
        cusparseHandle, myCusparseOperation, &alpha, subhandle->mat, vecX,
        &beta, vecY, myCudaDataType, subhandle->algo, &subhandle->bufferSize));
    KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMallocAsync(
        &subhandle->buffer, subhandle->bufferSize, exec.cuda_stream()));
    handle->is_set_up = true;
  }

  /* perform SpMV */
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseSpMV(
      cusparseHandle, myCusparseOperation, &alpha, subhandle->mat, vecX, &beta,
      vecY, myCudaDataType, subhandle->algo, subhandle->buffer));

  KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyDnVec(vecX));
  KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyDnVec(vecY));

#elif (9000 <= CUDA_VERSION)

  KokkosSparse::Impl::CuSparse9_SpMV_Data* subhandle;

  if (handle->is_set_up) {
    subhandle =
        dynamic_cast<KokkosSparse::Impl::CuSparse9_SpMV_Data*>(handle->tpl);
    if (!subhandle)
      throw std::runtime_error(
          "KokkosSparse::spmv: subhandle is not set up for cusparse");
  } else {
    /* create and set the subhandle and matrix descriptor */
    subhandle   = new KokkosSparse::Impl::CuSparse9_SpMV_Data(exec);
    handle->tpl = subhandle;
    cusparseMatDescr_t descrA = 0;
    KOKKOS_CUSPARSE_SAFE_CALL(cusparseCreateMatDescr(&subhandle->mat));
    KOKKOS_CUSPARSE_SAFE_CALL(
        cusparseSetMatType(subhandle->mat, CUSPARSE_MATRIX_TYPE_GENERAL));
    KOKKOS_CUSPARSE_SAFE_CALL(
        cusparseSetMatIndexBase(subhandle->mat, CUSPARSE_INDEX_BASE_ZERO));
    handle->is_set_up = true;
  }

  /* perform the actual SpMV operation */
    static_assert(std::is_same_v<int, offset_type>,
        "With cuSPARSE pre-10.0, offset type must be int. Something wrong with "
        "TPL avail logic.");
    if constexpr (std::is_same_v<value_type, float>) {
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseScsrmv(
          cusparseHandle, myCusparseOperation, A.numRows(), A.numCols(),
          A.nnz(), reinterpret_cast<float const*>(&alpha), subhandle->mat,
          reinterpret_cast<float const*>(A.values.data()),
          A.graph.row_map.data(), A.graph.entries.data(),
          reinterpret_cast<float const*>(x.data()),
          reinterpret_cast<float const*>(&beta),
          reinterpret_cast<float*>(y.data())));

    } else if constexpr(std::is_same_v<value_type, double>) {
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseDcsrmv(
          cusparseHandle, myCusparseOperation, A.numRows(), A.numCols(),
          A.nnz(), reinterpret_cast<double const*>(&alpha), subhandle->mat,
          reinterpret_cast<double const*>(A.values.data()),
          A.graph.row_map.data(), A.graph.entries.data(),
          reinterpret_cast<double const*>(x.data()),
          reinterpret_cast<double const*>(&beta),
          reinterpret_cast<double*>(y.data())));
    } else if constexpr(std::is_same_v<value_type, Kokkos::complex<float>>) {
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseCcsrmv(
          cusparseHandle, myCusparseOperation, A.numRows(), A.numCols(),
          A.nnz(), reinterpret_cast<cuComplex const*>(&alpha), subhandle->mat,
          reinterpret_cast<cuComplex const*>(A.values.data()),
          A.graph.row_map.data(), A.graph.entries.data(),
          reinterpret_cast<cuComplex const*>(x.data()),
          reinterpret_cast<cuComplex const*>(&beta),
          reinterpret_cast<cuComplex*>(y.data())));
    } else if constexpr(std::is_same_v<value_type, Kokkos::complex<double>>) {
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseZcsrmv(
          cusparseHandle, myCusparseOperation, A.numRows(), A.numCols(),
          A.nnz(), reinterpret_cast<cuDoubleComplex const*>(&alpha),
          subhandle->mat,
          reinterpret_cast<cuDoubleComplex const*>(A.values.data()),
          A.graph.row_map.data(), A.graph.entries.data(),
          reinterpret_cast<cuDoubleComplex const*>(x.data()),
          reinterpret_cast<cuDoubleComplex const*>(&beta),
          reinterpret_cast<cuDoubleComplex*>(y.data())));
    } else {
      static_assert(false,
          "Trying to call cusparse SpMV with a scalar type not float/double, "
          "nor complex of either!");
    }
#endif  // CUDA_VERSION
}

#define KOKKOSSPARSE_SPMV_CUSPARSE(SCALAR, ORDINAL, OFFSET, LAYOUT, SPACE,  \
                                   COMPILE_LIBRARY)                         \
  template <>                                                               \
  struct SPMV<                                                              \
      Kokkos::Cuda,                                                         \
      KokkosSparse::CrsMatrix<                                              \
          SCALAR const, ORDINAL const, Kokkos::Device<Kokkos::Cuda, SPACE>, \
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, OFFSET const>,           \
      Kokkos::View<                                                         \
          SCALAR const*, LAYOUT, Kokkos::Device<Kokkos::Cuda, SPACE>,       \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>,  \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::Cuda, SPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                \
      true, COMPILE_LIBRARY> {                                              \
    using device_type       = Kokkos::Device<Kokkos::Cuda, SPACE>;          \
    using memory_trait_type = Kokkos::MemoryTraits<Kokkos::Unmanaged>;      \
    using AMatrix = CrsMatrix<SCALAR const, ORDINAL const, device_type,     \
                              memory_trait_type, OFFSET const>;             \
    using XVector = Kokkos::View<                                           \
        SCALAR const*, LAYOUT, device_type,                                 \
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>;    \
    using YVector =                                                         \
        Kokkos::View<SCALAR*, LAYOUT, device_type, memory_trait_type>;      \
    using Controls = KokkosKernels::Experimental::Controls;                 \
                                                                            \
    using coefficient_type = typename YVector::non_const_value_type;        \
                                                                            \
    static void spmv(const Kokkos::Cuda& exec, const Controls& controls,    \
                     const char mode[], const coefficient_type& alpha,      \
                     const AMatrix& A, const XVector& x,                    \
                     const coefficient_type& beta, const YVector& y) {      \
      std::string label = "KokkosSparse::spmv[TPL_CUSPARSE," +              \
                          Kokkos::ArithTraits<SCALAR>::name() + "]";        \
      Kokkos::Profiling::pushRegion(label);                                 \
      spmv_cusparse(exec, controls, mode, alpha, A, x, beta, y);            \
      Kokkos::Profiling::popRegion();                                       \
    }                                                                       \
  };

#if (9000 <= CUDA_VERSION)
KOKKOSSPARSE_SPMV_CUSPARSE(double, int, int, Kokkos::LayoutLeft,
                           Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(double, int, int, Kokkos::LayoutRight,
                           Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(float, int, int, Kokkos::LayoutLeft,
                           Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(float, int, int, Kokkos::LayoutRight,
                           Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<double>, int, int,
                           Kokkos::LayoutLeft, Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<double>, int, int,
                           Kokkos::LayoutRight, Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<float>, int, int, Kokkos::LayoutLeft,
                           Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<float>, int, int,
                           Kokkos::LayoutRight, Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(double, int, int, Kokkos::LayoutLeft,
                           Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(double, int, int, Kokkos::LayoutRight,
                           Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(float, int, int, Kokkos::LayoutLeft,
                           Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(float, int, int, Kokkos::LayoutRight,
                           Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<double>, int, int,
                           Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<double>, int, int,
                           Kokkos::LayoutRight, Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<float>, int, int, Kokkos::LayoutLeft,
                           Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<float>, int, int,
                           Kokkos::LayoutRight, Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)

#if defined(CUSPARSE_VERSION) && (10300 <= CUSPARSE_VERSION)
KOKKOSSPARSE_SPMV_CUSPARSE(double, int64_t, size_t, Kokkos::LayoutLeft,
                           Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(double, int64_t, size_t, Kokkos::LayoutRight,
                           Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(float, int64_t, size_t, Kokkos::LayoutLeft,
                           Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(float, int64_t, size_t, Kokkos::LayoutRight,
                           Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<double>, int64_t, size_t,
                           Kokkos::LayoutLeft, Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<double>, int64_t, size_t,
                           Kokkos::LayoutRight, Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<float>, int64_t, size_t,
                           Kokkos::LayoutLeft, Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<float>, int64_t, size_t,
                           Kokkos::LayoutRight, Kokkos::CudaSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(double, int64_t, size_t, Kokkos::LayoutLeft,
                           Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(double, int64_t, size_t, Kokkos::LayoutRight,
                           Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(float, int64_t, size_t, Kokkos::LayoutLeft,
                           Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(float, int64_t, size_t, Kokkos::LayoutRight,
                           Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<double>, int64_t, size_t,
                           Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<double>, int64_t, size_t,
                           Kokkos::LayoutRight, Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<float>, int64_t, size_t,
                           Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_CUSPARSE(Kokkos::complex<float>, int64_t, size_t,
                           Kokkos::LayoutRight, Kokkos::CudaUVMSpace,
                           KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
#endif  // defined(CUSPARSE_VERSION) && (10300 <= CUSPARSE_VERSION)
#endif  // 9000 <= CUDA_VERSION

#undef KOKKOSSPARSE_SPMV_CUSPARSE

}  // namespace Impl
}  // namespace KokkosSparse
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUSPARSE

// rocSPARSE
#if defined(KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE)
#include "KokkosSparse_Utils_rocsparse.hpp"

namespace KokkosSparse {
namespace Impl {

template <class AMatrix, class XVector, class YVector>
void spmv_rocsparse(const Kokkos::HIP& exec,
                    KokkosSparse::SpMVHandle<Kokkos::HIP, AMatrix>* handle,
                    const char mode[],
                    typename YVector::non_const_value_type const& alpha,
                    const AMatrix& A, const XVector& x,
                    typename YVector::non_const_value_type const& beta,
                    const YVector& y) {
  using offset_type = typename AMatrix::non_const_size_type;
  using entry_type  = typename AMatrix::non_const_ordinal_type;
  using value_type  = typename AMatrix::non_const_value_type;

  /* initialize rocsparse library */
  rocsparse_handle rocsparseHandle =
      KokkosKernels::Impl::RocsparseSingleton::singleton().rocsparseHandle;
  /* Set rocsparse to use the given stream until this function exits */
  TemporarySetRocsparseStream(rocsparseHandle, exec);

  /* Set the operation mode */
  rocsparse_operation myRocsparseOperation = mode_kk_to_rocsparse(mode);

  /* Set the index type */
  rocsparse_indextype offset_index_type = rocsparse_index_type<offset_type>();
  rocsparse_indextype entry_index_type  = rocsparse_index_type<entry_type>();

  /* Set the scalar type */
  rocsparse_datatype compute_type = rocsparse_compute_type<value_type>();

  /* Create rocsparse dense vectors for X and Y */
  rocsparse_dnvec_descr vecX, vecY;
  void* x_data = static_cast<void*>(
      const_cast<typename XVector::non_const_value_type*>(x.data()));
  void* y_data = static_cast<void*>(
      const_cast<typename YVector::non_const_value_type*>(y.data()));
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_create_dnvec_descr(
      &vecX, x.extent_int(0), x_data,
      rocsparse_compute_type<typename XVector::non_const_value_type>()));
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_create_dnvec_descr(
      &vecY, y.extent_int(0), y_data,
      rocsparse_compute_type<typename YVector::non_const_value_type>()));

  rocsparse_spmv_alg alg = rocsparse_spmv_alg_default;

  KokkosSparse::Impl::RocSparse_CRS_SpMV_Data* subhandle;
  if (handle->is_set_up) {
    subhandle =
        dynamic_cast<KokkosSparse::Impl::RocSparse_CRS_SpMV_Data*>(handle->tpl);
    if (!subhandle)
      throw std::runtime_error(
          "KokkosSparse::spmv: subhandle is not set up for rocsparse CRS");
  } else {
    subhandle   = new KokkosSparse::Impl::RocSparse_CRS_SpMV_Data(exec);
    handle->tpl = subhandle;
    /* Create the rocsparse csr descr */
    // We need to do some casting to void*
    // Note that row_map is always a const view so const_cast is necessary,
    // however entries and values may not be const so we need to check first.
    void* csr_row_ptr =
        static_cast<void*>(const_cast<offset_type*>(A.graph.row_map.data()));
    void* csr_col_ind =
        static_cast<void*>(const_cast<entry_type*>(A.graph.entries.data()));
    void* csr_val =
        static_cast<void*>(const_cast<value_type*>(A.values.data()));

    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_create_csr_descr(
        &subhandle->mat, A.numRows(), A.numCols(), A.nnz(), csr_row_ptr,
        csr_col_ind, csr_val, offset_index_type, entry_index_type,
        rocsparse_index_base_zero, compute_type));

    /* Size and allocate buffer, and analyze the matrix */

#if KOKKOSSPARSE_IMPL_ROCM_VERSION >= 60000
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_spmv(
        rocsparseHandle, myRocsparseOperation, &alpha, subhandle->mat, vecX,
        &beta, vecY, compute_type, alg, rocsparse_spmv_stage_buffer_size,
        &subhandle->bufferSize, nullptr));
    KOKKOS_IMPL_HIP_SAFE_CALL(
        hipMalloc(&subhandle->buffer, subhandle->bufferSize));
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_spmv(
        rocsparseHandle, myRocsparseOperation, &alpha, subhandle->mat, vecX,
        &beta, vecY, compute_type, alg, rocsparse_spmv_stage_preprocess,
        &subhandle->bufferSize, subhandle->buffer));
#elif KOKKOSSPARSE_IMPL_ROCM_VERSION >= 50400
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_spmv_ex(
        rocsparseHandle, myRocsparseOperation, &alpha, subhandle->mat, vecX,
        &beta, vecY, compute_type, alg, rocsparse_spmv_stage_auto,
        &subhandle->bufferSize, nullptr));
    KOKKOS_IMPL_HIP_SAFE_CALL(
        hipMalloc(&subhandle->buffer, subhandle->bufferSize));
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_spmv_ex(
        rocsparseHandle, myRocsparseOperation, &alpha, subhandle->mat, vecX,
        &beta, vecY, compute_type, alg, rocsparse_spmv_stage_preprocess,
        &subhandle->bufferSize, subhandle->buffer));
#else
    KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_spmv(
        rocsparseHandle, myRocsparseOperation, &alpha, subhandle->mat, vecX,
        &beta, vecY, compute_type, alg, &subhandle->bufferSize, nullptr));
    KOKKOS_IMPL_HIP_SAFE_CALL(
        hipMalloc(&subhandle->buffer, subhandle->bufferSize));
#endif
    handle->is_set_up = true;
  }

  /* Perform the actual computation */

#if KOKKOSSPARSE_IMPL_ROCM_VERSION >= 60000
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_spmv(
      rocsparseHandle, myRocsparseOperation, &alpha, subhandle->mat, vecX,
      &beta, vecY, compute_type, alg, rocsparse_spmv_stage_compute,
      &subhandle->bufferSize, subhandle->buffer));
#elif KOKKOSSPARSE_IMPL_ROCM_VERSION >= 50400
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_spmv_ex(
      rocsparseHandle, myRocsparseOperation, &alpha, subhandle->mat, vecX,
      &beta, vecY, compute_type, alg, rocsparse_spmv_stage_compute,
      &subhandle->bufferSize, subhandle->buffer));
#else
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(
      rocsparse_spmv(rocsparseHandle, myRocsparseOperation, &alpha,
                     subhandle->mat, vecX, &beta, vecY, compute_type, alg,
                     &subhandle->bufferSize, subhandle->buffer));
#endif

  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_destroy_dnvec_descr(vecY));
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_destroy_dnvec_descr(vecX));
}

#define KOKKOSSPARSE_SPMV_ROCSPARSE(SCALAR, LAYOUT, COMPILE_LIBRARY)          \
  template <>                                                                 \
  struct SPMV<                                                                \
      Kokkos::HIP,                                                            \
      KokkosSparse::CrsMatrix<SCALAR const, rocsparse_int const,              \
                              Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,  \
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>,        \
                              rocsparse_int const>,                           \
      Kokkos::View<                                                           \
          SCALAR const*, LAYOUT,                                              \
          Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,                      \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>,    \
      Kokkos::View<SCALAR*, LAYOUT,                                           \
                   Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>,             \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                  \
      true, COMPILE_LIBRARY> {                                                \
    using device_type       = Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>;  \
    using memory_trait_type = Kokkos::MemoryTraits<Kokkos::Unmanaged>;        \
    using AMatrix = CrsMatrix<SCALAR const, rocsparse_int const, device_type, \
                              memory_trait_type, rocsparse_int const>;        \
    using XVector = Kokkos::View<                                             \
        SCALAR const*, LAYOUT, device_type,                                   \
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>;      \
    using YVector =                                                           \
        Kokkos::View<SCALAR*, LAYOUT, device_type, memory_trait_type>;        \
    using Controls = KokkosKernels::Experimental::Controls;                   \
                                                                              \
    using coefficient_type = typename YVector::non_const_value_type;          \
                                                                              \
    static void spmv(const Kokkos::HIP& exec, const Controls& controls,       \
                     const char mode[], const coefficient_type& alpha,        \
                     const AMatrix& A, const XVector& x,                      \
                     const coefficient_type& beta, const YVector& y) {        \
      std::string label = "KokkosSparse::spmv[TPL_ROCSPARSE," +               \
                          Kokkos::ArithTraits<SCALAR>::name() + "]";          \
      Kokkos::Profiling::pushRegion(label);                                   \
      spmv_rocsparse(exec, controls, mode, alpha, A, x, beta, y);             \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

KOKKOSSPARSE_SPMV_ROCSPARSE(double, Kokkos::LayoutLeft,
                            KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ROCSPARSE(double, Kokkos::LayoutRight,
                            KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ROCSPARSE(float, Kokkos::LayoutLeft,
                            KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ROCSPARSE(float, Kokkos::LayoutRight,
                            KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ROCSPARSE(Kokkos::complex<double>, Kokkos::LayoutLeft,
                            KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ROCSPARSE(Kokkos::complex<double>, Kokkos::LayoutRight,
                            KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ROCSPARSE(Kokkos::complex<float>, Kokkos::LayoutLeft,
                            KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ROCSPARSE(Kokkos::complex<float>, Kokkos::LayoutRight,
                            KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)

#undef KOKKOSSPARSE_SPMV_ROCSPARSE

}  // namespace Impl
}  // namespace KokkosSparse
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
#include <mkl.h>
#include "KokkosSparse_Utils_mkl.hpp"

namespace KokkosSparse {
namespace Impl {

#if (__INTEL_MKL__ > 2017)
// MKL 2018 and above: use new interface: sparse_matrix_t and mkl_sparse_?_mv()

template <typename ExecutionSpace, typename Scalar, typename Handle>
void spmv_mkl(Handle* handle, sparse_operation_t op, Scalar alpha, Scalar beta, 

inline void spmv_mkl(sparse_operation_t op, float alpha, float beta, MKL_INT m,
                     MKL_INT n, const MKL_INT* Arowptrs,
                     const MKL_INT* Aentries, const float* Avalues,
                     const float* x, float* y) {
  sparse_matrix_t A_mkl;
  matrix_descr A_descr;
  A_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  A_descr.mode = SPARSE_FILL_MODE_FULL;
  A_descr.diag = SPARSE_DIAG_NON_UNIT;
  KOKKOSKERNELS_MKL_SAFE_CALL(mkl_sparse_s_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, m, n, const_cast<MKL_INT*>(Arowptrs),
      const_cast<MKL_INT*>(Arowptrs + 1), const_cast<MKL_INT*>(Aentries),
      const_cast<float*>(Avalues)));
  KOKKOSKERNELS_MKL_SAFE_CALL(
      mkl_sparse_s_mv(op, alpha, A_mkl, A_descr, x, beta, y));
}

inline void spmv_mkl(sparse_operation_t op, double alpha, double beta,
                     MKL_INT m, MKL_INT n, const MKL_INT* Arowptrs,
                     const MKL_INT* Aentries, const double* Avalues,
                     const double* x, double* y) {
  sparse_matrix_t A_mkl;
  matrix_descr A_descr;
  A_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  A_descr.mode = SPARSE_FILL_MODE_FULL;
  A_descr.diag = SPARSE_DIAG_NON_UNIT;
  KOKKOSKERNELS_MKL_SAFE_CALL(mkl_sparse_d_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, m, n, const_cast<MKL_INT*>(Arowptrs),
      const_cast<MKL_INT*>(Arowptrs + 1), const_cast<MKL_INT*>(Aentries),
      const_cast<double*>(Avalues)));
  KOKKOSKERNELS_MKL_SAFE_CALL(
      mkl_sparse_d_mv(op, alpha, A_mkl, A_descr, x, beta, y));
}

inline void spmv_mkl(sparse_operation_t op, Kokkos::complex<float> alpha,
                     Kokkos::complex<float> beta, MKL_INT m, MKL_INT n,
                     const MKL_INT* Arowptrs, const MKL_INT* Aentries,
                     const Kokkos::complex<float>* Avalues,
                     const Kokkos::complex<float>* x,
                     Kokkos::complex<float>* y) {
  sparse_matrix_t A_mkl;
  matrix_descr A_descr;
  A_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  A_descr.mode = SPARSE_FILL_MODE_FULL;
  A_descr.diag = SPARSE_DIAG_NON_UNIT;
  KOKKOSKERNELS_MKL_SAFE_CALL(mkl_sparse_c_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, m, n, const_cast<MKL_INT*>(Arowptrs),
      const_cast<MKL_INT*>(Arowptrs + 1), const_cast<MKL_INT*>(Aentries),
      (MKL_Complex8*)Avalues));
  MKL_Complex8 alpha_mkl{alpha.real(), alpha.imag()};
  MKL_Complex8 beta_mkl{beta.real(), beta.imag()};
  KOKKOSKERNELS_MKL_SAFE_CALL(mkl_sparse_c_mv(
      op, alpha_mkl, A_mkl, A_descr, reinterpret_cast<const MKL_Complex8*>(x),
      beta_mkl, reinterpret_cast<MKL_Complex8*>(y)));
}

inline void spmv_mkl(sparse_operation_t op, Kokkos::complex<double> alpha,
                     Kokkos::complex<double> beta, MKL_INT m, MKL_INT n,
                     const MKL_INT* Arowptrs, const MKL_INT* Aentries,
                     const Kokkos::complex<double>* Avalues,
                     const Kokkos::complex<double>* x,
                     Kokkos::complex<double>* y) {
  sparse_matrix_t A_mkl;
  matrix_descr A_descr;
  A_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
  A_descr.mode = SPARSE_FILL_MODE_FULL;
  A_descr.diag = SPARSE_DIAG_NON_UNIT;
  KOKKOSKERNELS_MKL_SAFE_CALL(mkl_sparse_z_create_csr(
      &A_mkl, SPARSE_INDEX_BASE_ZERO, m, n, const_cast<MKL_INT*>(Arowptrs),
      const_cast<MKL_INT*>(Arowptrs + 1), const_cast<MKL_INT*>(Aentries),
      (MKL_Complex16*)Avalues));
  MKL_Complex16 alpha_mkl{alpha.real(), alpha.imag()};
  MKL_Complex16 beta_mkl{beta.real(), beta.imag()};
  KOKKOSKERNELS_MKL_SAFE_CALL(mkl_sparse_z_mv(
      op, alpha_mkl, A_mkl, A_descr, reinterpret_cast<const MKL_Complex16*>(x),
      beta_mkl, reinterpret_cast<MKL_Complex16*>(y)));
}

// Note: classic MKL runs on Serial/OpenMP but can't use our execution space
// instances
#define KOKKOSSPARSE_SPMV_MKL(SCALAR, EXECSPACE, COMPILE_LIBRARY)              \
  template <>                                                                  \
  struct SPMV<EXECSPACE,                                                       \
              KokkosSparse::CrsMatrix<                                         \
                  SCALAR const, MKL_INT const,                                 \
                  Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,                \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged>, MKL_INT const>,     \
              Kokkos::View<SCALAR const*, Kokkos::LayoutLeft,                  \
                           Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,       \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged |            \
                                                Kokkos::RandomAccess>>,        \
              Kokkos::View<SCALAR*, Kokkos::LayoutLeft,                        \
                           Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,       \
                           Kokkos::MemoryTraits<Kokkos::Unmanaged>>,           \
              true, COMPILE_LIBRARY> {                                         \
    using device_type = Kokkos::Device<EXECSPACE, Kokkos::HostSpace>;          \
    using AMatrix =                                                            \
        CrsMatrix<SCALAR const, MKL_INT const, device_type,                    \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged>, MKL_INT const>;     \
    using XVector = Kokkos::View<                                              \
        SCALAR const*, Kokkos::LayoutLeft, device_type,                        \
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>;       \
    using YVector = Kokkos::View<SCALAR*, Kokkos::LayoutLeft, device_type,     \
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;     \
    using coefficient_type = typename YVector::non_const_value_type;           \
    using Controls         = KokkosKernels::Experimental::Controls;            \
                                                                               \
    static void spmv(const EXECSPACE&, const Controls&, const char mode[],     \
                     const coefficient_type& alpha, const AMatrix& A,          \
                     const XVector& x, const coefficient_type& beta,           \
                     const YVector& y) {                                       \
      std::string label = "KokkosSparse::spmv[TPL_MKL," +                      \
                          Kokkos::ArithTraits<SCALAR>::name() + "]";           \
      Kokkos::Profiling::pushRegion(label);                                    \
      spmv_mkl(mode_kk_to_mkl(mode[0]), alpha, beta, A.numRows(), A.numCols(), \
               A.graph.row_map.data(), A.graph.entries.data(),                 \
               A.values.data(), x.data(), y.data());                           \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSSPARSE_SPMV_MKL(float, Kokkos::Serial, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_MKL(double, Kokkos::Serial,
                      KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_MKL(Kokkos::complex<float>, Kokkos::Serial,
                      KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_MKL(Kokkos::complex<double>, Kokkos::Serial,
                      KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSSPARSE_SPMV_MKL(float, Kokkos::OpenMP, KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_MKL(double, Kokkos::OpenMP,
                      KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_MKL(Kokkos::complex<float>, Kokkos::OpenMP,
                      KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_MKL(Kokkos::complex<double>, Kokkos::OpenMP,
                      KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
#endif

#undef KOKKOSSPARSE_SPMV_MKL
#endif

#if defined(KOKKOS_ENABLE_SYCL) && \
    !defined(KOKKOSKERNELS_ENABLE_TPL_MKL_SYCL_OVERRIDE)
inline oneapi::mkl::transpose mode_kk_to_onemkl(char mode_kk) {
  switch (toupper(mode_kk)) {
    case 'N': return oneapi::mkl::transpose::nontrans;
    case 'T': return oneapi::mkl::transpose::trans;
    case 'H': return oneapi::mkl::transpose::conjtrans;
    default:;
  }
  throw std::invalid_argument(
      "Invalid mode for oneMKL (should be one of N, T, H)");
}

template <bool is_complex>
struct spmv_onemkl_wrapper {};

template <>
struct spmv_onemkl_wrapper<false> {
  template <class execution_space, class matrix_type, class xview_type,
            class yview_type>
  static void spmv(const execution_space& exec, oneapi::mkl::transpose mkl_mode,
                   typename matrix_type::non_const_value_type const alpha,
                   const matrix_type& A, const xview_type& x,
                   typename matrix_type::non_const_value_type const beta,
                   const yview_type& y) {
    using scalar_type  = typename matrix_type::non_const_value_type;
    using ordinal_type = typename matrix_type::non_const_ordinal_type;

    // oneAPI doesn't directly support mode H with real values, but this is
    // equivalent to mode T
    if (mkl_mode == oneapi::mkl::transpose::conjtrans &&
        !Kokkos::ArithTraits<scalar_type>::isComplex)
      mkl_mode = oneapi::mkl::transpose::trans;

    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
    oneapi::mkl::sparse::init_matrix_handle(&handle);
    auto ev_set = oneapi::mkl::sparse::set_csr_data(
        exec.sycl_queue(), handle, A.numRows(), A.numCols(),
        oneapi::mkl::index_base::zero,
        const_cast<ordinal_type*>(A.graph.row_map.data()),
        const_cast<ordinal_type*>(A.graph.entries.data()),
        const_cast<scalar_type*>(A.values.data()));
    auto ev_gemv =
        oneapi::mkl::sparse::gemv(exec.sycl_queue(), mkl_mode, alpha, handle,
                                  x.data(), beta, y.data(), {ev_set});
// MKL 2023.2 and up make this release okay async even though it takes a
// pointer to a stack variable
#if INTEL_MKL_VERSION >= 20230200
    oneapi::mkl::sparse::release_matrix_handle(exec.sycl_queue(), &handle,
                                               {ev_gemv});
#else
    auto ev_release = oneapi::mkl::sparse::release_matrix_handle(
        exec.sycl_queue(), &handle, {ev_gemv});
    ev_release.wait();
#endif
}
};

template <>
struct spmv_onemkl_wrapper<true> {
  template <class execution_space, class matrix_type, class xview_type,
            class yview_type>
  static void spmv(const execution_space& exec, oneapi::mkl::transpose mkl_mode,
                   typename matrix_type::non_const_value_type const alpha,
                   const matrix_type& A, const xview_type& x,
                   typename matrix_type::non_const_value_type const beta,
                   const yview_type& y) {
    using scalar_type  = typename matrix_type::non_const_value_type;
    using ordinal_type = typename matrix_type::non_const_ordinal_type;
    using mag_type     = typename Kokkos::ArithTraits<scalar_type>::mag_type;

    oneapi::mkl::sparse::matrix_handle_t handle = nullptr;
    oneapi::mkl::sparse::init_matrix_handle(&handle);
    auto ev_set = oneapi::mkl::sparse::set_csr_data(
        exec.sycl_queue(), handle, static_cast<ordinal_type>(A.numRows()),
        static_cast<ordinal_type>(A.numCols()), oneapi::mkl::index_base::zero,
        const_cast<ordinal_type*>(A.graph.row_map.data()),
        const_cast<ordinal_type*>(A.graph.entries.data()),
        reinterpret_cast<std::complex<mag_type>*>(
            const_cast<scalar_type*>(A.values.data())));
    auto ev_gemv = oneapi::mkl::sparse::gemv(
        exec.sycl_queue(), mkl_mode, alpha, handle,
        reinterpret_cast<std::complex<mag_type>*>(
            const_cast<scalar_type*>(x.data())),
        beta, reinterpret_cast<std::complex<mag_type>*>(y.data()), {ev_set});
    // MKL 2023.2 and up make this release okay async even though it takes a
    // pointer to a stack variable
#if INTEL_MKL_VERSION >= 20230200
    oneapi::mkl::sparse::release_matrix_handle(exec.sycl_queue(), &handle,
                                               {ev_gemv});
#else
    auto ev_release = oneapi::mkl::sparse::release_matrix_handle(
        exec.sycl_queue(), &handle, {ev_gemv});
    ev_release.wait();
#endif
  }
};

#define KOKKOSSPARSE_SPMV_ONEMKL(SCALAR, ORDINAL, MEMSPACE, COMPILE_LIBRARY) \
  template <>                                                                \
  struct SPMV<                                                               \
      Kokkos::Experimental::SYCL,                                            \
      KokkosSparse::CrsMatrix<                                               \
          SCALAR const, ORDINAL const,                                       \
          Kokkos::Device<Kokkos::Experimental::SYCL, MEMSPACE>,              \
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, ORDINAL const>,           \
      Kokkos::View<                                                          \
          SCALAR const*, Kokkos::LayoutLeft,                                 \
          Kokkos::Device<Kokkos::Experimental::SYCL, MEMSPACE>,              \
          Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>,   \
      Kokkos::View<SCALAR*, Kokkos::LayoutLeft,                              \
                   Kokkos::Device<Kokkos::Experimental::SYCL, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,                 \
      true, COMPILE_LIBRARY> {                                               \
    using execution_space = Kokkos::Experimental::SYCL;                      \
    using device_type     = Kokkos::Device<execution_space, MEMSPACE>;       \
    using AMatrix =                                                          \
        CrsMatrix<SCALAR const, ORDINAL const, device_type,                  \
                  Kokkos::MemoryTraits<Kokkos::Unmanaged>, ORDINAL const>;   \
    using XVector = Kokkos::View<                                            \
        SCALAR const*, Kokkos::LayoutLeft, device_type,                      \
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>;     \
    using YVector = Kokkos::View<SCALAR*, Kokkos::LayoutLeft, device_type,   \
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged>>;   \
    using coefficient_type = typename YVector::non_const_value_type;         \
    using Controls         = KokkosKernels::Experimental::Controls;          \
                                                                             \
    static void spmv(const execution_space& exec, const Controls&,           \
                     const char mode[], const coefficient_type& alpha,       \
                     const AMatrix& A, const XVector& x,                     \
                     const coefficient_type& beta, const YVector& y) {       \
      std::string label = "KokkosSparse::spmv[TPL_ONEMKL," +                 \
                          Kokkos::ArithTraits<SCALAR>::name() + "]";         \
      Kokkos::Profiling::pushRegion(label);                                  \
      oneapi::mkl::transpose mkl_mode = mode_kk_to_onemkl(mode[0]);          \
      spmv_onemkl_wrapper<Kokkos::ArithTraits<SCALAR>::is_complex>::spmv(    \
          exec, mkl_mode, alpha, A, x, beta, y);                             \
      Kokkos::Profiling::popRegion();                                        \
    }                                                                        \
  };

KOKKOSSPARSE_SPMV_ONEMKL(float, std::int32_t,
                         Kokkos::Experimental::SYCLDeviceUSMSpace,
                         KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ONEMKL(double, std::int32_t,
                         Kokkos::Experimental::SYCLDeviceUSMSpace,
                         KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ONEMKL(Kokkos::complex<float>, std::int32_t,
                         Kokkos::Experimental::SYCLDeviceUSMSpace,
                         KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ONEMKL(Kokkos::complex<double>, std::int32_t,
                         Kokkos::Experimental::SYCLDeviceUSMSpace,
                         KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)

KOKKOSSPARSE_SPMV_ONEMKL(float, std::int64_t,
                         Kokkos::Experimental::SYCLDeviceUSMSpace,
                         KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ONEMKL(double, std::int64_t,
                         Kokkos::Experimental::SYCLDeviceUSMSpace,
                         KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ONEMKL(Kokkos::complex<float>, std::int64_t,
                         Kokkos::Experimental::SYCLDeviceUSMSpace,
                         KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
KOKKOSSPARSE_SPMV_ONEMKL(Kokkos::complex<double>, std::int64_t,
                         Kokkos::Experimental::SYCLDeviceUSMSpace,
                         KOKKOSKERNELS_IMPL_COMPILE_LIBRARY)
#endif
}  // namespace Impl
}  // namespace KokkosSparse
#endif

#endif  // KOKKOSPARSE_SPMV_TPL_SPEC_DECL_HPP_
