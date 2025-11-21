// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSPARSE_SPMV_SELLMATRIX_TPL_SPEC_DECL_HPP
#define KOKKOSPARSE_SPMV_SELLMATRIX_TPL_SPEC_DECL_HPP

// cuSPARSE
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include "cusparse.h"
#include "KokkosSparse_Utils_cusparse.hpp"
#include "KokkosKernels_tpl_handles_decl.hpp"

namespace KokkosSparse {
namespace Impl {

template <class AMatrix, class XVector, class YVector>
void spmv_sellmatrix_cusparse(const Kokkos::Cuda& exec, const char mode[], typename YVector::const_value_type& alpha,
                              const AMatrix& A, const XVector& x, typename YVector::const_value_type& beta,
                              const YVector& y) {
  using offset_type = typename AMatrix::non_const_size_type;
  using entry_type  = typename AMatrix::non_const_ordinal_type;
  using value_type  = typename AMatrix::non_const_value_type;

  /* initialize cusparse library */
  cusparseHandle_t cusparseHandle = KokkosKernels::Impl::CusparseSingleton::singleton().cusparseHandle;
  /* Set cuSPARSE to use the given stream until this function exits */
  TemporarySetCusparseStream tscs(cusparseHandle, exec);

  /* Set the operation mode */
  cusparseOperation_t myCusparseOperation;
  switch (toupper(mode[0])) {
    case 'N': myCusparseOperation = CUSPARSE_OPERATION_NON_TRANSPOSE; break;
    case 'T': myCusparseOperation = CUSPARSE_OPERATION_TRANSPOSE; break;
    case 'H': myCusparseOperation = CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE; break;
    default: {
      std::ostringstream out;
      out << "Mode " << mode << " invalid for cuSPARSE SpMV.\n";
      throw std::invalid_argument(out.str());
    }
  }
  // cuSPARSE doesn't directly support mode H with real values, but this is
  // equivalent to mode T
  if (myCusparseOperation == CUSPARSE_OPERATION_CONJUGATE_TRANSPOSE &&
      !KokkosKernels::ArithTraits<value_type>::isComplex)
    myCusparseOperation = CUSPARSE_OPERATION_TRANSPOSE;

  /* Check that cusparse can handle the types of the input Kokkos::CrsMatrix */
  const cudaDataType myCudaDataType              = cuda_data_type_from<value_type>();
  const cusparseIndexType_t myCusparseOffsetType = cusparse_index_type_t_from<offset_type>();
  const cusparseIndexType_t myCusparseEntryType  = cusparse_index_type_t_from<entry_type>();

  /* create lhs and rhs */
  cusparseConstDnVecDescr_t vecX;
  cusparseDnVecDescr_t vecY;
  KOKKOSSPARSE_IMPL_CUSPARSE_SAFE_CALL(
      cusparseCreateConstDnVec(&vecX, x.extent_int(0), (const void*)x.data(), myCudaDataType));
  KOKKOSSPARSE_IMPL_CUSPARSE_SAFE_CALL(cusparseCreateDnVec(&vecY, y.extent_int(0), (void*)y.data(), myCudaDataType));

  /* create matrix */
  cusparseConstSpMatDescr_t matA;
  KOKKOSSPARSE_IMPL_CUSPARSE_SAFE_CALL(cusparseCreateConstSlicedEll(
      &matA, A.num_rows, A.num_cols, A.nnz, A.sell_nnz, A.num_rows_per_slice, (const void*)A.slice_offsets.data(),
      (const void*)A.entries.data(), (const void*)A.values.data(), myCusparseOffsetType, myCusparseEntryType,
      CUSPARSE_INDEX_BASE_ZERO, myCudaDataType));

  /* size and allocate buffer */
  size_t bufferSize = 0;
  void* buffer      = nullptr;
  KOKKOSSPARSE_IMPL_CUSPARSE_SAFE_CALL(cusparseSpMV_bufferSize(cusparseHandle, myCusparseOperation, &alpha, matA, vecX,
                                                               &beta, vecY, myCudaDataType, CUSPARSE_SPMV_SELL_ALG1,
                                                               &bufferSize));
  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaMallocAsync(&buffer, bufferSize, exec.cuda_stream()));

  /* perform SpMV */
  KOKKOSSPARSE_IMPL_CUSPARSE_SAFE_CALL(cusparseSpMV(cusparseHandle, myCusparseOperation, &alpha, matA, vecX, &beta,
                                                    vecY, myCudaDataType, CUSPARSE_SPMV_SELL_ALG1, buffer));

  KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeAsync(buffer, exec.cuda_stream()));
  KOKKOSSPARSE_IMPL_CUSPARSE_SAFE_CALL(cusparseDestroySpMat(matA));
  KOKKOSSPARSE_IMPL_CUSPARSE_SAFE_CALL(cusparseDestroyDnVec(vecX));
  KOKKOSSPARSE_IMPL_CUSPARSE_SAFE_CALL(cusparseDestroyDnVec(vecY));
}

#define KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(SCALAR, ORDINAL, OFFSET, LAYOUT, SPACE)                              \
  template <>                                                                                                      \
  struct SPMV_SELLMATRIX<                                                                                          \
      Kokkos::Cuda,                                                                                                \
      KokkosSparse::Experimental::SellMatrix<SCALAR const, ORDINAL const, Kokkos::Device<Kokkos::Cuda, SPACE>,     \
                                             Kokkos::MemoryTraits<Kokkos::Unmanaged>, OFFSET const>,               \
      Kokkos::View<SCALAR const*, LAYOUT, Kokkos::Device<Kokkos::Cuda, SPACE>,                                     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>,                                \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<Kokkos::Cuda, SPACE>, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
      true> {                                                                                                      \
    using device_type       = Kokkos::Device<Kokkos::Cuda, SPACE>;                                                 \
    using memory_trait_type = Kokkos::MemoryTraits<Kokkos::Unmanaged>;                                             \
    using AMatrix           = KokkosSparse::Experimental::SellMatrix<SCALAR const, ORDINAL const, device_type,     \
                                                           memory_trait_type, OFFSET const>;             \
    using XVector           = Kokkos::View<SCALAR const*, LAYOUT, device_type,                                     \
                                 Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess>>;        \
    using YVector           = Kokkos::View<SCALAR*, LAYOUT, device_type, memory_trait_type>;                       \
    using coefficient_type  = typename YVector::non_const_value_type;                                              \
                                                                                                                   \
    static void spmv(const Kokkos::Cuda& exec, const char mode[], const coefficient_type& alpha, const AMatrix& A, \
                     const XVector& x, const coefficient_type& beta, const YVector& y) {                           \
      std::string label = "KokkosSparse::spmv[TPL_CUSPARSE," + KokkosKernels::ArithTraits<SCALAR>::name() + "]";   \
      Kokkos::Profiling::pushRegion(label);                                                                        \
      spmv_sellmatrix_cusparse(exec, mode, alpha, A, x, beta, y);                                                  \
      Kokkos::Profiling::popRegion();                                                                              \
    }                                                                                                              \
  };

KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(double, int, int, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(double, int, int, Kokkos::LayoutRight, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(float, int, int, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(float, int, int, Kokkos::LayoutRight, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<double>, int, int, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<double>, int, int, Kokkos::LayoutRight, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<float>, int, int, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<float>, int, int, Kokkos::LayoutRight, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(double, int, int, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(double, int, int, Kokkos::LayoutRight, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(float, int, int, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(float, int, int, Kokkos::LayoutRight, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<double>, int, int, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<double>, int, int, Kokkos::LayoutRight, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<float>, int, int, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<float>, int, int, Kokkos::LayoutRight, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(double, int64_t, size_t, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(double, int64_t, size_t, Kokkos::LayoutRight, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(float, int64_t, size_t, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(float, int64_t, size_t, Kokkos::LayoutRight, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<double>, int64_t, size_t, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<double>, int64_t, size_t, Kokkos::LayoutRight, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<float>, int64_t, size_t, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<float>, int64_t, size_t, Kokkos::LayoutRight, Kokkos::CudaSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(double, int64_t, size_t, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(double, int64_t, size_t, Kokkos::LayoutRight, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(float, int64_t, size_t, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(float, int64_t, size_t, Kokkos::LayoutRight, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<double>, int64_t, size_t, Kokkos::LayoutLeft,
                                      Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<double>, int64_t, size_t, Kokkos::LayoutRight,
                                      Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<float>, int64_t, size_t, Kokkos::LayoutLeft, Kokkos::CudaUVMSpace)
KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE(Kokkos::complex<float>, int64_t, size_t, Kokkos::LayoutRight,
                                      Kokkos::CudaUVMSpace)

#undef KOKKOSSPARSE_SPMV_SELLMATRIX_CUSPARSE

}  // namespace Impl
}  // namespace KokkosSparse
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUSPARSE

#endif  // KOKKOSPARSE_SPMV_SELLMATRIX_TPL_SPEC_DECL_HPP
