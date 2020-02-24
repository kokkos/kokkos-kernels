/*
//@HEADER
// ************************************************************************
//
//               KokkosKernels 0.9: Linear Algebra and Graph Kernels
//                 Copyright 2017 Sandia Corporation
//
// Under the terms of Contract DE-AC04-94AL85000 with Sandia Corporation,
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
// THIS SOFTWARE IS PROVIDED BY SANDIA CORPORATION "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL SANDIA CORPORATION OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFIS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOSPARSE_SPMV_TPL_SPEC_DECL_HPP_
#define KOKKOSPARSE_SPMV_TPL_SPEC_DECL_HPP_

// cuSPARSE
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#include "cusparse.h"

namespace KokkosSparse {
namespace Impl {

  template <class AMatrix, class XVector, class YVector>
  void spmv_cusparse(const char mode[],
		     typename YVector::non_const_value_type const & alpha,
		     const AMatrix& A,
		     const XVector& x,
		     typename YVector::non_const_value_type const & beta,
		     const YVector& y) {
    using offset_type  = typename AMatrix::non_const_size_type;
    // using ordinal_type = typename AMatrix::non_const_ordinal_type;
    using value_type   = typename AMatrix::non_const_value_type;

#if defined(CUSPARSE_VERSION) && (10300 <= CUSPARSE_VERSION)

    cudaError_t      cuError;
    cusparseStatus_t status;
    cusparseHandle_t handle=0;

    cusparseIndexType_t myCusparseIndexType;
    if(std::is_same<offset_type, int>::value)     {myCusparseIndexType = CUSPARSE_INDEX_32I;}
    if(std::is_same<offset_type, int64_t>::value) {myCusparseIndexType = CUSPARSE_INDEX_64I;}
    cudaDataType myCudaDataType;
    if(std::is_same<value_type, float>::value)  {myCudaDataType = CUDA_R_32F;}
    if(std::is_same<value_type, double>::value) {myCudaDataType = CUDA_R_64F;}

    /* initialize cusparse library */
    status = cusparseCreate(&handle);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cusparse was not initialized correctly");
    }

    /* create matrix */
    cusparseSpMatDescr_t A_cusparse;
    status = cusparseCreateCsr(&A_cusparse, A.numRows(), A.numCols(), A.nnz(),
			       const_cast<offset_type*>(A.graph.row_map.data()),
			       const_cast<offset_type*>(A.graph.entries.data()),
			       const_cast<value_type*>(A.values.data()),
			       myCusparseIndexType,
			       myCusparseIndexType,
			       CUSPARSE_INDEX_BASE_ZERO,
			       myCudaDataType);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cusparse matrix was not created correctly");
    }

    /* create lhs and rhs */
    cusparseDnVecDescr_t vecX, vecY;
    status = cusparseCreateDnVec(&vecX, x.extent_int(0), const_cast<value_type*>(x.data()), myCudaDataType);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cusparse vecX was not created correctly");
    }
    status = cusparseCreateDnVec(&vecY, y.extent_int(0), const_cast<value_type*>(y.data()), myCudaDataType);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cusparse vecY was not created correctly");
    }

    size_t bufferSize = 0;
    void*  dBuffer    = NULL;
    status = cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
				     &alpha, A_cusparse, vecX, &beta, vecY, myCudaDataType,
				     CUSPARSE_CSRMV_ALG1, &bufferSize);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cusparse bufferSize computation failed");
    }
    cuError = cudaMalloc(&dBuffer, bufferSize);
    if (cuError != cudaSuccess) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cuda buffer allocation failed");
    }

    /* perform SpMV */
    status = cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE,
			  &alpha, A_cusparse, vecX, &beta, vecY, myCudaDataType,
			  CUSPARSE_CSRMV_ALG1, dBuffer);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cusparseSpMV() failed");
    }

    cuError = cudaFree(dBuffer);
    if (cuError != cudaSuccess) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cuda buffer deallocation failed");
    }
    status = cusparseDestroyDnVec(vecX);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cusparse vecX was not destroyed correctly");
    }
    status = cusparseDestroyDnVec(vecY);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cusparse vecY was not destroyed correctly");
    }
    status = cusparseDestroySpMat(A_cusparse);
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cusparse matrix was not destroyed correctly");
    }
    status = cusparseDestroy(handle);
    handle = 0;
    if (status != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cusparse handle was not desctroyed correctly");
    }

#else

    /* Initialize cusparse */
    cusparseStatus_t cusparseStatus;
    cusparseHandle_t cusparseHandle=0;
    cusparseStatus = cusparseCreate(&cusparseHandle);
    if(cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: cannot initialize cusparse handle");
    }

    /* create and set the matrix descriptor */
    cusparseMatDescr_t descrA = 0;
    cusparseStatus = cusparseCreateMatDescr(&descrA);
    if(cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: error creating the matrix descriptor");
    }
    cusparseStatus = cusparseSetMatType(descrA, CUSPARSE_MATRIX_TYPE_GENERAL);
    if(cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: error setting the matrix type");
    }
    cusparseStatus = cusparseSetMatIndexBase(descrA, CUSPARSE_INDEX_BASE_ZERO);
    if(cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
      throw std::runtime_error("KokkosSparse::spmv[TPL_CUSPARSE,double]: error setting the matrix index base");
    }

    /* perform the actual SpMV operation */
    if(std::is_same<int, offset_type>::value) {
      if (std::is_same<value_type,float>::value) {
	cusparseStatus = cusparseScsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					A.numRows(), A.numCols(), A.nnz(),
					(const float *) &alpha, descrA,
					(const float *) A.values.data(), A.graph.row_map.data(), A.graph.entries.data(),
					(const float *) x.data(),
					(const float *) &beta,
					(float *) y.data());

      } else  if (std::is_same<value_type,double>::value) {
	cusparseStatus = cusparseDcsrmv(cusparseHandle, CUSPARSE_OPERATION_NON_TRANSPOSE,
					A.numRows(), A.numCols(), A.nnz(),
					(double const *) &alpha, descrA,
					(double const *) A.values.data(), A.graph.row_map.data(), A.graph.entries.data(),
					(double const *) x.data(),
					(double const *) &beta,
					(double *) y.data());
      } else {
	throw std::logic_error("Trying to call cusparse SpMV with a scalar type that is not float or double!");
      }
    } else {
      throw std::logic_error("Trying to call cusparse SpMV with an offset type that is not int!");
    }

    cusparseStatus = cusparseDestroyMatDescr(descrA);
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
      throw("KokkosSparse::spmv[TPL_CUSPARSE,double]: matrix descriptor was not desctroyed correctly");
    }
    cusparseStatus = cusparseDestroy(cusparseHandle);
    cusparseHandle = 0;
    if (cusparseStatus != CUSPARSE_STATUS_SUCCESS) {
      throw("KokkosSparse::spmv[TPL_CUSPARSE,double]: cusparse handle was not desctroyed correctly");
    }

#endif // CUSPARSE_VERSION
  }

#define KOKKOSSPARSE_SPMV_CUSPARSE(SCALAR, OFFSET, LAYOUT, COMPILE_LIBRARY) \
  template<>								\
  struct SPMV<SCALAR const,  OFFSET const, Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged>, OFFSET const, \
	      SCALAR const*, LAYOUT,       Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess>, \
	      SCALAR*,       LAYOUT,       Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>, Kokkos::MemoryTraits<Kokkos::Unmanaged>, \
	      true, COMPILE_LIBRARY> {					\
    using device_type = Kokkos::Device<Kokkos::Cuda, Kokkos::CudaSpace>; \
    using memory_trait_type = Kokkos::MemoryTraits<Kokkos::Unmanaged>;	\
    using AMatrix = CrsMatrix<SCALAR const, OFFSET const, device_type, memory_trait_type, OFFSET const>; \
    using XVector = Kokkos::View<SCALAR const*, LAYOUT,device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged|Kokkos::RandomAccess>>; \
    using YVector = Kokkos::View<SCALAR*, LAYOUT, device_type, memory_trait_type>; \
									\
    using coefficient_type = typename YVector::non_const_value_type;	\
									\
    static void spmv (const char mode[],				\
		      const coefficient_type& alpha,			\
		      const AMatrix& A,					\
		      const XVector& x,					\
		      const coefficient_type& beta,			\
		      const YVector& y) {				\
      Kokkos::Profiling::pushRegion("KokkosSparse::spmv[TPL_CUSPARSE,double]");	\
      spmv_cusparse(mode, alpha, A, x, beta, y);			\
      Kokkos::Profiling::popRegion();					\
    }									\
  };

  KOKKOSSPARSE_SPMV_CUSPARSE(double, int, Kokkos::LayoutLeft,  true)
  KOKKOSSPARSE_SPMV_CUSPARSE(double, int, Kokkos::LayoutLeft,  false)
  KOKKOSSPARSE_SPMV_CUSPARSE(double, int, Kokkos::LayoutRight, true)
  KOKKOSSPARSE_SPMV_CUSPARSE(double, int, Kokkos::LayoutRight, false)
  KOKKOSSPARSE_SPMV_CUSPARSE(float,  int, Kokkos::LayoutLeft,  true)
  KOKKOSSPARSE_SPMV_CUSPARSE(float,  int, Kokkos::LayoutLeft,  false)
  KOKKOSSPARSE_SPMV_CUSPARSE(float,  int, Kokkos::LayoutRight, true)
  KOKKOSSPARSE_SPMV_CUSPARSE(float,  int, Kokkos::LayoutRight, false)

#undef KOKKOSSPARSE_SPMV_CUSPARSE

} // namespace Impl
} // namespace KokkosSparse
#endif // KOKKOSKERNELS_ENABLE_TPL_CUSPARSE

#endif // KOKKOSPARSE_SPMV_TPL_SPEC_DECL_HPP_
