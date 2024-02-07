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

#include <Kokkos_Core.hpp>
// Use TPL utilities for safely finalizing matrix descriptors, etc.
#include "KokkosSparse_Utils_cusparse.hpp"
#include "KokkosSparse_Utils_rocsparse.hpp"
#include "KokkosSparse_Utils_mkl.hpp"

#ifndef KOKKOSSPARSE_SPMV_HANDLE_HPP_
#define KOKKOSSPARSE_SPMV_HANDLE_HPP_

namespace KokkosSparse {

enum SPMVAlgorithm {
  SPMV_DEFAULT,
  SPMV_FAST_SETUP,
  SPMV_NATIVE,
  SPMV_MERGE_PATH
}

namespace Impl
{
  template<typename ExecutionSpace>
  struct TPL_SpMV_Data
  {
    virtual ~TPL_SpMV_Data(){}
  };
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#if defined(CUSPARSE_VERSION) && (10300 <= CUSPARSE_VERSION)
  // Data used by cuSPARSE >=10.3
  struct CuSparse_SpMV_Data : public TPL_SpMV_Data<Kokkos::Cuda>
  {
    ~CuSparse_SpMV_Data()
    {
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroySpMat(mat));
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFree(buffer));
    }

    cusparseSpMVAlg_t algo;
    cusparseSpMatDescr_t mat;
    size_t bufferSize = 0;
    void* buffer = nullptr;
  };
#else
  // Data used by cuSPARSE <10.3
  struct CuSparse_SpMV_Data : public TPL_SpMV_Data<Kokkos::Cuda>
  {
    ~CuSparse_SpMV_Data()
    {
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyMatDescr(mat));
    }

    cusparseMatDescr_t mat;
  };
#endif
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
  struct RocSparse_SpMV_Data : public TPL_SpMV_Data
  {
    ~RocSparse_SpMV_Data()
    {
      KOKKOS_IMPL_HIP_SAFE_CALL(hipFree(buffer));
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_destroy_mat_descr(mat));
  KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_destroy_spmat_descr(spmat));
    }

  rocsparse_spmv_alg  algo;
    rocsparse_mat_descr mat;
    rocsparse_spmat_descr spmat;
    size_t bufferSize = 0;
    void* buffer = nullptr;
  };
#endif

//note: header defining __INTEL_MKL__ is pulled in above by Utils_mkl.hpp
#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
  
#if (__INTEL_MKL__ > 2017)
  template<typename ExecutionSpace>
  struct MKL_SpMV_Data : public TPL_SpMV_Data<ExecutionSpace>
  {
    ~MKL_SpMV_Data()
    {
  KOKKOSKERNELS_MKL_SAFE_CALL(mkl_sparse_destroy(mat));
  // descr is just a plain-old-data struct
    }

    sparse_matrix_t mat;
    matrix_descr descr;
  };
#endif

#if defined(KOKKOS_ENABLE_SYCL) && \
    !defined(KOKKOSKERNELS_ENABLE_TPL_MKL_SYCL_OVERRIDE)
  struct OneMKL_SpMV_Data : public TPL_SpMV_Data<Kokkos::Experimental::SYCL>
  {
    ~OneMKL_SpMV_Data()
    {
      // Make sure no spmv is still running with this handle, if exec uses an out-of-order queue (rare case)
      if(!exec.sycl_queue().is_in_order())
        exec.fence();
#if INTEL_MKL_VERSION >= 20230200
    // MKL 2023.2 and up make this async release okay even though it takes a
    // pointer to mat, which is going out of scope after this destructor
    oneapi::mkl::sparse::release_matrix_handle(exec.sycl_queue(), &mat);
#else
    // But in older versions, wait on ev_release before letting mat go out of scope
    auto ev_release = oneapi::mkl::sparse::release_matrix_handle(exec.sycl_queue(), &mat);
    ev_release.wait();
#endif
    }

    oneapi::mkl::sparse::matrix_handle_t mat;
    // Remember the most recent execution space instance to use this handle.
    Kokkos::Experimental::SYCL exec;
  };
#endif
#endif
}

template <class ExecutionSpace, class AMatrix>
class SPMVHandle
{
public:
  SPMVHandle(SPMVAlgorithm algo_ = SPMV_DEFAULT)
    : algo(algo_), tpl(nullptr)
  {}

  ~SPMVHandle()
  {
    if(tpl) delete tpl;
  }

  SPMVAlgorithm get_algorithm() const {return algo};

  void set_last_used_exec(const ExecutionSpace& exec)
  {
    if(tpl)
      tpl->set_exec_space(exec);
  }

private:
  SPMVAlgorithm algo;
  // All TPL "Data" structs are defined above even if the TPL is not enabled.
  // This way, no macro logic is required here.
  TPL_SpMV_Data<ExecutionSpace>* tpl;
  bool isSetUp;
};

}

#endif
