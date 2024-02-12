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

namespace Impl {
  // Execution spaces do not support operator== in public interface.
  // So this is a conservative check for whether e1 and e2 are known to be the
  // same. If it cannot be determined, assume they are different.
  template <typename ExecutionSpace>
  inline bool exec_spaces_same(const ExecutionSpace& e1,
                               const ExecutionSpace& e2) {
    return false;
  }

#ifdef KOKKOS_ENABLE_CUDA
  template <>
  inline bool exec_spaces_same<Kokkos::Cuda>(const Kokkos::Cuda& e1,
                                             const Kokkos::Cuda& e2) {
    return e1.impl_internal_space_instance() ==
           e2.impl_internal_space_instance();
  }
#endif
#ifdef KOKKOS_ENABLE_HIP
  template <>
  inline bool exec_spaces_same<Kokkos::HIP>(const Kokkos::HIP& e1,
                                            const Kokkos::HIP& e2) {
    return e1.impl_internal_space_instance() ==
           e2.impl_internal_space_instance();
  }
#endif
#ifdef KOKKOS_ENABLE_SYCL
  template <>
  inline bool exec_spaces_same<Kokkos::Experimental::SYCL>(
      const Kokkos::Experimental::SYCL& e1,
      const Kokkos::Experimental::SYCL& e2) {
    return e1.impl_internal_space_instance() ==
           e2.impl_internal_space_instance();
  }
#endif

  template <typename ExecutionSpace>
  struct TPL_SpMV_Data {
    // Disallow default construction: must provide the initial execution space
    TPL_SpMV_Data() = delete;
    TPL_SpMV_Data(const ExecutionSpace& exec_) : exec(exec) {}
    void set_apply_called() { apply_called = true; }
    void set_exec_space(const ExecutionSpace& new_exec) {
      // Check if new_exec is different from (old) exec.
      // If it is, fence the old exec now.
      // That way, SPMVHandle cleanup doesn't need
      // to worry about resources still being in use on the old exec.
      if (!exec_spaces_same(exec, new_exec)) {
        exec.fence();
        exec = new_exec;
      }
    }
    virtual ~TPL_SpMV_Data() {}
    ExecutionSpace exec;
  };
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#if defined(CUSPARSE_VERSION) && (10300 <= CUSPARSE_VERSION)
  // Data used by cuSPARSE >=10.3
  struct CuSparse_SpMV_Data : public TPL_SpMV_Data<Kokkos::Cuda> {
    CuSparse_SpMV_Data(const Kokkos::Cuda& exec) : TPL_SpMV_Data(exec) {}
    ~CuSparse_SpMV_Data() {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeAsync(buffer, exec.cuda_stream()));
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroySpMat(mat));
    }

    cusparseSpMVAlg_t algo;
    cusparseSpMatDescr_t mat;
    size_t bufferSize = 0;
    void* buffer      = nullptr;
  };
#else
  // Data used by cuSPARSE <10.3
  struct CuSparse_SpMV_Data : public TPL_SpMV_Data<Kokkos::Cuda> {
    CuSparse_SpMV_Data(const Kokkos::Cuda& exec) : TPL_SpMV_Data(exec) {}
    ~CuSparse_SpMV_Data() {
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyMatDescr(mat));
    }

    cusparseMatDescr_t mat;
  };
#endif
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
  struct RocSparse_SpMV_Data : public TPL_SpMV_Data<Kokkos::HIP> {
    RocSparse_SpMV_Data(const Kokkos::HIP& exec) : TPL_SpMV_Data(exec) {}
    ~RocSparse_SpMV_Data() {
      // note: hipFree includes an implicit device synchronize
      KOKKOS_IMPL_HIP_SAFE_CALL(hipFree(buffer));
      KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_destroy_spmat_descr(spmat));
    }

    rocsparse_spmat_descr mat;
    size_t bufferSize = 0;
    void* buffer      = nullptr;
  };
#endif

// note: header defining __INTEL_MKL__ is pulled in above by Utils_mkl.hpp
#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL

#if (__INTEL_MKL__ > 2017)
  template <typename ExecutionSpace>
  struct MKL_SpMV_Data : public TPL_SpMV_Data<ExecutionSpace> {
    MKL_SpMV_Data(const ExecutionSpac& exec) : TPL_SpMV_Data(exec) {}
    ~MKL_SpMV_Data() {
      KOKKOSKERNELS_MKL_SAFE_CALL(mkl_sparse_destroy(mat));
      // descr is just a plain-old-data struct, no cleanup to do
    }

    sparse_matrix_t mat;
    matrix_descr descr;
  };
#endif

#if defined(KOKKOS_ENABLE_SYCL) && \
    !defined(KOKKOSKERNELS_ENABLE_TPL_MKL_SYCL_OVERRIDE)
  struct OneMKL_SpMV_Data : public TPL_SpMV_Data<Kokkos::Experimental::SYCL> {
    OneMKL_SpMV_Data(const Kokkos::Experimental::SYCL& exec)
        : TPL_SpMV_Data(exec) {}
    ~OneMKL_SpMV_Data() {
      // Make sure no spmv is still running with this handle, if exec uses an
      // out-of-order queue (rare case)
      if (!exec.sycl_queue().is_in_order()) exec.fence();
#if INTEL_MKL_VERSION >= 20230200
      // MKL 2023.2 and up make this async release okay even though it takes a
      // pointer to mat, which is going out of scope after this destructor
      oneapi::mkl::sparse::release_matrix_handle(exec.sycl_queue(), &mat);
#else
      // But in older versions, wait on ev_release before letting mat go out of
      // scope
      auto ev_release =
          oneapi::mkl::sparse::release_matrix_handle(exec.sycl_queue(), &mat);
      ev_release.wait();
#endif
    }

    oneapi::mkl::sparse::matrix_handle_t mat;
  };
#endif
#endif

  template <class ExecutionSpace, class AMatrix>
  struct SPMVHandleImpl {
    SPMVHandleImpl(SPMVAlgorithm algo_) : algo(algo_) {}
    ~SPMVHandleImpl() {
      if (tpl) delete tpl;
    }
    void set_exec_space(const ExecutionSpace& exec) {
      if (tpl) tpl->set_exec_space(exec);
    }
    bool is_set_up;
    SPMVAlgorithm algo;
    TPL_SpMV_Data<ExecutionSpace>* tpl;
  };
}

/// \class SPMVHandle
/// \brief Opaque handle type for KokkosSparse::spmv. It passes the choice of
/// algorithm to the spmv
///    implementation, and also may store internal data which can be used to
///    speed up the spmv computation.
/// \tparam ExecutionSpace The space on which KokkosSparse::spmv will be run.
/// \tparam AMatrix A specialization of KokkosSparse::CrsMatrix or
/// KokkosSparse::BsrMatrix.
///
/// \warning All calls to spmv with a given instance of SPMVHandle must use the
/// same matrix.

template <class ExecutionSpace, class AMatrix, class XVector, class YVector>
class SPMVHandle : public Impl::SPMVHandleImpl<ExecutionSpace, AMatrix> {
  static_assert(
      std::is_same_v<ExecutionSpace, typename AMatrix::execution_space>,
      "SPMVHandle: ExecutionSpace must match AMatrix::execution_space.");
  static_assert(is_crs_matrix_v<AMatrix> || is_bsr_matrix_v<AMatrix>,
                "SPMVHandle: AMatrix must be a specialization of CrsMatrix or "
                "BsrMatrix.");

 public:
  SPMVHandle(SPMVAlgorithm algo_ = SPMV_DEFAULT)
      : Impl::SPMVHandleImpl(algo_) {}

  SPMVAlgorithm get_algorithm() const {return this->algo};
};

}  // namespace KokkosSparse

#endif
