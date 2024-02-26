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

/// SPMVAlgorithm values can be used to select different algorithms/methods for performing
/// SpMV computations.
enum SPMVAlgorithm {
  SPMV_DEFAULT,     /// Default algorithm: best overall performance for repeated applications of SpMV.
  SPMV_FAST_SETUP,  /// Best performance in the non-reuse case, where the handle is only used once.
  SPMV_NATIVE,      /// Use the best KokkosKernels implementation, even if a TPL implementation is available.
  SPMV_MERGE_PATH,  /// Use load-balancing merge path algorithm.
  SPMV_BSR_V41,     /// Use experimental version 4.1 algorithm (for BsrMatrix only)
  SPMV_BSR_V42,     /// Use experimental version 4.2 algorithm (for BsrMatrix only)
  SPMV_BSR_TC,      /// Use experimental tensor core algorithm (for BsrMatrix only)
}

namespace Impl {
  // Execution spaces do not support operator== in public interface, even though
  // in practice the major async/GPU spaces do have the feature.
  // This is a conservative check for whether e1 and e2 are known to be the
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
  // Data used by cuSPARSE >=10.3 for both single-vector (SpMV) and multi-vector (SpMM).
  // TODO: in future, this can also be used for BSR (cuSPARSE >=12.2)
  struct CuSparse10_SpMV_Data : public TPL_SpMV_Data<Kokkos::Cuda> {
    CuSparse10_SpMV_Data (const Kokkos::Cuda& exec) : TPL_SpMV_Data(exec) {}
    ~CuSparse10_SpMV_Data () {
      KOKKOS_IMPL_CUDA_SAFE_CALL(cudaFreeAsync(buffer, exec.cuda_stream()));
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroySpMat(mat));
    }

    // Algo for single vector case (SpMV)
    cusparseSpMVAlg_t algo;
    // Algo for multiple vector case (SpMM)
    cusparseSpMMAlg_t algo;
    cusparseSpMatDescr_t mat;
    size_t bufferSize = 0;
    void* buffer      = nullptr;
  };
#endif

  // Data used by cuSPARSE <10.3 for CRS, and >=9 for BSR
  struct CuSparse9_SpMV_Data : public TPL_SpMV_Data<Kokkos::Cuda> {
    CuSparse9_SpMV_Data(const Kokkos::Cuda& exec) : TPL_SpMV_Data(exec) {}
    ~CuSparse9_SpMV_Data() {
      KOKKOS_CUSPARSE_SAFE_CALL(cusparseDestroyMatDescr(mat));
    }

    cusparseMatDescr_t mat;
  };
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
  struct RocSparse_CRS_SpMV_Data : public TPL_SpMV_Data<Kokkos::HIP> {
    RocSparse_CRS_SpMV_Data(const Kokkos::HIP& exec) : TPL_SpMV_Data(exec) {}
    ~RocSparse_CRS_SpMV_Data() {
      // note: hipFree includes an implicit device synchronize
      KOKKOS_IMPL_HIP_SAFE_CALL(hipFree(buffer));
      KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_destroy_spmat_descr(spmat));
    }

    rocsparse_spmat_descr mat;
    size_t bufferSize = 0;
    void* buffer      = nullptr;
  };

  struct RocSparse_BSR_SpMV_Data : public TPL_SpMV_Data<Kokkos::HIP> {
    RocSparse_BSR_SpMV_Data(const Kokkos::HIP& exec) : TPL_SpMV_Data(exec) {}
    ~RocSparse_BSR_SpMV_Data() {
      // note: hipFree includes an implicit device synchronize
      KOKKOS_IMPL_HIP_SAFE_CALL(hipFree(buffer));
      KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(rocsparse_destroy_spmat_descr(spmat));
    }

   rocsparse_mat_descr mat;
    rocsparse_mat_info info;
    size_t bufferSize = 0;
    void* buffer      = nullptr;
  };
#endif

// note: header defining __INTEL_MKL__ is pulled in above by Utils_mkl.hpp
#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL

#if (__INTEL_MKL__ > 2017)
  // Data for classic MKL (both CRS and BSR)
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

  // IsBSR false: A is a CrsMatrix
  // IsBSR true: A is a BsrMatrix
  template <bool IsBSR, class ExecutionSpace, class MemorySpace, class Scalar, class Offset, class Ordinal>
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
///    algorithm to the spmv implementation, and also may store internal data which can be used to
///    speed up the spmv computation.
/// \tparam ExecutionSpace The space on which KokkosSparse::spmv will be run.
/// \tparam AMatrix A specialization of KokkosSparse::CrsMatrix or
/// KokkosSparse::BsrMatrix.
///
/// SPMVHandle's internal resources are lazily allocated and initialized by the first
/// spmv call.
///
/// SPMVHandle automatically cleans up all allocated resources when it is destructed.
/// No fencing by the user is required between the final spmv and cleanup.
///
/// A SPMVHandle instance can be used in any number of calls, with any execution space
/// instance and any X/Y vectors (with matching types) each call.
///
/// \warning However, all calls to spmv with a given instance of SPMVHandle must use the
/// same matrix.

template <class ExecutionSpace, class AMatrix, class XVector, class YVector>
class SPMVHandle : public Impl::SPMVHandleImpl<is_bsr_matrix_v<AMatrix>, ExecutionSpace, typename AMatrix::memory_space, typename AMatrix::non_const_value_type, typename AMatrix::non_const_size_type, typename AMatrix::non_const_ordinal_type>
{
  // Check all template parameters for compatibility with each other
  // NOTE: we do not require that ExecutionSpace matches AMatrix::execution_space.
  // For example, if the matrix's device is <Cuda, CudaHostPinnedSpace> it is allowed to run spmv on Serial.
  static_assert(is_crs_matrix_v<AMatrix> || is_bsr_matrix_v<AMatrix>,
                "SPMVHandle: AMatrix must be a specialization of CrsMatrix or "
                "BsrMatrix.");
  static_assert(Kokkos::is_view<XVector>::value,
                "SPMVHandle: XVector must be a Kokkos::View.");
  static_assert(Kokkos::is_view<YVector>::value,
                "SPMVHandle: YVector must be a Kokkos::View.");
  static_assert(XVector::rank() == YVector::rank(),
                "SPMVHandle: ranks of XVector and YVector must match.");
  static_assert(XVector::rank() == size_t(1) || YVector::rank() == size_t(2),
                "SPMVHandle: XVector and YVector must be both rank-1 or both rank-2.");
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename AMatrix::memory_space>::accessible,
      "SPMVHandle: AMatrix must be accessible from ExecutionSpace");
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename XVector::memory_space>::accessible,
      "SPMVHandle: XVector must be accessible from ExecutionSpace");
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename YVector::memory_space>::accessible,
      "SPMVHandle: YVector must be accessible from ExecutionSpace");
  // Prevent copying (this object does not support reference counting)
  SPMVHandle(const SPMVHandle&) = delete;
  SPMVHandle& operator=(const SPMVHandle&) = delete;

 public:
  /// \brief Create a new SPMVHandle using the given algorithm.
  ///        Depending on the TPLs 
  SPMVHandle(SPMVAlgorithm algo_ = SPMV_DEFAULT)
      : Impl::SPMVHandleImpl(algo_) {}

  SPMVAlgorithm get_algorithm() const {return this->algo}

  // Note: these typedef names cannot shadow template parameters
  using AMatrixType = AMatrix;
  using XVectorType = XVector;
  using YVectorType = YVector;
  using ExecutionSpaceType = ExecutionSpace;
};

}  // namespace KokkosSparse

#endif
