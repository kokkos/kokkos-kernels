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

#ifndef KOKKOSLAPACK_TPL_SPEC_HPP_
#define KOKKOSLAPACK_TPL_SPEC_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSOLVER
#include "cuda_runtime.h"
//#include "cublas_v2.h"
#include "cusolverDn.h"

namespace KokkosLapack {
namespace Impl {

struct CudaLapackSingleton {
  cusolverDnHandle_t handle;

  CudaLapackSingleton();

  static CudaLapackSingleton& singleton();
};

inline void cusolver_internal_error_throw(cusolverStatus_t cusolverState,
                                          const char* name, const char* file,
                                          const int line) {
  std::ostringstream out;
  // out << name << " error( " << cusolverGetStatusName(cusolverState)
  //     << "): " << cusolverGetStatusString(cusolverState);
  out << name << " error( ";
  switch (cusolverState) {
    case CUSOLVER_STATUS_NOT_INITIALIZED:
      out << "CUSOLVER_STATUS_NOT_INITIALIZED): the library was not "
             "initialized.";
      break;
    case CUSOLVER_STATUS_ALLOC_FAILED:
      out << "CUSOLVER_STATUS_ALLOC_FAILED): the resource allocation failed.";
      break;
    case CUSOLVER_STATUS_INVALID_VALUE:
      out << "CUSOLVER_STATUS_INVALID_VALUE): an invalid numerical value was "
             "used as an argument.";
      break;
    case CUSOLVER_STATUS_ARCH_MISMATCH:
      out << "CUSOLVER_STATUS_ARCH_MISMATCH): an absent device architectural "
             "feature is required.";
      break;
    case CUSOLVER_STATUS_MAPPING_ERROR:
      out << "CUSOLVER_STATUS_MAPPING_ERROR): an access to GPU memory space "
             "failed.";
      break;
    case CUSOLVER_STATUS_EXECUTION_FAILED:
      out << "CUSOLVER_STATUS_EXECUTION_FAILED): the GPU program failed to "
             "execute.";
      break;
    case CUSOLVER_STATUS_INTERNAL_ERROR:
      out << "CUSOLVER_STATUS_INTERNAL_ERROR): an internal operation failed.";
      break;
    case CUSOLVER_STATUS_NOT_SUPPORTED:
      out << "CUSOLVER_STATUS_NOT_SUPPORTED): the feature required is not "
             "supported.";
      break;
    default: out << "unrecognized error code): this is bad!"; break;
  }
  if (file) {
    out << " " << file << ":" << line;
  }
  throw std::runtime_error(out.str());
}

inline void cusolver_internal_safe_call(cusolverStatus_t cusolverState,
                                        const char* name,
                                        const char* file = nullptr,
                                        const int line   = 0) {
  if (CUSOLVER_STATUS_SUCCESS != cusolverState) {
    cusolver_internal_error_throw(cusolverState, name, file, line);
  }
}

// The macro below defines the interface for the safe cusolver calls.
// The functions themselves are protected by impl namespace and this
// is not meant to be used by external application or libraries.
#define KOKKOS_CUSOLVER_SAFE_CALL_IMPL(call)                             \
  KokkosLapack::Impl::cusolver_internal_safe_call(call, #call, __FILE__, \
                                                  __LINE__)

/// \brief This function converts KK transpose mode to cusolver transpose mode
inline cublasOperation_t trans_mode_kk_to_cusolver(const char kkMode[]) {
  cublasOperation_t trans;
  if ((kkMode[0] == 'N') || (kkMode[0] == 'n'))
    trans = CUBLAS_OP_N;
  else if ((kkMode[0] == 'T') || (kkMode[0] == 't'))
    trans = CUBLAS_OP_T;
  else
    trans = CUBLAS_OP_C;
  return trans;
}

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUSOLVER

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER
#include <rocsolver/rocsolver.h>

namespace KokkosLapack {
namespace Impl {

struct RocsolverSingleton {
  rocsolver_handle handle;

  RocsolverSingleton();

  static RocsolverSingleton& singleton();
};

inline void rocsolver_internal_error_throw(rocsolver_status rocsolverState,
                                           const char* name, const char* file,
                                           const int line) {
  std::ostringstream out;
  out << name << " error( ";
  switch (rocsolverState) {
    case rocsolver_status_invalid_handle:
      out << "rocsolver_status_invalid_handle): handle not initialized, "
             "invalid "
             "or null.";
      break;
    case rocsolver_status_not_implemented:
      out << "rocsolver_status_not_implemented): function is not implemented.";
      break;
    case rocsolver_status_invalid_pointer:
      out << "rocsolver_status_invalid_pointer): invalid pointer argument.";
      break;
    case rocsolver_status_invalid_size:
      out << "rocsolver_status_invalid_size): invalid size argument.";
      break;
    case rocsolver_status_memory_error:
      out << "rocsolver_status_memory_error): failed internal memory "
             "allocation, "
             "copy or dealloc.";
      break;
    case rocsolver_status_internal_error:
      out << "rocsolver_status_internal_error): other internal library "
             "failure.";
      break;
    case rocsolver_status_perf_degraded:
      out << "rocsolver_status_perf_degraded): performance degraded due to low "
             "device memory.";
      break;
    case rocsolver_status_size_query_mismatch:
      out << "unmatched start/stop size query): .";
      break;
    case rocsolver_status_size_increased:
      out << "rocsolver_status_size_increased): queried device memory size "
             "increased.";
      break;
    case rocsolver_status_size_unchanged:
      out << "rocsolver_status_size_unchanged): queried device memory size "
             "unchanged.";
      break;
    case rocsolver_status_invalid_value:
      out << "rocsolver_status_invalid_value): passed argument not valid.";
      break;
    case rocsolver_status_continue:
      out << "rocsolver_status_continue): nothing preventing function to "
             "proceed.";
      break;
    case rocsolver_status_check_numerics_fail:
      out << "rocsolver_status_check_numerics_fail): will be set if the "
             "vector/matrix has a NaN or an Infinity.";
      break;
    default: out << "unrecognized error code): this is bad!"; break;
  }
  if (file) {
    out << " " << file << ":" << line;
  }
  throw std::runtime_error(out.str());
}

inline void rocsolver_internal_safe_call(rocsolver_status rocsolverState,
                                         const char* name,
                                         const char* file = nullptr,
                                         const int line   = 0) {
  if (rocsolver_status_success != rocsolverState) {
    rocsolver_internal_error_throw(rocsolverState, name, file, line);
  }
}

// The macro below defines the interface for the safe rocsolver calls.
// The functions themselves are protected by impl namespace and this
// is not meant to be used by external application or libraries.
#define KOKKOS_ROCSOLVER_SAFE_CALL_IMPL(call)                             \
  KokkosLapack::Impl::rocsolver_internal_safe_call(call, #call, __FILE__, \
                                                   __LINE__)

/// \brief This function converts KK transpose mode to rocsolver transpose mode
inline rocsolver_operation trans_mode_kk_to_rocsolver(const char kkMode[]) {
  rocsolver_operation trans;
  if ((kkMode[0] == 'N') || (kkMode[0] == 'n'))
    trans = rocsolver_operation_none;
  else if ((kkMode[0] == 'T') || (kkMode[0] == 't'))
    trans = rocsolver_operation_transpose;
  else
    trans = rocsolver_operation_conjugate_transpose;
  return trans;
}

}  // namespace Impl
}  // namespace KokkosLapack

#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER

// If LAPACK TPL is enabled, it is preferred over magma's LAPACK
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include "magma_v2.h"

namespace KokkosLapack {
namespace Impl {

struct MagmaSingleton {
  MagmaSingleton();

  static MagmaSingleton& singleton();
};

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_MAGMA

#endif  // KOKKOSLAPACK_TPL_SPEC_HPP_
