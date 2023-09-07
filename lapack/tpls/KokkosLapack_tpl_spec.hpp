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

#ifdef KOKKOSKERNELS_ENABLE_TPL_CULAPACK
#include "cuda_runtime.h"
#include "culapack_v2.h"

namespace KokkosLapack {
namespace Impl {

struct CudaLapackSingleton {
  culapackHandle_t handle;

  CudaLapackSingleton();

  static CudaLapackSingleton& singleton();
};

inline void culapack_internal_error_throw(culapackStatus_t culapackState,
                                        const char* name, const char* file,
                                        const int line) {
  std::ostringstream out;
  // out << name << " error( " << culapackGetStatusName(culapackState)
  //     << "): " << culapackGetStatusString(culapackState);
  out << name << " error( ";
  switch (culapackState) {
    case CULAPACK_STATUS_NOT_INITIALIZED:
      out << "CULAPACK_STATUS_NOT_INITIALIZED): the library was not initialized.";
      break;
    case CULAPACK_STATUS_ALLOC_FAILED:
      out << "CULAPACK_STATUS_ALLOC_FAILED): the resource allocation failed.";
      break;
    case CULAPACK_STATUS_INVALID_VALUE:
      out << "CULAPACK_STATUS_INVALID_VALUE): an invalid numerical value was "
             "used as an argument.";
      break;
    case CULAPACK_STATUS_ARCH_MISMATCH:
      out << "CULAPACK_STATUS_ARCH_MISMATCH): an absent device architectural "
             "feature is required.";
      break;
    case CULAPACK_STATUS_MAPPING_ERROR:
      out << "CULAPACK_STATUS_MAPPING_ERROR): an access to GPU memory space "
             "failed.";
      break;
    case CULAPACK_STATUS_EXECUTION_FAILED:
      out << "CULAPACK_STATUS_EXECUTION_FAILED): the GPU program failed to "
             "execute.";
      break;
    case CULAPACK_STATUS_INTERNAL_ERROR:
      out << "CULAPACK_STATUS_INTERNAL_ERROR): an internal operation failed.";
      break;
    case CULAPACK_STATUS_NOT_SUPPORTED:
      out << "CULAPACK_STATUS_NOT_SUPPORTED): the feature required is not "
             "supported.";
      break;
    default: out << "unrecognized error code): this is bad!"; break;
  }
  if (file) {
    out << " " << file << ":" << line;
  }
  throw std::runtime_error(out.str());
}

inline void culapack_internal_safe_call(culapackStatus_t culapackState,
                                      const char* name,
                                      const char* file = nullptr,
                                      const int line   = 0) {
  if (CULAPACK_STATUS_SUCCESS != culapackState) {
    culapack_internal_error_throw(culapackState, name, file, line);
  }
}

// The macro below defines the interface for the safe culapack calls.
// The functions themselves are protected by impl namespace and this
// is not meant to be used by external application or libraries.
#define KOKKOS_CULAPACK_SAFE_CALL_IMPL(call) \
  KokkosLapack::Impl::culapack_internal_safe_call(call, #call, __FILE__, __LINE__)

/// \brief This function converts KK transpose mode to cuLAPACK transpose mode
inline culapackOperation_t trans_mode_kk_to_culapack(const char kkMode[]) {
  culapackOperation_t trans;
  if ((kkMode[0] == 'N') || (kkMode[0] == 'n'))
    trans = CULAPACK_OP_N;
  else if ((kkMode[0] == 'T') || (kkMode[0] == 't'))
    trans = CULAPACK_OP_T;
  else
    trans = CULAPACK_OP_C;
  return trans;
}

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_CULAPACK

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCLAPACK
#include <roclapack/roclapack.h>

namespace KokkosLapack {
namespace Impl {

struct RocLapackSingleton {
  roclapack_handle handle;

  RocLapackSingleton();

  static RocLapackSingleton& singleton();
};

inline void roclapack_internal_error_throw(roclapack_status roclapackState,
                                         const char* name, const char* file,
                                         const int line) {
  std::ostringstream out;
  out << name << " error( ";
  switch (roclapackState) {
    case roclapack_status_invalid_handle:
      out << "roclapack_status_invalid_handle): handle not initialized, invalid "
             "or null.";
      break;
    case roclapack_status_not_implemented:
      out << "roclapack_status_not_implemented): function is not implemented.";
      break;
    case roclapack_status_invalid_pointer:
      out << "roclapack_status_invalid_pointer): invalid pointer argument.";
      break;
    case roclapack_status_invalid_size:
      out << "roclapack_status_invalid_size): invalid size argument.";
      break;
    case roclapack_status_memory_error:
      out << "roclapack_status_memory_error): failed internal memory allocation, "
             "copy or dealloc.";
      break;
    case roclapack_status_internal_error:
      out << "roclapack_status_internal_error): other internal library failure.";
      break;
    case roclapack_status_perf_degraded:
      out << "roclapack_status_perf_degraded): performance degraded due to low "
             "device memory.";
      break;
    case roclapack_status_size_query_mismatch:
      out << "unmatched start/stop size query): .";
      break;
    case roclapack_status_size_increased:
      out << "roclapack_status_size_increased): queried device memory size "
             "increased.";
      break;
    case roclapack_status_size_unchanged:
      out << "roclapack_status_size_unchanged): queried device memory size "
             "unchanged.";
      break;
    case roclapack_status_invalid_value:
      out << "roclapack_status_invalid_value): passed argument not valid.";
      break;
    case roclapack_status_continue:
      out << "roclapack_status_continue): nothing preventing function to "
             "proceed.";
      break;
    case roclapack_status_check_numerics_fail:
      out << "roclapack_status_check_numerics_fail): will be set if the "
             "vector/matrix has a NaN or an Infinity.";
      break;
    default: out << "unrecognized error code): this is bad!"; break;
  }
  if (file) {
    out << " " << file << ":" << line;
  }
  throw std::runtime_error(out.str());
}

inline void roclapack_internal_safe_call(roclapack_status roclapackState,
                                       const char* name,
                                       const char* file = nullptr,
                                       const int line   = 0) {
  if (roclapack_status_success != roclapackState) {
    roclapack_internal_error_throw(roclapackState, name, file, line);
  }
}

// The macro below defines the interface for the safe roclapack calls.
// The functions themselves are protected by impl namespace and this
// is not meant to be used by external application or libraries.
#define KOKKOS_ROCLAPACK_SAFE_CALL_IMPL(call) \
  KokkosLapack::Impl::roclapack_internal_safe_call(call, #call, __FILE__, __LINE__)

/// \brief This function converts KK transpose mode to rocLAPACK transpose mode
inline roclapack_operation trans_mode_kk_to_roclapack(const char kkMode[]) {
  roclapack_operation trans;
  if ((kkMode[0] == 'N') || (kkMode[0] == 'n'))
    trans = roclapack_operation_none;
  else if ((kkMode[0] == 'T') || (kkMode[0] == 't'))
    trans = roclapack_operation_transpose;
  else
    trans = roclapack_operation_conjugate_transpose;
  return trans;
}

}  // namespace Impl
}  // namespace KokkosLapack

#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCLAPACK

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
