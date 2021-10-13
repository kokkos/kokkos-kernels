/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
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
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOSBLAS_TPL_SPEC_HPP_
#define KOKKOSBLAS_TPL_SPEC_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include "cuda_runtime.h"
#include "cublas_v2.h"

namespace KokkosBlas {
namespace Impl {

struct CudaBlasSingleton {
  cublasHandle_t handle;

  CudaBlasSingleton();

  static CudaBlasSingleton& singleton();
};

inline void cublas_internal_error_throw(cublasStatus_t cublasState,
                                        const char* name, const char* file,
                                        const int line) {
  std::ostringstream out;
  // out << name << " error( " << cublasGetStatusName(cublasState)
  //     << "): " << cublasGetStatusString(cublasState);
  out << name << " error( ";
  switch (cublasState) {
    case CUBLAS_STATUS_NOT_INITIALIZED:
      out << "CUBLAS_STATUS_NOT_INITIALIZED): the library was not initialized.";
      break;
    case CUBLAS_STATUS_ALLOC_FAILED:
      out << "CUBLAS_STATUS_ALLOC_FAILED): the resource allocation failed.";
      break;
    case CUBLAS_STATUS_INVALID_VALUE:
      out << "CUBLAS_STATUS_INVALID_VALUE): an invalid numerical value was "
             "used as an argument.";
      break;
    case CUBLAS_STATUS_ARCH_MISMATCH:
      out << "CUBLAS_STATUS_ARCH_MISMATCH): an absent device architectural "
             "feature is required.";
      break;
    case CUBLAS_STATUS_MAPPING_ERROR:
      out << "CUBLAS_STATUS_MAPPING_ERROR): an access to GPU memory space "
             "failed.";
      break;
    case CUBLAS_STATUS_EXECUTION_FAILED:
      out << "CUBLAS_STATUS_EXECUTION_FAILED): the GPU program failed to "
             "execute.";
      break;
    case CUBLAS_STATUS_INTERNAL_ERROR:
      out << "CUBLAS_STATUS_INTERNAL_ERROR): an internal operation failed.";
      break;
    case CUBLAS_STATUS_NOT_SUPPORTED:
      out << "CUBLAS_STATUS_NOT_SUPPORTED): the feature required is not "
             "supported.";
      break;
    default: out << "unrecognized error code): this is bad!"; break;
  }
  if (file) {
    out << " " << file << ":" << line;
  }
  throw std::runtime_error(out.str());
}

inline void cublas_internal_safe_call(cublasStatus_t cublasState,
                                      const char* name,
                                      const char* file = nullptr,
                                      const int line   = 0) {
  if (CUBLAS_STATUS_SUCCESS != cublasState) {
    cublas_internal_error_throw(cublasState, name, file, line);
  }
}

// The macro below defines the interface for the safe cublas calls.
// The functions themselves are protected by impl namespace and this
// is not meant to be used by external application or libraries.
#define KOKKOS_CUBLAS_SAFE_CALL_IMPL(call) \
  KokkosBlas::Impl::cublas_internal_safe_call(call, #call, __FILE__, __LINE__)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

// If LAPACK TPL is enabled, it is preferred over magma's LAPACK
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA
#include "magma_v2.h"

namespace KokkosBlas {
namespace Impl {

struct MagmaSingleton {
  MagmaSingleton();

  static MagmaSingleton& singleton();
};

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_MAGMA

#endif  // KOKKOSBLAS_TPL_SPEC_HPP_
