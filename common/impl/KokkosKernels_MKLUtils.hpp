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

#ifndef _KOKKOSKERNELS_UTILS_MKL_HPP
#define _KOKKOSKERNELS_UTILS_MKL_HPP

#include "KokkosKernels_config.h"

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL

#include <mkl.h>

#if defined(KOKKOSKERNELS_ENABLE_TPL_MKL)
#include "mkl_version.h"
#if __INTEL_MKL__ >= 2018
#define __KOKKOSBLAS_ENABLE_INTEL_MKL_COMPACT__ 1
#include "mkl_compact.h"
#endif
#endif

namespace KokkosKernels {
namespace Impl {

inline void mkl_internal_safe_call(sparse_status_t mkl_status, const char *name,
                                   const char *file = nullptr,
                                   const int line   = 0) {
  if (SPARSE_STATUS_SUCCESS != mkl_status) {
    std::ostringstream oss;
    oss << "MKL call \"" << name << "\" at " << file << ":" << line
        << " encountered error: ";
    switch (mkl_status) {
      case SPARSE_STATUS_NOT_INITIALIZED:
        oss << "SPARSE_STATUS_NOT_INITIALIZED (empty handle or matrix arrays)";
        break;
      case SPARSE_STATUS_ALLOC_FAILED:
        oss << "SPARSE_STATUS_ALLOC_FAILED (internal error: memory allocation "
               "failed)";
        break;
      case SPARSE_STATUS_INVALID_VALUE:
        oss << "SPARSE_STATUS_INVALID_VALUE (invalid input value)";
        break;
      case SPARSE_STATUS_EXECUTION_FAILED:
        oss << "SPARSE_STATUS_EXECUTION_FAILED (e.g. 0-diagonal element for "
               "triangular solver)";
        break;
      case SPARSE_STATUS_INTERNAL_ERROR:
        oss << "SPARSE_STATUS_INTERNAL_ERROR";
        break;
      case SPARSE_STATUS_NOT_SUPPORTED:
        oss << "SPARSE_STATUS_NOT_SUPPORTED (e.g. operation for double "
               "precision doesn't support other types)";
        break;
      default: oss << "unknown (code " << (int)mkl_status << ")"; break;
    }
    oss << '\n';
    Kokkos::abort(oss.str().c_str());
  }
}

#define KOKKOSKERNELS_MKL_SAFE_CALL(call) \
  KokkosKernels::Impl::mkl_internal_safe_call(call, #call, __FILE__, __LINE__)

}  // namespace Impl
}  // namespace KokkosKernels

#endif  // KOKKOSKERNELS_ENABLE_TPL_MKL

#endif  // _KOKKOSKERNELS_UTILS_MKL_HPP