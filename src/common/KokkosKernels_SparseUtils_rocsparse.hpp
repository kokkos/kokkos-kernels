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

#ifndef _KOKKOSKERNELS_SPARSEUTILS_ROCSPARSE_HPP
#define _KOKKOSKERNELS_SPARSEUTILS_ROCSPARSE_HPP

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#include "rocsparse.h"

namespace KokkosSparse {
namespace Impl {

inline void rocsparse_internal_error_throw(rocsparse_status rocsparseStatus,
                                          const char* name, const char* file,
                                          const int line) {
  std::ostringstream out;
  out << name << " error( ";
  switch (rocsparseStatus) {
    case rocsparse_status_invalid_handle:
      out << "rocsparse_status_invalid_handle): handle not initialized, invalid or null.";
      break;
    case rocsparse_status_not_implemented:
      out << "rocsparse_status_not_implemented): function is not implemented.";
      break;
    case rocsparse_status_invalid_pointer:
      out << "rocsparse_status_invalid_pointer): invalid pointer parameter.";
      break;
    case rocsparse_status_invalid_size:
      out << "rocsparse_status_invalid_size): invalid size parameter.";
      break;
    case rocsparse_status_memory_error:
      out << "rocsparse_status_memory_error): failed memory allocation, copy, dealloc.";
      break;
    case rocsparse_status_internal_error:
      out << "rocsparse_status_internal_error): other internal library failure.";
      break;
    case rocsparse_status_invalid_value:
      out << "rocsparse_status_invalid_value): invalid value parameter.";
      break;
    case rocsparse_status_arch_mismatch:
      out << "rocsparse_status_arch_mismatch): device arch is not supported.";
      break;
    case rocsparse_status_zero_pivot:
      out << "rocsparse_status_zero_pivot): encountered zero pivot.";
      break;
    case rocsparse_status_not_initialized:
      out << "rocsparse_status_not_initialized): descriptor has not been initialized.";
      break;
    case rocsparse_status_type_mismatch:
      out << "rocsparse_status_type_mismatch): index types do not match.";
      break;
    default: out << "unrecognized error code): this is bad!"; break;
  }
  if (file) {
    out << " " << file << ":" << line;
  }
  throw std::runtime_error(out.str());
}

inline void rocsparse_internal_safe_call(rocsparse_status rocsparseStatus,
                                        const char* name,
                                        const char* file = nullptr,
                                        const int line   = 0) {
  if (rocsparse_status_success != rocsparseStatus) {
    rocsparse_internal_error_throw(rocsparseStatus, name, file, line);
  }
}

// The macro below defines is the public interface for the safe cusparse calls.
// The functions themselves are protected by impl namespace.
#define KOKKOS_ROCSPARSE_SAFE_CALL_IMPL(call)				\
  KokkosSparse::Impl::rocsparse_internal_safe_call(call, #call, __FILE__, \
						   __LINE__)

}  // namespace Impl

}  // namespace KokkosSparse

#endif  // KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
#endif  // _KOKKOSKERNELS_SPARSEUTILS_CUSPARSE_HPP
