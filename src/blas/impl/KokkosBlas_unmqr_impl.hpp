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
#ifndef KOKKOSBLAS_IMPL_UNMQR_HPP_
#define KOKKOSBLAS_IMPL_UNMQR_HPP_

#include <Kokkos_Core.hpp>
#include <Kokkos_ArithTraits.hpp>
#include <KokkosKernels_config.h>
#include <type_traits>
#include <sstream>

namespace KokkosBlas {
namespace Impl {
// Put non TPL implementation here

template <class AVT, class TVT, class CVT, class WVT>
void execute_unmqr(char /*side*/, char /*trans*/, int /*k*/, AVT& /*A*/, TVT& /*tau*/, CVT& /*C*/,
                   WVT& /*workspace*/) {
  std::ostringstream os;
  os << "There is no ETI implementation of UNMQR. Compile with TPL (LAPACKE or "
        "CUSOLVER).\n";
  Kokkos::Impl::throw_runtime_exception(os.str());
}

template <class AVT, class TVT, class CVT>
int64_t execute_unmqr_workspace(char /*side*/, char /*trans*/, int /*k*/, AVT& /*A*/, TVT& /*tau*/, CVT& /*C*/) {
  std::ostringstream os;
  os << "There is no ETI implementation of UNMQR Workspace. Compile with TPL "
        "(LAPACKE or CUSOLVER).\n";
  Kokkos::Impl::throw_runtime_exception(os.str());
  return 0;
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOSBLAS_IMPL_UNMQR_HPP_
