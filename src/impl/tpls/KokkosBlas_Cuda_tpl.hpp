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
#ifndef KOKKOSBLAS_CUDA_TPL_HPP_
#define KOKKOSBLAS_CUDA_TPL_HPP_

#if defined(KOKKOSKERNELS_ENABLE_TPL_CUBLAS)
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

CudaBlasSingleton::CudaBlasSingleton() {
  cublasStatus_t stat = cublasCreate(&handle);
  if (stat != CUBLAS_STATUS_SUCCESS)
    Kokkos::abort("CUBLAS initialization failed\n");

  Kokkos::push_finalize_hook([&]() { cublasDestroy(handle); });
}

CudaBlasSingleton& CudaBlasSingleton::singleton() {
  static CudaBlasSingleton s;
  return s;
}

}  // namespace Impl
}  // namespace KokkosBlas
#endif

#if defined(KOKKOSKERNELS_ENABLE_TPL_CUSOLVER)
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {
CudaSolverSingleton::CudaSolverSingleton() {
  auto stat = cusolverDnCreate(&handle);
  if (stat != CUSOLVER_STATUS_SUCCESS)
    Kokkos::abort("CUSOLVER initialization failed\n");

  Kokkos::push_finalize_hook([&]() { cusolverDnDestroy(handle); });
}

CudaSolverSingleton& CudaSolverSingleton::singleton() {
  static CudaSolverSingleton s;
  return s;
}

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOS_KERNELS_ENABLE_TPL_CUSOLVER

#if defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA)
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

MagmaSingleton::MagmaSingleton() {
  magma_int_t stat = magma_init();
  if (stat != MAGMA_SUCCESS) Kokkos::abort("MAGMA initialization failed\n");

  Kokkos::push_finalize_hook([&]() { magma_finalize(); });
}

MagmaSingleton& MagmaSingleton::singleton() {
  static MagmaSingleton s;
  return s;
}

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // defined(KOKKOSKERNELS_ENABLE_TPL_MAGMA)

#endif  // KOKKOSBLAS_CUDA_TPL_HPP_
