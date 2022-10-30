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

#ifndef KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class ExecutionSpace, class SViewType, class MViewType>
struct rotg_tpl_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

namespace KokkosBlas {
namespace Impl {

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#define KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(SCALAR, LAYOUT, EXECSPACE,    \
                                             MEMSPACE)                     \
  template <>                                                              \
  struct rotg_tpl_spec_avail<                                              \
      EXECSPACE,                                                           \
      Kokkos::View<SCALAR, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,               \
      Kokkos::View<typename Kokkos::ArithTraits<SCALAR>::mag_type, LAYOUT, \
                   Kokkos::Device<EXECSPACE, MEMSPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {             \
    enum : bool { value = true };                                          \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::Serial,
                                     Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutRight,
                                     Kokkos::Serial, Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::Serial,
                                     Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutRight, Kokkos::Serial,
                                     Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>,
                                     Kokkos::LayoutLeft, Kokkos::Serial,
                                     Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>,
                                     Kokkos::LayoutRight, Kokkos::Serial,
                                     Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft,
                                     Kokkos::Serial, Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>,
                                     Kokkos::LayoutRight, Kokkos::Serial,
                                     Kokkos::HostSpace)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::OpenMP,
                                     Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutRight,
                                     Kokkos::OpenMP, Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::OpenMP,
                                     Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutRight, Kokkos::OpenMP,
                                     Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>,
                                     Kokkos::LayoutLeft, Kokkos::OpenMP,
                                     Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>,
                                     Kokkos::LayoutRight, Kokkos::OpenMP,
                                     Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft,
                                     Kokkos::OpenMP, Kokkos::HostSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>,
                                     Kokkos::LayoutRight, Kokkos::OpenMP,
                                     Kokkos::HostSpace)
#endif
#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#define KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUT, EXECSPACE, \
                                               MEMSPACE)                  \
  template <>                                                             \
  struct rotg_tpl_spec_avail<                                             \
      EXECSPACE,                                                          \
      Kokkos::View<SCALAR, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,              \
      Kokkos::View<Kokkos::ArithTraits<SCALAR>::mag_type, LAYOUT,         \
                   Kokkos::Device<EXECSPACE, MEMSPACE>,                   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {            \
    enum : bool { value = true };                                         \
  };

KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutRight,
                                       Kokkos::Cuda, Kokkos::CudaSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>,
                                       Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>,
                                       Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>,
                                       Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>,
                                       Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaSpace)

KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutRight,
                                       Kokkos::Cuda, Kokkos::CudaUVMSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>,
                                       Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>,
                                       Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>,
                                       Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>,
                                       Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
#endif

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#define KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_ROCBLAS(SCALAR, LAYOUT, EXECSPACE, \
                                                MEMSPACE)                  \
  template <>                                                              \
  struct rotg_tpl_spec_avail<                                              \
      EXECSPACE,                                                           \
      Kokkos::View<SCALAR, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,               \
      Kokkos::View<Kokkos::ArithTraits<SCALAR>::mag_type, LAYOUT,          \
                   Kokkos::Device<EXECSPACE, MEMSPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {             \
    enum : bool { value = true };                                          \
  };

KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutLeft, Kokkos::HIP,
                                        Kokkos::HIPSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutRight,
                                        Kokkos::HIP, Kokkos::HIPSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutLeft, Kokkos::HIP,
                                        Kokkos::HIPSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutRight, Kokkos::HIP,
                                        Kokkos::HIPSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>,
                                        Kokkos::LayoutLeft, Kokkos::HIP,
                                        Kokkos::HIPSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>,
                                        Kokkos::LayoutRight, Kokkos::HIP,
                                        Kokkos::HIPSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>,
                                        Kokkos::LayoutLeft, Kokkos::HIP,
                                        Kokkos::HIPSpace)
KOKKOSBLAS1_ROTG_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>,
                                        Kokkos::LayoutRight, Kokkos::HIP,
                                        Kokkos::HIPSpace)
#endif

}  // namespace Impl
}  // namespace KokkosBlas
#endif
