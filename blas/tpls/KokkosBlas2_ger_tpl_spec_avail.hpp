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

#ifndef KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class AT, class XT, class YT>
struct ger_tpl_spec_avail {
  enum : bool { value = false };
};

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS

#define KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_BLAS(SCALAR, LAYOUTX, LAYOUTY, LAYOUTA, MEMSPACE) \
  template <class ExecSpace>                                                             \
  struct ger_tpl_spec_avail< Kokkos::View< const SCALAR*                                 \
                                         , LAYOUTX                                       \
                                         , Kokkos::Device<ExecSpace, MEMSPACE>           \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged>       \
                                         >                                               \
                           , Kokkos::View< const SCALAR*                                 \
                                         , LAYOUTY                                       \
                                         , Kokkos::Device<ExecSpace, MEMSPACE>           \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged>       \
                                         >                                               \
                           , Kokkos::View< SCALAR**                                      \
                                         , LAYOUTA                                       \
                                         , Kokkos::Device<ExecSpace, MEMSPACE>,	         \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged>       \
                                         >                                               \
                           > {                                                           \
    enum : bool { value = true };                                                        \
  };

KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_BLAS(double,                  Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_BLAS(float,                   Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>,  Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::HostSpace)

KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_BLAS(double,                  Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_BLAS(float,                   Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>,  Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace)

#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS

#define KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUTX, LAYOUTY, LAYOUTA, MEMSPACE) \
  template <class ExecSpace>                                                               \
  struct ger_tpl_spec_avail< Kokkos::View< const SCALAR*                                   \
                                         , LAYOUTX                                         \
                                         , Kokkos::Device<ExecSpace, MEMSPACE>             \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged>         \
                                         >                                                 \
                           , Kokkos::View< const SCALAR*                                   \
                                         , LAYOUTY,                                        \
                                         , Kokkos::Device<ExecSpace, MEMSPACE>             \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged>         \
                                         >                                                 \
                           , Kokkos::View< SCALAR**                                        \
                                         , LAYOUTA                                         \
                                         , Kokkos::Device<ExecSpace, MEMSPACE>             \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged>         \
                                         >                                                 \
                           > {                                                             \
    enum : bool { value = true };                                                          \
  };

// We use the same layout for X, Y and Abecause the GER interface will
// switch the layouts of X and Y to that of A. So this TPL version will
// match any layout combination, as long as none are LayoutStride.

KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_CUBLAS(double,                  Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_CUBLAS(float,                   Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>,  Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::LayoutLeft, Kokkos::CudaSpace)

KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_CUBLAS(double,                  Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_CUBLAS(float,                   Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>,  Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace)

#endif

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

#define KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_ROCBLAS(SCALAR, LAYOUT)                            \
  template <>                                                                             \
  struct ger_tpl_spec_avail< Kokkos::View< const SCALAR*                                  \
                                         , LAYOUT                                         \
                                         , Kokkos::Device< Kokkos::Experimental::HIP,     \
                                                         , Kokkos::Experimental::HIPSpace \
                                                         >                                \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged>        \
                                         >                                                \
                           , Kokkos::View< const SCALAR*                                  \
                                         , LAYOUT                                         \
                                         , Kokkos::Device< Kokkos::Experimental::HIP      \
                                                         , Kokkos::Experimental::HIPSpace \
                                                         >                                \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged>        \
                                         >                                                \
                           , Kokkos::View< SCALAR**                                       \
                                         , LAYOUT                                         \
                                         , Kokkos::Device< Kokkos::Experimental::HIP      \
                                                         , Kokkos::Experimental::HIPSpace \
                                                         >                                \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged>        \
                                         >                                                \
                           > {                                                            \
    enum : bool { value = true };                                                         \
  };

KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_ROCBLAS(double,                  Kokkos::LayoutLeft)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_ROCBLAS(float,                   Kokkos::LayoutLeft)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutLeft)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>,  Kokkos::LayoutLeft)

KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_ROCBLAS(double,                  Kokkos::LayoutRight)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_ROCBLAS(float,                   Kokkos::LayoutRight)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>, Kokkos::LayoutRight)
KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float> , Kokkos::LayoutRight)

#endif
}  // namespace Impl
}  // namespace KokkosBlas

#endif // KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_HPP_
