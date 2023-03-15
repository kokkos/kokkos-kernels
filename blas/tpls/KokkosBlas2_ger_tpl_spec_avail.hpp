<<<<<<< HEAD
=======
/*
>>>>>>> d868047b (Renamed files)
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
<<<<<<< HEAD

#ifndef KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_HPP_
=======
*/

#ifndef KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_HPP_

// EEP
>>>>>>> d868047b (Renamed files)

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
<<<<<<< HEAD
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
                                         , Kokkos::Device<ExecSpace, MEMSPACE>           \
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

=======
template <class ExecutionSpace, class VectorView, class ScalarView>
struct ger_tpl_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

namespace KokkosBlas {
namespace Impl {

// Generic Host side BLAS (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#define KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_BLAS(SCALAR, LAYOUT, EXECSPACE) \
  template <>                                                           \
  struct ger_tpl_spec_avail<                                           \
      EXECSPACE,                                                        \
      Kokkos::View<SCALAR*, LAYOUT,                                     \
                   Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,            \
      Kokkos::View<SCALAR*, LAYOUT,                                     \
                   Kokkos::Device<EXECSPACE, Kokkos::HostSpace>,        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {          \
    enum : bool { value = true };                                       \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::Serial)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::Serial)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>,
                                     Kokkos::LayoutLeft, Kokkos::Serial)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft,
                                     Kokkos::Serial)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::OpenMP)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::OpenMP)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<double>,
                                     Kokkos::LayoutLeft, Kokkos::OpenMP)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_BLAS(Kokkos::complex<float>, Kokkos::LayoutLeft,
                                     Kokkos::OpenMP)
#endif
>>>>>>> d868047b (Renamed files)
#endif

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
<<<<<<< HEAD

#define KOKKOSBLAS2_GER_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUTX, LAYOUTY, LAYOUTA, MEMSPACE) \
  template <class ExecSpace>                                                               \
  struct ger_tpl_spec_avail< Kokkos::View< const SCALAR*                                   \
                                         , LAYOUTX                                         \
                                         , Kokkos::Device<ExecSpace, MEMSPACE>             \
                                         , Kokkos::MemoryTraits<Kokkos::Unmanaged>         \
                                         >                                                 \
                           , Kokkos::View< const SCALAR*                                   \
                                         , LAYOUTY                                         \
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

=======
#define KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUT, EXECSPACE, \
                                               MEMSPACE)                  \
  template <>                                                             \
  struct ger_tpl_spec_avail<                                             \
      EXECSPACE,                                                          \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,              \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {            \
    enum : bool { value = true };                                         \
  };

KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>,
                                       Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>,
                                       Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaSpace)

KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutRight,
                                       Kokkos::Cuda, Kokkos::CudaSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>,
                                       Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>,
                                       Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaSpace)

KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>,
                                       Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>,
                                       Kokkos::LayoutLeft, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)

KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutRight,
                                       Kokkos::Cuda, Kokkos::CudaUVMSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<double>,
                                       Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_CUBLAS(Kokkos::complex<float>,
                                       Kokkos::LayoutRight, Kokkos::Cuda,
                                       Kokkos::CudaUVMSpace)
>>>>>>> d868047b (Renamed files)
#endif

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
<<<<<<< HEAD

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
=======
#define KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_ROCBLAS(SCALAR, LAYOUT, EXECSPACE, \
                                                MEMSPACE)                  \
  template <>                                                              \
  struct ger_tpl_spec_avail<                                              \
      EXECSPACE,                                                           \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,               \
      Kokkos::View<SCALAR*, LAYOUT, Kokkos::Device<EXECSPACE, MEMSPACE>,   \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {             \
    enum : bool { value = true };                                          \
  };

KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutLeft, Kokkos::HIP,
                                        Kokkos::HIPSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutLeft, Kokkos::HIP,
                                        Kokkos::HIPSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>,
                                        Kokkos::LayoutLeft, Kokkos::HIP,
                                        Kokkos::HIPSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>,
                                        Kokkos::LayoutLeft, Kokkos::HIP,
                                        Kokkos::HIPSpace)

KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutRight,
                                        Kokkos::HIP, Kokkos::HIPSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutRight, Kokkos::HIP,
                                        Kokkos::HIPSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<double>,
                                        Kokkos::LayoutRight, Kokkos::HIP,
                                        Kokkos::HIPSpace)
KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_ROCBLAS(Kokkos::complex<float>,
                                        Kokkos::LayoutRight, Kokkos::HIP,
                                        Kokkos::HIPSpace)
#endif

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSBLAS1_GER_TPL_SPEC_AVAIL_HPP_
>>>>>>> d868047b (Renamed files)
