// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_HPP_
#define KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_HPP_

namespace KokkosBlas {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class execution_space, class VectorView, class ParamView>
struct rotm_tpl_spec_avail {
  enum : bool { value = false };
};
}  // namespace Impl
}  // namespace KokkosBlas

namespace KokkosBlas {
namespace Impl {

// Generic Host side BLAS (could be MKL or whatever)
// ARMPL is disabled as it does not detect some corner
// cases correctly which leads to failing unit-tests
#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS)
#define KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_BLAS(SCALAR, LAYOUT, EXEC_SPACE)                              \
  template <>                                                                                         \
  struct rotm_tpl_spec_avail<                                                                         \
      EXEC_SPACE, Kokkos::View<SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
      Kokkos::View<const SCALAR[5], LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {   \
    enum : bool { value = true };                                                                     \
  };

#ifdef KOKKOS_ENABLE_SERIAL
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::Serial)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::Serial)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutRight, Kokkos::Serial)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutRight, Kokkos::Serial)
#endif

#ifdef KOKKOS_ENABLE_OPENMP
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutLeft, Kokkos::OpenMP)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutLeft, Kokkos::OpenMP)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_BLAS(double, Kokkos::LayoutRight, Kokkos::OpenMP)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_BLAS(float, Kokkos::LayoutRight, Kokkos::OpenMP)
#endif
#endif  // KOKKOSKERNELS_ENABLE_TPL_BLAS

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#define KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_CUBLAS(SCALAR, LAYOUT, EXEC_SPACE)                            \
  template <>                                                                                         \
  struct rotm_tpl_spec_avail<                                                                         \
      EXEC_SPACE, Kokkos::View<SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
      Kokkos::View<const SCALAR[5], LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {   \
    enum : bool { value = true };                                                                     \
  };

KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutLeft, Kokkos::Cuda)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutLeft, Kokkos::Cuda)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_CUBLAS(double, Kokkos::LayoutRight, Kokkos::Cuda)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_CUBLAS(float, Kokkos::LayoutRight, Kokkos::Cuda)
#endif

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#define KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_ROCBLAS(SCALAR, LAYOUT, EXEC_SPACE)                           \
  template <>                                                                                         \
  struct rotm_tpl_spec_avail<                                                                         \
      EXEC_SPACE, Kokkos::View<SCALAR*, LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>, \
      Kokkos::View<const SCALAR[5], LAYOUT, EXEC_SPACE, Kokkos::MemoryTraits<Kokkos::Unmanaged>>> {   \
    enum : bool { value = true };                                                                     \
  };

KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutLeft, Kokkos::HIP)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutLeft, Kokkos::HIP)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_ROCBLAS(double, Kokkos::LayoutRight, Kokkos::HIP)
KOKKOSBLAS1_ROTM_TPL_SPEC_AVAIL_ROCBLAS(float, Kokkos::LayoutRight, Kokkos::HIP)
#endif

}  // namespace Impl
}  // namespace KokkosBlas
#endif
