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

#ifndef KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_HPP_
#define KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_HPP_

namespace KokkosLapack {
namespace Impl {
// Specialization struct which defines whether a specialization exists
template <class ExecutionSpace, class AMatrix, class BXMV, class IPIVV>
struct gesv_tpl_spec_avail {
  enum : bool { value = false };
};

// Generic Host side LAPACK (could be MKL or whatever)
#ifdef KOKKOSKERNELS_ENABLE_TPL_LAPACK

#define KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_LAPACK(SCALAR, LAYOUT, MEMSPACE) \
  template <class ExecSpace>                                              \
  struct gesv_tpl_spec_avail<                                             \
      ExecSpace,                                                          \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,             \
      Kokkos::View<int*, LAYOUT, Kokkos::Device<ExecSpace, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {          \
    enum : bool { value = true };                                         \
  };

KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_LAPACK(double, Kokkos::LayoutLeft,
                                        Kokkos::HostSpace)
KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_LAPACK(float, Kokkos::LayoutLeft,
                                        Kokkos::HostSpace)
KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_LAPACK(Kokkos::complex<double>,
                                        Kokkos::LayoutLeft, Kokkos::HostSpace)
KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_LAPACK(Kokkos::complex<float>,
                                        Kokkos::LayoutLeft, Kokkos::HostSpace)
#endif

// MAGMA
#ifdef KOKKOSKERNELS_ENABLE_TPL_MAGMA

#define KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_MAGMA(SCALAR, LAYOUT, MEMSPACE)     \
  template <>                                                                \
  struct gesv_tpl_spec_avail<                                                \
      Kokkos::Cuda,                                                          \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<int*, LAYOUT, Kokkos::HostSpace,                          \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {             \
    enum : bool { value = true };                                            \
  };

KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_MAGMA(double, Kokkos::LayoutLeft,
                                       Kokkos::CudaSpace)
KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_MAGMA(float, Kokkos::LayoutLeft,
                                       Kokkos::CudaSpace)
KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_MAGMA(Kokkos::complex<double>,
                                       Kokkos::LayoutLeft, Kokkos::CudaSpace)
KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_MAGMA(Kokkos::complex<float>,
                                       Kokkos::LayoutLeft, Kokkos::CudaSpace)
#endif
}  // namespace Impl
}  // namespace KokkosLapack

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER
#include <rocsolver/rocsolver.h>

namespace KokkosLapack {
namespace Impl {

#define KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_ROCSOLVER(SCALAR, LAYOUT, MEMSPACE) \
  template <>                                                                \
  struct gesv_tpl_spec_avail<                                                \
      Kokkos::HIP,                                                           \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEMSPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<SCALAR**, LAYOUT, Kokkos::Device<Kokkos::HIP, MEMSPACE>,  \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                \
      Kokkos::View<rocblas_int*, LAYOUT,                                     \
                   Kokkos::Device<Kokkos::HIP, MEMSPACE>,                    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> > > {             \
    enum : bool { value = true };                                            \
  };

KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_ROCSOLVER(double, Kokkos::LayoutLeft,
                                           Kokkos::HIPSpace)
KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_ROCSOLVER(float, Kokkos::LayoutLeft,
                                           Kokkos::HIPSpace)
KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_ROCSOLVER(Kokkos::complex<double>,
                                           Kokkos::LayoutLeft, Kokkos::HIPSpace)
KOKKOSLAPACK_GESV_TPL_SPEC_AVAIL_ROCSOLVER(Kokkos::complex<float>,
                                           Kokkos::LayoutLeft, Kokkos::HIPSpace)

}  // namespace Impl
}  // namespace KokkosLapack
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCSOLVER

#endif
