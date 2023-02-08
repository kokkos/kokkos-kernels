/*
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
*/

#ifndef KOKKOSPARSE_SPGEMM_NOREUSE_TPL_SPEC_AVAIL_HPP_
#define KOKKOSPARSE_SPGEMM_NOREUSE_TPL_SPEC_AVAIL_HPP_

namespace KokkosSparse {
namespace Impl {

// Specialization struct which defines whether a specialization exists
template <class CMatrix, class AMatrix, class BMatrix>
struct spgemm_noreuse_tpl_spec_avail {
  enum : bool { value = false };
};

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSPARSE
// NOTE: all versions of cuSPARSE 10.x and 11.x support exactly the same matrix
// types, so there is no ifdef'ing on versions needed in avail. Offset and
// Ordinal must both be 32-bit. Even though the "generic" API lets you specify
// offsets and ordinals independently as either 16, 32 or 64-bit, SpGEMM will
// just fail at runtime if you don't use 32 for both.

#define SPGEMM_NOREUSE_AVAIL_CUSPARSE(SCALAR, MEMSPACE)                    \
  template <>                                                              \
  struct spgemm_noreuse_tpl_spec_avail<                                    \
      KokkosSparse::CrsMatrix<                                             \
          SCALAR, int, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, void, int>, \
      KokkosSparse::CrsMatrix<                                             \
          const SCALAR, const int, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, \
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, const int>,             \
      KokkosSparse::CrsMatrix<                                             \
          const SCALAR, const int, Kokkos::Device<Kokkos::Cuda, MEMSPACE>, \
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, const int>> {           \
    enum : bool { value = true };                                          \
  };

#define SPGEMM_NOREUSE_AVAIL_CUSPARSE_S(SCALAR)            \
  SPGEMM_NOREUSE_AVAIL_CUSPARSE(SCALAR, Kokkos::CudaSpace) \
  SPGEMM_NOREUSE_AVAIL_CUSPARSE(SCALAR, Kokkos::CudaUVMSpace)

SPGEMM_NOREUSE_AVAIL_CUSPARSE_S(float)
SPGEMM_NOREUSE_AVAIL_CUSPARSE_S(double)
SPGEMM_NOREUSE_AVAIL_CUSPARSE_S(Kokkos::complex<float>)
SPGEMM_NOREUSE_AVAIL_CUSPARSE_S(Kokkos::complex<double>)

#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
#define SPGEMM_NOREUSE_AVAIL_ROCSPARSE(SCALAR)                               \
  template <>                                                                \
  struct spgemm_noreuse_tpl_spec_avail<                                      \
      KokkosSparse::CrsMatrix<SCALAR, int,                                   \
                              Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>, \
                              void, int>,                                    \
      KokkosSparse::CrsMatrix<const SCALAR, const int,                       \
                              Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>, \
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>,       \
                              const int>,                                    \
      KokkosSparse::CrsMatrix<const SCALAR, const int,                       \
                              Kokkos::Device<Kokkos::HIP, Kokkos::HIPSpace>, \
                              Kokkos::MemoryTraits<Kokkos::Unmanaged>,       \
                              const int>> {                                  \
    enum : bool { value = true };                                            \
  };

SPGEMM_NOREUSE_AVAIL_ROCSPARSE(float)
SPGEMM_NOREUSE_AVAIL_ROCSPARSE(double)
SPGEMM_NOREUSE_AVAIL_ROCSPARSE(Kokkos::complex<float>)
SPGEMM_NOREUSE_AVAIL_ROCSPARSE(Kokkos::complex<double>)
#endif

#ifdef KOKKOSKERNELS_ENABLE_TPL_MKL
#define SPGEMM_NOREUSE_AVAIL_MKL(SCALAR, EXEC)                              \
  template <>                                                               \
  struct spgemm_noreuse_tpl_spec_avail<                                     \
      KokkosSparse::CrsMatrix<                                              \
          SCALAR, int, Kokkos::Device<EXEC, Kokkos::HostSpace>, void, int>, \
      KokkosSparse::CrsMatrix<                                              \
          const SCALAR, const int, Kokkos::Device<EXEC, Kokkos::HostSpace>, \
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, const int>,              \
      KokkosSparse::CrsMatrix<                                              \
          const SCALAR, const int, Kokkos::Device<EXEC, Kokkos::HostSpace>, \
          Kokkos::MemoryTraits<Kokkos::Unmanaged>, const int>> {            \
    enum : bool { value = true };                                           \
  };

#define SPGEMM_NOREUSE_AVAIL_MKL_E(EXEC)                 \
  SPGEMM_NOREUSE_AVAIL_MKL(float, EXEC)                  \
  SPGEMM_NOREUSE_AVAIL_MKL(double, EXEC)                 \
  SPGEMM_NOREUSE_AVAIL_MKL(Kokkos::complex<float>, EXEC) \
  SPGEMM_NOREUSE_AVAIL_MKL(Kokkos::complex<double>, EXEC)

#ifdef KOKKOS_ENABLE_SERIAL
SPGEMM_NOREUSE_AVAIL_MKL_E(Kokkos::Serial)
#endif
#ifdef KOKKOS_ENABLE_OPENMP
SPGEMM_NOREUSE_AVAIL_MKL_E(Kokkos::OpenMP)
#endif
#endif

}  // namespace Impl
}  // namespace KokkosSparse

#endif
