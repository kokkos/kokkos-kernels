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

#ifndef KOKKOSBLAS2_GER_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS2_GER_TPL_SPEC_DECL_HPP_

#ifdef KOKKOSKERNELS_ENABLE_TPL_BLAS
#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_GER_DETERMINE_ARGS(LAYOUTA)                              \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;     \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);

#define KOKKOSBLAS2_DGER_BLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE,         \
                              ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                \
  struct GER<Kokkos::View<const double*, LAYOUTX,                           \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             Kokkos::View<const double*, LAYOUTY,                           \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             Kokkos::View<double**, LAYOUTA,                                \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             true, ETI_SPEC_AVAIL> {                                        \
    typedef double SCALAR;                                                  \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                            \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        XViewType;                                                          \
    typedef Kokkos::View<const SCALAR*, LAYOUTY,                            \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        YViewType;                                                          \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                 \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        AViewType;                                                          \
                                                                            \
    static void ger(const typename AViewType::execution_space& /* space */, \
                    typename AViewType::const_value_type& alpha,            \
                    const XViewType& X, const YViewType& Y,                 \
                    const AViewType& A) {                                   \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-dger-blas\n" );   \
      Kokkos::Profiling::pushRegion("KokkosBlas::ger[TPL_BLAS,double]");    \
      KOKKOSBLAS2_GER_DETERMINE_ARGS(LAYOUTA);                              \
      HostBlas<double>::ger(M, N, alpha, X.data(), one, Y.data(), one,      \
                            A.data(), LDA);                                 \
      Kokkos::Profiling::popRegion();                                       \
    }                                                                       \
  };

#define KOKKOSBLAS2_SGER_BLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE,         \
                              ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                \
  struct GER<Kokkos::View<const float*, LAYOUTX,                            \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             Kokkos::View<const float*, LAYOUTY,                            \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,		    \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             Kokkos::View<float**, LAYOUTA,                                 \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             true, ETI_SPEC_AVAIL> {                                        \
    typedef float SCALAR;                                                   \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                            \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        XViewType;                                                          \
    typedef Kokkos::View<const SCALAR*, LAYOUTY,                            \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        YViewType;                                                          \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                 \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        AViewType;                                                          \
                                                                            \
    static void ger(const typename AViewType::execution_space& /* space */, \
                    typename AViewType::const_value_type& alpha,            \
                    const XViewType& X, const YViewType& Y,                 \
                    const AViewType& A) {                                   \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-sger-blas\n" );   \
      Kokkos::Profiling::pushRegion("KokkosBlas::ger[TPL_BLAS,float]");     \
      KOKKOSBLAS2_GER_DETERMINE_ARGS(LAYOUTA);                              \
      HostBlas<float>::ger(M, N, alpha, X.data(), one, Y.data(), one,       \
                           A.data(), LDA);                                  \
      Kokkos::Profiling::popRegion();                                       \
    }                                                                       \
  };

#define KOKKOSBLAS2_ZGER_BLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE,         \
                              ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                \
  struct GER<Kokkos::View<const Kokkos::complex<double>*, LAYOUTX,          \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             Kokkos::View<const Kokkos::complex<double>*, LAYOUTY,          \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             Kokkos::View<Kokkos::complex<double>**, LAYOUTA,               \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<double> SCALAR;                                 \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                            \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        XViewType;                                                          \
    typedef Kokkos::View<const SCALAR*, LAYOUTY,                            \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        YViewType;                                                          \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                 \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        AViewType;                                                          \
                                                                            \
    static void ger(const typename AViewType::execution_space& /* space */, \
                    typename AViewType::const_value_type& alpha,            \
                    const XViewType& X, const YViewType& Y,                 \
                    const AViewType& A) {                                   \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-zger-blas\n" );   \
      Kokkos::Profiling::pushRegion(                                        \
          "KokkosBlas::ger[TPL_BLAS,complex<double>]");                     \
      KOKKOSBLAS2_GER_DETERMINE_ARGS(LAYOUTA);                              \
      const std::complex<double> alpha_val = alpha;                         \
      HostBlas<std::complex<double> >::ger(                                 \
          M, N, alpha_val,                                                  \
          reinterpret_cast<const std::complex<double>*>(X.data()), one,     \
          reinterpret_cast<const std::complex<double>*>(Y.data()), one,     \
          reinterpret_cast<std::complex<double>*>(A.data()), LDA);          \
      Kokkos::Profiling::popRegion();                                       \
    }                                                                       \
  };

#define KOKKOSBLAS2_CGER_BLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE,         \
                              ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                \
  struct GER<Kokkos::View<const Kokkos::complex<float>*, LAYOUTX,           \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             Kokkos::View<const Kokkos::complex<float>*, LAYOUTY,           \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             Kokkos::View<Kokkos::complex<float>**, LAYOUTA,                \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,             \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
             true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<float> SCALAR;                                  \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                            \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        XViewType;                                                          \
    typedef Kokkos::View<const SCALAR*, LAYOUTY,                            \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        YViewType;                                                          \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                 \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,              \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >          \
        AViewType;                                                          \
                                                                            \
    static void ger(const typename AViewType::execution_space& /* space */, \
                    typename AViewType::const_value_type& alpha,            \
                    const XViewType& X, const YViewType& Y,                 \
                    const AViewType& A) {                                   \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-cger-blas\n" );   \
      Kokkos::Profiling::pushRegion(                                        \
          "KokkosBlas::ger[TPL_BLAS,complex<float>]");                      \
      KOKKOSBLAS2_GER_DETERMINE_ARGS(LAYOUTA);                              \
      const std::complex<float> alpha_val = alpha;                          \
      HostBlas<std::complex<float> >::ger(                                  \
          M, N, alpha_val,                                                  \
          reinterpret_cast<const std::complex<float>*>(X.data()), one,      \
          reinterpret_cast<const std::complex<float>*>(Y.data()), one,      \
          reinterpret_cast<std::complex<float>*>(A.data()), LDA);           \
      Kokkos::Profiling::popRegion();                                       \
    }                                                                       \
  };

KOKKOSBLAS2_DGER_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true )
KOKKOSBLAS2_DGER_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
KOKKOSBLAS2_DGER_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true )
KOKKOSBLAS2_DGER_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS2_SGER_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true )
KOKKOSBLAS2_SGER_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
KOKKOSBLAS2_SGER_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true )
KOKKOSBLAS2_SGER_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS2_ZGER_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true )
KOKKOSBLAS2_ZGER_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
KOKKOSBLAS2_ZGER_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true )
KOKKOSBLAS2_ZGER_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS2_CGER_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, true )
KOKKOSBLAS2_CGER_BLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::HostSpace, false)
KOKKOSBLAS2_CGER_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, true )
KOKKOSBLAS2_CGER_BLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::HostSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_BLAS

// cuBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_CUBLAS
#include <KokkosBlas_tpl_spec.hpp>

//#include <test_common/KokkosKernels_TestUtils.hpp>
//typename multivector_layout_adapter<typename AViewType::HostMirror>::BaseType h_b_A = Kokkos::create_mirror_view(A);
//typename AViewType::HostMirror h_A = multivector_layout_adapter<typename AViewType::HostMirror>::view(h_b_A);

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_GER_CUBLAS_DETERMINE_ARGS(LAYOUTA)                       \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;     \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);

#define KOKKOSBLAS2_DGER_CUBLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE,         \
                                ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                  \
  struct GER<Kokkos::View<const double*, LAYOUTX,                             \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
             Kokkos::View<const double*, LAYOUTY,                             \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
             Kokkos::View<double**, LAYOUTA,                                  \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
             true, ETI_SPEC_AVAIL> {                                          \
    typedef double SCALAR;                                                    \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                              \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        XViewType;                                                            \
    typedef Kokkos::View<const SCALAR*, LAYOUTY,                              \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        YViewType;                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
                                                                              \
    static void ger(const typename AViewType::execution_space& space,         \
                    typename AViewType::const_value_type& alpha,              \
                    const XViewType& X, const YViewType& Y,                   \
                    const AViewType& A) {                                     \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-dger-cublas\n" );   \
      Kokkos::Profiling::pushRegion("KokkosBlas::ger[TPL_CUBLAS,double]");    \
      KOKKOSBLAS2_GER_CUBLAS_DETERMINE_ARGS(LAYOUTA);                         \
      KokkosBlas::Impl::CudaBlasSingleton& s =                                \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                   \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                           \
          cublasSetStream(s.handle, space.cuda_stream()));                    \
      if (A_is_lr) {                                                          \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasDger(s.handle, M, N, &alpha,         \
                                              Y.data(), one, X.data(), one,   \
                                              A.data(), LDA));                \
      }                                                                       \
      else {                                                                  \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasDger(s.handle, M, N, &alpha,         \
                                              X.data(), one, Y.data(), one,   \
                                              A.data(), LDA));                \
      }                                                                       \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasSetStream(s.handle, NULL));          \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS2_SGER_CUBLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE,         \
                                ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                  \
  struct GER<Kokkos::View<const float*, LAYOUTX,                              \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
             Kokkos::View<const float*, LAYOUTY,                              \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
             Kokkos::View<float**, LAYOUTA,                                   \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
             true, ETI_SPEC_AVAIL> {                                          \
    typedef float SCALAR;                                                     \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                              \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        XViewType;                                                            \
    typedef Kokkos::View<const SCALAR*, LAYOUTY,                              \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        YViewType;                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
                                                                              \
    static void ger(const typename AViewType::execution_space& space,         \
                    typename AViewType::const_value_type& alpha,              \
                    const XViewType& X, const YViewType& Y,                   \
                    const AViewType& A) {                                     \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-sger-cublas\n" );   \
      Kokkos::Profiling::pushRegion("KokkosBlas::ger[TPL_CUBLAS,float]");     \
      KOKKOSBLAS2_GER_CUBLAS_DETERMINE_ARGS(LAYOUTA);                         \
      KokkosBlas::Impl::CudaBlasSingleton& s =                                \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                   \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                           \
          cublasSetStream(s.handle, space.cuda_stream()));                    \
      if (A_is_lr) {                                                          \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasSger(s.handle, M, N, &alpha,         \
                                              Y.data(), one, X.data(), one,   \
                                              A.data(), LDA));                \
      }                                                                       \
      else {                                                                  \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasSger(s.handle, M, N, &alpha,         \
                                              X.data(), one, Y.data(), one,   \
                                              A.data(), LDA));                \
      }                                                                       \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasSetStream(s.handle, NULL));          \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS2_ZGER_CUBLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE,          \
                                ETI_SPEC_AVAIL)                                \
  template <class ExecSpace>                                                   \
  struct GER<Kokkos::View<const Kokkos::complex<double>*, LAYOUTX,             \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,           \
             Kokkos::View<const Kokkos::complex<double>*, LAYOUTY,             \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,           \
             Kokkos::View<Kokkos::complex<double>**, LAYOUTA,                  \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,           \
             true, ETI_SPEC_AVAIL> {                                           \
    typedef Kokkos::complex<double> SCALAR;                                    \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                               \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        XViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTY,                               \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        YViewType;                                                             \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                    \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
                                                                               \
    static void ger(const typename AViewType::execution_space& space,          \
                    typename AViewType::const_value_type& alpha,               \
                    const XViewType& X, const YViewType& Y,                    \
                    const AViewType& A) {                                      \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-zger-cublas\n" );    \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::ger[TPL_CUBLAS,complex<double>]");                      \
      KOKKOSBLAS2_GER_CUBLAS_DETERMINE_ARGS(LAYOUTA);                          \
      KokkosBlas::Impl::CudaBlasSingleton& s =                                 \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                    \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasSetStream(s.handle, space.cuda_stream()));                     \
      if (A_is_lr) {                                                           \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasZgeru(s.handle, M, N,                                          \
                      reinterpret_cast<const cuDoubleComplex*>(&alpha),        \
                      reinterpret_cast<const cuDoubleComplex*>(Y.data()), one, \
                      reinterpret_cast<const cuDoubleComplex*>(X.data()), one, \
                      reinterpret_cast<cuDoubleComplex*>(A.data()), LDA));     \
      }                                                                        \
      else {                                                                   \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                            \
          cublasZgeru(s.handle, M, N,                                          \
                      reinterpret_cast<const cuDoubleComplex*>(&alpha),        \
                      reinterpret_cast<const cuDoubleComplex*>(X.data()), one, \
                      reinterpret_cast<const cuDoubleComplex*>(Y.data()), one, \
                      reinterpret_cast<cuDoubleComplex*>(A.data()), LDA));     \
      }                                                                        \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasSetStream(s.handle, NULL));           \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS2_CGER_CUBLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE,         \
                                ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                  \
  struct GER<Kokkos::View<const Kokkos::complex<float>*, LAYOUTX,             \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
             Kokkos::View<const Kokkos::complex<float>*, LAYOUTY,             \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
             Kokkos::View<Kokkos::complex<float>**, LAYOUTA,                  \
                          Kokkos::Device<ExecSpace, MEM_SPACE>,               \
                          Kokkos::MemoryTraits<Kokkos::Unmanaged> >,          \
             true, ETI_SPEC_AVAIL> {					      \
    typedef Kokkos::complex<float> SCALAR;                                    \
    typedef Kokkos::View<const SCALAR*, LAYOUTX,                              \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        XViewType;                                                            \
    typedef Kokkos::View<const SCALAR*, LAYOUTY,                              \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        YViewType;                                                            \
    typedef Kokkos::View<SCALAR**, LAYOUTA,                                   \
                         Kokkos::Device<ExecSpace, MEM_SPACE>,                \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
                                                                              \
    static void ger(const typename AViewType::execution_space& space,         \
                    typename AViewType::const_value_type& alpha,              \
                    const XViewType& X, const YViewType& Y,                   \
                    const AViewType& A) {                                     \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-cger-cublas\n" );   \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosBlas::ger[TPL_CUBLAS,complex<float>]");                      \
      KOKKOSBLAS2_GER_CUBLAS_DETERMINE_ARGS(LAYOUTA);                         \
      KokkosBlas::Impl::CudaBlasSingleton& s =                                \
          KokkosBlas::Impl::CudaBlasSingleton::singleton();                   \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                           \
          cublasSetStream(s.handle, space.cuda_stream()));                    \
      if (A_is_lr) {                                                          \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                           \
          cublasCgeru(s.handle, M, N,                                         \
                      reinterpret_cast<const cuComplex*>(&alpha),             \
                      reinterpret_cast<const cuComplex*>(Y.data()), one,      \
                      reinterpret_cast<const cuComplex*>(X.data()), one,      \
                      reinterpret_cast<cuComplex*>(A.data()), LDA));          \
      }                                                                       \
      else {                                                                  \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(                                           \
          cublasCgeru(s.handle, M, N,                                         \
                      reinterpret_cast<const cuComplex*>(&alpha),             \
                      reinterpret_cast<const cuComplex*>(X.data()), one,      \
                      reinterpret_cast<const cuComplex*>(Y.data()), one,      \
                      reinterpret_cast<cuComplex*>(A.data()), LDA));          \
      }                                                                       \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL(cublasSetStream(s.handle, NULL));          \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

KOKKOSBLAS2_DGER_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_DGER_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_DGER_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_DGER_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_SGER_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_SGER_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_SGER_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_SGER_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_ZGER_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_ZGER_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_ZGER_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_ZGER_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_CGER_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_CGER_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_CGER_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_CGER_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_CUBLAS

// rocBLAS
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCBLAS
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_GER_ROCBLAS_DETERMINE_ARGS(LAYOUT)                       \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUT>::value;      \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);

#define KOKKOSBLAS2_DGER_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <>                                                                  \
  struct GER<                                                                  \
      Kokkos::View<const double*, LAYOUT,                                      \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const double*, LAYOUT,                                      \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<double**, LAYOUT,                                           \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef double SCALAR;                                                     \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                                \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        XViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                                \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        YViewType;                                                             \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                     \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
                                                                               \
    static void ger(const typename AViewType::execution_space& space,          \
                    typename AViewType::const_value_type& alpha,               \
                    const XViewType& X, const YViewType& Y,                    \
                    const AViewType& A) {                                      \
      Kokkos::Profiling::pushRegion("KokkosBlas::ger[TPL_ROCBLAS,double]");    \
      KOKKOSBLAS2_GER_ROCBLAS_DETERMINE_ARGS(LAYOUT);                          \
      KokkosBlas::Impl::RocBlasSingleton& s =                                  \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                     \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                           \
          rocblas_set_stream(s.handle, space.hip_stream()));                   \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                           \
          rocblas_dger(s.handle, M, N, &alpha, X.data(), one, Y.data(), one,   \
                       A.data(), LDA));                                        \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));       \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS2_SGER_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <>                                                                  \
  struct GER<                                                                  \
      Kokkos::View<const float*, LAYOUT,                                       \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const float*, LAYOUT,                                       \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float**, LAYOUT,                                            \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef float SCALAR;                                                      \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                                \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        XViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                                \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        YViewType;                                                             \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                     \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
                                                                               \
    static void ger(const typename AViewType::execution_space& space,          \
                    typename AViewType::const_value_type& alpha,               \
                    const XViewType& X, const YViewType& Y,                    \
                    const AViewType& A) {                                      \
      Kokkos::Profiling::pushRegion("KokkosBlas::ger[TPL_ROCBLAS,float]");     \
      KOKKOSBLAS2_GER_ROCBLAS_DETERMINE_ARGS(LAYOUT);                          \
      KokkosBlas::Impl::RocBlasSingleton& s =                                  \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                     \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                           \
          rocblas_set_stream(s.handle, space.hip_stream()));                   \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                           \
          rocblas_sger(s.handle, M, N, &alpha, X.data(), one, Y.data(), one,   \
                       A.data(), LDA));                                        \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));       \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS2_ZGER_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <>                                                                  \
  struct GER<                                                                  \
      Kokkos::View<const Kokkos::complex<double>*, LAYOUT,                     \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const Kokkos::complex<double>*, LAYOUT,                     \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<Kokkos::complex<double>**, LAYOUT,                          \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef Kokkos::complex<double> SCALAR;                                    \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                                \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        XViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                                \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        YViewType;                                                             \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                     \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
                                                                               \
    static void ger(const typename AViewType::execution_space& space,          \
                    typename AViewType::const_value_type& alpha,               \
                    const XViewType& X, const YViewType& Y,                    \
                    const AViewType& A) {                                      \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::ger[TPL_ROCBLAS,complex<double>]");                     \
      KOKKOSBLAS2_GER_ROCBLAS_DETERMINE_ARGS(LAYOUT);                          \
      KokkosBlas::Impl::RocBlasSingleton& s =                                  \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                     \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                           \
          rocblas_set_stream(s.handle, space.hip_stream()));                   \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_zger(                              \
          s.handle, M, N,                                                      \
          reinterpret_cast<const rocblas_double_complex*>(&alpha),             \
          reinterpret_cast<const rocblas_double_complex*>(X.data()), one,      \
          reinterpret_cast<const rocblas_double_complex*>(Y.data()), one,      \
          reinterpret_cast<rocblas_double_complex*>(A.data()), LDA));          \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));       \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS2_CGER_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)            \
  template <>                                                                  \
  struct GER<                                                                  \
      Kokkos::View<const Kokkos::complex<float>*, LAYOUT,                      \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const Kokkos::complex<float>*, LAYOUT,                      \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<Kokkos::complex<float>**, LAYOUT,                           \
                   Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef Kokkos::complex<float> SCALAR;                                     \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                                \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        XViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUT,                                \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        YViewType;                                                             \
    typedef Kokkos::View<SCALAR**, LAYOUT,                                     \
                         Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>, \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
                                                                               \
    static void ger(const typename AViewType::execution_space& space,          \
                    typename AViewType::const_value_type& alpha,               \
                    const XViewType& X, const YViewType& Y,                    \
                    const AViewType& A) {                                      \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::ger[TPL_ROCBLAS,complex<float>]");                      \
      KOKKOSBLAS2_GER_ROCBLAS_DETERMINE_ARGS(LAYOUT);                          \
      KokkosBlas::Impl::RocBlasSingleton& s =                                  \
          KokkosBlas::Impl::RocBlasSingleton::singleton();                     \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(                                           \
          rocblas_set_stream(s.handle, space.hip_stream()));                   \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_cger(                              \
          s.handle, M, N,                                                      \
          reinterpret_cast<const rocblas_float_complex*>(&alpha),              \
          reinterpret_cast<const rocblas_float_complex*>(X.data()), one,       \
          reinterpret_cast<const rocblas_float_complex*>(Y.data()), one,       \
          reinterpret_cast<rocblas_float_complex*>(A.data()), LDA));           \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));       \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

KOKKOSBLAS2_DGER_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_DGER_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, false)
KOKKOSBLAS2_DGER_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_DGER_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, false)

KOKKOSBLAS2_SGER_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_SGER_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, false)
KOKKOSBLAS2_SGER_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_SGER_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, false)

KOKKOSBLAS2_ZGER_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_ZGER_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, false)
KOKKOSBLAS2_ZGER_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_ZGER_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, false)

KOKKOSBLAS2_CGER_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_CGER_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, false)
KOKKOSBLAS2_CGER_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_CGER_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas
#endif  // KOKKOSKERNELS_ENABLE_TPL_ROCBLAS

#endif
