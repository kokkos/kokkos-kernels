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

#ifndef KOKKOSBLAS2_SYR_TPL_SPEC_DECL_CUBLAS_HPP_
#define KOKKOSBLAS2_SYR_TPL_SPEC_DECL_CUBLAS_HPP_

#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_SYR_CUBLAS_DETERMINE_ARGS(LAYOUT)                        \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUT>::value;      \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);

#define KOKKOSBLAS2_DSYR_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                                      \
  struct SYR< Kokkos::View< const double*                                                         \
                          , LAYOUT                                                                \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                  \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                               \
                          >                                                                       \
            , Kokkos::View< double**                                                              \
                          , LAYOUT                                                                \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                  \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                               \
                          >                                                                       \
            , true                                                                                \
            , ETI_SPEC_AVAIL                                                                      \
            > {                                                                                   \
    typedef double SCALAR;                                                                        \
    typedef Kokkos::View< const SCALAR*                                                           \
                        , LAYOUT                                                                  \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>	                                  \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                 \
                        > XViewType;                                                              \
    typedef Kokkos::View< SCALAR**                                                                \
                        , LAYOUT                                                                  \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                                    \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                 \
                        > AViewType;                                                              \
                                                                                                  \
    static void syr( const typename AViewType::execution_space  & space                           \
                   , const          char                          trans[]                         \
                   , const          char                          uplo[]                          \
                   , typename       AViewType::const_value_type & alpha                           \
                   , const          XViewType                   & X                               \
                   , const          AViewType                   & A                               \
                   ) {                                                                            \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing through tpl-dsyr-cublas\n" );                       \
      Kokkos::Profiling::pushRegion("KokkosBlas::syr[TPL_CUBLAS,double]");                        \
      KOKKOSBLAS2_SYR_CUBLAS_DETERMINE_ARGS(LAYOUT);                                              \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, space.cuda_stream()) );             \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasDsyr( s.handle                                          \
                                              , uplo                                              \
                                              , N                                                 \
                                              , &alpha                                            \
                                              , X.data()                                          \
                                              , one                                               \
                                              , A.data()                                          \
                                              , LDA                                               \
                                              )                                                   \
                                    );                                                            \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, NULL) );                            \
      Kokkos::Profiling::popRegion();                                                             \
    }                                                                                             \
  };

#define KOKKOSBLAS2_SSYR_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                                      \
  struct SYR< Kokkos::View< const float*                                                          \
                          , LAYOUT                                                                \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                  \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                               \
                          >                                                                       \
            , Kokkos::View< float**                                                               \
                          , LAYOUT                                                                \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                  \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                               \
                          >                                                                       \
            , true                                                                                \
            , ETI_SPEC_AVAIL                                                                      \
            > {                                                                                   \
    typedef float SCALAR;                                                                         \
    typedef Kokkos::View< const SCALAR*                                                           \
                        , LAYOUT                                                                  \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>	                                  \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                 \
                        > XViewType;                                                              \
    typedef Kokkos::View< SCALAR**                                                                \
                        , LAYOUT                                                                  \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                                    \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                 \
                        > AViewType;                                                              \
                                                                                                  \
    static void syr( const typename AViewType::execution_space  & space                           \
                   , const          char                          trans[]                         \
                   , const          char                          uplo[]                          \
                   , typename       AViewType::const_value_type & alpha                           \
                   , const          XViewType                   & X                               \
                   , const          AViewType                   & A                               \
                   ) {                                                                            \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing through tpl-ssyr-cublas\n" );                       \
      Kokkos::Profiling::pushRegion("KokkosBlas::syr[TPL_CUBLAS,float]");                         \
      KOKKOSBLAS2_SYR_CUBLAS_DETERMINE_ARGS(LAYOUT);                                              \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, space.cuda_stream()) );             \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSsyr( s.handle                                          \
                                              , uplo                                              \
                                              , N                                                 \
                                              , &alpha                                            \
                                              , X.data()                                          \
                                              , one                                               \
                                              , A.data()                                          \
                                              , LDA                                               \
                                              )                                                   \
                                  );                                                              \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, NULL) );                            \
      Kokkos::Profiling::popRegion();                                                             \
    }                                                                                             \
  };

#define KOKKOSBLAS2_ZSYR_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                   \
  template <class ExecSpace>                                                                            \
  struct SYR< Kokkos::View< const Kokkos::complex<double>*                                              \
                          , LAYOUT                                                                      \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                        \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                     \
                          >                                                                             \
            , Kokkos::View< Kokkos::complex<double>**                                                   \
                          , LAYOUT                                                                      \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                        \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                     \
                          >                                                                             \
            , true                                                                                      \
            , ETI_SPEC_AVAIL                                                                            \
            > {                                                                                         \
    typedef Kokkos::complex<double> SCALAR;                                                             \
    typedef Kokkos::View< const SCALAR*                                                                 \
                        , LAYOUT                                                                        \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>	                                        \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                       \
                        > XViewType;                                                                    \
    typedef Kokkos::View< SCALAR**                                                                      \
                        , LAYOUT                                                                        \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                                          \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                       \
                        > AViewType;                                                                    \
                                                                                                        \
    static void syr( const typename AViewType::execution_space  & space                                 \
                   , const          char                          trans[]                               \
                   , const          char                          uplo[]                                \
                   , typename       AViewType::const_value_type & alpha                                 \
                   , const          XViewType                   & X                                     \
                   , const          AViewType                   & A                                     \
                   ) {                                                                                  \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing through tpl-zsyr-cublas\n" );                             \
      Kokkos::Profiling::pushRegion("KokkosBlas::syr[TPL_CUBLAS,complex<double>]");                     \
      KOKKOSBLAS2_SYR_CUBLAS_DETERMINE_ARGS(LAYOUT);                                                    \
      bool justTranspose = (trans[0] == 'T') || (trans[0] == 't');                                      \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton();       \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, space.cuda_stream()) );                   \
      if (justTranspose) {                                                                              \
        kk_syr( space, trans, uplo, alpha, X, A);                                                       \
        KOKKOS_IMPL_DO_NOT_USE_PRINTF("cublasZsyru() is not supported\n"); /* AquiEPP */                \
        throw std::runtime_error("Error: cublasZsyru() is not supported.");                             \
      }                                                                                                 \
      else {                                                                                            \
        KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasZsyrc( s.handle                                             \
                                                 , uplo                                                 \
                                                 , N                                                    \
                                                 , reinterpret_cast<const cuDoubleComplex*>(&alpha)     \
                                                 , reinterpret_cast<const cuDoubleComplex*>(X.data())   \
                                                 , one                                                  \
                                                 , reinterpret_cast<cuDoubleComplex*>(A.data())         \
                                                 , LDA                                                  \
                                                 )                                                      \
                                    );                                                                  \
      }                                                                                                 \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, NULL) );                                  \
      Kokkos::Profiling::popRegion();                                                                   \
    }                                                                                                   \
  };

#define KOKKOSBLAS2_CSYR_CUBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)               \
  template <class ExecSpace>                                                                        \
  struct SYR< Kokkos::View< const Kokkos::complex<float>*                                           \
                          , LAYOUT                                                                  \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                    \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                 \
                          >                                                                         \
            , Kokkos::View< Kokkos::complex<float>**                                                \
                          , LAYOUT                                                                  \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                    \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                 \
                          >                                                                         \
            , true                                                                                  \
            , ETI_SPEC_AVAIL                                                                        \
            > {                                                                                     \
    typedef Kokkos::complex<float> SCALAR;                                                          \
    typedef Kokkos::View< const SCALAR*                                                             \
                        , LAYOUT                                                                    \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>	                                    \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                   \
                        > XViewType;                                                                \
    typedef Kokkos::View< SCALAR**                                                                  \
                        , LAYOUT                                                                    \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                                      \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                   \
                        > AViewType;                                                                \
                                                                                                    \
    static void syr( const typename AViewType::execution_space  & space                             \
                   , const          char                          trans[]                           \
                   , const          char                          uplo[]                            \
                   , typename       AViewType::const_value_type & alpha                             \
                   , const          XViewType                   & X                                 \
                   , const          AViewType                   & A                                 \
                   ) {                                                                              \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing through tpl-csyr-cublas\n" );                         \
      Kokkos::Profiling::pushRegion("KokkosBlas::syr[TPL_CUBLAS,complex<float>]");                  \
      KOKKOSBLAS2_SYR_CUBLAS_DETERMINE_ARGS(LAYOUT);                                                \
      bool justTranspose = (trans[0] == 'T') || (trans[0] == 't');                                  \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton();   \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, space.cuda_stream()) );               \
      if (justTranspose) {                                                                          \
        kk_syr( space, trans, uplo, alpha, X, A);                                                   \
        KOKKOS_IMPL_DO_NOT_USE_PRINTF("cublasCsyru() is not supported\n"); /* AquiEPP */            \
        throw std::runtime_error("Error: cublasCsyru() is not supported.");                         \
      }                                                                                             \
      else {                                                                                        \
        KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasCsyrc( s.handle                                         \
                                                 , uplo                                             \
                                                 , N                                                \
                                                 , reinterpret_cast<const cuComplex*>(&alpha)       \
                                                 , reinterpret_cast<const cuComplex*>(X.data())     \
                                                 , one                                              \
                                                 , reinterpret_cast<cuComplex*>(A.data())           \
                                                 , LDA                                              \
                                                 )                                                  \
                                    );                                                              \
      }                                                                                             \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, NULL) );                              \
      Kokkos::Profiling::popRegion();                                                               \
    }                                                                                               \
  };

KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, true )
KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, false)
KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true )
KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, true )
KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, false)
KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true )
KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, true )
KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, false)
KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true )
KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, true )
KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::CudaUVMSpace, false)
KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaUVMSpace, true )
KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::CudaUVMSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif
