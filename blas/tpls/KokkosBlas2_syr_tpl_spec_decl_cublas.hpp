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

#define KOKKOSBLAS2_SYR_CUBLAS_DETERMINE_ARGS(LAYOUTA)                       \
  bool A_is_ll      = std::is_same<Kokkos::LayoutLeft, LAYOUTA>::value;      \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;     \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);                   \
  if (( trans[0] == 'T' ) ||                                                 \
      ( trans[0] == 't' ) ||                                                 \
      ( trans[0] == 'H' ) ||                                                 \
      ( trans[0] == 'h' )) {                                                 \
  }                                                                          \
  else {                                                                     \
    throw std::runtime_error("Error: invalid 'trans' for cudaBlas::syr()");  \
  }                                                                          \
  if (( uplo[0] == 'U' ) ||                                                  \
      ( uplo[0] == 'u' ) ||                                                  \
      ( uplo[0] == 'L' ) ||                                                  \
      ( uplo[0] == 'l' )) {                                                  \
  }                                                                          \
  else {                                                                     \
    throw std::runtime_error("Error: invalid 'uplo' for cudaBlas::syr()");   \
  }

#define KOKKOSBLAS2_DSYR_CUBLAS(LAYOUTX, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                                      \
  struct SYR< Kokkos::View< const double*                                                         \
                          , LAYOUTX                                                               \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                  \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                               \
                          >                                                                       \
            , Kokkos::View< double**                                                              \
                          , LAYOUTA                                                               \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                  \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                               \
                          >                                                                       \
            , true                                                                                \
            , ETI_SPEC_AVAIL                                                                      \
            > {                                                                                   \
    typedef double SCALAR;                                                                        \
    typedef Kokkos::View< const SCALAR*                                                           \
                        , LAYOUTX                                                                 \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>	                                  \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                 \
                        > XViewType;                                                              \
    typedef Kokkos::View< SCALAR**                                                                \
                        , LAYOUTA                                                                 \
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
      KOKKOSBLAS2_SYR_CUBLAS_DETERMINE_ARGS(LAYOUTA);                                             \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, space.cuda_stream()) );             \
      if (A_is_ll) {                                                                              \
        KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasDsyr( s.handle                                        \
                                                , uplo                                            \
                                                , N                                               \
                                                , &alpha                                          \
                                                , X.data()                                        \
                                                , one                                             \
                                                , A.data()                                        \
                                                , LDA                                             \
                                                )                                                 \
                                    );                                                            \
      }                                                                                           \
      else {                                                                                      \
        KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasDsyr( s.handle                                        \
                                                , uplo                                            \
                                                , N                                               \
                                                , &alpha                                          \
                                                , X.data()                                        \
                                                , one                                             \
                                                , A.data()                                        \
                                                , LDA                                             \
                                                )                                                 \
                                    );                                                            \
      }                                                                                           \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, NULL) );                            \
      Kokkos::Profiling::popRegion();                                                             \
    }                                                                                             \
  };

#define KOKKOSBLAS2_SSYR_CUBLAS(LAYOUTX, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL)             \
  template <class ExecSpace>                                                                      \
  struct SYR< Kokkos::View< const float*                                                          \
                          , LAYOUTX                                                               \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                  \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                               \
                          >                                                                       \
            , Kokkos::View< float**                                                               \
                          , LAYOUTA                                                               \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                  \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                               \
                          >                                                                       \
            , true                                                                                \
            , ETI_SPEC_AVAIL                                                                      \
            > {                                                                                   \
    typedef float SCALAR;                                                                         \
    typedef Kokkos::View< const SCALAR*                                                           \
                        , LAYOUTX                                                                 \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>	                                  \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                 \
                        > XViewType;                                                              \
    typedef Kokkos::View< SCALAR**                                                                \
                        , LAYOUTA                                                                 \
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
      KOKKOSBLAS2_SYR_CUBLAS_DETERMINE_ARGS(LAYOUTA);                                             \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton(); \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, space.cuda_stream()) );             \
      if (A_is_ll) {                                                                              \
        KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSsyr( s.handle                                        \
                                                , uplo                                            \
                                                , N                                               \
                                                , &alpha                                          \
                                                , X.data()                                        \
                                                , one                                             \
                                                , A.data()                                        \
                                                , LDA                                             \
                                                )                                                 \
                                    );                                                            \
      }                                                                                           \
      else {                                                                                      \
        KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSsyr( s.handle                                        \
                                                , uplo                                            \
                                                , N                                               \
                                                , &alpha                                          \
                                                , X.data()                                        \
                                                , one                                             \
                                                , A.data()                                        \
                                                , LDA                                             \
                                                )                                                 \
                                    );                                                            \
      }                                                                                           \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, NULL) );                            \
      Kokkos::Profiling::popRegion();                                                             \
    }                                                                                             \
  };

#define KOKKOSBLAS2_ZSYR_CUBLAS(LAYOUTX, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL)                   \
  template <class ExecSpace>                                                                            \
  struct SYR< Kokkos::View< const Kokkos::complex<double>*                                              \
                          , LAYOUTX                                                                     \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                        \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                     \
                          >                                                                             \
            , Kokkos::View< Kokkos::complex<double>**                                                   \
                          , LAYOUTA                                                                     \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                        \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                     \
                          >                                                                             \
            , true                                                                                      \
            , ETI_SPEC_AVAIL                                                                            \
            > {                                                                                         \
    typedef Kokkos::complex<double> SCALAR;                                                             \
    typedef Kokkos::View< const SCALAR*                                                                 \
                        , LAYOUTX                                                                       \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>	                                        \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                       \
                        > XViewType;                                                                    \
    typedef Kokkos::View< SCALAR**                                                                      \
                        , LAYOUTA                                                                       \
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
      KOKKOSBLAS2_SYR_CUBLAS_DETERMINE_ARGS(LAYOUTA);                                                   \
      bool justTranspose = (trans[0] == 'T') || (trans[0] == 't');                                      \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton();       \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, space.cuda_stream()) );                   \
      if (A_is_ll) {                                                                                    \
	if (justTranspose) {                                                                            \
          KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasZsyru( s.handle                                           \
                                                   , uplo                                               \
                                                   , N                                                  \
                                                   , reinterpret_cast<const cuDoubleComplex*>(&alpha)   \
                                                   , reinterpret_cast<const cuDoubleComplex*>(X.data()) \
                                                   , one                                                \
                                                   , reinterpret_cast<cuDoubleComplex*>(A.data())       \
                                                   , LDA                                                \
                                                   )                                                    \
                                      );                                                                \
        }                                                                                               \
	else {                                                                                          \
          KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasZsyrc( s.handle                                           \
                                                   , uplo                                               \
                                                   , N                                                  \
                                                   , reinterpret_cast<const cuDoubleComplex*>(&alpha)   \
                                                   , reinterpret_cast<const cuDoubleComplex*>(X.data()) \
                                                   , one                                                \
                                                   , reinterpret_cast<cuDoubleComplex*>(A.data())       \
                                                   , LDA                                                \
                                                   )                                                    \
                                      );                                                                \
	}                                                                                               \
      }                                                                                                 \
      else {                                                                                            \
	if (justTranspose) {                                                                            \
          KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasZsyru( s.handle                                           \
                                                   , uplo                                               \
                                                   , N                                                  \
                                                   , reinterpret_cast<const cuDoubleComplex*>(&alpha)   \
                                                   , reinterpret_cast<const cuDoubleComplex*>(X.data()) \
                                                   , one                                                \
                                                   , reinterpret_cast<cuDoubleComplex*>(A.data())       \
                                                   , LDA                                                \
                                                   )                                                    \
                                      );                                                                \
        }                                                                                               \
	else {                                                                                          \
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("cublasZsyrc() requires LayoutLeft: throwing exception\n");     \
          throw std::runtime_error("Error: cublasZsyrc() requires LayoutLeft views.");                  \
	}                                                                                               \
      }                                                                                                 \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, NULL) );                                  \
      Kokkos::Profiling::popRegion();                                                                   \
    }                                                                                                   \
  };

#define KOKKOSBLAS2_CSYR_CUBLAS(LAYOUTX, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL)               \
  template <class ExecSpace>                                                                        \
  struct SYR< Kokkos::View< const Kokkos::complex<float>*                                           \
                          , LAYOUTX                                                                 \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                    \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                 \
                          >                                                                         \
            , Kokkos::View< Kokkos::complex<float>**                                                \
                          , LAYOUTA                                                                 \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                    \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                 \
                          >                                                                         \
            , true                                                                                  \
            , ETI_SPEC_AVAIL                                                                        \
            > {                                                                                     \
    typedef Kokkos::complex<float> SCALAR;                                                          \
    typedef Kokkos::View< const SCALAR*                                                             \
                        , LAYOUTX                                                                   \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>	                                    \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                   \
                        > XViewType;                                                                \
    typedef Kokkos::View< SCALAR**                                                                  \
                        , LAYOUTA                                                                   \
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
      KOKKOSBLAS2_SYR_CUBLAS_DETERMINE_ARGS(LAYOUTA);                                               \
      bool justTranspose = (trans[0] == 'T') || (trans[0] == 't');                                  \
      KokkosBlas::Impl::CudaBlasSingleton & s = KokkosBlas::Impl::CudaBlasSingleton::singleton();   \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, space.cuda_stream()) );               \
      if (A_is_ll) {                                                                                \
	if (justTranspose) {                                                                        \
          KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasCsyru( s.handle                                       \
                                                   , uplo                                           \
                                                   , N                                              \
                                                   , reinterpret_cast<const cuComplex*>(&alpha)     \
                                                   , reinterpret_cast<const cuComplex*>(X.data())   \
                                                   , one                                            \
                                                   , reinterpret_cast<cuComplex*>(A.data())         \
                                                   , LDA                                            \
                                                   )                                                \
                                      );                                                            \
        }                                                                                           \
	else {                                                                                      \
          KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasCsyrc( s.handle                                       \
                                                   , uplo                                           \
                                                   , N                                              \
                                                   , reinterpret_cast<const cuComplex*>(&alpha)     \
                                                   , reinterpret_cast<const cuComplex*>(X.data())   \
                                                   , one                                            \
                                                   , reinterpret_cast<cuComplex*>(A.data())         \
                                                   , LDA                                            \
                                                   )                                                \
                                      );                                                            \
	}                                                                                           \
      }                                                                                             \
      else {                                                                                        \
	if (justTranspose) {                                                                        \
          KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasCsyru( s.handle                                       \
                                                   , uplo                                           \
                                                   , N                                              \
                                                   , reinterpret_cast<const cuComplex*>(&alpha)     \
                                                   , reinterpret_cast<const cuComplex*>(X.data())   \
                                                   , one                                            \
                                                   , reinterpret_cast<cuComplex*>(A.data())         \
                                                   , LDA                                            \
                                                   )                                                \
                                      );                                                            \
        }                                                                                           \
	else {                                                                                      \
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("cublasCsyrc() requires LayoutLeft: throwing exception\n"); \
          throw std::runtime_error("Error: cublasCsyrc() requires LayoutLeft views.");              \
	}                                                                                           \
      }                                                                                             \
      KOKKOS_CUBLAS_SAFE_CALL_IMPL( cublasSetStream(s.handle, NULL) );                              \
      Kokkos::Profiling::popRegion();                                                               \
    }                                                                                               \
  };

KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_DSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_SSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_ZSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, true )
KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::LayoutLeft,  Kokkos::CudaSpace, false)
KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, true )
KOKKOSBLAS2_CSYR_CUBLAS(Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::LayoutRight, Kokkos::CudaSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif
