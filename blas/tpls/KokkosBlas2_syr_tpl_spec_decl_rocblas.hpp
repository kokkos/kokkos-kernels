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

#ifndef KOKKOSBLAS2_SYR_TPL_SPEC_DECL_ROCBLAS_HPP_
#define KOKKOSBLAS2_SYR_TPL_SPEC_DECL_ROCBLAS_HPP_

#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_SYR_ROCBLAS_DETERMINE_ARGS(LAYOUT)                       \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUT>::value;      \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);

#define KOKKOSBLAS2_DSYR_ROCBLAS(LAYOUT, EXEC_SPACE, MEM_SPACE, ETI_SPEC_AVAIL)                \
  template <>                                                                                  \
  struct SYR< EXEC_SPACE                                                                       \
            , Kokkos::View< const double*                                                      \
                          , LAYOUT                                                             \
                          , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                              \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                            \
                          >                                                                    \
            , Kokkos::View< double**                                                           \
                          , LAYOUT                                                             \
                          , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                              \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                            \
                          >                                                                    \
            , true                                                                             \
            , ETI_SPEC_AVAIL                                                                   \
            > {                                                                                \
    typedef double SCALAR;                                                                     \
    typedef Kokkos::View< const SCALAR*                                                        \
                        , LAYOUT                                                               \
                        , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                              \
                        > XViewType;                                                           \
    typedef Kokkos::View< SCALAR**                                                             \
                        , LAYOUT                                                               \
                        , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                              \
                        > AViewType;                                                           \
                                                                                               \
    static void syr( const typename AViewType::execution_space  & space                        \
                   , const          char                          trans[]                      \
                   , const          char                          uplo[]                       \
                   , typename       AViewType::const_value_type & alpha                        \
                   , const          XViewType                   & X                            \
                   , const          AViewType                   & A                            \
                   ) {                                                                         \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing through tpl-dsyr-rocblas\n" );                   \
      Kokkos::Profiling::pushRegion("KokkosBlas::syr[TPL_ROCBLAS,double]");                    \
      KOKKOSBLAS2_SYR_ROCBLAS_DETERMINE_ARGS(LAYOUT);                                          \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton(); \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_set_stream(s.handle, space.hip_stream()) );       \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_dsyr( s.handle                                    \
                                                 , uplo                                        \
                                                 , N                                           \
                                                 , &alpha                                      \
                                                 , X.data()                                    \
                                                 , one                                         \
                                                 , A.data()                                    \
                                                 , LDA                                         \
                                                 )                                             \
                                   );                                                          \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_set_stream(s.handle, NULL) );                     \
      Kokkos::Profiling::popRegion();                                                          \
    }                                                                                          \
  };

#define KOKKOSBLAS2_SSYR_ROCBLAS(LAYOUT, EXEC_SPACE, MEM_SPACE, ETI_SPEC_AVAIL)                \
  template <>                                                                                  \
  struct SYR< EXEC_SPACE                                                                       \
            , Kokkos::View< const float*                                                       \
                          , LAYOUT                                                             \
                          , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                              \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                            \
                          >                                                                    \
            , Kokkos::View< float**                                                            \
                          , LAYOUT                                                             \
                          , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                              \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                            \
                          >                                                                    \
            , true                                                                             \
            , ETI_SPEC_AVAIL                                                                   \
            > {                                                                                \
    typedef float SCALAR;                                                                      \
    typedef Kokkos::View< const SCALAR*                                                        \
                        , LAYOUT                                                               \
                        , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                              \
                        > XViewType;                                                           \
    typedef Kokkos::View< SCALAR**                                                             \
                        , LAYOUT                                                               \
                        , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                              \
                        > AViewType;                                                           \
                                                                                               \
    static void syr( const typename AViewType::execution_space  & space                        \
                   , const          char                          trans[]                      \
                   , const          char                          uplo[]                       \
                   , typename       AViewType::const_value_type & alpha                        \
                   , const          XViewType                   & X                            \
                   , const          AViewType                   & A                            \
                   ) {                                                                         \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing through tpl-ssyr-rocblas\n" );                   \
      Kokkos::Profiling::pushRegion("KokkosBlas::syr[TPL_ROCBLAS,float]");                     \
      KOKKOSBLAS2_SYR_ROCBLAS_DETERMINE_ARGS(LAYOUT);                                          \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton(); \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_set_stream(s.handle, space.hip_stream()) );       \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_ssyr( s.handle                                    \
                                                 , uplo                                        \
                                                 , N                                           \
                                                 , &alpha                                      \
                                                 , X.data()                                    \
                                                 , one                                         \
                                                 , A.data()                                    \
                                                 , LDA                                         \
                                                 )                                             \
                                   );                                                          \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_set_stream(s.handle, NULL) );                     \
      Kokkos::Profiling::popRegion();                                                          \
    }                                                                                          \
  };

#define KOKKOSBLAS2_ZSYR_ROCBLAS(LAYOUT, EXEC_SPACE, MEM_SPACE, ETI_SPEC_AVAIL)                                 \
  template <>                                                                                                   \
  struct SYR< EXEC_SPACE                                                                                        \
            , Kokkos::View< const Kokkos::complex<double>*                                                      \
                          , LAYOUT                                                                              \
                          , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                               \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                             \
                          >                                                                                     \
            , Kokkos::View< Kokkos::complex<double>**                                                           \
                          , LAYOUT                                                                              \
                          , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                               \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                             \
                          >                                                                                     \
            , true                                                                                              \
            , ETI_SPEC_AVAIL                                                                                    \
            > {                                                                                                 \
    typedef Kokkos::complex<double> SCALAR;                                                                     \
    typedef Kokkos::View< const SCALAR*                                                                         \
                        , LAYOUT                                                                                \
                        , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                                 \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                               \
                        > XViewType;                                                                            \
    typedef Kokkos::View< SCALAR**                                                                              \
                        , LAYOUT                                                                                \
                        , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                                 \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                               \
                        > AViewType;                                                                            \
                                                                                                                \
    static void syr( const typename AViewType::execution_space  & space                                         \
                   , const          char                          trans[]                                       \
                   , const          char                          uplo[]                                        \
                   , typename       AViewType::const_value_type & alpha                                         \
                   , const          XViewType                   & X                                             \
                   , const          AViewType                   & A                                             \
                   ) {                                                                                          \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing through tpl-zsyr-rocblas\n" );                                    \
      Kokkos::Profiling::pushRegion("KokkosBlas::syr[TPL_ROCBLAS,complex<double>]");                            \
      KOKKOSBLAS2_SYR_ROCBLAS_DETERMINE_ARGS(LAYOUT);                                                           \
      bool justTranspose = (trans[0] == 'T') || (trans[0] == 't');                                              \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                  \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_set_stream(s.handle, space.hip_stream()) );                        \
      if (justTranspose) {                                                                                      \
        kk_syr( space, trans, uplo, alpha, X, A);                                                               \
        KOKKOS_IMPL_DO_NOT_USE_PRINTF("rocblasZsyru() is not supported\n"); /* AquiEPP */                       \
        throw std::runtime_error("Error: rocblasZsyru() is not supported.");                                    \
      }                                                                                                         \
      else {                                                                                                    \
        KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_zsyrc( s.handle                                                  \
                                                    , uplo                                                      \
                                                    , N                                                         \
                                                    , reinterpret_cast<const rocblas_double_complex*>(&alpha)   \
                                                    , reinterpret_cast<const rocblas_double_complex*>(X.data()) \
                                                    , one                                                       \
                                                    , reinterpret_cast<rocblas_double_complex*>(A.data())       \
                                                    , LDA                                                       \
                                                    )                                                           \
                                     );                                                                         \
      }                                                                                                         \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));                                        \
      Kokkos::Profiling::popRegion();                                                                           \
    }                                                                                                           \
  };

#define KOKKOSBLAS2_CSYR_ROCBLAS(LAYOUT, EXEC_SPACE, MEM_SPACE, ETI_SPEC_AVAIL)                                \
  template <>                                                                                                  \
  struct SYR< EXEC_SPACE                                                                                       \
            , Kokkos::View< const Kokkos::complex<float>*                                                      \
                          , LAYOUT                                                                             \
                          , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                              \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                            \
                          >                                                                                    \
            , Kokkos::View< Kokkos::complex<float>**                                                           \
                          , LAYOUT                                                                             \
                          , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                              \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                            \
                          >                                                                                    \
            , true                                                                                             \
            , ETI_SPEC_AVAIL                                                                                   \
            > {                                                                                                \
    typedef Kokkos::complex<float> SCALAR;                                                                     \
    typedef Kokkos::View< const SCALAR*                                                                        \
                        , LAYOUT                                                                               \
                        , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                                \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                              \
                        > XViewType;                                                                           \
    typedef Kokkos::View< SCALAR**                                                                             \
                        , LAYOUT                                                                               \
                        , Kokkos::Device<EXEC_SPACE, MEM_SPACE>                                                \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                              \
                        > AViewType;                                                                           \
                                                                                                               \
    static void syr( const typename AViewType::execution_space  & space                                        \
                   , const          char                          trans[]                                      \
                   , const          char                          uplo[]                                       \
                   , typename       AViewType::const_value_type & alpha                                        \
                   , const          XViewType                   & X                                            \
                   , const          AViewType                   & A                                            \
                   ) {                                                                                         \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing through tpl-csyr-rocblas\n" );                                   \
      Kokkos::Profiling::pushRegion("KokkosBlas::syr[TPL_ROCBLAS,complex<float>]");                            \
      KOKKOSBLAS2_SYR_ROCBLAS_DETERMINE_ARGS(LAYOUT);                                                          \
      bool justTranspose = (trans[0] == 'T') || (trans[0] == 't');                                             \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                 \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_set_stream(s.handle, space.hip_stream()) );                       \
      if (justTranspose) {                                                                                     \
        kk_syr( space, trans, uplo, alpha, X, A);                                                              \
        KOKKOS_IMPL_DO_NOT_USE_PRINTF("rocblasCsyru() is not supported\n"); /* AquiEPP */                      \
        throw std::runtime_error("Error: rocblasCsyru() is not supported.");                                   \
      }                                                                                                        \
      else {                                                                                                   \
        KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_csyrc( s.handle                                                 \
                                                    , uplo                                                     \
                                                    , N                                                        \
                                                    , reinterpret_cast<const rocblas_float_complex*>(&alpha)   \
                                                    , reinterpret_cast<const rocblas_float_complex*>(X.data()) \
                                                    , one                                                      \
                                                    , reinterpret_cast<rocblas_float_complex*>(A.data())       \
                                                    , LDA                                                      \
                                                    )                                                          \
                                     );                                                                        \
      }                                                                                                        \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));                                       \
      Kokkos::Profiling::popRegion();                                                                          \
    }                                                                                                          \
  };

KOKKOSBLAS2_DSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::HIP, Kokkos::HIPSpace, true )
KOKKOSBLAS2_DSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::HIP, Kokkos::HIPSpace, false)
KOKKOSBLAS2_DSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIP, Kokkos::HIPSpace, true )
KOKKOSBLAS2_DSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIP, Kokkos::HIPSpace, false)

KOKKOSBLAS2_SSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::HIP, Kokkos::HIPSpace, true )
KOKKOSBLAS2_SSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::HIP, Kokkos::HIPSpace, false)
KOKKOSBLAS2_SSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIP, Kokkos::HIPSpace, true )
KOKKOSBLAS2_SSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIP, Kokkos::HIPSpace, false)

KOKKOSBLAS2_ZSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::HIP, Kokkos::HIPSpace, true )
KOKKOSBLAS2_ZSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::HIP, Kokkos::HIPSpace, false)
KOKKOSBLAS2_ZSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIP, Kokkos::HIPSpace, true )
KOKKOSBLAS2_ZSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIP, Kokkos::HIPSpace, false)

KOKKOSBLAS2_CSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::HIP, Kokkos::HIPSpace, true )
KOKKOSBLAS2_CSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::HIP, Kokkos::HIPSpace, false)
KOKKOSBLAS2_CSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIP, Kokkos::HIPSpace, true )
KOKKOSBLAS2_CSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::HIP, Kokkos::HIPSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif
