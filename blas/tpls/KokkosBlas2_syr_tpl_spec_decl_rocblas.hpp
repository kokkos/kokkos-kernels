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
  bool A_is_ll      = std::is_same<Kokkos::LayoutLeft, LAYOUT>::value;       \
  bool A_is_lr      = std::is_same<Kokkos::LayoutRight, LAYOUT>::value;      \
  const int M       = static_cast<int>(A_is_lr ? A.extent(1) : A.extent(0)); \
  const int N       = static_cast<int>(A_is_lr ? A.extent(0) : A.extent(1)); \
  constexpr int one = 1;                                                     \
  const int LDA     = A_is_lr ? A.stride(0) : A.stride(1);

#define KOKKOSBLAS2_DSYR_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                            \
  template <>                                                                                  \
  struct SYR< Kokkos::View< const double*                                                      \
                          , LAYOUT                                                             \
                          , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>               \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                            \
                          >                                                                    \
            , Kokkos::View< double**                                                           \
                          , LAYOUT                                                             \
                          , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>               \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                            \
                          >                                                                    \
            , true                                                                             \
            , ETI_SPEC_AVAIL                                                                   \
            > {                                                                                \
    typedef double SCALAR;                                                                     \
    typedef Kokkos::View< const SCALAR*                                                        \
                        , LAYOUT                                                               \
                        , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                 \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                              \
                        > XViewType;                                                           \
    typedef Kokkos::View< SCALAR**                                                             \
                        , LAYOUT                                                               \
                        , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                 \
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
      if (A_is_ll) {                                                                           \
        KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_dsyr( s.handle                                  \
                                                   , uplo                                      \
                                                   , N                                         \
                                                   , &alpha                                    \
                                                   , X.data()                                  \
                                                   , one                                       \
                                                   , A.data()                                  \
                                                   , LDA                                       \
                                                   )                                           \
                                     );                                                        \
      }                                                                                        \
      else {                                                                                   \
        KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_dsyr( s.handle                                  \
                                                   , uplo                                      \
                                                   , N                                         \
                                                   , &alpha                                    \
                                                   , X.data()                                  \
                                                   , one                                       \
                                                   , A.data()                                  \
                                                   , LDA                                       \
                                                   )                                           \
                                     );                                                        \
      }                                                                                        \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_set_stream(s.handle, NULL) );                     \
      Kokkos::Profiling::popRegion();                                                          \
    }                                                                                          \
  };

#define KOKKOSBLAS2_SSYR_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                            \
  template <>                                                                                  \
  struct SYR< Kokkos::View< const float*                                                       \
                          , LAYOUT                                                             \
                          , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>               \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                            \
                          >                                                                    \
            , Kokkos::View< const float*                                                       \
                          , LAYOUT                                                             \
                          , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>               \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                            \
                          >                                                                    \
            , Kokkos::View< float**                                                            \
                          , LAYOUT                                                             \
                          , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>               \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                            \
                          >                                                                    \
            , true                                                                             \
            , ETI_SPEC_AVAIL                                                                   \
            > {                                                                                \
    typedef float SCALAR;                                                                      \
    typedef Kokkos::View< const SCALAR*                                                        \
                        , LAYOUT                                                               \
                        , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                 \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                              \
                        > XViewType;                                                           \
    typedef Kokkos::View< SCALAR**                                                             \
                        , LAYOUT                                                               \
                        , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                 \
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
      if (A_is_ll) {                                                                           \
        KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_ssyr( s.handle                                  \
                                                   , uplo                                      \
                                                   , N                                         \
                                                   , &alpha                                    \
                                                   , X.data()                                  \
                                                   , one                                       \
                                                   , A.data()                                  \
                                                   , LDA                                       \
                                                   )                                           \
                                     );                                                        \
      }                                                                                        \
      else {                                                                                   \
        KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_ssyr( s.handle                                  \
                                                   , uplo                                      \
                                                   , N                                         \
                                                   , &alpha                                    \
                                                   , X.data()                                  \
                                                   , one                                       \
                                                   , A.data()                                  \
                                                   , LDA                                       \
                                                   )                                           \
                                     );                                                        \
      }                                                                                        \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_set_stream(s.handle, NULL) );                     \
      Kokkos::Profiling::popRegion();                                                          \
    }                                                                                          \
  };

#define KOKKOSBLAS2_ZSYR_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                                               \
  template <>                                                                                                     \
  struct SYR< Kokkos::View< const Kokkos::complex<double>*                                                        \
                          , LAYOUT                                                                                \
                          , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                                  \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                               \
                          >                                                                                       \
            , Kokkos::View< const Kokkos::complex<double>*                                                        \
                          , LAYOUT                                                                                \
                          , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                                  \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                               \
                          >                                                                                       \
            , Kokkos::View< Kokkos::complex<double>**                                                             \
                          , LAYOUT                                                                                \
                          , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                                  \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                               \
                          >                                                                                       \
            , true                                                                                                \
            , ETI_SPEC_AVAIL                                                                                      \
            > {                                                                                                   \
    typedef Kokkos::complex<double> SCALAR;                                                                       \
    typedef Kokkos::View< const SCALAR*                                                                           \
                        , LAYOUT                                                                                  \
                        , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                                    \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                                 \
                        > XViewType;                                                                              \
    typedef Kokkos::View< SCALAR**                                                                                \
                        , LAYOUT                                                                                  \
                        , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                                    \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                                 \
                        > AViewType;                                                                              \
                                                                                                                  \
    static void syr( const typename AViewType::execution_space  & space                                           \
                   , const          char                          trans[]                                         \
                   , const          char                          uplo[]                                          \
                   , typename       AViewType::const_value_type & alpha                                           \
                   , const          XViewType                   & X                                               \
                   , const          AViewType                   & A                                               \
                   ) {                                                                                            \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing through tpl-zsyr-rocblas\n" );                                      \
      Kokkos::Profiling::pushRegion("KokkosBlas::syr[TPL_ROCBLAS,complex<double>]");                              \
      KOKKOSBLAS2_SYR_ROCBLAS_DETERMINE_ARGS(LAYOUT);                                                             \
      bool justTranspose = (trans[0] == 'T') || (trans[0] == 't');                                                \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                    \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_set_stream(s.handle, space.hip_stream()) );                          \
      if (A_is_ll) {                                                                                              \
        if (justTranspose) {                                                                                      \
          KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_zsyru( s.handle                                                  \
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
      }                                                                                                           \
      else {                                                                                                      \
        if (justTranspose) {                                                                                      \
          KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_zsyru( s.handle                                                  \
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
        else {                                                                                                    \
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("rocblasZsyrc() requires LayoutLeft: throwing exception\n");              \
          throw std::runtime_error("Error: rocblasZsyrc() requires LayoutLeft views.");                           \
        }                                                                                                         \
      }                                                                                                           \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));                                          \
      Kokkos::Profiling::popRegion();                                                                             \
    }                                                                                                             \
  };

#define KOKKOSBLAS2_CSYR_ROCBLAS(LAYOUT, MEM_SPACE, ETI_SPEC_AVAIL)                                              \
  template <>                                                                                                    \
  struct SYR< Kokkos::View< const Kokkos::complex<float>*                                                        \
                          , LAYOUT                                                                               \
                          , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                                 \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                              \
                          >                                                                                      \
            , Kokkos::View< const Kokkos::complex<float>*                                                        \
                          , LAYOUT                                                                               \
                          , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                                 \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                              \
                          >                                                                                      \
            , Kokkos::View< Kokkos::complex<float>**                                                             \
                          , LAYOUT                                                                               \
                          , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                                 \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                              \
                          >                                                                                      \
            , true                                                                                               \
            , ETI_SPEC_AVAIL                                                                                     \
            > {                                                                                                  \
    typedef Kokkos::complex<float> SCALAR;                                                                       \
    typedef Kokkos::View< const SCALAR*                                                                          \
                        , LAYOUT                                                                                 \
                        , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                                   \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                                \
                        > XViewType;                                                                             \
    typedef Kokkos::View< SCALAR**                                                                               \
                        , LAYOUT                                                                                 \
                        , Kokkos::Device<Kokkos::Experimental::HIP, MEM_SPACE>                                   \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                                \
                        > AViewType;                                                                             \
                                                                                                                 \
    static void syr( const typename AViewType::execution_space  & space                                          \
                   , const          char                          trans[]                                        \
                   , const          char                          uplo[]                                         \
                   , typename       AViewType::const_value_type & alpha                                          \
                   , const          XViewType                   & X                                              \
                   , const          AViewType                   & A                                              \
                   ) {                                                                                           \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing through tpl-csyr-rocblas\n" );                                     \
      Kokkos::Profiling::pushRegion("KokkosBlas::syr[TPL_ROCBLAS,complex<float>]");                              \
      KOKKOSBLAS2_SYR_ROCBLAS_DETERMINE_ARGS(LAYOUT);                                                            \
      bool justTranspose = (trans[0] == 'T') || (trans[0] == 't');                                               \
      KokkosBlas::Impl::RocBlasSingleton& s = KokkosBlas::Impl::RocBlasSingleton::singleton();                   \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_set_stream(s.handle, space.hip_stream()) );                         \
      if (A_is_ll) {                                                                                             \
        if (justTranspose) {                                                                                     \
          KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_csyru( s.handle                                                 \
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
      }                                                                                                          \
      else {                                                                                                     \
        if (justTranspose) {                                                                                     \
          KOKKOS_ROCBLAS_SAFE_CALL_IMPL( rocblas_csyru( s.handle                                                 \
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
        else {                                                                                                   \
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("rocblasCsyrc() requires LayoutLeft: throwing exception\n");             \
          throw std::runtime_error("Error: rocblasCgec() requires LayoutLeft views.");                           \
        }                                                                                                        \
      }                                                                                                          \
      KOKKOS_ROCBLAS_SAFE_CALL_IMPL(rocblas_set_stream(s.handle, NULL));                                         \
      Kokkos::Profiling::popRegion();                                                                            \
    }                                                                                                            \
  };

KOKKOSBLAS2_DSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_DSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, false)
KOKKOSBLAS2_DSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_DSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, false)

KOKKOSBLAS2_SSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_SSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, false)
KOKKOSBLAS2_SSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_SSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, false)

KOKKOSBLAS2_ZSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_ZSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, false)
KOKKOSBLAS2_ZSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_ZSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, false)

KOKKOSBLAS2_CSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_CSYR_ROCBLAS(Kokkos::LayoutLeft,  Kokkos::Experimental::HIPSpace, false)
KOKKOSBLAS2_CSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, true )
KOKKOSBLAS2_CSYR_ROCBLAS(Kokkos::LayoutRight, Kokkos::Experimental::HIPSpace, false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif
