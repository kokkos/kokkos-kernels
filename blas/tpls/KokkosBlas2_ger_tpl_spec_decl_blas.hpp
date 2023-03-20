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

#ifndef KOKKOSBLAS2_GER_TPL_SPEC_DECL_BLAS_HPP_
#define KOKKOSBLAS2_GER_TPL_SPEC_DECL_BLAS_HPP_

#include "KokkosBlas_Host_tpl.hpp"

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS2_GER_DETERMINE_ARGS(LAYOUTA)                              \
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
    throw std::runtime_error("Error: invalid 'trans' for HostBlas::ger()");  \
  }

#define KOKKOSBLAS2_DGER_BLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL) \
  template <class ExecSpace>                                                        \
  struct GER< Kokkos::View< const double*                                           \
                          , LAYOUTX                                                 \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                    \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                 \
                          >                                                         \
            , Kokkos::View< const double*                                           \
                          , LAYOUTY                                                 \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                    \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                 \
                          >                                                         \
            , Kokkos::View< double**                                                \
                          , LAYOUTA                                                 \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                    \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                 \
                          >                                                         \
            , true                                                                  \
            , ETI_SPEC_AVAIL                                                        \
            > {                                                                     \
    typedef double SCALAR;                                                          \
    typedef Kokkos::View< const SCALAR*                                             \
                        , LAYOUTX                                                   \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                      \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                   \
                        > XViewType;                                                \
    typedef Kokkos::View< const SCALAR*                                             \
                        , LAYOUTY                                                   \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                      \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                   \
                        > YViewType;                                                \
    typedef Kokkos::View< SCALAR**                                                  \
                        , LAYOUTA                                                   \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                      \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                   \
                        > AViewType;                                                \
                                                                                    \
    static void ger( const typename AViewType::execution_space  & /* space */       \
                   , const          char                          trans[]           \
                   , typename       AViewType::const_value_type & alpha             \
                   , const          XViewType                   & X                 \
                   , const          YViewType                   & Y                 \
                   , const          AViewType                   & A                 \
                   ) {                                                              \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-dger-blas\n" );           \
      Kokkos::Profiling::pushRegion("KokkosBlas::ger[TPL_BLAS,double]");            \
      KOKKOSBLAS2_GER_DETERMINE_ARGS(LAYOUTA);                                      \
      if (A_is_lr) {                                                                \
        HostBlas<SCALAR>::ger( M                                                    \
                             , N                                                    \
                             , alpha                                                \
                             , X.data()                                             \
                             , one                                                  \
                             , Y.data()                                             \
                             , one                                                  \
                             , A.data()                                             \
                             , LDA                                                  \
                             );                                                     \
      }                                                                             \
      else {                                                                        \
        HostBlas<SCALAR>::ger( M                                                    \
                             , N                                                    \
                             , alpha                                                \
                             , Y.data()                                             \
                             , one                                                  \
                             , X.data()                                             \
                             , one                                                  \
                             , A.data()                                             \
                             , LDA                                                  \
                             );                                                     \
      }                                                                             \
      Kokkos::Profiling::popRegion();                                               \
    }                                                                               \
  };

#define KOKKOSBLAS2_SGER_BLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL) \
  template <class ExecSpace>                                                        \
  struct GER< Kokkos::View< const float*                                            \
                          , LAYOUTX                                                 \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                    \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                 \
                          >                                                         \
            , Kokkos::View< const float*                                            \
                          , LAYOUTY                                                 \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                    \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                 \
                          >                                                         \
            , Kokkos::View< float**                                                 \
                          , LAYOUTA                                                 \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                    \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                 \
                          >                                                         \
            , true                                                                  \
            , ETI_SPEC_AVAIL                                                        \
            > {                                                                     \
    typedef float SCALAR;                                                           \
    typedef Kokkos::View< const SCALAR*                                             \
                        , LAYOUTX                                                   \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                      \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                   \
                        > XViewType;                                                \
    typedef Kokkos::View< const SCALAR*                                             \
                        , LAYOUTY                                                   \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                      \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                   \
                        > YViewType;                                                \
    typedef Kokkos::View< SCALAR**                                                  \
                        , LAYOUTA                                                   \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                      \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                   \
                        > AViewType;                                                \
                                                                                    \
    static void ger( const typename AViewType::execution_space  & /* space */       \
                   , const          char                          trans[]           \
                   , typename       AViewType::const_value_type & alpha             \
                   , const          XViewType                   & X                 \
                   , const          YViewType                   & Y                 \
                   , const          AViewType                   & A                 \
                   ) {                                                              \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-sger-blas\n" );           \
      Kokkos::Profiling::pushRegion("KokkosBlas::ger[TPL_BLAS,float]");             \
      KOKKOSBLAS2_GER_DETERMINE_ARGS(LAYOUTA);                                      \
      if (A_is_lr) {                                                                \
        HostBlas<SCALAR>::ger( M                                                    \
                             , N                                                    \
                             , alpha                                                \
                             , X.data()                                             \
                             , one                                                  \
                             , Y.data()                                             \
                             , one                                                  \
                             , A.data()                                             \
                             , LDA                                                  \
                             );                                                     \
      }                                                                             \
      else {                                                                        \
        HostBlas<SCALAR>::ger( M                                                    \
                             , N                                                    \
                             , alpha                                                \
                             , Y.data()                                             \
                             , one                                                  \
                             , X.data()                                             \
                             , one                                                  \
                             , A.data()                                             \
                             , LDA                                                  \
                             );                                                     \
      }                                                                             \
      Kokkos::Profiling::popRegion();                                               \
    }                                                                               \
  };

#define KOKKOSBLAS2_ZGER_BLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL)                      \
  template <class ExecSpace>                                                                             \
  struct GER< Kokkos::View< const Kokkos::complex<double>*                                               \
                          , LAYOUTX                                                                      \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                         \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                      \
                          >                                                                              \
            , Kokkos::View< const Kokkos::complex<double>*                                               \
                          , LAYOUTY                                                                      \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                         \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                      \
                          >                                                                              \
            , Kokkos::View< Kokkos::complex<double>**                                                    \
                          , LAYOUTA                                                                      \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                         \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                      \
                          >                                                                              \
            , true                                                                                       \
            , ETI_SPEC_AVAIL                                                                             \
            > {                                                                                          \
    typedef Kokkos::complex<double> SCALAR;                                                              \
    typedef Kokkos::View< const SCALAR*                                                                  \
                        , LAYOUTX                                                                        \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                                           \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                        \
                        > XViewType;                                                                     \
    typedef Kokkos::View< const SCALAR*                                                                  \
                        , LAYOUTY                                                                        \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                                           \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                        \
                        > YViewType;                                                                     \
    typedef Kokkos::View< SCALAR**                                                                       \
                        , LAYOUTA                                                                        \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                                           \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                        \
                        > AViewType;                                                                     \
                                                                                                         \
    static void ger( const typename AViewType::execution_space  & /* space */                            \
                   , const          char                          trans[]                                \
                   , typename       AViewType::const_value_type & alpha                                  \
                   , const          XViewType                   & X                                      \
                   , const          YViewType                   & Y                                      \
                   , const          AViewType                   & A                                      \
                   ) {                                                                                   \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-zger-blas\n" );                                \
      Kokkos::Profiling::pushRegion("KokkosBlas::ger[TPL_BLAS,complex<double>");                         \
      KOKKOSBLAS2_GER_DETERMINE_ARGS(LAYOUTA);                                                           \
      const std::complex<double> alpha_val = static_cast<const std::complex<double>>(alpha);             \
      bool justTranspose = (trans[0] == 'T') || (trans[0] == 't');                                       \
      if (A_is_lr) {                                                                                     \
        if (justTranspose) {                                                                             \
          HostBlas<std::complex<double>>::geru( M                                                        \
                                              , N                                                        \
                                              , alpha_val                                                \
                                              , reinterpret_cast<const std::complex<double>*>(X.data())  \
                                              , one                                                      \
                                              , reinterpret_cast<const std::complex<double>*>(Y.data())  \
                                              , one                                                      \
                                              , reinterpret_cast<std::complex<double>*>(A.data())        \
                                              , LDA                                                      \
                                              );                                                         \
        }                                                                                                \
        else {                                                                                           \
          HostBlas<std::complex<double>>::gerc( M                                                        \
                                              , N                                                        \
                                              , alpha_val                                                \
                                              , reinterpret_cast<const std::complex<double>*>(X.data())  \
                                              , one                                                      \
                                              , reinterpret_cast<const std::complex<double>*>(Y.data())  \
                                              , one                                                      \
                                              , reinterpret_cast<std::complex<double>*>(A.data())        \
                                              , LDA                                                      \
                                              );                                                         \
        }                                                                                                \
      }                                                                                                  \
      else {                                                                                             \
        if (justTranspose) {                                                                             \
          HostBlas<std::complex<double>>::geru( M                                                        \
                                              , N                                                        \
                                              , alpha_val                                                \
                                              , reinterpret_cast<const std::complex<double>*>(Y.data())  \
                                              , one                                                      \
                                              , reinterpret_cast<const std::complex<double>*>(X.data())  \
                                              , one                                                      \
                                              , reinterpret_cast<std::complex<double>*>(A.data())        \
                                              , LDA                                                      \
                                              );                                                         \
        }                                                                                                \
        else {                                                                                           \
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("blasZgerc() requires LayoutRight: throwing exception\n");       \
          throw std::runtime_error("Error: blasZgerc() requires LayoutRight views.");                    \
        }                                                                                                \
      }                                                                                                  \
      Kokkos::Profiling::popRegion();                                                                    \
    }                                                                                                    \
  };

#define KOKKOSBLAS2_CGER_BLAS(LAYOUTX, LAYOUTY, LAYOUTA, MEM_SPACE, ETI_SPEC_AVAIL)                   \
  template <class ExecSpace>                                                                          \
  struct GER< Kokkos::View< const Kokkos::complex<float>*                                             \
                          , LAYOUTX                                                                   \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                      \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                   \
                          >                                                                           \
            , Kokkos::View< const Kokkos::complex<float>*                                             \
                          , LAYOUTY                                                                   \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                      \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                   \
                          >                                                                           \
            , Kokkos::View< Kokkos::complex<float>**                                                  \
                          , LAYOUTA                                                                   \
                          , Kokkos::Device<ExecSpace, MEM_SPACE>                                      \
                          , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                   \
                          >                                                                           \
            , true                                                                                    \
            , ETI_SPEC_AVAIL                                                                          \
            > {                                                                                       \
    typedef Kokkos::complex<float> SCALAR;                                                            \
    typedef Kokkos::View< const SCALAR*                                                               \
                        , LAYOUTX                                                                     \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                                        \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                     \
                        > XViewType;                                                                  \
    typedef Kokkos::View< const SCALAR*                                                               \
                        , LAYOUTY                                                                     \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                                        \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                     \
                        > YViewType;                                                                  \
    typedef Kokkos::View< SCALAR**                                                                    \
                        , LAYOUTA                                                                     \
                        , Kokkos::Device<ExecSpace, MEM_SPACE>                                        \
                        , Kokkos::MemoryTraits<Kokkos::Unmanaged>                                     \
                        > AViewType;                                                                  \
                                                                                                      \
    static void ger( const typename AViewType::execution_space  & /* space */                         \
                   , const          char                          trans[]                             \
                   , typename       AViewType::const_value_type & alpha                               \
                   , const          XViewType                   & X                                   \
                   , const          YViewType                   & Y                                   \
                   , const          AViewType                   & A                                   \
                   ) {                                                                                \
      KOKKOS_IMPL_DO_NOT_USE_PRINTF( "Passing throuhg tpl-cger-blas\n" );                             \
      Kokkos::Profiling::pushRegion("KokkosBlas::ger[TPL_BLAS,complex<float>");                       \
      KOKKOSBLAS2_GER_DETERMINE_ARGS(LAYOUTA);                                                        \
      const std::complex<float> alpha_val = static_cast<const std::complex<float>>(alpha);            \
      bool justTranspose = (trans[0] == 'T') || (trans[0] == 't');                                    \
      if (A_is_lr) {                                                                                  \
        if (justTranspose) {                                                                          \
          HostBlas<std::complex<float>>::geru( M                                                      \
                                             , N                                                      \
                                             , alpha_val                                              \
                                             , reinterpret_cast<const std::complex<float>*>(X.data()) \
                                             , one                                                    \
                                             , reinterpret_cast<const std::complex<float>*>(Y.data()) \
                                             , one                                                    \
                                             , reinterpret_cast<std::complex<float>*>(A.data())       \
                                             , LDA                                                    \
                                             );                                                       \
        }                                                                                             \
        else {                                                                                        \
          HostBlas<std::complex<float>>::gerc( M                                                      \
                                             , N                                                      \
                                             , alpha_val                                              \
                                             , reinterpret_cast<const std::complex<float>*>(X.data()) \
                                             , one                                                    \
                                             , reinterpret_cast<const std::complex<float>*>(Y.data()) \
                                             , one                                                    \
                                             , reinterpret_cast<std::complex<float>*>(A.data())       \
                                             , LDA                                                    \
                                             );                                                       \
        }                                                                                             \
      }                                                                                               \
      else {                                                                                          \
        if (justTranspose) {                                                                          \
          HostBlas<std::complex<float>>::geru( M                                                      \
                                             , N                                                      \
                                             , alpha_val                                              \
                                             , reinterpret_cast<const std::complex<float>*>(Y.data()) \
                                             , one                                                    \
                                             , reinterpret_cast<const std::complex<float>*>(X.data()) \
                                             , one                                                    \
                                             , reinterpret_cast<std::complex<float>*>(A.data())       \
                                             , LDA                                                    \
                                             );                                                       \
        }                                                                                             \
        else {                                                                                        \
          KOKKOS_IMPL_DO_NOT_USE_PRINTF("blasCgerc() requires LayoutRight: throwing exception\n");    \
          throw std::runtime_error("Error: blasCgerc() requires LayoutRight views.");                 \
        }                                                                                             \
      }                                                                                               \
      Kokkos::Profiling::popRegion();                                                                 \
    }                                                                                                 \
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

#endif
