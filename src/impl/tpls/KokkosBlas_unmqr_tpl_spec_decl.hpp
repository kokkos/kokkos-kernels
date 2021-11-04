#ifndef KOKKOSBLAS_UNMQR_TPL_SPEC_DECL_HPP_
#define KOKKOSBLAS_UNMQR_TPL_SPEC_DECL_HPP_

#if defined(KOKKOSKERNELS_ENABLE_TPL_BLAS) && \
    defined(KOKKOSKERNELS_ENABLE_TPL_LAPACKE)
#include "KokkosBlas_Host_tpl.hpp"
#include "KokkosLapack_Host_tpl.hpp"
#include <stdio.h>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS_DUNMQR_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE,         \
                                 ETI_SPEC_AVAIL)                              \
  template <class ExecSpace>                                                  \
  struct UNMQR<                                                               \
      Kokkos::View<const double**, LAYOUTA,                                   \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<const double*, LAYOUTB,                                    \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<double**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>,    \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      Kokkos::View<double*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                 \
      true, ETI_SPEC_AVAIL> {                                                 \
    typedef double SCALAR;                                                    \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                             \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        CViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTB,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WViewType;                                                            \
                                                                              \
    static void unmqr(char side, char trans, int k, AViewType& A,             \
                      TauViewType& tau, CViewType& C, WViewType& workspace) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosLapack::unmqr[TPL_LAPACK, double]");                         \
      int M           = C.extent(0);                                          \
      int N           = C.extent(1);                                          \
      bool A_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;    \
      bool C_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;    \
      const int AST   = A_is_lr ? A.stride(0) : A.stride(1),                  \
                LDA   = AST == 0 ? 1 : AST;                                   \
      const int CST   = C_is_lr ? C.stride(0) : C.stride(1),                  \
                LDC   = CST == 0 ? 1 : CST;                                   \
      const int lwork = workspace.extent(0);                                  \
      HostLapack<double>::unmqr(A_is_lr, side, trans, M, N, k, A.data(), LDA, \
                                tau.data(), C.data(), LDC, workspace.data(),  \
                                lwork);                                       \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_SUNMQR_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE,          \
                                 ETI_SPEC_AVAIL)                               \
  template <class ExecSpace>                                                   \
  struct UNMQR<                                                                \
      Kokkos::View<const float**, LAYOUTA,                                     \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const float*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef float SCALAR;                                                      \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTB,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        WViewType;                                                             \
                                                                               \
    static void unmqr(char side, char trans, int k, AViewType& A,              \
                      TauViewType& tau, CViewType& C, WViewType& workspace) {  \
      Kokkos::Profiling::pushRegion("KokkosLapack::unmqr[TPL_LAPACK, float]"); \
      int M           = C.extent(0);                                           \
      int N           = C.extent(1);                                           \
      bool A_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;     \
      bool C_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;     \
      const int AST   = A_is_lr ? A.stride(0) : A.stride(1),                   \
                LDA   = AST == 0 ? 1 : AST;                                    \
      const int CST   = C_is_lr ? C.stride(0) : C.stride(1),                   \
                LDC   = CST == 0 ? 1 : CST;                                    \
      const int lwork = workspace.extent(0);                                   \
      char ctrans     = (side == 'T' || side == 't') ? 'C' : side;             \
      HostLapack<float>::unmqr(A_is_lr, side, trans, M, N, k, A.data(), LDA,   \
                               tau.data(), C.data(), LDC, workspace.data(),    \
                               lwork);                                         \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS_ZUNMQR_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE,         \
                                 ETI_SPEC_AVAIL)                              \
  template <class ExecSpace>                                                  \
  struct UNMQR<Kokkos::View<const Kokkos::complex<double>**, LAYOUTA,         \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<const Kokkos::complex<double>*, LAYOUTB,          \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<double>**, LAYOUTC,               \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<double>*, LAYOUTB,                \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<double> SCALAR;                                   \
    typedef std::complex<double> S2;                                          \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                             \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        CViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTB,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WViewType;                                                            \
    static void unmqr(char side, char trans, int k, AViewType& A,             \
                      TauViewType& tau, CViewType& C, WViewType& workspace) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosLapack::unmqr[TPL_LAPACK, complex<double>]");                \
      int M           = C.extent(0);                                          \
      int N           = C.extent(1);                                          \
      bool A_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;    \
      bool C_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;    \
      const int AST   = A_is_lr ? A.stride(0) : A.stride(1),                  \
                LDA   = AST == 0 ? 1 : AST;                                   \
      const int CST   = C_is_lr ? C.stride(0) : C.stride(1),                  \
                LDC   = CST == 0 ? 1 : CST;                                   \
      const int lwork = workspace.extent(0);                                  \
      char ctrans     = (trans == 'T' || trans == 't') ? 'C' : trans;         \
      HostLapack<S2>::unmqr(A_is_lr, side, ctrans, M, N, k,                   \
                            reinterpret_cast<const S2*>(A.data()), LDA,       \
                            reinterpret_cast<const S2*>(tau.data()),          \
                            reinterpret_cast<S2*>(C.data()), LDC,             \
                            reinterpret_cast<S2*>(workspace.data()), lwork);  \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

#define KOKKOSBLAS_CUNMQR_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE,         \
                                 ETI_SPEC_AVAIL)                              \
  template <class ExecSpace>                                                  \
  struct UNMQR<Kokkos::View<const Kokkos::complex<float>**, LAYOUTA,          \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<const Kokkos::complex<float>*, LAYOUTB,           \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<float>**, LAYOUTC,                \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               Kokkos::View<Kokkos::complex<float>*, LAYOUTB,                 \
                            Kokkos::Device<ExecSpace, MEMSPACE>,              \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,        \
               true, ETI_SPEC_AVAIL> {                                        \
    typedef Kokkos::complex<float> SCALAR;                                    \
    typedef std::complex<float> S2;                                           \
    typedef int ORDINAL;                                                      \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                             \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        AViewType;                                                            \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        TauViewType;                                                          \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                   \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        CViewType;                                                            \
    typedef Kokkos::View<SCALAR*, LAYOUTB,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                 \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >            \
        WViewType;                                                            \
                                                                              \
    static void unmqr(char side, char trans, int k, AViewType& A,             \
                      TauViewType& tau, CViewType& C, WViewType& workspace) { \
      Kokkos::Profiling::pushRegion(                                          \
          "KokkosLapack::unmqr[TPL_LAPACK, complex<float>]");                 \
      int M           = C.extent(0);                                          \
      int N           = C.extent(1);                                          \
      bool A_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;    \
      bool C_is_lr    = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;    \
      const int AST   = A_is_lr ? A.stride(0) : A.stride(1),                  \
                LDA   = AST == 0 ? 1 : AST;                                   \
      const int CST   = C_is_lr ? C.stride(0) : C.stride(1),                  \
                LDC   = CST == 0 ? 1 : CST;                                   \
      const int lwork = workspace.extent(0);                                  \
      char ctrans     = (trans == 'T' || trans == 't') ? 'C' : trans;         \
      HostLapack<S2>::unmqr(A_is_lr, side, ctrans, M, N, k,                   \
                            reinterpret_cast<const S2*>(A.data()), LDA,       \
                            reinterpret_cast<const S2*>(tau.data()),          \
                            reinterpret_cast<S2*>(C.data()), LDC,             \
                            reinterpret_cast<S2*>(workspace.data()), lwork);  \
      Kokkos::Profiling::popRegion();                                         \
    }                                                                         \
  };

// WORKSPACE QUERIES

#define KOKKOSBLAS_DUNMQR_WORKSPACE_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC,          \
                                           MEMSPACE, ETI_SPEC_AVAIL)           \
  template <class ExecSpace>                                                   \
  struct UNMQR_WORKSPACE<                                                      \
      Kokkos::View<const double**, LAYOUTA,                                    \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const double*, LAYOUTB,                                     \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<double**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef double SCALAR;                                                     \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
                                                                               \
    static int64_t unmqr_workspace(char side, char trans, int k, AViewType& A, \
                                   TauViewType& tau, CViewType& C) {           \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosLapack::unmqr[TPL_LAPACK, double]");                          \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      int lwork     = -1;                                                      \
      SCALAR query  = 0;                                                       \
      HostLapack<SCALAR>::unmqr(A_is_lr, side, trans, M, N, k, A.data(), LDA,  \
                                tau.data(), C.data(), LDC, &query, lwork);     \
      Kokkos::Profiling::popRegion();                                          \
      return (int64_t)query;                                                   \
    }                                                                          \
  };

#define KOKKOSBLAS_SUNMQR_WORKSPACE_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC,          \
                                           MEMSPACE, ETI_SPEC_AVAIL)           \
  template <class ExecSpace>                                                   \
  struct UNMQR_WORKSPACE<                                                      \
      Kokkos::View<const float**, LAYOUTA,                                     \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const float*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef float SCALAR;                                                      \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
                                                                               \
    static int64_t unmqr_workspace(char side, char trans, int k, AViewType& A, \
                                   TauViewType& tau, CViewType& C) {           \
      Kokkos::Profiling::pushRegion("KokkosLapack::unmqr[TPL_LAPACK, float]"); \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      int lwork     = -1;                                                      \
      SCALAR query  = 0;                                                       \
      HostLapack<SCALAR>::unmqr(A_is_lr, side, trans, M, N, k, A.data(), LDA,  \
                                tau.data(), C.data(), LDC, &query, lwork);     \
      Kokkos::Profiling::popRegion();                                          \
      return (int64_t)query;                                                   \
    }                                                                          \
  };

#define KOKKOSBLAS_ZUNMQR_WORKSPACE_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC,          \
                                           MEMSPACE, ETI_SPEC_AVAIL)           \
  template <class ExecSpace>                                                   \
  struct UNMQR_WORKSPACE<                                                      \
      Kokkos::View<const Kokkos::complex<double>**, LAYOUTA,                   \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const Kokkos::complex<double>*, LAYOUTB,                    \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<Kokkos::complex<double>**, LAYOUTC,                         \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef Kokkos::complex<double> SCALAR;                                    \
    typedef std::complex<double> S2;                                           \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTB,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        WViewType;                                                             \
    static int64_t unmqr_workspace(char side, char trans, int k, AViewType& A, \
                                   TauViewType& tau, CViewType& C) {           \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosLapack::unmqr[TPL_LAPACK, complex<double>]");                 \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      int lwork     = -1;                                                      \
      SCALAR query  = 0;                                                       \
      char ctrans   = (trans == 'T' || trans == 't') ? 'C' : trans;            \
      HostLapack<S2>::unmqr(A_is_lr, side, ctrans, M, N, k,                    \
                            reinterpret_cast<const S2*>(A.data()), LDA,        \
                            reinterpret_cast<const S2*>(tau.data()),           \
                            reinterpret_cast<S2*>(C.data()), LDC,              \
                            reinterpret_cast<S2*>(&query), lwork);             \
      Kokkos::Profiling::popRegion();                                          \
      return (int64_t)query.real();                                            \
    }                                                                          \
  };

#define KOKKOSBLAS_CUNMQR_WORKSPACE_LAPACK(LAYOUTA, LAYOUTB, LAYOUTC,          \
                                           MEMSPACE, ETI_SPEC_AVAIL)           \
  template <class ExecSpace>                                                   \
  struct UNMQR_WORKSPACE<                                                      \
      Kokkos::View<const Kokkos::complex<float>**, LAYOUTA,                    \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const Kokkos::complex<float>*, LAYOUTB,                     \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<Kokkos::complex<float>**, LAYOUTC,                          \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef Kokkos::complex<float> SCALAR;                                     \
    typedef std::complex<float> S2;                                            \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
                                                                               \
    static int64_t unmqr_workspace(char side, char trans, int k, AViewType& A, \
                                   TauViewType& tau, CViewType& C) {           \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosLapack::unmqr[TPL_LAPACK, complex<float>]");                  \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      int lwork     = -1;                                                      \
      SCALAR query  = 0;                                                       \
      char ctrans   = (trans == 'T' || trans == 't') ? 'C' : trans;            \
      HostLapack<S2>::unmqr(A_is_lr, side, ctrans, M, N, k,                    \
                            reinterpret_cast<const S2*>(A.data()), LDA,        \
                            reinterpret_cast<const S2*>(tau.data()),           \
                            reinterpret_cast<S2*>(C.data()), LDC,              \
                            reinterpret_cast<S2*>(&query), lwork);             \
      Kokkos::Profiling::popRegion();                                          \
      return (int64_t)query.real();                                            \
    }                                                                          \
  };

KOKKOSBLAS_DUNMQR_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_DUNMQR_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_DUNMQR_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_DUNMQR_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS_SUNMQR_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_SUNMQR_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_SUNMQR_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_SUNMQR_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS_ZUNMQR_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_ZUNMQR_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_ZUNMQR_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_ZUNMQR_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS_CUNMQR_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_CUNMQR_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                         Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_CUNMQR_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_CUNMQR_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                         Kokkos::LayoutRight, Kokkos::HostSpace, false)

KOKKOSBLAS_DUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                   Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_DUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                   Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_DUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                                   Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_DUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                                   Kokkos::LayoutRight, Kokkos::HostSpace,
                                   false)

KOKKOSBLAS_SUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                   Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_SUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                   Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_SUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                                   Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_SUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                                   Kokkos::LayoutRight, Kokkos::HostSpace,
                                   false)

KOKKOSBLAS_ZUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                   Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_ZUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                   Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_ZUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                                   Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_ZUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                                   Kokkos::LayoutRight, Kokkos::HostSpace,
                                   false)

KOKKOSBLAS_CUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                   Kokkos::LayoutLeft, Kokkos::HostSpace, true)
KOKKOSBLAS_CUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                   Kokkos::LayoutLeft, Kokkos::HostSpace, false)
KOKKOSBLAS_CUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                                   Kokkos::LayoutRight, Kokkos::HostSpace, true)
KOKKOSBLAS_CUNMQR_WORKSPACE_LAPACK(Kokkos::LayoutRight, Kokkos::LayoutRight,
                                   Kokkos::LayoutRight, Kokkos::HostSpace,
                                   false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // ENABLE BLAS/LAPACK

// CUSOLVER

#ifdef KOKKOSKERNELS_ENABLE_TPL_CUSOLVER
#include <KokkosBlas_tpl_spec.hpp>

namespace KokkosBlas {
namespace Impl {

#define KOKKOSBLAS_DUNMQR_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE,        \
                                   ETI_SPEC_AVAIL)                             \
  template <class ExecSpace>                                                   \
  struct UNMQR<                                                                \
      Kokkos::View<const double**, LAYOUTA,                                    \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const double*, LAYOUTB,                                     \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<double**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<double*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef double SCALAR;                                                     \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTB,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        WViewType;                                                             \
                                                                               \
    static void unmqr(char side, char trans, int k, AViewType& A,              \
                      TauViewType& tau, CViewType& C, WViewType& workspace) {  \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::unmqr[TPL_CUSOLVER, double]");                          \
      int devinfo   = 0;                                                       \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      cublasSideMode_t m_side =                                                \
          (side == 'L' || side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT; \
      cublasOperation_t m_trans =                                              \
          (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c')       \
              ? CUBLAS_OP_T                                                    \
              : CUBLAS_OP_N;                                                   \
      const int lwork = workspace.extent(0);                                   \
      KokkosBlas::Impl::CudaSolverSingleton& s =                               \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                  \
      cusolverDnDormqr(s.handle, m_side, m_trans, M, N, k, A.data(), LDA,      \
                       tau.data(), C.data(), LDC, workspace.data(), lwork,     \
                       &devinfo);                                              \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS_SUNMQR_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE,        \
                                   ETI_SPEC_AVAIL)                             \
  template <class ExecSpace>                                                   \
  struct UNMQR<                                                                \
      Kokkos::View<const float**, LAYOUTA,                                     \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const float*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>,       \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef float SCALAR;                                                      \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTB,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        WViewType;                                                             \
                                                                               \
    static void unmqr(char side, char trans, int k, AViewType& A,              \
                      TauViewType& tau, CViewType& C, WViewType& workspace) {  \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::unmqr[TPL_CUSOLVER, single]");                          \
      int devinfo   = 0;                                                       \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      cublasSideMode_t m_side =                                                \
          (side == 'L' || side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT; \
      cublasOperation_t m_trans =                                              \
          (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c')       \
              ? CUBLAS_OP_T                                                    \
              : CUBLAS_OP_N;                                                   \
      const int lwork = workspace.extent(0);                                   \
      KokkosBlas::Impl::CudaSolverSingleton& s =                               \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                  \
      cusolverDnSormqr(s.handle, m_side, m_trans, M, N, k, A.data(), LDA,      \
                       tau.data(), C.data(), LDC, workspace.data(), lwork,     \
                       &devinfo);                                              \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS_ZUNMQR_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE,        \
                                   ETI_SPEC_AVAIL)                             \
  template <class ExecSpace>                                                   \
  struct UNMQR<Kokkos::View<const Kokkos::complex<double>**, LAYOUTA,          \
                            Kokkos::Device<ExecSpace, MEMSPACE>,               \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
               Kokkos::View<const Kokkos::complex<double>*, LAYOUTB,           \
                            Kokkos::Device<ExecSpace, MEMSPACE>,               \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
               Kokkos::View<Kokkos::complex<double>**, LAYOUTC,                \
                            Kokkos::Device<ExecSpace, MEMSPACE>,               \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
               Kokkos::View<Kokkos::complex<double>*, LAYOUTB,                 \
                            Kokkos::Device<ExecSpace, MEMSPACE>,               \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
               true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<double> SCALAR;                                    \
    typedef double PRECISION;                                                  \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTB,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        WViewType;                                                             \
                                                                               \
    static void unmqr(char side, char trans, int k, AViewType& A,              \
                      TauViewType& tau, CViewType& C, WViewType& workspace) {  \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::unmqr[TPL_CUSOLVER, Kokkos::complex<double>]");         \
      int devinfo   = 0;                                                       \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      cublasSideMode_t m_side =                                                \
          (side == 'L' || side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT; \
      cublasOperation_t m_trans =                                              \
          (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c')       \
              ? CUBLAS_OP_C                                                    \
              : CUBLAS_OP_N;                                                   \
      const int lwork = workspace.extent(0);                                   \
      KokkosBlas::Impl::CudaSolverSingleton& s =                               \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                  \
      cusolverDnZunmqr(s.handle, m_side, m_trans, M, N, k,                     \
                       reinterpret_cast<const cuDoubleComplex*>(A.data()),     \
                       LDA,                                                    \
                       reinterpret_cast<const cuDoubleComplex*>(tau.data()),   \
                       reinterpret_cast<cuDoubleComplex*>(C.data()), LDC,      \
                       reinterpret_cast<cuDoubleComplex*>(workspace.data()),   \
                       lwork, &devinfo);                                       \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

#define KOKKOSBLAS_CUNMQR_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC, MEMSPACE,        \
                                   ETI_SPEC_AVAIL)                             \
  template <class ExecSpace>                                                   \
  struct UNMQR<Kokkos::View<const Kokkos::complex<float>**, LAYOUTA,           \
                            Kokkos::Device<ExecSpace, MEMSPACE>,               \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
               Kokkos::View<const Kokkos::complex<float>*, LAYOUTB,            \
                            Kokkos::Device<ExecSpace, MEMSPACE>,               \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
               Kokkos::View<Kokkos::complex<float>**, LAYOUTC,                 \
                            Kokkos::Device<ExecSpace, MEMSPACE>,               \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
               Kokkos::View<Kokkos::complex<float>*, LAYOUTB,                  \
                            Kokkos::Device<ExecSpace, MEMSPACE>,               \
                            Kokkos::MemoryTraits<Kokkos::Unmanaged> >,         \
               true, ETI_SPEC_AVAIL> {                                         \
    typedef Kokkos::complex<float> SCALAR;                                     \
    typedef float PRECISION;                                                   \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
    typedef Kokkos::View<SCALAR*, LAYOUTB,                                     \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        WViewType;                                                             \
                                                                               \
    static void unmqr(char side, char trans, int k, AViewType& A,              \
                      TauViewType& tau, CViewType& C, WViewType& workspace) {  \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::unmqr[TPL_CUSOLVER, Kokkos::complex<float>]");          \
      int devinfo   = 0;                                                       \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      cublasSideMode_t m_side =                                                \
          (side == 'L' || side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT; \
      cublasOperation_t m_trans =                                              \
          (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c')       \
              ? CUBLAS_OP_C                                                    \
              : CUBLAS_OP_N;                                                   \
      const int lwork = workspace.extent(0);                                   \
      KokkosBlas::Impl::CudaSolverSingleton& s =                               \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                  \
      cusolverDnCunmqr(s.handle, m_side, m_trans, M, N, k,                     \
                       reinterpret_cast<const cuComplex*>(A.data()), LDA,      \
                       reinterpret_cast<const cuComplex*>(tau.data()),         \
                       reinterpret_cast<cuComplex*>(C.data()), LDC,            \
                       reinterpret_cast<cuComplex*>(workspace.data()), lwork,  \
                       &devinfo);                                              \
      Kokkos::Profiling::popRegion();                                          \
    }                                                                          \
  };

// WORKSPACE QUERIES

#define KOKKOSBLAS_DUNMQR_WORKSPACE_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC,        \
                                             MEMSPACE, ETI_SPEC_AVAIL)         \
  template <class ExecSpace>                                                   \
  struct UNMQR_WORKSPACE<                                                      \
      Kokkos::View<const double**, LAYOUTA,                                    \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const double*, LAYOUTB,                                     \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<double**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>,     \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef double SCALAR;                                                     \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
                                                                               \
    static int64_t unmqr_workspace(char side, char trans, int k, AViewType& A, \
                                   TauViewType& tau, CViewType& C) {           \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::unmqr[TPL_CUSOLVER, double]");                          \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      cublasSideMode_t m_side =                                                \
          (side == 'L' || side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT; \
      cublasOperation_t m_trans =                                              \
          (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c')       \
              ? CUBLAS_OP_T                                                    \
              : CUBLAS_OP_N;                                                   \
      int lwork = 0;                                                           \
      KokkosBlas::Impl::CudaSolverSingleton& s =                               \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                  \
      cusolverDnDormqr_bufferSize(s.handle, m_side, m_trans, M, N, k,          \
                                  A.data(), LDA, tau.data(), C.data(), LDC,    \
                                  &lwork);                                     \
      Kokkos::Profiling::popRegion();                                          \
      return (int64_t)lwork;                                                   \
    }                                                                          \
  };

#define KOKKOSBLAS_SUNMQR_WORKSPACE_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC,        \
                                             MEMSPACE, ETI_SPEC_AVAIL)         \
  template <class ExecSpace>                                                   \
  struct UNMQR_WORKSPACE<                                                      \
      Kokkos::View<const float**, LAYOUTA,                                     \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const float*, LAYOUTB, Kokkos::Device<ExecSpace, MEMSPACE>, \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<float**, LAYOUTC, Kokkos::Device<ExecSpace, MEMSPACE>,      \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef float SCALAR;                                                      \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
                                                                               \
    static int64_t unmqr_workspace(char side, char trans, int k, AViewType& A, \
                                   TauViewType& tau, CViewType& C) {           \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::unmqr[TPL_CUSOLVER, single]");                          \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      cublasSideMode_t m_side =                                                \
          (side == 'L' || side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT; \
      cublasOperation_t m_trans =                                              \
          (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c')       \
              ? CUBLAS_OP_T                                                    \
              : CUBLAS_OP_N;                                                   \
      int lwork = 0;                                                           \
      KokkosBlas::Impl::CudaSolverSingleton& s =                               \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                  \
      cusolverDnSormqr_bufferSize(s.handle, m_side, m_trans, M, N, k,          \
                                  A.data(), LDA, tau.data(), C.data(), LDC,    \
                                  &lwork);                                     \
      Kokkos::Profiling::popRegion();                                          \
      return (int64_t)lwork;                                                   \
    }                                                                          \
  };

#define KOKKOSBLAS_ZUNMQR_WORKSPACE_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC,        \
                                             MEMSPACE, ETI_SPEC_AVAIL)         \
  template <class ExecSpace>                                                   \
  struct UNMQR_WORKSPACE<                                                      \
      Kokkos::View<const Kokkos::complex<double>**, LAYOUTA,                   \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const Kokkos::complex<double>*, LAYOUTB,                    \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<Kokkos::complex<double>**, LAYOUTC,                         \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef Kokkos::complex<double> SCALAR;                                    \
    typedef double PRECISION;                                                  \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
                                                                               \
    static int64_t unmqr_workspace(char side, char trans, int k, AViewType& A, \
                                   TauViewType& tau, CViewType& C) {           \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::unmqr[TPL_CUSOLVER, Kokkos::complex<double>]");         \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      cublasSideMode_t m_side =                                                \
          (side == 'L' || side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT; \
      cublasOperation_t m_trans =                                              \
          (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c')       \
              ? CUBLAS_OP_C                                                    \
              : CUBLAS_OP_N;                                                   \
      int lwork = 0;                                                           \
      KokkosBlas::Impl::CudaSolverSingleton& s =                               \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                  \
      cusolverDnZunmqr_bufferSize(                                             \
          s.handle, m_side, m_trans, M, N, k,                                  \
          reinterpret_cast<const cuDoubleComplex*>(A.data()), LDA,             \
          reinterpret_cast<const cuDoubleComplex*>(tau.data()),                \
          reinterpret_cast<cuDoubleComplex*>(C.data()), LDC, &lwork);          \
      Kokkos::Profiling::popRegion();                                          \
      return (int64_t)lwork;                                                   \
    }                                                                          \
  };

#define KOKKOSBLAS_CUNMQR_WORKSPACE_CUSOLVER(LAYOUTA, LAYOUTB, LAYOUTC,        \
                                             MEMSPACE, ETI_SPEC_AVAIL)         \
  template <class ExecSpace>                                                   \
  struct UNMQR_WORKSPACE<                                                      \
      Kokkos::View<const Kokkos::complex<float>**, LAYOUTA,                    \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<const Kokkos::complex<float>*, LAYOUTB,                     \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      Kokkos::View<Kokkos::complex<float>**, LAYOUTC,                          \
                   Kokkos::Device<ExecSpace, MEMSPACE>,                        \
                   Kokkos::MemoryTraits<Kokkos::Unmanaged> >,                  \
      true, ETI_SPEC_AVAIL> {                                                  \
    typedef Kokkos::complex<float> SCALAR;                                     \
    typedef float PRECISION;                                                   \
    typedef int ORDINAL;                                                       \
    typedef Kokkos::View<const SCALAR**, LAYOUTA,                              \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        AViewType;                                                             \
    typedef Kokkos::View<const SCALAR*, LAYOUTB,                               \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        TauViewType;                                                           \
    typedef Kokkos::View<SCALAR**, LAYOUTC,                                    \
                         Kokkos::Device<ExecSpace, MEMSPACE>,                  \
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >             \
        CViewType;                                                             \
                                                                               \
    static int64_t unmqr_workspace(char side, char trans, int k, AViewType& A, \
                                   TauViewType& tau, CViewType& C) {           \
      Kokkos::Profiling::pushRegion(                                           \
          "KokkosBlas::unmqr[TPL_CUSOLVER, Kokkos::complex<float>]");          \
      int M         = C.extent(0);                                             \
      int N         = C.extent(1);                                             \
      bool A_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTA>::value;       \
      bool C_is_lr  = std::is_same<Kokkos::LayoutRight, LAYOUTC>::value;       \
      const int AST = A_is_lr ? A.stride(0) : A.stride(1),                     \
                LDA = AST == 0 ? 1 : AST;                                      \
      const int CST = C_is_lr ? C.stride(0) : C.stride(1),                     \
                LDC = CST == 0 ? 1 : CST;                                      \
      cublasSideMode_t m_side =                                                \
          (side == 'L' || side == 'l') ? CUBLAS_SIDE_LEFT : CUBLAS_SIDE_RIGHT; \
      cublasOperation_t m_trans =                                              \
          (trans == 'T' || trans == 't' || trans == 'C' || trans == 'c')       \
              ? CUBLAS_OP_C                                                    \
              : CUBLAS_OP_N;                                                   \
      int lwork = 0;                                                           \
      KokkosBlas::Impl::CudaSolverSingleton& s =                               \
          KokkosBlas::Impl::CudaSolverSingleton::singleton();                  \
      cusolverDnCunmqr_bufferSize(                                             \
          s.handle, m_side, m_trans, M, N, k,                                  \
          reinterpret_cast<const cuComplex*>(A.data()), LDA,                   \
          reinterpret_cast<const cuComplex*>(tau.data()),                      \
          reinterpret_cast<cuComplex*>(C.data()), LDC, &lwork);                \
      Kokkos::Profiling::popRegion();                                          \
      return (int64_t)lwork;                                                   \
    }                                                                          \
  };

// CudaSpace
KOKKOSBLAS_DUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_DUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_SUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_SUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_ZUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_ZUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_CUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaSpace, true)
KOKKOSBLAS_CUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaSpace, false)

KOKKOSBLAS_DUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     true)
KOKKOSBLAS_DUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     false)

KOKKOSBLAS_SUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     true)
KOKKOSBLAS_SUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     false)

KOKKOSBLAS_ZUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     true)
KOKKOSBLAS_ZUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     false)

KOKKOSBLAS_CUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     true)
KOKKOSBLAS_CUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaSpace,
                                     false)

// CUDA UVM Space
KOKKOSBLAS_DUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS_DUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS_SUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS_SUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS_ZUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS_ZUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS_CUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, true)
KOKKOSBLAS_CUNMQR_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                           Kokkos::LayoutLeft, Kokkos::CudaUVMSpace, false)

KOKKOSBLAS_DUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     true)
KOKKOSBLAS_DUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft,
                                     Kokkos::CudCudaUVMSpaceaSpace, false)

KOKKOSBLAS_SUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     true)
KOKKOSBLAS_SUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     false)

KOKKOSBLAS_ZUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     true)
KOKKOSBLAS_ZUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     false)

KOKKOSBLAS_CUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     true)
KOKKOSBLAS_CUNMQR_WORKSPACE_CUSOLVER(Kokkos::LayoutLeft, Kokkos::LayoutLeft,
                                     Kokkos::LayoutLeft, Kokkos::CudaUVMSpace,
                                     false)

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // IF CUSOLVER && CUBLAS

#endif  // KOKKOSBLAS_UNMQR_TPL_SPEC_DECL_HPP_
