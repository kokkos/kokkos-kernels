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

/// \file
/// \brief Deprecated interfaces for the Kokkos sparse-matrix-vector multiply
///

#ifndef KOKKOSSPARSE_SPMV_DEPRECATED_HPP_
#define KOKKOSSPARSE_SPMV_DEPRECATED_HPP_

#include "KokkosKernels_Controls.hpp"

namespace KokkosSparse {
namespace Impl {
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector, class XLayout = typename XVector::array_layout>
struct SPMV2D1D {
  static bool spmv2d1d(const char mode[], const AlphaType& alpha,
                       const AMatrix& A, const XVector& x, const BetaType& beta,
                       const YVector& y);

  template <typename ExecutionSpace>
  static bool spmv2d1d(const ExecutionSpace& space, const char mode[],
                       const AlphaType& alpha, const AMatrix& A,
                       const XVector& x, const BetaType& beta,
                       const YVector& y);
};

#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || !defined(KOKKOSKERNELS_ETI_ONLY)
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D<AlphaType, AMatrix, XVector, BetaType, YVector,
                Kokkos::LayoutStride> {
  static bool spmv2d1d(const char mode[], const AlphaType& alpha,
                       const AMatrix& A, const XVector& x, const BetaType& beta,
                       const YVector& y) {
    spmv(typename AMatrix::execution_space{}, mode, alpha, A, x, beta, y);
    return true;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d(const ExecutionSpace& space, const char mode[],
                       const AlphaType& alpha, const AMatrix& A,
                       const XVector& x, const BetaType& beta,
                       const YVector& y) {
    spmv(space, mode, alpha, A, x, beta, y);
    return true;
  }
};

#else

template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D<AlphaType, AMatrix, XVector, BetaType, YVector,
                Kokkos::LayoutStride> {
  static bool spmv2d1d(const char /*mode*/[], const AlphaType& /*alpha*/,
                       const AMatrix& /*A*/, const XVector& /*x*/,
                       const BetaType& /*beta*/, const YVector& /*y*/) {
    return false;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d(const ExecutionSpace& /* space */, const char /*mode*/[],
                       const AlphaType& /*alpha*/, const AMatrix& /*A*/,
                       const XVector& /*x*/, const BetaType& /*beta*/,
                       const YVector& /*y*/) {
    return false;
  }
};
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || !defined(KOKKOSKERNELS_ETI_ONLY)
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D<AlphaType, AMatrix, XVector, BetaType, YVector,
                Kokkos::LayoutLeft> {
  static bool spmv2d1d(const char mode[], const AlphaType& alpha,
                       const AMatrix& A, const XVector& x, const BetaType& beta,
                       const YVector& y) {
    spmv(typename AMatrix::execution_space{}, mode, alpha, A, x, beta, y);
    return true;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d(const ExecutionSpace& space, const char mode[],
                       const AlphaType& alpha, const AMatrix& A,
                       const XVector& x, const BetaType& beta,
                       const YVector& y) {
    spmv(space, mode, alpha, A, x, beta, y);
    return true;
  }
};

#else

template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D<AlphaType, AMatrix, XVector, BetaType, YVector,
                Kokkos::LayoutLeft> {
  static bool spmv2d1d(const char /*mode*/[], const AlphaType& /*alpha*/,
                       const AMatrix& /*A*/, const XVector& /*x*/,
                       const BetaType& /*beta*/, const YVector& /*y*/) {
    return false;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d(const ExecutionSpace& /* space */, const char /*mode*/[],
                       const AlphaType& /*alpha*/, const AMatrix& /*A*/,
                       const XVector& /*x*/, const BetaType& /*beta*/,
                       const YVector& /*y*/) {
    return false;
  }
};
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || !defined(KOKKOSKERNELS_ETI_ONLY)
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D<AlphaType, AMatrix, XVector, BetaType, YVector,
                Kokkos::LayoutRight> {
  static bool spmv2d1d(const char mode[], const AlphaType& alpha,
                       const AMatrix& A, const XVector& x, const BetaType& beta,
                       const YVector& y) {
    spmv(typename AMatrix::execution_space{}, mode, alpha, A, x, beta, y);
    return true;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d(const ExecutionSpace& space, const char mode[],
                       const AlphaType& alpha, const AMatrix& A,
                       const XVector& x, const BetaType& beta,
                       const YVector& y) {
    spmv(space, mode, alpha, A, x, beta, y);
    return true;
  }
};

#else

template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D<AlphaType, AMatrix, XVector, BetaType, YVector,
                Kokkos::LayoutRight> {
  static bool spmv2d1d(const char /*mode*/[], const AlphaType& /*alpha*/,
                       const AMatrix& /*A*/, const XVector& /*x*/,
                       const BetaType& /*beta*/, const YVector& /*y*/) {
    return false;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d(const ExecutionSpace& /* space */, const char /*mode*/[],
                       const AlphaType& /*alpha*/, const AMatrix& /*A*/,
                       const XVector& /*x*/, const BetaType& /*beta*/,
                       const YVector& /*y*/) {
    return false;
  }
};
#endif
}  // namespace Impl

template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector, class XLayout = typename XVector::array_layout>
using SPMV2D1D
    [[deprecated("KokkosSparse::SPMV2D1D is not part of the public interface - "
                 "use KokkosSparse::spmv instead")]] =
        Impl::SPMV2D1D<AlphaType, AMatrix, XVector, BetaType, YVector>;

template <class ExecutionSpace, class AlphaType, class AMatrix, class XVector,
          class BetaType, class YVector>
[
    [deprecated("Use the version of spmv that takes a SPMVHandle instead of "
                "Controls")]] void
spmv(const ExecutionSpace& space,
     KokkosKernels::Experimental::Controls controls, const char mode[],
     const AlphaType& alpha, const AMatrix& A, const XVector& x,
     const BetaType& beta, const YVector& y) {
  // Translate the algorithm choice in controls to a SPMVHandle algo enum.
  // Also since this interface does not allow reuse, use native instead of
  // rocSPARSE
  SPMVAlgorithm algo = SPMV_FAST_SETUP;
#ifdef KOKKOSKERNELS_ENABLE_TPL_ROCSPARSE
  if constexpr (std::is_same_v<typename AMatrix::execution_space, Kokkos::HIP>)
    algo = SPMV_NATIVE;
#endif
  if (controls.isParameter("algorithm")) {
    if (controls.getParameter("algorithm") != "tpl") algo = SPMV_NATIVE;
  }
  KokkosSparse::SPMVHandle<ExecutionSpace, AMatrix, XVector, YVector> handle(
      algo);
  spmv(space, &handle, mode, alpha, A, x, beta, y);
}

template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
[
    [deprecated("Use the version of spmv that takes a SPMVHandle instead of "
                "Controls")]] void
spmv(KokkosKernels::Experimental::Controls controls, const char mode[],
     const AlphaType& alpha, const AMatrix& A, const XVector& x,
     const BetaType& beta, const YVector& y) {
  spmv(typename AMatrix::execution_space{}, controls, mode, alpha, A, x, beta,
       y);
}

template <class ExecutionSpace, class AlphaType, class AMatrix, class XVector,
          class BetaType, class YVector>
[
    [deprecated("Use the version of spmv that takes a SPMVHandle instead of "
                "Controls")]] void
spmv(const ExecutionSpace& space,
     KokkosKernels::Experimental::Controls controls, const char mode[],
     const AlphaType& alpha, const AMatrix& A, const XVector& x,
     const BetaType& beta, const YVector& y, const RANK_ONE&) {
  spmv(space, controls, mode, alpha, A, x, beta, y);
}

template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
[
    [deprecated("Use the version of spmv that takes a SPMVHandle instead of "
                "Controls")]] void
spmv(KokkosKernels::Experimental::Controls controls, const char mode[],
     const AlphaType& alpha, const AMatrix& A, const XVector& x,
     const BetaType& beta, const YVector& y, const RANK_ONE&) {
  spmv(controls, mode, alpha, A, x, beta, y);
}

template <class ExecutionSpace, class AlphaType, class AMatrix, class XVector,
          class BetaType, class YVector>
[
    [deprecated("Use the version of spmv that takes a SPMVHandle instead of "
                "Controls")]] void
spmv(const ExecutionSpace& space,
     KokkosKernels::Experimental::Controls controls, const char mode[],
     const AlphaType& alpha, const AMatrix& A, const XVector& x,
     const BetaType& beta, const YVector& y, const RANK_TWO&) {
  spmv(space, controls, mode, alpha, A, x, beta, y);
}

template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
[
    [deprecated("Use the version of spmv that takes a SPMVHandle instead of "
                "Controls")]] void
spmv(KokkosKernels::Experimental::Controls controls, const char mode[],
     const AlphaType& alpha, const AMatrix& A, const XVector& x,
     const BetaType& beta, const YVector& y, const RANK_TWO&) {
  spmv(controls, mode, alpha, A, x, beta, y);
}

}  // namespace KokkosSparse

#endif
