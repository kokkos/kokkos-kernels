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
/// \brief Interfaces for the Kokkos sparse-matrix-vector multiply
///

#ifndef KOKKOSSPARSE_SPMV_HPP_
#define KOKKOSSPARSE_SPMV_HPP_

#include "KokkosKernels_helpers.hpp"
#include "KokkosSparse_spmv_handle.hpp"
#include "KokkosSparse_spmv_spec.hpp"
#include "KokkosSparse_spmv_struct_spec.hpp"
#include "KokkosSparse_spmv_bsrmatrix_spec.hpp"
#include <type_traits>
#include "KokkosSparse_BsrMatrix.hpp"
#include "KokkosSparse_CrsMatrix.hpp"
#include "KokkosBlas1_scal.hpp"
#include "KokkosKernels_Utils.hpp"
#include "KokkosKernels_Error.hpp"

namespace KokkosSparse {

namespace {
struct RANK_ONE {};
struct RANK_TWO {};
}  // namespace



/// \brief Kokkos sparse matrix-vector multiply.
///   Computes y := alpha*Op(A)*x + beta*y, where Op(A) is controlled by mode
///   (see below).
///
/// \tparam AlphaType Type of coefficient alpha. Must be convertible to
/// YVector::value_type. \tparam AMatrix A KokkosSparse::CrsMatrix, or
/// KokkosSparse::Experimental::BsrMatrix \tparam XVector Type of x, must be a
/// rank-2 Kokkos::View \tparam BetaType Type of coefficient beta. Must be
/// convertible to YVector::value_type. \tparam YVector Type of y, must be a
/// rank-2 Kokkos::View and its rank must match that of XVector
///
/// \param mode [in] Select A's operator mode: "N" for normal, "T" for
/// transpose, "C" for conjugate or "H" for conjugate transpose. \param alpha
/// [in] Scalar multiplier for the matrix A. \param A [in] The sparse matrix A.
/// \param x [in] A vector to multiply on the left by A.
/// \param beta [in] Scalar multiplier for the vector y.
/// \param y [in/out] Result vector.
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
void spmv(const char mode[], const AlphaType& alpha, const AMatrix& A,
          const XVector& x, const BetaType& beta, const YVector& y) {
  KokkosSparse::SPMVHandle<AMatrix> sh(KokkosSparse::SPMV_FAST_SETUP);
  spmv(&sh, mode, alpha, A, x, beta, y);
}

/// \brief Kokkos sparse matrix-vector multiply.
///   Computes y := alpha*Op(A)*x + beta*y, where Op(A) is controlled by mode
///   (see below).
///
/// \tparam ExecutionSpace A Kokkos execution space. Must be able to access
///   the memory spaces of A, x, and y.
/// \tparam AlphaType Type of coefficient alpha. Must be convertible to
/// YVector::value_type. \tparam AMatrix A KokkosSparse::CrsMatrix, or
/// KokkosSparse::Experimental::BsrMatrix \tparam XVector Type of x, must be a
/// rank-2 Kokkos::View \tparam BetaType Type of coefficient beta. Must be
/// convertible to YVector::value_type. \tparam YVector Type of y, must be a
/// rank-2 Kokkos::View and its rank must match that of XVector
///
/// \param space [in] The execution space instance on which to run the
///   kernel.
/// \param mode [in] Select A's operator mode: "N" for normal, "T" for
/// transpose, "C" for conjugate or "H" for conjugate transpose. \param alpha
/// [in] Scalar multiplier for the matrix A. \param A [in] The sparse matrix A.
/// \param x [in] A vector to multiply on the left by A.
/// \param beta [in] Scalar multiplier for the vector y.
/// \param y [in/out] Result vector.
template <class ExecutionSpace, class AlphaType, class AMatrix, class XVector,
          class BetaType, class YVector>
void spmv(const ExecutionSpace& space, const char mode[],
          const AlphaType& alpha, const AMatrix& A, const XVector& x,
          const BetaType& beta, const YVector& y) {
  KokkosSparse::SPMVHandle<AMatrix> sh(KokkosSparse::SPMV_FAST_SETUP);
  spmv(space, &sh, controls, mode, alpha, A, x, beta, y);
}

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

namespace Experimental {

template <class ExecutionSpace, class AlphaType, class AMatrix, class XVector,
          class BetaType, class YVector>
void spmv_struct(const ExecutionSpace& space, const char mode[],
                 const int stencil_type,
                 const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                                    Kokkos::HostSpace>& structure,
                 const AlphaType& alpha, const AMatrix& A, const XVector& x,
                 const BetaType& beta, const YVector& y,
                 [[maybe_unused]] const RANK_ONE& tag) {
  // Make sure that both x and y have the same rank.
  static_assert((int)XVector::rank == (int)YVector::rank,
                "KokkosSparse::spmv_struct: Vector ranks do not match.");
  // Make sure A, x, y are accessible to ExecutionSpace
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename AMatrix::memory_space>::accessible,
      "KokkosSparse::spmv_struct: AMatrix must be accessible from "
      "ExecutionSpace");
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename XVector::memory_space>::accessible,
      "KokkosSparse::spmv_struct: XVector must be accessible from "
      "ExecutionSpace");
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename YVector::memory_space>::accessible,
      "KokkosSparse::spmv_struct: YVector must be accessible from "
      "ExecutionSpace");
  // Make sure that x (and therefore y) is rank 1.
  static_assert(
      (int)XVector::rank == 1,
      "KokkosSparse::spmv_struct: Both Vector inputs must have rank 1 in "
      "order to call this specialization of spmv.");
  // Make sure that y is non-const.
  static_assert(std::is_same<typename YVector::value_type,
                             typename YVector::non_const_value_type>::value,
                "KokkosSparse::spmv_struct: Output Vector must be non-const.");

  // Check compatibility of dimensions at run time.
  if ((mode[0] == NoTranspose[0]) || (mode[0] == Conjugate[0])) {
    if ((x.extent(1) != y.extent(1)) ||
        (static_cast<size_t>(A.numCols()) > static_cast<size_t>(x.extent(0))) ||
        (static_cast<size_t>(A.numRows()) > static_cast<size_t>(y.extent(0)))) {
      std::ostringstream os;
      os << "KokkosSparse::spmv_struct: Dimensions do not match: "
         << ", A: " << A.numRows() << " x " << A.numCols()
         << ", x: " << x.extent(0) << " x " << x.extent(1)
         << ", y: " << y.extent(0) << " x " << y.extent(1);

      KokkosKernels::Impl::throw_runtime_exception(os.str());
    }
  } else {
    if ((x.extent(1) != y.extent(1)) ||
        (static_cast<size_t>(A.numCols()) > static_cast<size_t>(y.extent(0))) ||
        (static_cast<size_t>(A.numRows()) > static_cast<size_t>(x.extent(0)))) {
      std::ostringstream os;
      os << "KokkosSparse::spmv_struct: Dimensions do not match (transpose): "
         << ", A: " << A.numRows() << " x " << A.numCols()
         << ", x: " << x.extent(0) << " x " << x.extent(1)
         << ", y: " << y.extent(0) << " x " << y.extent(1);

      KokkosKernels::Impl::throw_runtime_exception(os.str());
    }
  }

  typedef KokkosSparse::CrsMatrix<
      typename AMatrix::const_value_type, typename AMatrix::const_ordinal_type,
      typename AMatrix::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      typename AMatrix::const_size_type>
      AMatrix_Internal;

  typedef Kokkos::View<
      typename XVector::const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<XVector>::array_layout,
      typename XVector::device_type,
      Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >
      XVector_Internal;

  typedef Kokkos::View<
      typename YVector::non_const_value_type*,
      typename KokkosKernels::Impl::GetUnifiedLayout<YVector>::array_layout,
      typename YVector::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
      YVector_Internal;

  AMatrix_Internal A_i = A;
  XVector_Internal x_i = x;
  YVector_Internal y_i = y;

  return KokkosSparse::Impl::SPMV_STRUCT<
      ExecutionSpace, AMatrix_Internal, XVector_Internal,
      YVector_Internal>::spmv_struct(space, mode, stencil_type, structure,
                                     alpha, A_i, x_i, beta, y_i);
}

template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
void spmv_struct(const char mode[], const int stencil_type,
                 const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                                    Kokkos::HostSpace>& structure,
                 const AlphaType& alpha, const AMatrix& A, const XVector& x,
                 const BetaType& beta, const YVector& y, const RANK_ONE& tag) {
  spmv_struct(typename AMatrix::execution_space{}, mode, stencil_type,
              structure, alpha, A, x, beta, y, tag);
}

namespace Impl {
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector, class XLayout = typename XVector::array_layout>
struct SPMV2D1D_STRUCT {
  static bool spmv2d1d_struct(
      const char mode[], const int stencil_type,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& structure,
      const AlphaType& alpha, const AMatrix& A, const XVector& x,
      const BetaType& beta, const YVector& y);

  template <typename ExecutionSpace>
  static bool spmv2d1d_struct(
      const ExecutionSpace& space, const char mode[], const int stencil_type,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& structure,
      const AlphaType& alpha, const AMatrix& A, const XVector& x,
      const BetaType& beta, const YVector& y);
};

#if defined(KOKKOSKERNELS_INST_LAYOUTSTRIDE) || !defined(KOKKOSKERNELS_ETI_ONLY)
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D_STRUCT<AlphaType, AMatrix, XVector, BetaType, YVector,
                       Kokkos::LayoutStride> {
  static bool spmv2d1d_struct(
      const char mode[], const int stencil_type,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& structure,
      const AlphaType& alpha, const AMatrix& A, const XVector& x,
      const BetaType& beta, const YVector& y) {
    spmv_struct(mode, stencil_type, structure, alpha, A, x, beta, y,
                RANK_ONE());
    return true;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d_struct(
      const ExecutionSpace& space, const char mode[], const int stencil_type,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& structure,
      const AlphaType& alpha, const AMatrix& A, const XVector& x,
      const BetaType& beta, const YVector& y) {
    spmv_struct(space, mode, stencil_type, structure, alpha, A, x, beta, y,
                RANK_ONE());
    return true;
  }
};
#else
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D_STRUCT<AlphaType, AMatrix, XVector, BetaType, YVector,
                       Kokkos::LayoutStride> {
  static bool spmv2d1d_struct(
      const char /*mode*/[], const int /*stencil_type*/,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& /*structure*/,
      const AlphaType& /*alpha*/, const AMatrix& /*A*/, const XVector& /*x*/,
      const BetaType& /*beta*/, const YVector& /*y*/) {
    return false;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d_struct(
      const ExecutionSpace& /* space*/, const char /*mode*/[],
      const int /*stencil_type*/,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& /*structure*/,
      const AlphaType& /*alpha*/, const AMatrix& /*A*/, const XVector& /*x*/,
      const BetaType& /*beta*/, const YVector& /*y*/) {
    return false;
  }
};
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT) || !defined(KOKKOSKERNELS_ETI_ONLY)
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D_STRUCT<AlphaType, AMatrix, XVector, BetaType, YVector,
                       Kokkos::LayoutLeft> {
  static bool spmv2d1d_struct(
      const char mode[], const int stencil_type,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& structure,
      const AlphaType& alpha, const AMatrix& A, const XVector& x,
      const BetaType& beta, const YVector& y) {
    spmv_struct(mode, stencil_type, structure, alpha, A, x, beta, y,
                RANK_ONE());
    return true;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d_struct(
      const ExecutionSpace& space, const char mode[], const int stencil_type,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& structure,
      const AlphaType& alpha, const AMatrix& A, const XVector& x,
      const BetaType& beta, const YVector& y) {
    spmv_struct(space, mode, stencil_type, structure, alpha, A, x, beta, y,
                RANK_ONE());
    return true;
  }
};
#else
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D_STRUCT<AlphaType, AMatrix, XVector, BetaType, YVector,
                       Kokkos::LayoutLeft> {
  static bool spmv2d1d_struct(
      const char /*mode*/[], const int /*stencil_type*/,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& /*structure*/,
      const AlphaType& /*alpha*/, const AMatrix& /*A*/, const XVector& /*x*/,
      const BetaType& /*beta*/, const YVector& /*y*/) {
    return false;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d_struct(
      const ExecutionSpace /*space*/, const char /*mode*/[],
      const int /*stencil_type*/,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& /*structure*/,
      const AlphaType& /*alpha*/, const AMatrix& /*A*/, const XVector& /*x*/,
      const BetaType& /*beta*/, const YVector& /*y*/) {
    return false;
  }
};
#endif

#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT) || !defined(KOKKOSKERNELS_ETI_ONLY)
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D_STRUCT<AlphaType, AMatrix, XVector, BetaType, YVector,
                       Kokkos::LayoutRight> {
  static bool spmv2d1d_struct(
      const char mode[], const int stencil_type,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& structure,
      const AlphaType& alpha, const AMatrix& A, const XVector& x,
      const BetaType& beta, const YVector& y) {
    spmv_struct(mode, stencil_type, structure, alpha, A, x, beta, y,
                RANK_ONE());
    return true;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d_struct(
      const ExecutionSpace& space, const char mode[], const int stencil_type,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& structure,
      const AlphaType& alpha, const AMatrix& A, const XVector& x,
      const BetaType& beta, const YVector& y) {
    spmv_struct(space, mode, stencil_type, structure, alpha, A, x, beta, y,
                RANK_ONE());
    return true;
  }
};
#else
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
struct SPMV2D1D_STRUCT<AlphaType, AMatrix, XVector, BetaType, YVector,
                       Kokkos::LayoutRight> {
  static bool spmv2d1d_struct(
      const char /*mode*/[], const int /*stencil_type*/,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& /*structure*/,
      const AlphaType& /*alpha*/, const AMatrix& /*A*/, const XVector& /*x*/,
      const BetaType& /*beta*/, const YVector& /*y*/) {
    return false;
  }

  template <typename ExecutionSpace>
  static bool spmv2d1d_struct(
      const ExecutionSpace& /*space*/, const char /*mode*/[],
      const int /*stencil_type*/,
      const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                         Kokkos::HostSpace>& /*structure*/,
      const AlphaType& /*alpha*/, const AMatrix& /*A*/, const XVector& /*x*/,
      const BetaType& /*beta*/, const YVector& /*y*/) {
    return false;
  }
};
#endif
}  // namespace Impl

template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector, class XLayout = typename XVector::array_layout>
using SPMV2D1D_STRUCT
    [[deprecated("KokkosSparse::SPMV2D1D_STRUCT is not part of the public "
                 "interface - use KokkosSparse::spmv_struct instead")]] =
        Impl::SPMV2D1D_STRUCT<AlphaType, AMatrix, XVector, BetaType, YVector>;

template <class ExecutionSpace, class AlphaType, class AMatrix, class XVector,
          class BetaType, class YVector>
void spmv_struct(const ExecutionSpace& space, const char mode[],
                 const int stencil_type,
                 const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                                    Kokkos::HostSpace>& structure,
                 const AlphaType& alpha, const AMatrix& A, const XVector& x,
                 const BetaType& beta, const YVector& y,
                 [[maybe_unused]] const RANK_TWO& tag) {
  // Make sure A, x, y are accessible to ExecutionSpace
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename AMatrix::memory_space>::accessible,
      "KokkosSparse::spmv_struct: AMatrix must be accessible from "
      "ExecutionSpace");
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename XVector::memory_space>::accessible,
      "KokkosSparse::spmv_struct: XVector must be accessible from "
      "ExecutionSpace");
  static_assert(
      Kokkos::SpaceAccessibility<ExecutionSpace,
                                 typename YVector::memory_space>::accessible,
      "KokkosSparse::spmv_struct: YVector must be accessible from "
      "ExecutionSpace");
  // Make sure that both x and y have the same rank.
  static_assert(XVector::rank == YVector::rank,
                "KokkosSparse::spmv: Vector ranks do not match.");
  // Make sure that y is non-const.
  static_assert(std::is_same<typename YVector::value_type,
                             typename YVector::non_const_value_type>::value,
                "KokkosSparse::spmv: Output Vector must be non-const.");

  // Check compatibility of dimensions at run time.
  if ((mode[0] == NoTranspose[0]) || (mode[0] == Conjugate[0])) {
    if ((x.extent(1) != y.extent(1)) ||
        (static_cast<size_t>(A.numCols()) > static_cast<size_t>(x.extent(0))) ||
        (static_cast<size_t>(A.numRows()) > static_cast<size_t>(y.extent(0)))) {
      std::ostringstream os;
      os << "KokkosSparse::spmv: Dimensions do not match: "
         << ", A: " << A.numRows() << " x " << A.numCols()
         << ", x: " << x.extent(0) << " x " << x.extent(1)
         << ", y: " << y.extent(0) << " x " << y.extent(1);
      KokkosKernels::Impl::throw_runtime_exception(os.str());
    }
  } else {
    if ((x.extent(1) != y.extent(1)) ||
        (static_cast<size_t>(A.numCols()) > static_cast<size_t>(y.extent(0))) ||
        (static_cast<size_t>(A.numRows()) > static_cast<size_t>(x.extent(0)))) {
      std::ostringstream os;
      os << "KokkosSparse::spmv: Dimensions do not match (transpose): "
         << ", A: " << A.numRows() << " x " << A.numCols()
         << ", x: " << x.extent(0) << " x " << x.extent(1)
         << ", y: " << y.extent(0) << " x " << y.extent(1);
      KokkosKernels::Impl::throw_runtime_exception(os.str());
    }
  }

  typedef KokkosSparse::CrsMatrix<
      typename AMatrix::const_value_type, typename AMatrix::const_ordinal_type,
      typename AMatrix::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged>,
      typename AMatrix::const_size_type>
      AMatrix_Internal;

  AMatrix_Internal A_i = A;

  // Call single-vector version if appropriate
  if (x.extent(1) == 1) {
    typedef Kokkos::View<
        typename XVector::const_value_type*, typename YVector::array_layout,
        typename XVector::device_type,
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >
        XVector_SubInternal;
    typedef Kokkos::View<
        typename YVector::non_const_value_type*, typename YVector::array_layout,
        typename YVector::device_type, Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        YVector_SubInternal;

    XVector_SubInternal x_i = Kokkos::subview(x, Kokkos::ALL(), 0);
    YVector_SubInternal y_i = Kokkos::subview(y, Kokkos::ALL(), 0);

    // spmv_struct (mode, alpha, A, x_i, beta, y_i);
    if (Impl::SPMV2D1D_STRUCT<AlphaType, AMatrix_Internal, XVector_SubInternal,
                              BetaType, YVector_SubInternal,
                              typename XVector_SubInternal::array_layout>::
            spmv2d1d_struct(space, mode, stencil_type, structure, alpha, A, x_i,
                            beta, y_i)) {
      return;
    }
  }

  // Call true rank 2 vector implementation
  {
    typedef Kokkos::View<
        typename XVector::const_value_type**, typename XVector::array_layout,
        typename XVector::device_type,
        Kokkos::MemoryTraits<Kokkos::Unmanaged | Kokkos::RandomAccess> >
        XVector_Internal;

    typedef Kokkos::View<typename YVector::non_const_value_type**,
                         typename YVector::array_layout,
                         typename YVector::device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged> >
        YVector_Internal;

    XVector_Internal x_i = x;
    YVector_Internal y_i = y;

    return KokkosSparse::Impl::SPMV_MV<
        ExecutionSpace, AMatrix_Internal, XVector_Internal,
        YVector_Internal>::spmv_mv(space,
                                   KokkosKernels::Experimental::Controls(),
                                   mode, alpha, A_i, x_i, beta, y_i);
  }
}

template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
void spmv_struct(const char mode[], const int stencil_type,
                 const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                                    Kokkos::HostSpace>& structure,
                 const AlphaType& alpha, const AMatrix& A, const XVector& x,
                 const BetaType& beta, const YVector& y, const RANK_TWO& tag) {
  spmv_struct(typename AMatrix::execution_space{}, mode, stencil_type,
              structure, alpha, A, x, beta, y, tag);
}

/// \brief Public interface to structured local sparse matrix-vector multiply.
///
/// Compute y = beta*y + alpha*Op(A)*x, where x and y are either both
/// rank 1 (single vectors) or rank 2 (multivectors) Kokkos::View
/// instances, A is a KokkosSparse::CrsMatrix, and Op(A) is determined
/// by \c mode.  If beta == 0, ignore and overwrite the initial
/// entries of y; if alpha == 0, ignore the entries of A and x.
///
/// \param mode [in] "N" for no transpose, "T" for transpose, or "C"
///             for conjugate transpose.
/// \param stencil_type
/// \param structure [in] this 1D view stores the # rows in each dimension
///                  (i,j,k)
/// \param alpha [in] Scalar multiplier for the matrix A.
/// \param A [in] The sparse matrix; KokkosSparse::CrsMatrix instance.
/// \param x [in] Either a
///                single vector (rank-1 Kokkos::View) or
///                multivector (rank-2 Kokkos::View).
/// \param beta [in] Scalar multiplier for the (multi)vector y.
/// \param y [in/out] Either a single vector (rank-1 Kokkos::View) or
///   multivector (rank-2 Kokkos::View).  It must have the same number
///   of columns as x.
template <class AlphaType, class AMatrix, class XVector, class BetaType,
          class YVector>
void spmv_struct(const char mode[], const int stencil_type,
                 const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                                    Kokkos::HostSpace>& structure,
                 const AlphaType& alpha, const AMatrix& A, const XVector& x,
                 const BetaType& beta, const YVector& y) {
  typedef
      typename std::conditional<XVector::rank == 2, RANK_TWO, RANK_ONE>::type
          RANK_SPECIALISE;
  spmv_struct(mode, stencil_type, structure, alpha, A, x, beta, y,
              RANK_SPECIALISE());
}

/// \brief Public interface to structured local sparse matrix-vector multiply.
///
/// Compute y = beta*y + alpha*Op(A)*x, where x and y are either both
/// rank 1 (single vectors) or rank 2 (multivectors) Kokkos::View
/// instances, A is a KokkosSparse::CrsMatrix, and Op(A) is determined
/// by \c mode.  If beta == 0, ignore and overwrite the initial
/// entries of y; if alpha == 0, ignore the entries of A and x.
///
/// \param space [in] The execution space instance on which to run the
///   kernel.
/// \param mode [in] "N" for no transpose, "T" for transpose, or "C"
///             for conjugate transpose.
/// \param stencil_type
/// \param structure [in] this 1D view stores the # rows in each dimension
///                  (i,j,k)
/// \param alpha [in] Scalar multiplier for the matrix A.
/// \param A [in] The sparse matrix; KokkosSparse::CrsMatrix instance.
/// \param x [in] Either a
///                single vector (rank-1 Kokkos::View) or
///                multivector (rank-2 Kokkos::View).
/// \param beta [in] Scalar multiplier for the (multi)vector y.
/// \param y [in/out] Either a single vector (rank-1 Kokkos::View) or
///   multivector (rank-2 Kokkos::View).  It must have the same number
///   of columns as x.
template <class ExecutionSpace, class AlphaType, class AMatrix, class XVector,
          class BetaType, class YVector>
void spmv_struct(const ExecutionSpace& space, const char mode[],
                 const int stencil_type,
                 const Kokkos::View<typename AMatrix::non_const_ordinal_type*,
                                    Kokkos::HostSpace>& structure,
                 const AlphaType& alpha, const AMatrix& A, const XVector& x,
                 const BetaType& beta, const YVector& y) {
  typedef
      typename std::conditional<XVector::rank == 2, RANK_TWO, RANK_ONE>::type
          RANK_SPECIALISE;
  spmv_struct(space, mode, stencil_type, structure, alpha, A, x, beta, y,
              RANK_SPECIALISE());
}

}  // namespace Experimental
}  // namespace KokkosSparse

// Pull in all the deprecated versions of spmv
// It's included here (and not at the top) because it uses definitions in this file.
#include "KokkosSparse_spmv_deprecated.hpp"

#endif
