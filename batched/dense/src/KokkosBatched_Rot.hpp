// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project
#ifndef KOKKOSBATCHED_ROT_HPP_
#define KOKKOSBATCHED_ROT_HPP_

#include <KokkosBatched_Util.hpp>

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

namespace KokkosBatched {

/// \brief Serial Batched Rot:
/// Applies a plane rotation to vectors x and y:
///   x(i) := c*x(i) + s*y(i)
///   y(i) := c*y(i) - s*x(i)          (Trans::Transpose, drot/zdrot)
///   y(i) := c*y(i) - conj(s)*x(i)    (Trans::ConjTranspose, zrot)
///
/// \tparam ArgTrans: Type indicating whether s is used directly (Trans::Transpose)
/// or its conjugate is used (Trans::ConjTranspose)
///
/// \tparam XViewType: Input/output type for the vector x, needs to be a 1D view
/// \tparam YViewType: Input/output type for the vector y, needs to be a 1D view
/// \tparam CType: Input type for the cosine c (typically real)
/// \tparam SType: Input type for the sine s (real or complex)
///
/// \param[inout] x: x is a length n vector, a rank 1 view
/// \param[inout] y: y is a length n vector, a rank 1 view
/// \param[in] c: cosine of the rotation (real scalar)
/// \param[in] s: sine of the rotation (real or complex scalar)
///
/// No nested parallel_for is used inside of the function.
///
template <typename ArgTrans>
struct SerialRot {
  static_assert(std::is_same_v<ArgTrans, Trans::Transpose> || std::is_same_v<ArgTrans, Trans::ConjTranspose>,
                "KokkosBatched::rot: Use Trans::Transpose for {s,d,cs,zd}rot or Trans::ConjTranspose for {c,z}rot");

  template <typename XViewType, typename YViewType, typename CType, typename SType>
  KOKKOS_INLINE_FUNCTION static int invoke(const XViewType &x, const YViewType &y, const CType c, const SType s);
};
}  // namespace KokkosBatched

#include "KokkosBatched_Rot_Serial_Impl.hpp"

#endif  // KOKKOSBATCHED_ROT_HPP_
