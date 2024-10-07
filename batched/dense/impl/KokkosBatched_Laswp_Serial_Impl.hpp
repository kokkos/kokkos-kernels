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

#ifndef KOKKOSBATCHED_LASWP_SERIAL_IMPL_HPP_
#define KOKKOSBATCHED_LASWP_SERIAL_IMPL_HPP_

/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Laswp_Serial_Internal.hpp"

namespace KokkosBatched {

///
/// Serial Internal Impl
/// ========================

///
//// Forward pivot apply
///

/// row swap
template <>
struct SerialLaswp<Side::Left, Direct::Forward> {
  template <typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int piv, const AViewType &A) {
    if constexpr (AViewType::rank == 1) {
      const int as0 = A.stride(0);
      SerialLaswpVectorForwardInternal::invoke(piv, A.data(), as0);
    } else if constexpr (AViewType::rank == 2) {
      const int n = A.extent(1), as0 = A.stride(0), as1 = A.stride(1);
      SerialLaswpMatrixForwardInternal::invoke(n, piv, A.data(), as0, as1);
    }
    return 0;
  }

  template <typename PivViewType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const PivViewType piv, const AViewType &A) {
    if constexpr (AViewType::rank == 1) {
      const int plen = piv.extent(0), ps0 = piv.stride(0), as0 = A.stride(0);
      SerialLaswpVectorForwardInternal::invoke(plen, piv.data(), ps0, A.data(), as0);
    } else if constexpr (AViewType::rank == 2) {
      // row permutation
      const int plen = piv.extent(0), ps0 = piv.stride(0), n = A.extent(1), as0 = A.stride(0), as1 = A.stride(1);
      SerialLaswpMatrixForwardInternal::invoke(n, plen, piv.data(), ps0, A.data(), as0, as1);
    }
    return 0;
  }
};

/// column swap
template <>
struct SerialLaswp<Side::Right, Direct::Forward> {
  template <typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int piv, const AViewType &A) {
    if constexpr (AViewType::rank == 1) {
      const int as0 = A.stride(0);
      SerialLaswpVectorForwardInternal::invoke(piv, A.data(), as0);
    } else if constexpr (AViewType::rank == 2) {
      const int m = A.extent(0), as0 = A.stride(0), as1 = A.stride(1);
      SerialLaswpMatrixForwardInternal::invoke(m, piv, A.data(), as1, as0);
    }
    return 0;
  }

  template <typename PivViewType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const PivViewType &piv, const AViewType &A) {
    if constexpr (AViewType::rank == 1) {
      const int plen = piv.extent(0), ps = piv.stride(0), as0 = A.stride(0);
      SerialLaswpVectorForwardInternal ::invoke(plen, piv.data(), ps, A.data(), as0);
    } else if constexpr (AViewType::rank == 2) {
      // column permutation
      const int plen = piv.extent(0), ps = piv.stride(0), m = A.extent(0), as0 = A.stride(0), as1 = A.stride(1);
      SerialLaswpMatrixForwardInternal ::invoke(m, plen, piv.data(), ps, A.data(), as1, as0);
    }
    return 0;
  }
};

///
/// Backward pivot apply
///

/// row swap
template <>
struct SerialLaswp<Side::Left, Direct::Backward> {
  template <typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int piv, const AViewType &A) {
    if constexpr (AViewType::rank == 1) {
      const int as0 = A.stride(0);
      SerialLaswpVectorBackwardInternal::invoke(piv, A.data(), as0);
    } else if constexpr (AViewType::rank == 2) {
      const int n = A.extent(1), as0 = A.stride(0), as1 = A.stride(1);
      SerialLaswpMatrixBackwardInternal::invoke(n, piv, A.data(), as0, as1);
    }
    return 0;
  }

  template <typename PivViewType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const PivViewType piv, const AViewType &A) {
    if constexpr (AViewType::rank == 1) {
      const int plen = piv.extent(0), ps0 = piv.stride(0), as0 = A.stride(0);
      SerialLaswpVectorBackwardInternal::invoke(plen, piv.data(), ps0, A.data(), as0);
    } else if constexpr (AViewType::rank == 2) {
      // row permutation
      const int plen = piv.extent(0), ps0 = piv.stride(0), n = A.extent(1), as0 = A.stride(0), as1 = A.stride(1);
      SerialLaswpMatrixBackwardInternal::invoke(n, plen, piv.data(), ps0, A.data(), as0, as1);
    }
    return 0;
  }
};

/// column swap
template <>
struct SerialLaswp<Side::Right, Direct::Backward> {
  template <typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const int piv, const AViewType &A) {
    if constexpr (AViewType::rank == 1) {
      const int as0 = A.stride(0);
      SerialLaswpVectorBackwardInternal::invoke(piv, A.data(), as0);
    } else if constexpr (AViewType::rank == 2) {
      const int m = A.extent(0), as0 = A.stride(0), as1 = A.stride(1);
      SerialLaswpMatrixBackwardInternal::invoke(m, piv, A.data(), as1, as0);
    }
    return 0;
  }

  template <typename PivViewType, typename AViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const PivViewType &piv, const AViewType &A) {
    if constexpr (AViewType::rank == 1) {
      const int plen = piv.extent(0), ps = piv.stride(0), as0 = A.stride(0);
      SerialLaswpVectorBackwardInternal ::invoke(plen, piv.data(), ps, A.data(), as0);
    } else if constexpr (AViewType::rank == 2) {
      // column permutation
      const int plen = piv.extent(0), ps = piv.stride(0), m = A.extent(0), as0 = A.stride(0), as1 = A.stride(1);
      SerialLaswpMatrixBackwardInternal ::invoke(m, plen, piv.data(), ps, A.data(), as1, as0);
    }
    return 0;
  }
};

}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_LASWP_SERIAL_IMPL_HPP_
