/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are
// met:
//
// 1. Redistributions of source code must retain the above copyright
// notice, this list of conditions and the following disclaimer.
//
// 2. Redistributions in binary form must reproduce the above copyright
// notice, this list of conditions and the following disclaimer in the
// documentation and/or other materials provided with the distribution.
//
// 3. Neither the name of the Corporation nor the names of the
// contributors may be used to endorse or promote products derived from
// this software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY NTESS "AS IS" AND ANY
// EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR
// PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL NTESS OR THE
// CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
// EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
// PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
// PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
// LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//
// Questions? Contact Brian Kelley (bmkelle@sandia.gov)
//
// ************************************************************************
//@HEADER
*/

#ifndef KOKKOS_BLAS_UTIL_HPP
#define KOKKOS_BLAS_UTIL_HPP

#include "Kokkos_ArithTraits.hpp"

namespace KokkosBlas {

//////// Tags for BLAS ////////
struct Trans {
  struct Transpose {};
  struct NoTranspose {};
  struct ConjTranspose {};
};

#if !defined(KOKKOS_IF_ON_HOST)

namespace Impl {

template <class>
struct algo_level3_blocked_mb_impl;
template <>
struct algo_level3_blocked_mb_impl<Kokkos::HostSpace> {
  static constexpr int value = 4;
};
#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct algo_level3_blocked_mb_impl<Kokkos::CudaSpace> {
  static constexpr int value = 2;
};
#endif
#if defined(KOKKOS_ENABLE_HIP)
template <>
struct algo_level3_blocked_mb_impl<Kokkos::Experimental::HIPSpace> {
  static constexpr int value = 2;
};
#endif
#if defined(KOKKOS_ENABLE_SYCL)
template <>
struct algo_level3_blocked_mb_impl<Kokkos::Experimental::SYCLDeviceUSMSpace> {
  static constexpr int value = 2;
};
#endif

template <class>
struct algo_level2_blocked_mb_impl;
template <>
struct algo_level2_blocked_mb_impl<Kokkos::HostSpace> {
  static constexpr int value = 4;
};
#if defined(KOKKOS_ENABLE_CUDA)
template <>
struct algo_level2_blocked_mb_impl<Kokkos::CudaSpace> {
  static constexpr int value = 1;
};
#endif
#if defined(KOKKOS_ENABLE_HIP)
template <>
struct algo_level2_blocked_mb_impl<Kokkos::Experimental::HIPSpace> {
  static constexpr int value = 1;
};
#endif
#if defined(KOKKOS_ENABLE_SYCL)
template <>
struct algo_level2_blocked_mb_impl<Kokkos::Experimental::SYCLDeviceUSMSpace> {
  static constexpr int value = 1;
};
#endif

}  // namespace Impl
#endif

struct Algo {
  struct Level3 {
    struct Unblocked {
      static const char *name() { return "Unblocked"; }
    };
    struct Blocked {
      static const char *name() { return "Blocked"; }
      // TODO:: for now harwire the blocksizes; this should reflect
      // register blocking (not about team parallelism).
      // this mb should vary according to
      // - team policy (smaller) or range policy (bigger)
      // - space (gpu vs host)
      // - blocksize input (blk <= 4 mb = 2, otherwise mb = 4), etc.
#if defined(KOKKOS_IF_ON_HOST)
      static constexpr KOKKOS_FUNCTION int mb() {
        KOKKOS_IF_ON_HOST((return 4;))
        KOKKOS_IF_ON_DEVICE((return 2;))
      }

#else  // FIXME remove when requiring minimum version of Kokkos 3.6
      static constexpr KOKKOS_FUNCTION int mb() {
        return algo_level3_blocked_mb_impl<
            Kokkos::Impl::ActiveExecutionMemorySpace>::value;
      }

#endif
    };
    struct MKL {
      static const char *name() { return "MKL"; }
    };
    struct CompactMKL {
      static const char *name() { return "CompactMKL"; }
    };

    // When this is first developed, unblocked algorithm is a naive
    // implementation and blocked algorithm uses register blocking variant of
    // algorithm (manual unrolling). This distinction is almost meaningless and
    // it just adds more complications. Eventually, the blocked version will be
    // removed and we only use the default algorithm. For testing and
    // development purpose, we still leave algorithm tag in the template
    // arguments.
    using Default = Unblocked;
  };

  using Gemm      = Level3;
  using Trsm      = Level3;
  using Trmm      = Level3;
  using Trtri     = Level3;
  using LU        = Level3;
  using InverseLU = Level3;
  using SolveLU   = Level3;
  using QR        = Level3;
  using UTV       = Level3;

  struct Level2 {
    struct Unblocked {};
    struct Blocked {
      // TODO:: for now hardwire the blocksizes; this should reflect
      // register blocking (not about team parallelism).
      // this mb should vary according to
      // - team policy (smaller) or range policy (bigger)
      // - space (cuda vs host)
      // - blocksize input (blk <= 4 mb = 2, otherwise mb = 4), etc.
#if defined(KOKKOS_IF_ON_HOST)
      static constexpr KOKKOS_FUNCTION int mb() {
        KOKKOS_IF_ON_HOST((return 4;))
        KOKKOS_IF_ON_DEVICE((return 1;))
      }

#else  // FIXME remove when requiring minimum version of Kokkos 3.6
      static constexpr KOKKOS_FUNCTION int mb() {
        return algo_level2_blocked_mb_impl<
            Kokkos::Impl::ActiveExecutionMemorySpace>::value;
      }

#endif
    };
    struct MKL {};
    struct CompactMKL {};

    // When this is first developed, unblocked algorithm is a naive
    // implementation and blocked algorithm uses register blocking variant of
    // algorithm (manual unrolling). This distinction is almost meaningless and
    // it just adds more complications. Eventually, the blocked version will be
    // removed and we only use the default algorithm. For testing and
    // development purpose, we still leave algorithm tag in the template
    // arguments.
    using Default = Unblocked;
  };

  using Gemv   = Level2;
  using Trsv   = Level2;
  using ApplyQ = Level2;
};

namespace Impl {

// Helper to choose the work distribution for a TeamPolicy computing multiple
// reductions. Each team computes a partial reduction and atomically contributes
// to the final result.
//
// This was originally written for dot-based GEMM, but can also be applied to
// multivector dots/norms.

// Input params:
//  * length: size of each vector to reduce
//  * numReductions: number of reductions to compute
// Output params:
//  * teamsPerReduction: number of teams to use for each reduction
template <typename ExecSpace, typename size_type>
void multipleReductionWorkDistribution(size_type length,
                                       size_type numReductions,
                                       size_type &teamsPerDot) {
  constexpr size_type workPerTeam = 4096;  // Amount of work per team
  size_type appxNumTeams =
      (length * numReductions) / workPerTeam;  // Estimation for appxNumTeams

  // Adjust appxNumTeams in case it is too small or too large
  if (appxNumTeams < 1) appxNumTeams = 1;
  if (appxNumTeams > 1024) appxNumTeams = 1024;

  // If there are more reductions than the number of teams,
  // then set the number of teams to be number of reductions.
  // We don't want a team to contribute to more than one reduction.
  if (numReductions >= appxNumTeams) {
    teamsPerDot = 1;
  }
  // If there are more teams than reductions, each reduction can
  // potentially be performed by multiple teams. First, compute
  // teamsPerDot as an integer (take the floor, not ceiling), then,
  // compute actual number of teams by using this factor.
  else {
    teamsPerDot = appxNumTeams / numReductions;
  }
}

// Functor to apply sqrt() to each element of a 1D view.

template <class RV>
struct TakeSqrtFunctor {
  TakeSqrtFunctor(const RV &r_) : r(r_) {}

  KOKKOS_INLINE_FUNCTION void operator()(int i) const {
    r(i) = Kokkos::ArithTraits<typename RV::non_const_value_type>::sqrt(r(i));
  }

  RV r;
};

}  // namespace Impl
}  // namespace KokkosBlas

#endif
