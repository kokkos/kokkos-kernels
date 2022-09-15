/*
//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.0
//       Copyright (2020) National Technology & Engineering
//               Solutions of Sandia, LLC (NTESS).
//
// Under the terms of Contract DE-NA0003525 with NTESS,
// the U.S. Government retains certain rights in this software./
/
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

#ifndef KOKKOSBLAS_EXECUTION_POLICY_HPP
#define KOKKOSBLAS_EXECUTION_POLICY_HPP

namespace KokkosBlas {

//////// Tags for BLAS ////////
struct Mode {
  struct Serial {
    static const char *name() { return "Serial"; }
  };
  struct Team {
    static const char *name() { return "Team"; }
  };
  struct TeamVector {
    static const char *name() { return "TeamVector"; }
  };
};

//////// execution_policy ////////
template <class ExecResource, class Algo = void>
struct execution_policy {
  using execution_resource = ExecResource;
  using algorithm          = Algo;

  // The execution resource describes what processing
  // resources are available to the algorithm. For
  // instance it might be an execution_space if we are
  // launching a kernel from host or it might be one of
  // the Mode (Serial, Team or TeamVector) if the kernel
  // is launched from a parallel_for.
  const execution_resource &exec;

  // Algo allows to directly request an specific kernel
  // implementation, such as Blocked or Unblocked. More
  // algo can be used depending on kernels implementations.
  const algorithm &alg;

  KOKKOS_FUNCTION
  execution_policy(const execution_resource &exec_, const algorithm &alg_)
      : exec(exec_), alg(alg_) {}
};

template <class ExecResource>
struct execution_policy<ExecResource, void> {
  using execution_resource = ExecResource;

  // Only member actually used, this is currently the case
  // when calling a device level implementation
  const execution_resource &exec;

  // In this case, alg is not specified so we store is as nullptr
  const void *alg;

  KOKKOS_FUNCTION
  execution_policy(const execution_resource &exec_)
      : exec(exec_), alg(nullptr) {}
};

struct Trans {
  struct Transpose {};
  struct NoTranspose {};
  struct ConjTranspose {};
};

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

} // namespace KokkosBlas

#endif
