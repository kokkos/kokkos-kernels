//@HEADER
// ************************************************************************
//
//                        Kokkos v. 3.4
//       Copyright (2021) National Technology & Engineering
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
// Questions? Contact Siva Rajamanickam (srajama@sandia.gov)
//
// ************************************************************************
//@HEADER

//
// Created by Harvey, Evan on 7/13/21.
//

#ifndef KOKKOSKERNELS_KOKKOSBATCHED_KERNEL_HEADER_HPP
#define KOKKOSKERNELS_KOKKOSBATCHED_KERNEL_HEADER_HPP

namespace KokkosBatched {

/// \brief Heuristic algorithm types. See BatchedKernelHandle for details.
namespace BaseHeuristicAlgos {
enum BASE_HEURISTIC_ALGOS : int { SQUARE = 0, TALL, WIDE, N };
}

/// \brief Tpl algorithm types. See BatchedKernelHandle for details.
namespace BaseTplAlgos {
enum BASE_TPL_ALGOS : int { ARMPL = BaseHeuristicAlgos::N, MKL, SYCL, N };
}

/// \brief KokkosBatched algorithm types. See BatchedKernelHandle for details.
namespace BaseKokkosBatchedAlgos {
enum BASE_KOKKOS_BATCHED_ALGOS : int { KK_SERIAL = BaseTplAlgos::N, N };
}

#define N_BASE_ALGOS BaseKokkosBatchedAlgos::N

// clang-format off
/// \brief Handle for selecting runtime behavior of the BatchedGemm interface.
///
/// \var kernelAlgoType  Specifies which algorithm to use for invocation (default, SQUARE).
///
///                    Specifies whether to select optimal invocations based on inputs and
///                    heuristics:
///                      SQUARE select invocations based on square matrix heuristics where M=N
///                      TALL   select invocations based on tall   matrix heuristics where M>N
///                      WIDE   select invocations based on wide   matrix heuristics where M<N
///                    Note: If the heuristics indicate SIMD views are required for optimal
///                    performance, notify the user that SIMD views are required for
///                    optimal performance.
///
///                    Specifies which cmake-enabled tpl algorithm to invoke:
///                      ARMPL    Invoke the ArmPL TPL interface
///                      MKL      Invoke the MKL TPL interface
///                      SYCL     Invoke the SYCL TPL interface
///                    Note: Requires that input views for A, B, and C reside on either host
///                    or device depending on the TPL selected.
///                    Note: If the user selects a TPL, an error will be thrown if:
///                       1. The TPL is not enabled via cmake
///                       2. The input views do not reside on the host/device as needed
///
///                    Specifies which kokkos-kernels (KK) algorithm to invoke:
///                      KK_SERIAL            Invoke SerialFUNC     via RangePolicy(BatchSz)
/// \var teamSz        Specifies the team size that will affect any KK algorithm which uses
///                    TeamPolicy (default, Kokkos::AUTO).
///                    Note: Only applied if useAlgo_type == KK_*
/// \var vecLen        Specifies the vector length that will affect any KK algorithm which
///                    uses TeamPolicy and Kokkos::ThreadVectorRange or Kokkos::TeamVectorRange
///                    (default, Kokkos::AUTO).
///                    Note: Only applied if useAlgo_type == KK_*
// clang-format on
class BatchedKernelHandle {
 public:
  int kernelAlgoType = BaseHeuristicAlgos::SQUARE;
  int teamSz         = 0;
  int vecLen = 0;

  /// \var enabledDebug toggle debug messages.
  /// \var tplHandle    a handle specific to the TPL API.
  ///                   managed internally unless provided by user via
  ///                   constructor overload
 private:
  bool enableDebug = false;
  // TODO: TplHandle tplHandle;
};

}  // namespace KokkosBatched

#endif  // KOKKOSKERNELS_KOKKOSBATCHED_KERNEL_HEADER_HPP
