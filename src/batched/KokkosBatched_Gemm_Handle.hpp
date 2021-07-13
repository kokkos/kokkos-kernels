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

#ifndef KOKKOSKERNELS_KOKKOSBATCHED_GEMM_HANDLE_HPP
#define KOKKOSKERNELS_KOKKOSBATCHED_GEMM_HANDLE_HPP

#include "KokkosBatched_Kernel_Handle.hpp"

namespace KokkosBatched {

///                      KK_SERIAL            Invoke SerialFUNC     via
///                      RangePolicy(BatchSz) KK_TEAM              Invoke
///                      TeamFUNC       via TeamPolicy(BatchSz) KK_TEAMVECTOR
///                      Invoke TeamVectorFUNC via TeamPolicy(BatchSz)
///                      KK_SERIALSIMD        Invoke SerialFUNC     via
///                      TeamPolicy(BatchSz) KK_TEAMSIMD          Invoke
///                      TeamFUNC       via TeamPolicy(BatchSz) KK_SERIAL_OPT2
///                      Invoke SerialFUNC     via
///                                           RangePolicy(BatchSz*N*M)
///                      KK_TEAMVECTOR_SHMEM  Invoke TeamVectorFUNC via
///                      TeamPolicy(BatchSz)
///                                           Copies A and B to shared memory
///                                           before GEMM.
///                      KK_TEAMVECTOR_DBLBUF Invoke TeamVectorFUNC via
///                                           TeamPolicy(BatchSz*TILES)
///                                           Uses tiling and double buffering
///                                           via shared memory and register
///                                           buffers.
class GemmAlgos : public KernelAlgos {
 public:
  // Additional TPL Algos
  class CUBLAS : KernelAlgo {};
  class MAGMA : KernelAlgo {};

  // Additional KokkosKernels batched GEMM Algos
  class KK_TEAM : KernelAlgo {};
  class KK_TEAMVECTOR : KernelAlgo {};
  class KK_SERIALSIMD : KernelAlgo {};
  class KK_TEAMSIMD : KernelAlgo {};
  class KK_SERIAL_OPT2 : KernelAlgo {};
  class KK_TEAMVECTOR_SHMEM : KernelAlgo {};
  class KK_TEAMVECTOR_DBLBUF : KernelAlgo {};
};

/// \brief Handle for selecting runtime behavior of the BatchedGemm interface.
///
/// \var kernelAlgoType  Specifies which algorithm to use for invocation
/// (default, SQUARE).
///
///                    Specifies whether to select optimal invocations based on
///                    inputs and heuristics:
///                      SQUARE select invocations based on square matrix
///                      heuristics where M=N TALL   select invocations based on
///                      tall   matrix heuristics where M>N WIDE   select
///                      invocations based on wide   matrix heuristics where M<N
///                    Note: If the heuristics indicate SIMD views are required
///                    for optimal performance, notify the user that SIMD views
///                    are required for optimal performance.
///
///                    Specifies which cmake-enabled tpl algorithm to invoke:
///                      ARMPL    Invoke the ArmPL TPL interface
///                      MKL      Invoke the MKL TPL interface
///                      SYCL     Invoke the SYCL TPL interface
///                      CUBLAS   Invoke the CuBLAS TPL interface
///                      MAGMA    Invoke the Magma TPL interface
///                    Note: Requires that input views for A, B, and C reside on
///                    either host or device depending on the TPL selected.
///                    Note: If the user selects a TPL, an error will be thrown
///                    if:
///                       1. The TPL is not enabled via cmake
///                       2. The input views do not reside on the host/device as
///                       needed
///
///                    Specifies which kokkos-kernels (KK) algorithm to invoke:
///                      KK_SERIAL            Invoke SerialFUNC     via
///                      RangePolicy(BatchSz) KK_TEAM              Invoke
///                      TeamFUNC       via TeamPolicy(BatchSz) KK_TEAMVECTOR
///                      Invoke TeamVectorFUNC via TeamPolicy(BatchSz)
///                      KK_SERIALSIMD        Invoke SerialFUNC     via
///                      TeamPolicy(BatchSz) KK_TEAMSIMD          Invoke
///                      TeamFUNC       via TeamPolicy(BatchSz) KK_SERIAL_OPT2
///                      Invoke SerialFUNC     via
///                                           RangePolicy(BatchSz*N*M)
///                      KK_TEAMVECTOR_SHMEM  Invoke TeamVectorFUNC via
///                      TeamPolicy(BatchSz)
///                                           Copies A and B to shared memory
///                                           before GEMM.
///                      KK_TEAMVECTOR_DBLBUF Invoke TeamVectorFUNC via
///                                           TeamPolicy(BatchSz*TILES)
///                                           Uses tiling and double buffering
///                                           via shared memory and register
///                                           buffers.
/// \var teamSz        Specifies the team size that will affect any KK algorithm
/// which uses
///                    TeamPolicy (default, Kokkos::AUTO).
///                    Note: Only applied if useAlgo_type == KK_*
/// \var vecLen        Specifies the vector length that will affect any KK
/// algorithm which
///                    uses TeamPolicy and Kokkos::ThreadVectorRange or
///                    Kokkos::TeamVectorRange (default, Kokkos::AUTO). Note:
///                    Only applied if useAlgo_type == KK_*
template <class KernelAlgoType = GemmAlgos::SQUARE>
class BatchedGemmHandle : public BatchedKernelHandle<KernelAlgoType> {
 public:
  KernelAlgoType kernelAlgoType;
};

}  // namespace KokkosBatched

#endif  // KOKKOSKERNELS_KOKKOSBATCHED_GEMM_HANDLE_HPP
