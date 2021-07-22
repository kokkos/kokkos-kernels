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
#ifndef __KOKKOSBATCHED_GEMM_DECL_HPP__
#define __KOKKOSBATCHED_GEMM_DECL_HPP__

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

// Includes for non-functor-level routines
#include <KokkosBatched_Gemm_Handle.hpp>
#include <KokkosKernels_ExecSpaceUtils.hpp>

namespace KokkosBatched {
/********************* BEGIN functor-level routines *********************/
///
/// Serial Gemm
///

template <typename ArgTransA, typename ArgTransB, typename ArgAlgo>
struct SerialGemm {
  template <typename ScalarType, typename AViewType, typename BViewType,
            typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType alpha,
                                           const AViewType &A,
                                           const BViewType &B,
                                           const ScalarType beta,
                                           const CViewType &C);
};

///
/// Team Gemm
///

template <typename MemberType, typename ArgTransA, typename ArgTransB,
          typename ArgAlgo>
struct TeamGemm {
  template <typename ScalarType, typename AViewType, typename BViewType,
            typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member, const ScalarType alpha, const AViewType &A,
      const BViewType &B, const ScalarType beta, const CViewType &C);
};

///
/// TeamVector Gemm
///

template <typename MemberType, typename ArgTransA, typename ArgTransB,
          typename ArgAlgo>
struct TeamVectorGemm {
  template <typename ScalarType, typename AViewType, typename BViewType,
            typename CViewType>
  KOKKOS_INLINE_FUNCTION static int invoke(
      const MemberType &member, const ScalarType alpha, const AViewType &A,
      const BViewType &B, const ScalarType beta, const CViewType &C);
};

///
/// Selective Interface
///
template <typename MemberType, typename ArgTransA, typename ArgTransB,
          typename ArgMode, typename ArgAlgo>
struct Gemm {
  template <typename ScalarType, typename AViewType, typename BViewType,
            typename CViewType>
  KOKKOS_FORCEINLINE_FUNCTION static int invoke(
      const MemberType &member, const ScalarType alpha, const AViewType &A,
      const BViewType &B, const ScalarType beta, const CViewType &C) {
    int r_val = 0;
    if (std::is_same<ArgMode, Mode::Serial>::value) {
      r_val = SerialGemm<ArgTransA, ArgTransB, ArgAlgo>::invoke(alpha, A, B,
                                                                beta, C);
    } else if (std::is_same<ArgMode, Mode::Team>::value) {
      r_val = TeamGemm<MemberType, ArgTransA, ArgTransB, ArgAlgo>::invoke(
          member, alpha, A, B, beta, C);
    }
    return r_val;
  }
};
/********************* END functor-level routines *********************/

/********************* BEGIN non-functor-level routines *********************/

/********************* BEGIN forward declarations *********************/
template <class ArgTransA, class ArgTransB, class ArgMode, class ArgBatchSzDim,
          class ArgResultsPerThread, class ScalarType, class AViewType,
          class BViewType, class CViewType>
class BatchedSerialGemm;
/********************* END forward declarations *********************/

// clang-format off
/// \brief Non-blocking solve of general matrix multiply on a batch of
/// uniform matrices.
///
/// Note: If a TPL is selected, this interface follows the blocking
/// behavior (either blocking or non-blocking) of the TPL vendor's API.
///
/// Note: To leverage SIMD instructions, 4-rank views must be selected via the
/// template parameters documented below.
///
///        C = alpha * op(A) * op(B) + beta * C
///
/// \tparam ArgTransA      Specifies what op does to A:
///                        Trans::NoTranspose   for non-transpose
///                        Trans::Transpose     for transpose
///                        Trans::ConjTranspose for conjugate transpose
/// \tparam ArgTransB      Specifies what op does to B:
///                        Trans::NoTranspose   for non-transpose
///                        Trans::Transpose     for transpose
///                        Trans::ConjTranspose for conjugate transpose
/// \tparam ArgBatchSzDim  Specifies where the batch dimension is allocated in
///                        AViewType, BViewType, and CViewType:
///                        BatchSzDim::Left  Batch dimension is leftmost
///                        BatchSzDim::Right Batch dimension is rightmost
/// \tparam ScalarType     Specifies the scalar type of alpha and beta
/// \tparam AViewType      Input matrix, as either a 3-rank Kokkos::View or a
///                        4-rank Kokkos::View for SIMD operations.
/// \tparam BViewType      Input matrix, as either a 3-rank Kokkos::View or a
///                        4-rank Kokkos::View for SIMD operations.
/// \tparam CViewType      Input(RHS)/Output(LHS) matrix, as either a 3-rank
///                        Kokkos::View or a 4-rank Kokkos::View for SIMD
///                        operations.
///
/// \param handle [in]     A handle which specifies how to invoke the batched
///                        gemm.
///                        See struct BatchedGemmHandle for details.
/// \param alpha [in]      Input coefficient used for multiplication with A
/// \param A [in]          Input matrix, as a 3-rank Kokkos::View
///                        If ArgBatchSzDim == "BatchSzDim::Right", matrix A is MxKxB
///                        If ArgBatchSzDim == "BatchSzDim::Left",  matrix A is BxMxK
/// \param B [in]          Input matrix, as a 3-rank Kokkos::View
///                        If ArgBatchSzDim == "BatchSzDim::Right", matrix B is KxNxB
///                        If ArgBatchSzDim == "BatchSzDim::Left",  matrix B is BxKxN
/// \param beta [in]       Input coefficient used for multiplication with C
/// \param C [in/out]      Input/Output matrix, as a 3-rank Kokkos::View
///                        If ArgBatchSzDim == "BatchSzDim::Right", matrix C is MxNxB
///                        If ArgBatchSzDim == "BatchSzDim::Left",  matrix C is BxMxN
/// \return 0 upon success, non-zero otherwise
// clang-format on
template <typename ArgTransA, typename ArgTransB, typename ArgBatchSzDim,
          typename BatchedGemmHandleType, typename ScalarType,
          typename AViewType, typename BViewType, typename CViewType>
int BatchedGemm(const BatchedGemmHandleType *handle, const ScalarType alpha,
                const AViewType &A, const BViewType &B, const ScalarType beta,
                const CViewType &C) {
  int ret             = 0;
  using ViewValueType = typename CViewType::value_type;
  // Check for valid input views
  static_assert(Kokkos::Impl::is_view<AViewType>::value,
                "AViewType must be a Kokkos::View.");
  static_assert(Kokkos::Impl::is_view<BViewType>::value,
                "BViewType must be a Kokkos::View.");
  static_assert(Kokkos::Impl::is_view<CViewType>::value,
                "CViewType must be a Kokkos::View.");
  if (is_vector<ViewValueType>::value) {
    // Check ranks of view with underlying SIMD value types
    // For SIMD views, we can have either 3-rank or 4-ranks inputs.
    switch (handle->get_kernel_algo_type()) {
      case BaseKokkosBatchedAlgos::KK_SERIAL:
        static_assert(static_cast<int>(AViewType::rank) == 3,
                      "AViewType must have rank 3.");
        static_assert(static_cast<int>(BViewType::rank) == 3,
                      "BViewType must have rank 3.");
        static_assert(static_cast<int>(CViewType::rank) == 3,
                      "CViewType must have rank 3.");
        break;

        // TODO: check this once KK_TEAM is supported
        //        case GemmKokkosBatchedAlgos::KK_TEAM:
        //          static_assert(static_cast<int>(AViewType::rank) == 4,
        //                        "AViewType must have rank 4.");
        //          static_assert(static_cast<int>(BViewType::rank) == 4,
        //                        "BViewType must have rank 4.");
        //          static_assert(static_cast<int>(CViewType::rank) == 4,
        //                        "CViewType must have rank 4.");
        //          break;

      default:
        std::ostringstream os;
        os << "KokkosBatched::BatchedGemm does not support kernelAlgoType = "
           << std::to_string(handle->get_kernel_algo_type())
           << " with SIMD views." << std::endl;
        Kokkos::Impl::throw_runtime_exception(os.str());
        break;
    }
  } else {
    // Check ranks of views with underlying scalar value types
    static_assert(static_cast<int>(AViewType::rank) == 3,
                  "AViewType must have rank 3.");
    static_assert(static_cast<int>(BViewType::rank) == 3,
                  "BViewType must have rank 3.");
    static_assert(static_cast<int>(CViewType::rank) == 3,
                  "CViewType must have rank 3.");
  }

  // Check for valid data access patterns
  // Skip checking a_layout == b_layout == c_layout
  // Skip checking for LayoutStride
  using c_layout = typename CViewType::array_layout;
  if (std::is_same<c_layout, Kokkos::LayoutLeft>::value &&
      !std::is_same<ArgBatchSzDim, BatchLayout::Right>::value) {
    throw std::runtime_error(
        "Error: LayoutLeft views require BatchLayout::Right");
  }
  if (std::is_same<c_layout, Kokkos::LayoutRight>::value &&
      !std::is_same<ArgBatchSzDim, BatchLayout::Left>::value) {
    throw std::runtime_error(
        "Error: LayoutRight views require BatchLayout::Left");
  }

  // Begin checking conditions for optimal BatchedGemm invocation.
  using view_scalar_type   = typename CViewType::value_type;
  constexpr bool is_vector = KokkosBatched::is_vector<view_scalar_type>::value;
#if defined(KOKKOS_ENABLE_CUDA)
  constexpr bool on_gpu = KokkosKernels::Impl::kk_is_gpu_exec_space<
      typename CViewType::execution_space>();
#endif  // KOKKOS_ENABLE_CUDA
#if defined(KOKKOS_ENABLE_HIP)
  constexpr bool on_gpu = std::is_same<typename CViewType::execution_space,
                                       Kokkos::Experimental::HIP>::value;
#endif  // KOKKOS_ENABLE_HIP

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
  constexpr bool on_gpu = false;
#endif

#if __x86_64__
  constexpr bool on_x86_64 =
      std::is_same<typename CViewType::execution_space::memory_space,
                   Kokkos::HostSpace>::value;
#else
  constexpr bool on_x86_64 = false;
#endif  // Intel architectures

#if defined(__ARM_ARCH_ISA_A64)
  constexpr bool on_a64fx =
      std::is_same<typename CViewType::execution_space::memory_space,
                   Kokkos::HostSpace>::value;
#else
  constexpr bool on_a64fx = false;
#endif

  // Select whether to calculate a rank-0 or rank-2 result per thread
  using resultsPerThread =
      typename std::conditional<!is_vector && on_gpu, ResultsPerThread::Rank0,
                                ResultsPerThread::Rank2>::type;

  // Selects whether to use the register blocking or non-register blocking
  // implementation of SerialGemm.
  using mode_type = typename std::conditional<
      is_vector,
      typename std::conditional<on_gpu || on_x86_64, Algo::Gemm::Blocked,
                                Algo::Gemm::Unblocked>::type,
      typename std::conditional<
          on_gpu, Algo::Gemm::Unblocked,
          typename std::conditional<on_a64fx, Algo::Gemm::Unblocked,
                                    Algo::Gemm::Blocked>::type>::type>::type;

  if (handle->enableDebug) {
    std::cout << "view_scalar_type:" << typeid(view_scalar_type).name()
              << std::endl
              << "execution_space:"
              << typeid(typename CViewType::execution_space).name() << std::endl
              << "resultsPerThread:" << typeid(resultsPerThread).name()
              << std::endl
              << "mode_type:" << typeid(mode_type).name() << std::endl
              << "is_vector:" << is_vector << std::endl
              << "on_gpu:" << on_gpu << std::endl
              << "on_x86_64:" << on_x86_64 << std::endl
              << "on_a64fx:" << on_a64fx << std::endl;
  }

  switch (handle->get_kernel_algo_type()) {
    case BaseKokkosBatchedAlgos::KK_SERIAL:
      ret = BatchedSerialGemm<ArgTransA, ArgTransB, mode_type, ArgBatchSzDim,
                              resultsPerThread, ScalarType, AViewType,
                              BViewType, CViewType>(alpha, A, B, beta, C)
                .invoke();
      break;

    ////////////// HEURISTIC ALGOS //////////////
    case BaseHeuristicAlgos::SQUARE:

    case BaseHeuristicAlgos::TALL:

    case BaseHeuristicAlgos::WIDE:

    ////////////// TPL ALGOS //////////////
    case BaseTplAlgos::ARMPL:

    case BaseTplAlgos::MKL:

    case GemmTplAlgos::CUBLAS:

    case GemmTplAlgos::MAGMA:

      ////////////// KokkosBatched ALGOS //////////////

    case GemmKokkosBatchedAlgos::KK_TEAM:

    case GemmKokkosBatchedAlgos::KK_TEAMVECTOR:

    case GemmKokkosBatchedAlgos::KK_SERIALSIMD:

    case GemmKokkosBatchedAlgos::KK_TEAMSIMD:

    case GemmKokkosBatchedAlgos::KK_SERIAL_OPT2:

    case GemmKokkosBatchedAlgos::KK_TEAMVECTOR_SHMEM:

    case GemmKokkosBatchedAlgos::KK_TEAMVECTOR_DBLBUF:

    default:
      std::ostringstream os;
      os << "KokkosBatched::BatchedGemm does not support kernelAlgoType = "
         << std::to_string(handle->get_kernel_algo_type()) << "." << std::endl;
      Kokkos::Impl::throw_runtime_exception(os.str());
      break;
  }
  return ret;
}
/********************* END non-functor-level routines *********************/
}  // namespace KokkosBatched

#include "KokkosBatched_Gemm_Serial_Impl.hpp"
#include "KokkosBatched_Gemm_Team_Impl.hpp"
#include "KokkosBatched_Gemm_TeamVector_Impl.hpp"

#endif
