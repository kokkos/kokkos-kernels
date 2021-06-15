#ifndef __KOKKOSBATCHED_GEMM_DECL_HPP__
#define __KOKKOSBATCHED_GEMM_DECL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Vector.hpp"

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
namespace Experimental {

// Forward declare BatchedSerialGemm
template <typename ArgTransA, typename ArgTransB, typename ArgMode,
          typename ArgBatchLayout, typename ArgResultsPerThread>
struct BatchedSerialGemm;

/// \brief Non-blocking solve of general matrix multiply on a batch of
/// matrices.
///        C = alpha * op(A) * op(B) + beta * C
///
/// \tparam ArgTransA      Specifies what op does to A:
///                        Trans::NoTranspose for non-transpose
///                        Trans::Transpose for transpose
///                        Trans::ConjTranspose for conjugate transpose
/// \tparam ArgTransB      Specifies what op does to B:
///                        Trans::NoTranspose   for non-transpose
///                        Trans::Transpose     for transpose
///                        Trans::ConjTranspose for conjugate transpose
/// \tparam ArgBatchLayout Specifies where the batch dimension is allocated in
///                        AViewType, BViewType, and CViewType:
///                        BatchLayout::Left  Batch dimension is the leftmost
///                        dimension BatchLayout::Right Batch dimension is the
///                        rightmost dimension
/// \tparam ScalarType     Specifies the scalar type of alpha and beta
/// \tparam AViewType      Input matrix, as a 3-rank Kokkos::View
/// \tparam BViewType      Input matrix, as a 3-rank Kokkos::View
/// \tparam CViewType      Input(RHS)/Output(LHS) matrix, as a 3-rank
/// Kokkos::View
///
/// \param alpha [in]      Input coefficient used for multiplication with A
/// \param A [in]          Input matrix, as a 3-rank Kokkos::View
///                        If ArgBatchLayout == "BatchLayout::Right", matrix A
///                        is MxKxB If ArgBatchLayout == "BatchLayout::Left",
///                        matrix A is BxMxK
/// \param B [in]          Input matrix, as a 3-rank Kokkos::View
///                        If ArgBatchLayout == "BatchLayout::Right", matrix B
///                        is KxNxB
/// \param beta [in]       Input coefficient used for multiplication with C
///                        If ArgBatchLayout == "BatchLayout::Left",  matrix A
///                        is BxKxN
/// \param C [in/out]      Input/Output matrix, as a 3-rank Kokkos::View
///                        If ArgBatchLayout == "BatchLayout::Right", matrix C
///                        is MxNxB If ArgBatchLayout == "BatchLayout::Left",
///                        matrix C is BxMxN
/// \return 0 upon success, non-zero otherwise
template <typename ArgTransA, typename ArgTransB, typename ArgBatchLayout>
struct BatchedGemm {
  template <typename ScalarType, typename AViewType, typename BViewType,
            typename CViewType>
  static int invoke(const ScalarType alpha, const AViewType &A,
                    const BViewType &B, const ScalarType beta,
                    const CViewType &C) {
    int ret = 0;
    // Check for valid input views
    static_assert(Kokkos::Impl::is_view<AViewType>::value,
                  "AViewType must be a Kokkos::View.");
    static_assert(Kokkos::Impl::is_view<BViewType>::value,
                  "BViewType must be a Kokkos::View.");
    static_assert(Kokkos::Impl::is_view<CViewType>::value,
                  "CViewType must be a Kokkos::View.");
    static_assert(static_cast<int>(AViewType::rank) == 3,
                  "AViewType must have rank 3.");
    static_assert(static_cast<int>(BViewType::rank) == 3,
                  "BViewType must have rank 3.");
    static_assert(static_cast<int>(CViewType::rank) == 3,
                  "CViewType must have rank 3.");

    // Check for valid data access patterns
    // Skip checking a_layout == b_layout == c_layout
    // Skip checking for LayoutStride
    using c_layout = typename CViewType::array_layout;
    if (std::is_same<c_layout, Kokkos::LayoutLeft>::value &&
        !std::is_same<ArgBatchLayout, BatchLayout::Right>::value) {
      throw std::runtime_error(
          "Error: LayoutLeft views require BatchLayout::Right");
    }
    if (std::is_same<c_layout, Kokkos::LayoutRight>::value &&
        !std::is_same<ArgBatchLayout, BatchLayout::Left>::value) {
      throw std::runtime_error(
          "Error: LayoutRight views require BatchLayout::Left");
    }

    // Begin checking conditions for optimal BatchedGemm invocation.
    using view_scalar_type = typename CViewType::value_type;
    constexpr bool is_vector =
        KokkosBatched::is_vector<view_scalar_type>::value;
#if defined(KOKKOS_ENABLE_CUDA)
    constexpr bool on_gpu =
        std::is_same<typename CViewType::execution_space, Kokkos::Cuda>::value;
#endif  // KOKKOS_ENABLE_CUDA
#if defined(KOKKOS_ENABLE_HIP)
    constexpr bool on_gpu = std::is_same<typename CViewType::execution_space,
                                         Kokkos::Experimental::HIP>::value;
#endif  // KOKKOS_ENABLE_HIP

#if !defined(KOKKOS_ENABLE_CUDA) && !defined(KOKKOS_ENABLE_HIP)
    constexpr bool on_gpu = false;
#endif

#if __x86_64__
    constexpr bool on_intel =
        std::is_same<typename CViewType::execution_space::memory_space,
                     Kokkos::HostSpace>::value;
#else
    constexpr bool on_intel = false;
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
        typename std::conditional<on_gpu || on_intel, Algo::Gemm::Blocked,
                                  Algo::Gemm::Unblocked>::type,
        typename std::conditional<
            on_gpu, Algo::Gemm::Unblocked,
            typename std::conditional<on_a64fx, Algo::Gemm::Unblocked,
                                      Algo::Gemm::Blocked>::type>::type>::type;

#if 0
      std::cout << "view_scalar_type:" << typeid(view_scalar_type).name() << std::endl <<
  	        "execution_space:" << typeid(typename CViewType::execution_space).name() << std::endl <<
                "resultsPerThread:" << typeid(resultsPerThread).name() << std::endl <<
                "mode_type:" << typeid(mode_type).name() << std::endl <<
                "is_vector:" << is_vector << std::endl <<
                "on_gpu:" << on_gpu << std::endl <<
                "on_intel:" << on_intel << std::endl <<
                "on_a64fx:" << on_a64fx << std::endl;
#endif
    ret = BatchedSerialGemm<ArgTransA, ArgTransB, mode_type, ArgBatchLayout,
                            resultsPerThread>::invoke(alpha, A, B, beta, C);
    return ret;
  }
};
/********************* END non-functor-level routines *********************/
}  // namespace Experimental
}  // namespace KokkosBatched

#include "KokkosBatched_Gemm_Serial_Impl.hpp"
#include "KokkosBatched_Gemm_Team_Impl.hpp"
#include "KokkosBatched_Gemm_TeamVector_Impl.hpp"

#endif
