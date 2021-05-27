#ifndef __KOKKOSBATCHED_GEMM_SERIAL_IMPL_HPP__
#define __KOKKOSBATCHED_GEMM_SERIAL_IMPL_HPP__

/// \author Kyungjoo Kim (kyukim@sandia.gov)

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Gemm_Serial_Internal.hpp"

namespace KokkosBatched {
/********************* BEGIN functor-level routines *********************/
///
/// Serial Impl
/// ===========

///
/// Implemented:
/// NT/NT, T/NT, NT/T, T/T
///
/// Not yet immplemented (ConjTranspose):
/// CT/NT, NT/CT, CT/CT
///

///
/// NT/NT
///

#if defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL__) &&         \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_BATCHED__) && \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_COMPACT_BATCHED__)
template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::NoTranspose,
           Algo::Gemm::CompactMKL>::invoke(const ScalarType alpha,
                                           const AViewType &A,
                                           const BViewType &B,
                                           const ScalarType beta,
                                           const CViewType &C) {
  typedef typename CViewType::value_type vector_type;
  // typedef typename vector_type::value_type value_type;

  const int m = C.extent(0), n = C.extent(1), k = A.extent(1);

  static_assert(is_vector<vector_type>::value, "value type is not vector type");
  static_assert(
      vector_type::vector_length == 4 || vector_type::vector_length == 8,
      "AVX, AVX2 and AVX512 is supported");
  const MKL_COMPACT_PACK format =
      vector_type::vector_length == 8 ? MKL_COMPACT_AVX512 : MKL_COMPACT_AVX;

  // no error check
  int r_val = 0;
  if (A.stride_0() == 1 && B.stride_0() == 1 && C.stride_0() == 1) {
    mkl_dgemm_compact(MKL_COL_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_1(),
                      (const double *)B.data(), B.stride_1(), beta,
                      (double *)C.data(), C.stride_1(), format,
                      (MKL_INT)vector_type::vector_length);
  } else if (A.stride_1() == 1 && B.stride_1() == 1 && C.stride_1() == 1) {
    mkl_dgemm_compact(MKL_ROW_MAJOR, MKL_NOTRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_0(),
                      (const double *)B.data(), B.stride_0(), beta,
                      (double *)C.data(), C.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else {
    r_val = -1;
  }
  return r_val;
}
#endif

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::NoTranspose,
           Algo::Gemm::Unblocked>::invoke(const ScalarType alpha,
                                          const AViewType &A,
                                          const BViewType &B,
                                          const ScalarType beta,
                                          const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return SerialGemmInternal<Algo::Gemm::Unblocked>::invoke(
      C.extent(0), C.extent(1), A.extent(1), alpha, A.data(), A.stride_0(),
      A.stride_1(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::NoTranspose, Algo::Gemm::Blocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return SerialGemmInternal<Algo::Gemm::Blocked>::invoke(
      C.extent(0), C.extent(1), A.extent(1), alpha, A.data(), A.stride_0(),
      A.stride_1(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

///
/// T/NT
///

#if defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL__) &&         \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_BATCHED__) && \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_COMPACT_BATCHED__)
template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::NoTranspose,
           Algo::Gemm::CompactMKL>::invoke(const ScalarType alpha,
                                           const AViewType &A,
                                           const BViewType &B,
                                           const ScalarType beta,
                                           const CViewType &C) {
  typedef typename CViewType::value_type vector_type;
  // typedef typename vector_type::value_type value_type;

  const int m = C.extent(0), n = C.extent(1), k = A.extent(0);

  static_assert(is_vector<vector_type>::value, "value type is not vector type");
  static_assert(
      vector_type::vector_length == 4 || vector_type::vector_length == 8,
      "AVX, AVX2 and AVX512 is supported");
  const MKL_COMPACT_PACK format =
      vector_type::vector_length == 8 ? MKL_COMPACT_AVX512 : MKL_COMPACT_AVX;

  // no error check
  int r_val = 0;
  if (A.stride_0() == 1 && B.stride_0() == 1 && C.stride_0() == 1) {
    mkl_dgemm_compact(MKL_COL_MAJOR, MKL_TRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_1(),
                      (const double *)B.data(), B.stride_1(), beta,
                      (double *)C.data(), C.stride_1(), format,
                      (MKL_INT)vector_type::vector_length);
  } else if (A.stride_1() == 1 && B.stride_1() == 1 && C.stride_1() == 1) {
    mkl_dgemm_compact(MKL_ROW_MAJOR, MKL_TRANS, MKL_NOTRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_0(),
                      (const double *)B.data(), B.stride_0(), beta,
                      (double *)C.data(), C.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else {
    r_val = -1;
  }
  return r_val;
}
#endif

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Unblocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return SerialGemmInternal<Algo::Gemm::Unblocked>::invoke(
      C.extent(0), C.extent(1), A.extent(0), alpha, A.data(), A.stride_1(),
      A.stride_0(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::NoTranspose, Algo::Gemm::Blocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return SerialGemmInternal<Algo::Gemm::Blocked>::invoke(
      C.extent(0), C.extent(1), A.extent(0), alpha, A.data(), A.stride_1(),
      A.stride_0(), B.data(), B.stride_0(), B.stride_1(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

///
/// NT/T
///

#if defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL__) &&         \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_BATCHED__) && \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_COMPACT_BATCHED__)
template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::Transpose,
           Algo::Gemm::CompactMKL>::invoke(const ScalarType alpha,
                                           const AViewType &A,
                                           const BViewType &B,
                                           const ScalarType beta,
                                           const CViewType &C) {
  typedef typename CViewType::value_type vector_type;
  // typedef typename vector_type::value_type value_type;

  const int m = C.extent(0), n = C.extent(1), k = A.extent(1);

  static_assert(is_vector<vector_type>::value, "value type is not vector type");
  static_assert(
      vector_type::vector_length == 4 || vector_type::vector_length == 8,
      "AVX, AVX2 and AVX512 is supported");
  const MKL_COMPACT_PACK format =
      vector_type::vector_length == 8 ? MKL_COMPACT_AVX512 : MKL_COMPACT_AVX;

  // no error check
  int r_val = 0;
  if (A.stride_0() == 1 && B.stride_0() == 1 && C.stride_0() == 1) {
    mkl_dgemm_compact(MKL_COL_MAJOR, MKL_NOTRANS, MKL_TRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_1(),
                      (const double *)B.data(), B.stride_1(), beta,
                      (double *)C.data(), C.stride_1(), format,
                      (MKL_INT)vector_type::vector_length);
  } else if (A.stride_1() == 1 && B.stride_1() == 1 && C.stride_1() == 1) {
    mkl_dgemm_compact(MKL_ROW_MAJOR, MKL_NOTRANS, MKL_TRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_0(),
                      (const double *)B.data(), B.stride_0(), beta,
                      (double *)C.data(), C.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else {
    r_val = -1;
  }
  return r_val;
}
#endif

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Unblocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return SerialGemmInternal<Algo::Gemm::Unblocked>::invoke(
      C.extent(0), C.extent(1), A.extent(1), alpha, A.data(), A.stride_0(),
      A.stride_1(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::NoTranspose, Trans::Transpose, Algo::Gemm::Blocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return SerialGemmInternal<Algo::Gemm::Blocked>::invoke(
      C.extent(0), C.extent(1), A.extent(1), alpha, A.data(), A.stride_0(),
      A.stride_1(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

///
/// T/T
///

#if defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL__) &&         \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_BATCHED__) && \
    defined(__KOKKOSBATCHED_ENABLE_INTEL_MKL_COMPACT_BATCHED__)
template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::Transpose, Algo::Gemm::CompactMKL>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  typedef typename CViewType::value_type vector_type;
  // typedef typename vector_type::value_type value_type;

  const int m = C.extent(0), n = C.extent(1), k = A.extent(0);

  static_assert(is_vector<vector_type>::value, "value type is not vector type");
  static_assert(
      vector_type::vector_length == 4 || vector_type::vector_length == 8,
      "AVX, AVX2 and AVX512 is supported");
  const MKL_COMPACT_PACK format =
      vector_type::vector_length == 8 ? MKL_COMPACT_AVX512 : MKL_COMPACT_AVX;

  // no error check
  int r_val = 0;
  if (A.stride_0() == 1 && B.stride_0() == 1 && C.stride_0() == 1) {
    mkl_dgemm_compact(MKL_COL_MAJOR, MKL_TRANS, MKL_TRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_1(),
                      (const double *)B.data(), B.stride_1(), beta,
                      (double *)C.data(), C.stride_1(), format,
                      (MKL_INT)vector_type::vector_length);
  } else if (A.stride_1() == 1 && B.stride_1() == 1 && C.stride_1() == 1) {
    mkl_dgemm_compact(MKL_ROW_MAJOR, MKL_TRANS, MKL_TRANS, m, n, k, alpha,
                      (const double *)A.data(), A.stride_0(),
                      (const double *)B.data(), B.stride_0(), beta,
                      (double *)C.data(), C.stride_0(), format,
                      (MKL_INT)vector_type::vector_length);
  } else {
    r_val = -1;
  }
  return r_val;
}
#endif

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::Transpose, Algo::Gemm::Unblocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return SerialGemmInternal<Algo::Gemm::Unblocked>::invoke(
      C.extent(0), C.extent(1), A.extent(0), alpha, A.data(), A.stride_1(),
      A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(),
      C.stride_0(), C.stride_1());
}

template <>
template <typename ScalarType, typename AViewType, typename BViewType,
          typename CViewType>
KOKKOS_INLINE_FUNCTION int
SerialGemm<Trans::Transpose, Trans::Transpose, Algo::Gemm::Blocked>::invoke(
    const ScalarType alpha, const AViewType &A, const BViewType &B,
    const ScalarType beta, const CViewType &C) {
  // C = beta C + alpha A B
  // C (m x n), A(m x k), B(k x n)
  return SerialGemmInternal<Algo::Gemm::Blocked>::invoke(
      C.extent(0), C.extent(1), A.extent(0), alpha, A.data(), A.stride_1(),
      A.stride_0(), B.data(), B.stride_1(), B.stride_0(), beta, C.data(),
      C.stride_0(), C.stride_1());
}
/********************* END functor-level routines *********************/

/********************* BEGIN non-functor-level routines *********************/
namespace Experimental {
template <class ScalarType, class AViewType, class BViewType, class CViewType,
          class ArgTransA, class ArgTransB, class ArgMode, class ArgBatchLayout,
          class ArgResultsPerThread>
struct BatchedSerialGemmFunctor {
  AViewType A;
  BViewType B;
  CViewType C;
  ScalarType alpha, beta;
  size_t divisor, c_cols, batch_size;
  ArgBatchLayout batch_layout_tag;
  ArgTransA transA_tag;
  ArgTransB transB_tag;

  BatchedSerialGemmFunctor(ScalarType _alpha, AViewType _A, BViewType _B,
                           ScalarType _beta, CViewType _C)
      : A(_A), B(_B), C(_C), alpha(_alpha), beta(_beta) {}

  // subview_wrapper overloads for handling 3-rank BatchLayout::Left views
  template <class ViewType, class IdxType1, class IdxType2, class IdxType3>
  KOKKOS_INLINE_FUNCTION auto subview_wrapper(ViewType v, IdxType1 i1,
                                              IdxType2 i2, IdxType3 i3,
                                              const BatchLayout::Left &) const {
    return Kokkos::subview(v, i1, i2, i3);
  }
  template <class ViewType, class IdxType1, class IdxType2, class IdxType3>
  KOKKOS_INLINE_FUNCTION auto subview_wrapper(
      ViewType v, IdxType1 i1, IdxType2 i2, IdxType3 i3,
      const BatchLayout::Left &layout_tag, const Trans::NoTranspose) const {
    return subview_wrapper(v, i1, i2, i3, layout_tag);
  }
  template <class ViewType, class IdxType1, class IdxType2, class IdxType3>
  KOKKOS_INLINE_FUNCTION auto subview_wrapper(
      ViewType v, IdxType1 i1, IdxType2 i2, IdxType3 i3,
      const BatchLayout::Left &layout_tag, const Trans::Transpose) const {
    return subview_wrapper(v, i1, i3, i2, layout_tag);
  }

  // subview_wrapper overloads for handling 3-rank BatchLayout::Right views
  template <class ViewType, class IdxType1, class IdxType2, class IdxType3>
  KOKKOS_INLINE_FUNCTION auto subview_wrapper(
      ViewType v, IdxType1 i1, IdxType2 i2, IdxType3 i3,
      const BatchLayout::Right &) const {
    return Kokkos::subview(v, i2, i3, i1);
  }
  template <class ViewType, class IdxType1, class IdxType2, class IdxType3>
  KOKKOS_INLINE_FUNCTION auto subview_wrapper(
      ViewType v, IdxType1 i1, IdxType2 i2, IdxType3 i3,
      const BatchLayout::Right &layout_tag, const Trans::NoTranspose &) const {
    return subview_wrapper(v, i1, i2, i3, layout_tag);
  }
  template <class ViewType, class IdxType1, class IdxType2, class IdxType3>
  KOKKOS_INLINE_FUNCTION auto subview_wrapper(
      ViewType v, IdxType1 i1, IdxType2 i2, IdxType3 i3,
      const BatchLayout::Right &layout_tag, const Trans::Transpose &) const {
    return subview_wrapper(v, i1, i3, i2, layout_tag);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ResultsPerThread::Rank0 &, const int &i) const {
    // Here, the batch_idx is strided by c_rows * c_cols
    auto batch_idx = i / divisor;
    // For every batch, we need mod in [0, c_rows*c_cols-1]
    auto mod = i % divisor;
    // For every mod, we need a column index in [0, c_cols-1]
    auto col_idx = mod % c_cols;
    // For every mod, we need a row index in [0, c_rows-1]
    auto row_idx = mod / c_cols;

    // Due to taking 1-rank subviews out, we must handle transpose here.
    // Use overloads of subview_wrapper to handle transpose at compile time.
    auto svA_row = subview_wrapper(A, batch_idx, row_idx, Kokkos::ALL(),
                                   batch_layout_tag, transA_tag);
    auto svB_col = subview_wrapper(B, batch_idx, Kokkos::ALL(), col_idx,
                                   batch_layout_tag, transB_tag);
    auto svC_ele =
        subview_wrapper(C, batch_idx, row_idx, col_idx, batch_layout_tag);

    // Kokkos::subview(scalar, ALL) or Kokkos::subview(ALL, scalar) always
    // returns a column vector. Since the subviews above handle the
    // matrix transpositions, here we must perform the GEMM on:
    // row_vec x col_vec, which is svA_row' x svB_col to compute the element
    // of C.
    KokkosBatched::SerialGemm<Trans::Transpose, Trans::NoTranspose,
                              ArgMode>::invoke(alpha, svA_row, svB_col, beta,
                                               svC_ele);
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const ResultsPerThread::Rank2 &, const int &i) const {
    auto svA =
        subview_wrapper(A, i, Kokkos::ALL(), Kokkos::ALL(), batch_layout_tag);
    auto svB =
        subview_wrapper(B, i, Kokkos::ALL(), Kokkos::ALL(), batch_layout_tag);
    auto svC =
        subview_wrapper(C, i, Kokkos::ALL(), Kokkos::ALL(), batch_layout_tag);

    KokkosBatched::SerialGemm<ArgTransA, ArgTransB, ArgMode>::invoke(
        alpha, svA, svB, beta, svC);
  }

  void run() {
    using execution_space = typename CViewType::device_type::execution_space;
    using policy_type =
        Kokkos::RangePolicy<ArgResultsPerThread, execution_space>;
    Kokkos::parallel_for("BatchedSerialGemm", policy_type(0, batch_size),
                         *this);
  }
};

template <class ArgTransA, class ArgTransB, class ArgMode, class ArgBatchLayout,
          class ArgResultsPerThread>
struct BatchedSerialGemm {
  template <class ScalarType, class AViewType, class BViewType, class CViewType>
  static int invoke(const ScalarType alpha, const AViewType &A,
                    const BViewType &B, const ScalarType beta,
                    const CViewType &C) {
    if (std::is_same<ArgResultsPerThread, ResultsPerThread::Rank0>::value) {
      BatchedSerialGemmFunctor<ScalarType, AViewType, BViewType, CViewType,
                               ArgTransA, ArgTransB, ArgMode, ArgBatchLayout,
                               ArgResultsPerThread>
          functor(alpha, A, B, beta, C);

      // Set members for ResultsPerThread::Rank0 operator; these members allow
      // each thread to calculate its C output index
      if (std::is_same<ArgBatchLayout, BatchLayout::Left>::value) {
        functor.batch_size = C.extent(0);
        functor.divisor    = C.extent(1) * C.extent(2);
        functor.c_cols     = C.extent(2);
      } else {
        functor.batch_size = C.extent(2);
        functor.divisor    = C.extent(0) * C.extent(1);
        functor.c_cols     = C.extent(1);
      }

      // Increase the number of threads by the divisor
      functor.batch_size *= functor.divisor;

      functor.run();
    } else if (std::is_same<ArgResultsPerThread,
                            ResultsPerThread::Rank2>::value) {
      using argTransA = ArgTransA;
      BatchedSerialGemmFunctor<ScalarType, AViewType, BViewType, CViewType,
                               argTransA, ArgTransB, ArgMode, ArgBatchLayout,
                               ArgResultsPerThread>
          functor(alpha, A, B, beta, C);
      if (std::is_same<ArgBatchLayout, BatchLayout::Left>::value)
        functor.batch_size = C.extent(0);
      else
        functor.batch_size = C.extent(2);

      functor.run();
    } else {
      std::cerr << "Error: ArgResultsPerThread not supported" << std::endl;
      return -1;
    }
    return 0;
  }
};
/********************* END non-functor-level routines *********************/
}  // namespace Experimental
}  // namespace KokkosBatched

#endif
