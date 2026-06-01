// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#ifndef KOKKOSBATCHED_NRM_INTERNAL_HPP_
#define KOKKOSBATCHED_NRM_INTERNAL_HPP_

#include <KokkosBatched_Util.hpp>

namespace KokkosBatched {
namespace Impl {

template <typename NrmValueType, typename ValueType>
KOKKOS_INLINE_FUNCTION NrmValueType asum(const ValueType &x) {
  if constexpr (KokkosKernels::ArithTraits<ValueType>::is_complex) {
    return KokkosKernels::ArithTraits<NrmValueType>::abs(KokkosKernels::ArithTraits<ValueType>::real(x)) +
           KokkosKernels::ArithTraits<NrmValueType>::abs(KokkosKernels::ArithTraits<ValueType>::imag(x));
  } else {
    return KokkosKernels::ArithTraits<NrmValueType>::abs(x);
  }
}

template <typename NrmValueType, typename ValueType>
KOKKOS_INLINE_FUNCTION NrmValueType square(const ValueType &x) {
  if constexpr (KokkosKernels::ArithTraits<ValueType>::is_complex) {
    return KokkosKernels::ArithTraits<NrmValueType>::real(KokkosKernels::ArithTraits<ValueType>::conj(x) * x);
  } else {
    return x * x;
  }
}

///
/// Serial Internal Impl
/// ====================
template <typename NrmType>
struct SerialNrmInternal {
  template <typename ValueType, typename NrmValueType>
  KOKKOS_INLINE_FUNCTION static void invoke(const int n, const ValueType *KOKKOS_RESTRICT x, const int xs0,
                                            NrmValueType *KOKKOS_RESTRICT norm);
};

template <typename NrmType>
template <typename ValueType, typename NrmValueType>
KOKKOS_INLINE_FUNCTION void SerialNrmInternal<NrmType>::invoke(const int n, const ValueType *KOKKOS_RESTRICT x,
                                                               const int xs0, NrmValueType *KOKKOS_RESTRICT norm) {
  NrmValueType nrm = 0;

  if constexpr (std::is_same_v<NrmType, Norm::L1>) {
    for (int i = 0; i < n; ++i) {
      nrm += asum<NrmValueType>(x[i * xs0]);
    }
  } else if constexpr (std::is_same_v<NrmType, Norm::L2>) {
    for (int i = 0; i < n; ++i) {
      nrm += square<NrmValueType>(x[i * xs0]);
    }
    nrm = KokkosKernels::ArithTraits<NrmValueType>::sqrt(nrm);
  } else if constexpr (std::is_same_v<NrmType, Norm::LInf>) {
    for (int i = 0; i < n; ++i) {
      const NrmValueType abs_val = KokkosKernels::ArithTraits<ValueType>::abs(x[i * xs0]);
      if (abs_val > nrm) nrm = abs_val;
    }
  }

  *norm = nrm;
}

///
/// Team Internal Impl
/// ==================
template <typename MemberType, typename NrmType>
struct TeamNrmInternal {
  template <typename ValueType, typename NrmValueType>
  KOKKOS_INLINE_FUNCTION static void invoke(const MemberType &member, const int n, const ValueType *KOKKOS_RESTRICT x,
                                            const int xs0, NrmValueType *KOKKOS_RESTRICT norm);
};

template <typename MemberType, typename NrmType>
template <typename ValueType, typename NrmValueType>
KOKKOS_INLINE_FUNCTION void TeamNrmInternal<MemberType, NrmType>::invoke(const MemberType &member, const int n,
                                                                         const ValueType *KOKKOS_RESTRICT x,
                                                                         const int xs0,
                                                                         NrmValueType *KOKKOS_RESTRICT norm) {
  NrmValueType nrm = 0;

  if constexpr (std::is_same_v<NrmType, Norm::L1>) {
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, n),
        [&](const int i, NrmValueType &thread_nrm) { thread_nrm += asum<NrmValueType>(x[i * xs0]); }, nrm);
  } else if constexpr (std::is_same_v<NrmType, Norm::L2>) {
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, n),
        [&](const int i, NrmValueType &thread_nrm) { thread_nrm += square<NrmValueType>(x[i * xs0]); }, nrm);
    nrm = KokkosKernels::ArithTraits<NrmValueType>::sqrt(nrm);
  } else if constexpr (std::is_same_v<NrmType, Norm::LInf>) {
    Kokkos::Max<NrmValueType, typename MemberType::execution_space> max_nrm(nrm);
    Kokkos::parallel_reduce(
        Kokkos::TeamThreadRange(member, n),
        [&](const int i, NrmValueType &thread_nrm) {
          const NrmValueType abs_val = KokkosKernels::ArithTraits<ValueType>::abs(x[i * xs0]);
          if (abs_val > thread_nrm) thread_nrm = abs_val;
        },
        max_nrm);
  }

  *norm = nrm;
}

///
/// TeamVector Internal Impl
/// ==================
template <typename MemberType, typename NrmType>
struct TeamVectorNrmInternal {
  template <typename ValueType, typename NrmValueType>
  KOKKOS_INLINE_FUNCTION static void invoke(const MemberType &member, const int n, const ValueType *KOKKOS_RESTRICT x,
                                            const int xs0, NrmValueType *KOKKOS_RESTRICT norm);
};

template <typename MemberType, typename NrmType>
template <typename ValueType, typename NrmValueType>
KOKKOS_INLINE_FUNCTION void TeamVectorNrmInternal<MemberType, NrmType>::invoke(const MemberType &member, const int n,
                                                                               const ValueType *KOKKOS_RESTRICT x,
                                                                               const int xs0,
                                                                               NrmValueType *KOKKOS_RESTRICT norm) {
  NrmValueType nrm = 0;

  if constexpr (std::is_same_v<NrmType, Norm::L1>) {
    Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, n),
        [&](const int i, NrmValueType &thread_nrm) { thread_nrm += asum<NrmValueType>(x[i * xs0]); }, nrm);
  } else if constexpr (std::is_same_v<NrmType, Norm::L2>) {
    Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, n),
        [&](const int i, NrmValueType &thread_nrm) { thread_nrm += square<NrmValueType>(x[i * xs0]); }, nrm);
    nrm = KokkosKernels::ArithTraits<NrmValueType>::sqrt(nrm);
  } else if constexpr (std::is_same_v<NrmType, Norm::LInf>) {
    Kokkos::Max<NrmValueType, typename MemberType::execution_space> max_nrm(nrm);
    Kokkos::parallel_reduce(
        Kokkos::TeamVectorRange(member, n),
        [&](const int i, NrmValueType &thread_nrm) {
          const NrmValueType abs_val = KokkosKernels::ArithTraits<ValueType>::abs(x[i * xs0]);
          if (abs_val > thread_nrm) thread_nrm = abs_val;
        },
        max_nrm);
  }
  *norm = nrm;
}

}  // namespace Impl
}  // namespace KokkosBatched

#endif  // KOKKOSBATCHED_NRM_INTERNAL_HPP_
