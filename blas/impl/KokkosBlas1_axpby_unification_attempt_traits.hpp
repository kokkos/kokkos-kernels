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
#ifndef KOKKOS_BLAS1_AXPBY_UNIFICATION_ATTEMPT_TRAITS_HPP_
#define KOKKOS_BLAS1_AXPBY_UNIFICATION_ATTEMPT_TRAITS_HPP_

#include <KokkosKernels_helpers.hpp>
#include <KokkosKernels_ExecSpaceUtils.hpp>
#include <sstream>

namespace KokkosBlas {
namespace Impl {

// --------------------------------

template <class T>
constexpr int typeRank() {
  if constexpr (Kokkos::is_view_v<T>) {
    return T::rank;
  }
  return -1;
}

// --------------------------------

template <class T>
constexpr typename std::enable_if<Kokkos::is_view_v<T>, bool>::type Tr0_val() {
  return (T::rank == 0);
}

template <class T>
constexpr typename std::enable_if<!Kokkos::is_view_v<T>, bool>::type Tr0_val() {
  return false;
}

// --------------------------------

template <class T>
constexpr typename std::enable_if<Kokkos::is_view_v<T>, bool>::type Tr1s_val() {
  return (T::rank == 1) && (T::rank_dynamic == 0);
}

template <class T>
constexpr typename std::enable_if<!Kokkos::is_view_v<T>, bool>::type
Tr1s_val() {
  return false;
}

// --------------------------------

template <class T>
constexpr typename std::enable_if<Kokkos::is_view_v<T>, bool>::type Tr1d_val() {
  return (T::rank == 1) && (T::rank_dynamic == 1);
}

template <class T>
constexpr typename std::enable_if<!Kokkos::is_view_v<T>, bool>::type
Tr1d_val() {
  return false;
}

// --------------------------------

template <typename T, bool Enable = false>
struct getScalarTypeFromView {
  using type = void;
};

template <typename T>
struct getScalarTypeFromView<T, true> {
  using type = typename T::value_type;
};

// --------------------------------

template <typename T, bool Enable = false>
struct getLayoutFromView {
  using type = void;
};

template <typename T>
struct getLayoutFromView<T, true> {
  using type = typename T::array_layout;
};

// --------------------------------

template <typename T>
constexpr bool isTypeComplex() {
  return (std::is_same_v<T, Kokkos::complex<float>> ||
          std::is_same_v<T, Kokkos::complex<double>> ||
          std::is_same_v<T, Kokkos::complex<long double>> ||
          std::is_same_v<T, Kokkos::complex<int>> ||
          std::is_same_v<T, Kokkos::complex<unsigned int>> ||
          std::is_same_v<T, Kokkos::complex<long int>> ||
          std::is_same_v<T, Kokkos::complex<unsigned long int>> ||
          std::is_same_v<T, Kokkos::complex<size_t>> ||
          std::is_same_v<T, Kokkos::complex<std::int32_t>> ||
          std::is_same_v<T, Kokkos::complex<std::uint32_t>> ||
          std::is_same_v<T, Kokkos::complex<std::int64_t>> ||
          std::is_same_v<T, Kokkos::complex<std::uint64_t>>);
}

// --------------------------------

template <class tExecSpace, class AV, class XMV, class BV, class YMV>
struct AxpbyUnificationAttemptTraits {
  static constexpr bool atDevCase =
      KokkosKernels::Impl::kk_is_gpu_exec_space<tExecSpace>();
  static constexpr bool atHostCase = !atDevCase;

  static constexpr bool Asc  = !Kokkos::is_view_v<AV>;
  static constexpr bool Ar0  = Tr0_val<AV>();
  static constexpr bool Ar1s = Tr1s_val<AV>();
  static constexpr bool Ar1d = Tr1d_val<AV>();
  static constexpr bool Avi  = Ar0 || Ar1s || Ar1d;

  static constexpr bool Xr1 = Kokkos::is_view_v<XMV> && (XMV::rank == 1);
  static constexpr bool Xr2 = Kokkos::is_view_v<XMV> && (XMV::rank == 2);

  static constexpr bool Bsc  = !Kokkos::is_view_v<BV>;
  static constexpr bool Br0  = Tr0_val<BV>();
  static constexpr bool Br1s = Tr1s_val<BV>();
  static constexpr bool Br1d = Tr1d_val<BV>();
  static constexpr bool Bvi  = Br0 || Br1s || Br1d;

  static constexpr bool Yr1 = Kokkos::is_view_v<YMV> && (YMV::rank == 1);
  static constexpr bool Yr2 = Kokkos::is_view_v<YMV> && (YMV::rank == 2);

  static constexpr bool xyRank1Case = Xr1 && Yr1;
  static constexpr bool xyRank2Case = Xr2 && Yr2;

  // ********************************************************************
  // In order to better understand the lines between now and right before
  // the constructor, assume that all constructor checks.
  // ********************************************************************

  // ********************************************************************
  // Declare 'AtInputScalarTypeA'
  // ********************************************************************
  using ScalarTypeA2_atDev =
      typename getScalarTypeFromView<AV, Avi && atDevCase>::type;
  using ScalarTypeA1_atDev =
      std::conditional_t<Asc && atDevCase, AV, ScalarTypeA2_atDev>;

  using ScalarTypeA2_atHost =
      typename getScalarTypeFromView<AV, Avi && atHostCase>::type;
  using ScalarTypeA1_atHost =
      std::conditional_t<Asc && atHostCase, AV, ScalarTypeA2_atHost>;

  using AtInputScalarTypeA =
      std::conditional_t<atHostCase,  // 'const' not removed if present
                         ScalarTypeA1_atHost, ScalarTypeA1_atDev>;

  using AtInputScalarTypeA_nonConst = typename std::conditional_t<
      std::is_const_v<AtInputScalarTypeA>,
      typename std::remove_const<AtInputScalarTypeA>::type, AtInputScalarTypeA>;

  static constexpr bool atInputScalarTypeA_isComplex =
      isTypeComplex<AtInputScalarTypeA_nonConst>();

  // ********************************************************************
  // Declare 'AtInputScalarTypeX'
  // ********************************************************************
  using AtInputScalarTypeX =
      typename XMV::value_type;  // 'const' not removed if present

  using AtInputScalarTypeX_nonConst = typename std::conditional_t<
      std::is_const_v<AtInputScalarTypeX>,
      typename std::remove_const<AtInputScalarTypeX>::type, AtInputScalarTypeX>;

  static constexpr bool atInputScalarTypeX_isComplex =
      isTypeComplex<AtInputScalarTypeX_nonConst>();

  // ********************************************************************
  // Declare 'AtInputScalarTypeB'
  // ********************************************************************
  using ScalarTypeB2_atDev =
      typename getScalarTypeFromView<BV, Bvi && atDevCase>::type;
  using ScalarTypeB1_atDev =
      std::conditional_t<Bsc && atDevCase, BV, ScalarTypeB2_atDev>;

  using ScalarTypeB2_atHost =
      typename getScalarTypeFromView<BV, Bvi && atHostCase>::type;
  using ScalarTypeB1_atHost =
      std::conditional_t<Bsc && atHostCase, BV, ScalarTypeB2_atHost>;

  using AtInputScalarTypeB =
      std::conditional_t<atHostCase,  // 'const' not removed if present
                         ScalarTypeB1_atHost, ScalarTypeB1_atDev>;

  using AtInputScalarTypeB_nonConst = typename std::conditional_t<
      std::is_const_v<AtInputScalarTypeB>,
      typename std::remove_const<AtInputScalarTypeB>::type, AtInputScalarTypeB>;

  static constexpr bool atInputScalarTypeB_isComplex =
      isTypeComplex<AtInputScalarTypeB_nonConst>();

  // ********************************************************************
  // Declare 'AtInputScalarTypeY'
  // ********************************************************************
  using AtInputScalarTypeY =
      typename YMV::value_type;  // 'const' not removed if present

  using AtInputScalarTypeY_nonConst = typename std::conditional_t<
      std::is_const_v<AtInputScalarTypeY>,
      typename std::remove_const<AtInputScalarTypeY>::type, AtInputScalarTypeY>;

  static constexpr bool atInputScalarTypeY_isComplex =
      isTypeComplex<AtInputScalarTypeY_nonConst>();

  // ********************************************************************
  // Declare internal layouts
  // ********************************************************************
  using InternalLayoutX =
      typename KokkosKernels::Impl::GetUnifiedLayout<XMV>::array_layout;
  using InternalLayoutY =
      typename KokkosKernels::Impl::GetUnifiedLayoutPreferring<
          YMV, InternalLayoutX>::array_layout;

  // ********************************************************************
  // Declare 'InternalTypeA_tmp'
  // ********************************************************************
  using AtInputLayoutA = typename getLayoutFromView<AV, Avi>::type;
  static constexpr bool atInputLayoutA_isStride =
      std::is_same_v<AtInputLayoutA, Kokkos::LayoutStride>;
  using InternalLayoutA =
      std::conditional_t<(Ar1d || Ar1s) && atInputLayoutA_isStride,
                         AtInputLayoutA, InternalLayoutX>;

  static constexpr bool atInputScalarTypeA_mustRemain =
      atInputScalarTypeA_isComplex && !atInputScalarTypeX_isComplex;

  using InternalScalarTypeA = std::conditional_t<
      atInputScalarTypeA_mustRemain || ((Ar1d || Ar1s) && xyRank2Case),
      AtInputScalarTypeA_nonConst  // Yes, keep the input scalar type
      ,
      AtInputScalarTypeX_nonConst  // Yes, instead of
                                   // 'AtInputScalarTypeA_nonConst'
      >;

  using InternalTypeA_atDev =
      Kokkos::View<const InternalScalarTypeA*, InternalLayoutA,
                   typename XMV::device_type,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using InternalTypeA_atHost = std::conditional_t<
      (Ar1d || Ar1s) && xyRank2Case && atHostCase,
      Kokkos::View<const InternalScalarTypeA*, InternalLayoutA,
                   typename XMV::device_type,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,
      InternalScalarTypeA>;

  using InternalTypeA_tmp =
      std::conditional_t<atHostCase, InternalTypeA_atHost, InternalTypeA_atDev>;

  // ********************************************************************
  // Declare 'InternalTypeX'
  // ********************************************************************
  using InternalTypeX = std::conditional_t<
      Xr2,
      Kokkos::View<const AtInputScalarTypeX_nonConst**, InternalLayoutX,
                   typename XMV::device_type,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,
      Kokkos::View<const AtInputScalarTypeX_nonConst*, InternalLayoutX,
                   typename XMV::device_type,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;

  // ********************************************************************
  // Declare 'InternalTypeB_tmp'
  // ********************************************************************
  using AtInputLayoutB = typename getLayoutFromView<BV, Bvi>::type;
  static constexpr bool atInputLayoutB_isStride =
      std::is_same_v<AtInputLayoutB, Kokkos::LayoutStride>;
  using InternalLayoutB =
      std::conditional_t<(Br1d || Br1s) && atInputLayoutB_isStride,
                         AtInputLayoutB, InternalLayoutY>;

  static constexpr bool atInputScalarTypeB_mustRemain =
      atInputScalarTypeB_isComplex && !atInputScalarTypeY_isComplex;

  using InternalScalarTypeB = std::conditional_t<
      atInputScalarTypeB_mustRemain || ((Br1d || Br1s) && xyRank2Case),
      AtInputScalarTypeB_nonConst  // Yes, keep the input scalar type
      ,
      AtInputScalarTypeY_nonConst  // Yes, instead of
                                   // 'AtInputScalarTypeB_nonConst'
      >;

  using InternalTypeB_atDev =
      Kokkos::View<const InternalScalarTypeB*, InternalLayoutB,
                   typename YMV::device_type,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>;

  using InternalTypeB_atHost = std::conditional_t<
      ((Br1d || Br1s) && xyRank2Case && atHostCase),
      Kokkos::View<const InternalScalarTypeB*, InternalLayoutB,
                   typename YMV::device_type,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,
      InternalScalarTypeB>;

  using InternalTypeB_tmp =
      std::conditional_t<atHostCase, InternalTypeB_atHost, InternalTypeB_atDev>;

  // ********************************************************************
  // Declare 'InternalTypeY'
  // ********************************************************************
  using InternalTypeY = std::conditional_t<
      Yr2,
      Kokkos::View<AtInputScalarTypeY_nonConst**, InternalLayoutY,
                   typename YMV::device_type,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,
      Kokkos::View<AtInputScalarTypeY_nonConst*, InternalLayoutY,
                   typename YMV::device_type,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;

  // ********************************************************************
  // Declare 'InternalTypeA': if 'InternalTypeB_tmp' is a view then
  // make sure 'InternalTypeA' is a view as well
  // ********************************************************************
  using InternalTypeA = std::conditional_t<
      !Kokkos::is_view_v<InternalTypeA_tmp> &&
          Kokkos::is_view_v<InternalTypeB_tmp>,
      Kokkos::View<const InternalScalarTypeA*, InternalLayoutA,
                   typename XMV::device_type,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,
      InternalTypeA_tmp>;

  // ********************************************************************
  // Declare 'InternalTypeA_managed' with the same scalar type in
  // 'InternalTypeA'
  // ********************************************************************
  using InternalLayoutA_managed = InternalLayoutA;
  using InternalTypeA_managed   = std::conditional_t<
      Kokkos::is_view_v<InternalTypeA>,
      Kokkos::View<InternalScalarTypeA*, InternalLayoutA_managed,
                   typename XMV::device_type>,
      void>;

  // ********************************************************************
  // Declare 'InternalTypeB' if 'InternalTypeA_tmp' is a view then
  // make sure 'InternalTypeB' is a view as well
  // ********************************************************************
  using InternalTypeB = std::conditional_t<
      Kokkos::is_view_v<InternalTypeA_tmp> &&
          !Kokkos::is_view_v<InternalTypeB_tmp>,
      Kokkos::View<const InternalScalarTypeB*, InternalLayoutB,
                   typename YMV::device_type,
                   Kokkos::MemoryTraits<Kokkos::Unmanaged>>,
      InternalTypeB_tmp>;

  // ********************************************************************
  // Declare 'InternalTypeB_managed' with the same scalar type in
  // 'InternalTypeB'
  // ********************************************************************
  using InternalLayoutB_managed = InternalLayoutB;
  using InternalTypeB_managed   = std::conditional_t<
      Kokkos::is_view_v<InternalTypeB>,
      Kokkos::View<InternalScalarTypeB*, InternalLayoutB_managed,
                   typename YMV::device_type>,
      void>;

  // ********************************************************************
  // Auxiliary Boolean results on internal types
  // ********************************************************************
  static constexpr bool internalTypeA_sc  = !Kokkos::is_view_v<InternalTypeA>;
  static constexpr bool internalTypeA_r1d = Tr1d_val<InternalTypeA>();

  static constexpr bool internalTypeB_sc  = !Kokkos::is_view_v<InternalTypeB>;
  static constexpr bool internalTypeB_r1d = Tr1d_val<InternalTypeB>();

  static constexpr bool internalTypesAB_bothScalars =
      (internalTypeA_sc && internalTypeB_sc);
  static constexpr bool internalTypesAB_bothViews =
      (internalTypeA_r1d && internalTypeB_r1d);

  static void performChecks(const AV& a, const XMV& X, const BV& b,
                            const YMV& Y) {
    // ******************************************************************
    // Check 1/6: General checks
    // ******************************************************************
    static_assert(Kokkos::is_execution_space_v<tExecSpace>,
                  "KokkosBlas::Impl::AxpbyUnificationAttemptTraits()"
                  ": tExecSpace must be a valid Kokkos execution space.");

    if constexpr ((xyRank1Case && !xyRank2Case) ||
                  (!xyRank1Case && xyRank2Case)) {
      // Ok
    } else {
      std::ostringstream msg;
      msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 1/6"
          << ", invalid general case"
          << ": xyRank1Case = " << xyRank1Case
          << ", xyRank2Case = " << xyRank2Case;
      KokkosKernels::Impl::throw_runtime_exception(msg.str());
    }

    if constexpr (atInputScalarTypeY_isComplex == false) {
      if constexpr ((atInputScalarTypeA_isComplex == false) &&
                    (atInputScalarTypeX_isComplex == false) &&
                    (atInputScalarTypeB_isComplex == false)) {
        // Ok
      } else {
        std::ostringstream msg;
        msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 1/6"
            << ", invalid combination on scalar types: if Y is not complex, "
               "then A, X and B cannot be complex"
            << ": AtInputScalarTypeA = " << typeid(AtInputScalarTypeA).name()
            << ", AtInputScalarTypeX = " << typeid(AtInputScalarTypeX).name()
            << ", AtInputScalarTypeB = " << typeid(AtInputScalarTypeB).name()
            << ", AtInputScalarTypeY = " << typeid(AtInputScalarTypeY).name();
        KokkosKernels::Impl::throw_runtime_exception(msg.str());
      }
    }

    // ******************************************************************
    // Check 2/6: YMV is valid
    // ******************************************************************
    static_assert(Kokkos::is_view<YMV>::value,
                  "KokkosBlas::Impl::AxpbyUnificationAttemptTraits()"
                  ": Y is not a Kokkos::View.");
    static_assert(std::is_same<typename YMV::value_type,
                               typename YMV::non_const_value_type>::value,
                  "KokkosBlas::Impl::AxpbyUnificationAttemptTraits()"
                  ": Y is const.  It must be nonconst, "
                  "because it is an output argument "
                  "(we must be able to write to its entries).");
    static_assert(
        Kokkos::SpaceAccessibility<tExecSpace,
                                   typename YMV::memory_space>::accessible,
        "KokkosBlas::Impl::AxpbyUnificationAttemptTraits()"
        ": XMV must be accessible from tExecSpace");

    if constexpr ((Yr1 && !Yr2) || (!Yr1 && Yr2)) {
      // Ok
    } else {
      std::ostringstream msg;
      msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 2/6"
          << ", invalid YMV"
          << ": Yr1 = " << Yr1 << ", Yr2 = " << Yr2;
      KokkosKernels::Impl::throw_runtime_exception(msg.str());
    }

    // ******************************************************************
    // Check 3/6: XMV is valid
    // ******************************************************************
    static_assert(Kokkos::is_view<XMV>::value,
                  "KokkosBlas::Impl::AxpbyUnificationAttemptTraits()"
                  ": X is not a Kokkos::View.");
    static_assert(
        Kokkos::SpaceAccessibility<tExecSpace,
                                   typename XMV::memory_space>::accessible,
        "KokkosBlas::Impl::AxpbyUnificationAttemptTraits()"
        ": XMV must be accessible from tExecSpace");

    if constexpr ((Xr1 && !Xr2) || (!Xr1 && Xr2)) {
      // Ok
    } else {
      std::ostringstream msg;
      msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 3/6"
          << ", invalid XMV"
          << ": Xr1 = " << Xr1 << ", Xr2 = " << Xr2;
      KokkosKernels::Impl::throw_runtime_exception(msg.str());
    }

    if constexpr (xyRank1Case) {
      if (X.extent(0) == Y.extent(0)) {
        // Ok
      } else {
        std::ostringstream msg;
        msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 3/6"
            << ", invalid rank-1 X extent"
            << ": X.extent(0) = " << X.extent(0);
        KokkosKernels::Impl::throw_runtime_exception(msg.str());
      }
    } else {
      if ((X.extent(0) == Y.extent(0)) && (X.extent(1) == Y.extent(1))) {
        // Ok
      } else {
        std::ostringstream msg;
        msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 3/6"
            << ", invalid rank-2 X extents"
            << ": X.extent(0) = " << X.extent(0)
            << ", X.extent(1) = " << X.extent(1)
            << ", Y.extent(0) = " << Y.extent(0)
            << ", Y.extent(1) = " << Y.extent(1);
        KokkosKernels::Impl::throw_runtime_exception(msg.str());
      }
    }

    // ******************************************************************
    // Check 4/6: AV is valid
    // ******************************************************************
    if constexpr ((Asc && !Ar0 && !Ar1s && !Ar1d) ||
                  (!Asc && Ar0 && !Ar1s && !Ar1d) ||
                  (!Asc && !Ar0 && Ar1s && !Ar1d) ||
                  (!Asc && !Ar0 && !Ar1s && Ar1d)) {
      // Ok
    } else {
      std::ostringstream msg;
      msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 4/6"
          << ", invalid AV = " << typeid(AV).name() << ": Asc = " << Asc
          << ", Ar0 = " << Ar0 << ", Ar1s = " << Ar1s << ", Ar1d = " << Ar1d;
      KokkosKernels::Impl::throw_runtime_exception(msg.str());
    }

    if constexpr (Asc || Avi) {
      // Ok
    } else {
      std::ostringstream msg;
      msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 4/6"
          << ", AV memory must be either scalar or view"
          << ": Asc = " << Asc << ", Avi = " << Avi;
      KokkosKernels::Impl::throw_runtime_exception(msg.str());
    }

    if constexpr (Ar1d || Ar1s) {
      if constexpr (xyRank1Case) {
        if (a.extent(0) == 1) {
          // Ok
        } else {
          std::ostringstream msg;
          msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 4/6"
              << ", view 'a' must have extent(0) == 1 for xyRank1Case"
              << ": a.extent(0) = " << a.extent(0);
          KokkosKernels::Impl::throw_runtime_exception(msg.str());
        }
      } else {
        if ((a.extent(0) == 1) ||
            (a.extent(0) == Y.extent(1))) {  // Yes, 'Y' is the reference
          // Ok
        } else {
          std::ostringstream msg;
          msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 4/6"
              << ", view 'a' must have extent(0) == 1 or Y.extent(1) for "
                 "xyRank2Case"
              << ": a.extent(0) = " << a.extent(0)
              << ", Y.extent(0) = " << Y.extent(0)
              << ", Y.extent(1) = " << Y.extent(1);
          KokkosKernels::Impl::throw_runtime_exception(msg.str());
        }
      }  // if (rank1Case) else
    }    // if Ar1d

    // ******************************************************************
    // Check 5/6: BV is valid
    // ******************************************************************
    if constexpr ((Bsc && !Br0 && !Br1s && !Br1d) ||
                  (!Bsc && Br0 && !Br1s && !Br1d) ||
                  (!Bsc && !Br0 && Br1s && !Br1d) ||
                  (!Bsc && !Br0 && !Br1s && Br1d)) {
      // Ok
    } else {
      std::ostringstream msg;
      msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 5/6"
          << ", invalid BV"
          << ": Bsc = " << Bsc << ", Br0 = " << Br0 << ", Br1s = " << Br1s
          << ", Br1d = " << Br1d;
      KokkosKernels::Impl::throw_runtime_exception(msg.str());
    }

    if constexpr (Bsc || Bvi) {
      // Ok
    } else {
      std::ostringstream msg;
      msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 5/6"
          << ", BV memory must be either scalar or view"
          << ": Bsc = " << Bsc << ", Bvi = " << Bvi;
      KokkosKernels::Impl::throw_runtime_exception(msg.str());
    }

    if constexpr (Br1d || Br1s) {
      if constexpr (xyRank1Case) {
        if (b.extent(0) == 1) {
          // Ok
        } else {
          std::ostringstream msg;
          msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 5/6"
              << ", view 'b' must have extent(0) == 1 for xyRank1Case"
              << ": b.extent(0) = " << b.extent(0);
          KokkosKernels::Impl::throw_runtime_exception(msg.str());
        }
      } else {
        if ((b.extent(0) == 1) || (b.extent(0) == Y.extent(1))) {
          // Ok
        } else {
          std::ostringstream msg;
          msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 5/6"
              << ", view 'b' must have extent(0) == 1 or Y.extent(1) for "
                 "xyRank2Case"
              << ": b.extent(0) = " << b.extent(0)
              << ", Y.extent(0) = " << Y.extent(0)
              << ", Y.extent(1) = " << Y.extent(1);
          KokkosKernels::Impl::throw_runtime_exception(msg.str());
        }
      }  // if (rank1Case) else
    }    // if Br1d

    // ******************************************************************
    // Check 6/6: Checks on InternalTypeA, X, B, Y
    // ******************************************************************
    if constexpr (atHostCase) {
      if constexpr (xyRank1Case) {
        constexpr bool internalTypeA_isOk =
            (internalTypeA_sc || internalTypeA_r1d);
        constexpr bool internalTypeX_isOk = std::is_same_v<
            InternalTypeX,
            Kokkos::View<const AtInputScalarTypeX_nonConst*, InternalLayoutX,
                         typename XMV::device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;
        constexpr bool internalTypeB_isOk =
            (internalTypeB_sc || internalTypeB_r1d);
        constexpr bool internalTypeY_isOk = std::is_same_v<
            InternalTypeY,
            Kokkos::View<AtInputScalarTypeY_nonConst*, InternalLayoutY,
                         typename YMV::device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;
        if constexpr (internalTypeA_isOk && internalTypeX_isOk &&
                      internalTypeB_isOk && internalTypeY_isOk) {
          // Ok
        } else {
          std::ostringstream msg;
          msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check "
                 "6.1/6"
              << ", invalid internal types"
              << ": atHostCase = " << atHostCase
              << ", atDevCase = " << atDevCase
              << ", xyRank1Case= " << xyRank1Case
              << ", xyRank2Case= " << xyRank2Case
              << ", InternalTypeA = " << typeid(InternalTypeA).name()
              << ", InternalTypeX = " << typeid(InternalTypeX).name()
              << ", InternalTypeB = " << typeid(InternalTypeB).name()
              << ", InternalTypeY = " << typeid(InternalTypeY).name();
          KokkosKernels::Impl::throw_runtime_exception(msg.str());
        }
      } else {
        constexpr bool internalTypeA_isOk =
            (internalTypeA_sc || internalTypeA_r1d);
        constexpr bool internalTypeX_isOk = std::is_same_v<
            InternalTypeX,
            Kokkos::View<const AtInputScalarTypeX_nonConst**, InternalLayoutX,
                         typename XMV::device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;
        constexpr bool internalTypeB_isOk =
            (internalTypeB_sc || internalTypeB_r1d);
        constexpr bool internalTypeY_isOk = std::is_same_v<
            InternalTypeY,
            Kokkos::View<AtInputScalarTypeY_nonConst**, InternalLayoutY,
                         typename YMV::device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;
        if constexpr (internalTypeA_isOk && internalTypeX_isOk &&
                      internalTypeB_isOk && internalTypeY_isOk) {
          // Ok
        } else {
          std::ostringstream msg;
          msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check "
                 "6.2/6"
              << ", invalid internal types"
              << ": atHostCase = " << atHostCase
              << ", atDevCase = " << atDevCase
              << ", xyRank1Case= " << xyRank1Case
              << ", xyRank2Case= " << xyRank2Case
              << ", InternalTypeA = " << typeid(InternalTypeA).name()
              << ", InternalTypeX = " << typeid(InternalTypeX).name()
              << ", InternalTypeB = " << typeid(InternalTypeB).name()
              << ", InternalTypeY = " << typeid(InternalTypeY).name();
          KokkosKernels::Impl::throw_runtime_exception(msg.str());
        }
      }
    } else {
      if constexpr (xyRank1Case) {
        constexpr bool internalTypeA_isOk = internalTypeA_r1d;
        constexpr bool internalTypeX_isOk = std::is_same_v<
            InternalTypeX,
            Kokkos::View<const AtInputScalarTypeX_nonConst*, InternalLayoutX,
                         typename XMV::device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;
        constexpr bool internalTypeB_isOk = internalTypeB_r1d;
        constexpr bool internalTypeY_isOk = std::is_same_v<
            InternalTypeY,
            Kokkos::View<AtInputScalarTypeY_nonConst*, InternalLayoutY,
                         typename YMV::device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;
        if constexpr (internalTypeA_isOk && internalTypeX_isOk &&
                      internalTypeB_isOk && internalTypeY_isOk) {
          // Ok
        } else {
          std::ostringstream msg;
          msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check "
                 "6.3/6"
              << ", invalid internal types"
              << ": atHostCase = " << atHostCase
              << ", atDevCase = " << atDevCase
              << ", xyRank1Case= " << xyRank1Case
              << ", xyRank2Case= " << xyRank2Case
              << ", InternalTypeA = " << typeid(InternalTypeA).name()
              << ", InternalTypeX = " << typeid(InternalTypeX).name()
              << ", InternalTypeB = " << typeid(InternalTypeB).name()
              << ", InternalTypeY = " << typeid(InternalTypeY).name();
          KokkosKernels::Impl::throw_runtime_exception(msg.str());
        }
      } else {
        constexpr bool internalTypeA_isOk = internalTypeA_r1d;
        constexpr bool internalTypeX_isOk = std::is_same_v<
            InternalTypeX,
            Kokkos::View<const AtInputScalarTypeX_nonConst**, InternalLayoutX,
                         typename XMV::device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;
        constexpr bool internalTypeB_isOk = internalTypeB_r1d;
        constexpr bool internalTypeY_isOk = std::is_same_v<
            InternalTypeY,
            Kokkos::View<AtInputScalarTypeY_nonConst**, InternalLayoutY,
                         typename YMV::device_type,
                         Kokkos::MemoryTraits<Kokkos::Unmanaged>>>;
        if constexpr (internalTypeA_isOk && internalTypeX_isOk &&
                      internalTypeB_isOk && internalTypeY_isOk) {
          // Ok
        } else {
          std::ostringstream msg;
          msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check "
                 "6.4/6"
              << ", invalid internal types"
              << ": atHostCase = " << atHostCase
              << ", atDevCase = " << atDevCase
              << ", xyRank1Case= " << xyRank1Case
              << ", xyRank2Case= " << xyRank2Case
              << ", InternalTypeA = " << typeid(InternalTypeA).name()
              << ", InternalTypeX = " << typeid(InternalTypeX).name()
              << ", InternalTypeB = " << typeid(InternalTypeB).name()
              << ", InternalTypeY = " << typeid(InternalTypeY).name();
          KokkosKernels::Impl::throw_runtime_exception(msg.str());
        }
      }
    }

    if constexpr (atHostCase) {
      // ****************************************************************
      // We are in the 'atHostCase' case, with 2 possible subcases::
      //
      // 1) xyRank1Case, with the following possible situations:
      // - [InternalTypeA, B] = [S_a, S_b], or
      // - [InternalTypeA, B] = [view<S_a*,1>, view<S_b*,1>]
      //
      // or
      //
      // 2) xyRank2Case, with the following possible situations:
      // - [InternalTypeA, B] = [S_a, S_b], or
      // - [InternalTypeA, B] = [view<S_a*,1 / m>, view<S_b*,1 / m>]
      // ****************************************************************
      static_assert(
          internalTypesAB_bothScalars || internalTypesAB_bothViews,
          "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), atHostCase, "
          "invalid combination of types");
    }  // If atHostCase
    else if constexpr (atDevCase) {
      // ****************************************************************
      // We are in the 'atDevCase' case, with 2 possible subcases:
      //
      // 1) xyRank1Case, with only one possible situation:
      // - [InternalTypeA / B] = [view<S_a*,1>, view<S_b*,1>]
      //
      // or
      //
      // 2) xyRank2Case, with only one possible situation:
      // - [InternalTypeA / B] = [view<S_a*,1 / m>, view<S_b*,1 / m>]
      // ****************************************************************
      static_assert(
          internalTypesAB_bothViews,
          "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), atDevCase, "
          "invalid combination of types");
    }

    if constexpr (xyRank2Case && (Ar1d || Ar1s) && atInputLayoutA_isStride) {
      if (std::is_same_v<
              typename getLayoutFromView<
                  InternalTypeA, Kokkos::is_view_v<InternalTypeA>>::type,
              Kokkos::LayoutStride>) {
        // Ok
      } else {
        std::ostringstream msg;
        msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 6.5/6"
            << ", xyRank2Case = " << xyRank2Case
            << ", coeff 'a' is rank-1 and has LayoutStride at input, but no "
               "LayoutStride internally"
            << ", AV = " << typeid(AV).name()
            << ", InternalTypeA = " << typeid(InternalTypeA).name();
        KokkosKernels::Impl::throw_runtime_exception(msg.str());
      }
    }

    if constexpr (xyRank2Case && (Br1d || Br1s) && atInputLayoutB_isStride) {
      if (std::is_same_v<
              typename getLayoutFromView<
                  InternalTypeB, Kokkos::is_view_v<InternalTypeB>>::type,
              Kokkos::LayoutStride>) {
        // Ok
      } else {
        std::ostringstream msg;
        msg << "KokkosBlas::Impl::AxpbyUnificationAttemptTraits(), check 6.6/6"
            << ", xyRank2Case = " << xyRank2Case
            << ", coeff 'a' is rank-1 and has LayoutStride at input, but no "
               "LayoutStride internally"
            << ", BV = " << typeid(BV).name()
            << ", InternalTypeB = " << typeid(InternalTypeB).name();
        KokkosKernels::Impl::throw_runtime_exception(msg.str());
      }
    }
  }  // Constructor

  static void printInformation(std::ostream& os, std::string const& headerMsg) {
    os << headerMsg << ": AV = "
       << typeid(AV).name()
       //<< ", AV::const_data_type = "     << typeid(AV::const_data_type).name()
       //<< ", AV::non_const_data_type = " <<
       // typeid(AV::non_const_data_type).name()
       << ", AtInputScalarTypeA = " << typeid(AtInputScalarTypeA).name()
       << ", isConst = "
       << std::is_const_v<AtInputScalarTypeA> << ", isComplex = "
       << atInputScalarTypeA_isComplex << ", AtInputScalarTypeA_nonConst = "
       << typeid(AtInputScalarTypeA_nonConst).name()
       << ", InternalTypeA = " << typeid(InternalTypeA).name() << "\n"
       << ", InternalTypeA_managed = " << typeid(InternalTypeA_managed).name()
       << "\n"
       << "\n"
       << "XMV = " << typeid(XMV).name() << "\n"
       << "XMV::value_type = " << typeid(typename XMV::value_type).name()
       << "\n"
       << "XMV::const_data_type = "
       << typeid(typename XMV::const_data_type).name() << "\n"
       << "XMV::non_const_data_type = "
       << typeid(typename XMV::non_const_data_type).name() << "\n"
       << "AtInputScalarTypeX = " << typeid(AtInputScalarTypeX).name() << "\n"
       << "isConst = " << std::is_const_v<AtInputScalarTypeX> << "\n"
       << "isComplex = " << atInputScalarTypeX_isComplex << "\n"
       << "AtInputScalarTypeX_nonConst = "
       << typeid(AtInputScalarTypeX_nonConst).name() << "\n"
       << "InternalTypeX = " << typeid(InternalTypeX).name() << "\n"
       << "\n"
       << "BV = "
       << typeid(BV).name()
       //<< ", BV::const_data_type = "     << typeid(BV::const_data_type).name()
       //<< ", BV::non_const_data_type = " <<
       // typeid(BV::non_const_data_type).name()
       << ", AtInputScalarTypeB = " << typeid(AtInputScalarTypeB).name()
       << ", isConst = "
       << std::is_const_v<AtInputScalarTypeB> << ", isComplex = "
       << atInputScalarTypeB_isComplex << ", AtInputScalarTypeB_nonConst = "
       << typeid(AtInputScalarTypeB_nonConst).name()
       << ", InternalTypeB = " << typeid(InternalTypeB).name() << "\n"
       << ", InternalTypeB_managed = " << typeid(InternalTypeB_managed).name()
       << "\n"
       << "\n"
       << "YMV = " << typeid(YMV).name() << "\n"
       << "YMV::value_type = " << typeid(typename YMV::value_type).name()
       << "\n"
       << "YMV::const_data_type = "
       << typeid(typename YMV::const_data_type).name() << "\n"
       << "YMV::non_const_data_type = "
       << typeid(typename YMV::non_const_data_type).name() << "\n"
       << "AtInputScalarTypeY = " << typeid(AtInputScalarTypeY).name() << "\n"
       << "isConst = " << std::is_const_v<AtInputScalarTypeY> << "\n"
       << "isComplex = " << atInputScalarTypeY_isComplex << "\n"
       << "AtInputScalarTypeY_nonConst = "
       << typeid(AtInputScalarTypeY_nonConst).name() << "\n"
       << "InternalTypeY = " << typeid(InternalTypeY).name() << "\n"
       << std::endl;
  }
};  // struct AxpbyUnificationAttemptTraits

// --------------------------------

template <typename T, int rankT>
struct getScalarValueFromVariableAtHost {
  getScalarValueFromVariableAtHost() {
    static_assert((rankT == -1) || (rankT == 0) || (rankT == 1),
                  "Generic struct should not have been invoked!");
  }
};

template <typename T>
struct getScalarValueFromVariableAtHost<T, -1> {
  static T getValue(T const& var) { return var; }
};

template <typename T>
struct getScalarValueFromVariableAtHost<T, 0> {
  static typename T::value_type getValue(T const& var) { return var(); }
};

template <class T>
struct getScalarValueFromVariableAtHost<T, 1> {
  static typename T::value_type getValue(T const& var) { return var[0]; }
};

// --------------------------------

template <typename T>
size_t getAmountOfScalarsInCoefficient(T const& coeff) {
  size_t result = 1;
  if constexpr (Kokkos::is_view_v<T>) {
    if constexpr (T::rank == 1) {
      result = coeff.extent(0);
    }
  }
  return result;
}

// --------------------------------

template <typename T>
size_t getStrideInCoefficient(T const& coeff) {
  size_t result = 1;
  if constexpr (Kokkos::is_view_v<T>) {
    if constexpr ((T::rank == 1) && (std::is_same_v<typename T::array_layout,
                                                    Kokkos::LayoutStride>)) {
      result = coeff.stride_0();
    }
  }
  return result;
}

// --------------------------------

template <class T_in, class T_out>
static void populateRank1Stride1ViewWithScalarOrNonStrideView(
    T_in const& coeff_in, T_out& coeff_out) {
  // ***********************************************************************
  // 'coeff_out' is assumed to be rank-1, of LayoutLeft or LayoutRight
  //
  // One has to be careful with situations like the following:
  // - a coeff_in that deals with 'double', and
  // - a coeff_out deals with 'complex<double>'
  // ***********************************************************************
  using ScalarOutType =
      typename std::remove_const<typename T_out::value_type>::type;

  if constexpr (!Kokkos::is_view_v<T_in>) {
    // *********************************************************************
    // 'coeff_in' is scalar
    // *********************************************************************
    ScalarOutType scalarValue(coeff_in);
    Kokkos::deep_copy(coeff_out, scalarValue);
  } else if constexpr (T_in::rank == 0) {
    // *********************************************************************
    // 'coeff_in' is rank-0
    // *********************************************************************
    typename T_in::HostMirror h_coeff_in("h_coeff_in");
    Kokkos::deep_copy(h_coeff_in, coeff_in);
    ScalarOutType scalarValue(h_coeff_in());
    Kokkos::deep_copy(coeff_out, scalarValue);
  } else {
    // *********************************************************************
    // 'coeff_in' is also rank-1
    // *********************************************************************
    if (coeff_out.extent(0) != coeff_in.extent(0)) {
      std::ostringstream msg;
      msg << "In populateRank1Stride1ViewWithScalarOrNonStrideView()"
          << ": 'in' and 'out' should have the same extent(0)"
          << ", T_in = " << typeid(T_in).name()
          << ", coeff_in.label() = " << coeff_in.label()
          << ", coeff_in.extent(0) = " << coeff_in.extent(0)
          << ", T_out = " << typeid(T_out).name()
          << ", coeff_out.label() = " << coeff_out.label()
          << ", coeff_out.extent(0) = " << coeff_out.extent(0);
      KokkosKernels::Impl::throw_runtime_exception(msg.str());
    }

    using ScalarInType =
        typename std::remove_const<typename T_in::value_type>::type;
    if constexpr (std::is_same_v<ScalarInType, ScalarOutType>) {
      coeff_out = coeff_in;
    } else if (coeff_out.extent(0) == 1) {
      typename T_in::HostMirror h_coeff_in("h_coeff_in");
      Kokkos::deep_copy(h_coeff_in, coeff_in);
      ScalarOutType scalarValue(h_coeff_in[0]);
      Kokkos::deep_copy(coeff_out, scalarValue);
    } else {
      std::ostringstream msg;
      msg << "In populateRank1Stride1ViewWithScalarOrNonStrideView()"
          << ": scalar types 'in' and 'out' should be the same"
          << ", T_in = " << typeid(T_in).name()
          << ", ScalarInType = " << typeid(ScalarInType).name()
          << ", coeff_in.label() = " << coeff_in.label()
          << ", coeff_in.extent(0) = " << coeff_in.extent(0)
          << ", T_out = " << typeid(T_out).name()
          << ", ScalarOutType = " << typeid(ScalarOutType).name()
          << ", coeff_out.label() = " << coeff_out.label()
          << ", coeff_out.extent(0) = " << coeff_out.extent(0);
      KokkosKernels::Impl::throw_runtime_exception(msg.str());
    }
  }
}  // populateRank1Stride1ViewWithScalarOrNonStrideView()

}  // namespace Impl
}  // namespace KokkosBlas

#endif  // KOKKOS_BLAS1_AXPBY_UNIFICATION_ATTEMPT_TRAITS_HPP_
