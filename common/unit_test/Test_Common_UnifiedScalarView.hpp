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


#ifndef TEST_KOKKOSKERNELS_UNIFIEDSCALARVIEW_HPP
#define TEST_KOKKOSKERNELS_UNIFIEDSCALARVIEW_HPP

#include <KokkosKernels_UnifiedScalarView.hpp>

#include <string>

template <typename ExecSpace, typename ValueType>
void test_is_unified_scalar() {

    static_assert(
        KokkosKernels::Impl::is_scalar_or_scalar_view<ValueType>,
        ""
    );
    static_assert(
        KokkosKernels::Impl::is_scalar_or_scalar_view<const ValueType>,
        ""
    );
    static_assert(
        KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<ValueType>>,
        ""
    );
    static_assert(
        KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<const ValueType>>,
        ""
    );
    static_assert(
        KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<ValueType[1]>>,
        ""
    );
    static_assert(
        KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<const ValueType[1]>>,
        ""
    );
    // false cases

    static_assert(
        !KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<ValueType*>>,
        ""
    );
    static_assert(
        !KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<ValueType[2]>>,
        ""
    );

    static_assert(
        !KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<ValueType*[1]>>,
        ""
    );
    static_assert(
        !KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<ValueType*[2]>>,
        ""
    );
    // could support this one, but seems unlikely to come up in practice
    static_assert(
        !KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<const ValueType[1][1]>>,
        ""
    );
    static_assert(
        !KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<const ValueType[2][1]>>,
        ""
    );
    static_assert(
        !KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<const ValueType[1][2]>>,
        ""
    );
    static_assert(
        !KokkosKernels::Impl::is_scalar_or_scalar_view<Kokkos::View<ValueType**>>,
        ""
    );

    static_assert(
        !KokkosKernels::Impl::is_scalar_or_scalar_view<std::string>,
        ""
    );

    static_assert(
        !KokkosKernels::Impl::is_scalar_or_scalar_view<void>,
        ""
    );
}

template <typename ExecSpace, typename ValueType>
void test_unified_scalar() {

    // test scalars
    static_assert(
        std::is_same_v<ValueType, KokkosKernels::Impl::unified_scalar_t<ValueType>>,
        ""
    );
    static_assert(
        std::is_same_v<const ValueType, KokkosKernels::Impl::unified_scalar_t<const ValueType>>,
        ""
    );
    static_assert(
        std::is_same_v<ValueType, KokkosKernels::Impl::non_const_unified_scalar_t<ValueType>>,
        ""
    );
    static_assert(
        std::is_same_v<ValueType, KokkosKernels::Impl::non_const_unified_scalar_t<const ValueType>>,
        ""
    );

    // test 0D views
    static_assert(
        std::is_same_v<ValueType, KokkosKernels::Impl::unified_scalar_t<Kokkos::View<ValueType>>>,
        ""
    );
    static_assert(
        std::is_same_v<const ValueType, KokkosKernels::Impl::unified_scalar_t<Kokkos::View<const ValueType>>>,
        ""
    );
    static_assert(
        std::is_same_v<ValueType, KokkosKernels::Impl::non_const_unified_scalar_t<Kokkos::View<ValueType>>>,
        ""
    );
    static_assert(
        std::is_same_v<ValueType, KokkosKernels::Impl::non_const_unified_scalar_t<Kokkos::View<const ValueType>>>,
        ""
    );

    // test 1D views
    static_assert(
        std::is_same_v<ValueType, KokkosKernels::Impl::unified_scalar_t<Kokkos::View<ValueType[1]>>>,
        ""
    );
    static_assert(
        std::is_same_v<const ValueType, KokkosKernels::Impl::unified_scalar_t<Kokkos::View<const ValueType[1]>>>,
        ""
    );
    static_assert(
        std::is_same_v<ValueType, KokkosKernels::Impl::non_const_unified_scalar_t<Kokkos::View<ValueType[1]>>>,
        ""
    );
    static_assert(
        std::is_same_v<ValueType, KokkosKernels::Impl::non_const_unified_scalar_t<Kokkos::View<const ValueType[1]>>>,
        ""
    );
}

template <typename View>
struct Expect0DEqual {
    using exp_type = typename View::value_type;
    Expect0DEqual(const View &v, const exp_type &exp) : v_(v), exp_(exp) {}
    void operator()(size_t i, int &lsum) const {
        if (v_() != exp_) {
            ++lsum;
        }
    }
    View v_;
    exp_type exp_;
};

template <typename View>
struct Expect1DEqual {
    using exp_type = typename View::value_type;
    Expect1DEqual(const View &v, const exp_type &exp) : v_(v), exp_(exp) {}
    void operator()(size_t i, int &lsum) const {
        if (v_(0) != exp_) {
            ++lsum;
        }
    }
    View v_;
    exp_type exp_;
};


template <typename ExecSpace, typename ValueType>
void test_get_scalar() {

    // constexpr context
    static_assert(3 == KokkosKernels::Impl::get_scalar(3), "");

    // 0D
    {
        Kokkos::View<ValueType, ExecSpace> v;
        Kokkos::deep_copy(v, ValueType(4));
        Kokkos::View<const ValueType, ExecSpace> cv = v;

        const Kokkos::RangePolicy<ExecSpace> policy(0, 1); // one thread

        {
            Expect0DEqual op(v, ValueType(4));
            int err;
            Kokkos::parallel_reduce("", policy, op, Kokkos::Sum<int>(err));
            EXPECT_EQ(err, 0);
        }

        {
            Expect0DEqual op(cv, ValueType(4));
            int err;
            Kokkos::parallel_reduce("", policy, op, Kokkos::Sum<int>(err));
            EXPECT_EQ(err, 0);
        }        
    }

    // 1D
    {
        Kokkos::View<ValueType[1], ExecSpace> v;
        Kokkos::deep_copy(v, 4);
        Kokkos::View<const ValueType[1], ExecSpace> cv = v;

        const Kokkos::RangePolicy<ExecSpace> policy(0, 1); // one thread

        {
            Expect1DEqual op(v, ValueType(4));
            int err;
            Kokkos::parallel_reduce("", policy, op, Kokkos::Sum<int>(err));
            EXPECT_EQ(err, 0);
        }

        {
            Expect1DEqual op(cv, ValueType(4));
            int err;
            Kokkos::parallel_reduce("", policy, op, Kokkos::Sum<int>(err));
            EXPECT_EQ(err, 0);
        }     
    }

}

TEST_F(TestCategory, common_device_unifiedscalarview) {
  // Test device-level bitonic with some larger arrays

  test_is_unified_scalar<TestExecSpace, float>();
  test_is_unified_scalar<TestExecSpace, Kokkos::complex<float>>();
  test_unified_scalar<TestExecSpace, double>();
  test_unified_scalar<TestExecSpace, Kokkos::complex<double>>();
  test_get_scalar<TestExecSpace, double>();
  test_get_scalar<TestExecSpace, Kokkos::complex<double>>();
}

#endif // TEST_KOKKOSKERNELS_UNIFIEDSCALARVIEW_HPP
