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
/// \author Yuuichi Asahi (yuuichi.asahi@cea.fr)
#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include <Kokkos_Random.hpp>

#include "KokkosBatched_Util.hpp"
#include "KokkosBatched_Pttrf.hpp"
#include "Test_Batched_DenseUtils.hpp"

using namespace KokkosBatched;

namespace Test {
namespace Pttrf {

template <typename T>
struct real_type {
  using type = T;
};

template <typename T>
struct real_type<Kokkos::complex<T>> {
  using type = T;
};

template <typename DeviceType, typename DViewType, typename EViewType,
          typename AlgoTagType>
struct Functor_BatchedSerialPttrf {
  using execution_space = typename DeviceType::execution_space;
  DViewType _d;
  EViewType _e;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialPttrf(const DViewType &d, const EViewType &e)
      : _d(d), _e(e) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto dd = Kokkos::subview(_d, k, Kokkos::ALL());
    auto ee = Kokkos::subview(_e, k, Kokkos::ALL());

    KokkosBatched::SerialPttrf<AlgoTagType>::invoke(dd, ee);
  }

  inline void run() {
    using value_type = typename DViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialPttrf");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::Profiling::pushRegion(name.c_str());
    Kokkos::RangePolicy<execution_space> policy(0, _d.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
    Kokkos::Profiling::popRegion();
  }
};

template <typename DeviceType, typename ScalarType, typename AViewType,
          typename BViewType, typename CViewType, typename ArgTransB>
struct Functor_BatchedSerialGemm {
  using execution_space = typename DeviceType::execution_space;
  AViewType _a;
  BViewType _b;
  CViewType _c;
  ScalarType _alpha, _beta;

  KOKKOS_INLINE_FUNCTION
  Functor_BatchedSerialGemm(const ScalarType alpha, const AViewType &a,
                            const BViewType &b, const ScalarType beta,
                            const CViewType &c)
      : _alpha(alpha), _a(a), _b(b), _beta(beta), _c(c) {}

  KOKKOS_INLINE_FUNCTION
  void operator()(const int k) const {
    auto aa = Kokkos::subview(_a, k, Kokkos::ALL(), Kokkos::ALL());
    auto bb = Kokkos::subview(_b, k, Kokkos::ALL(), Kokkos::ALL());
    auto cc = Kokkos::subview(_c, k, Kokkos::ALL(), Kokkos::ALL());

    KokkosBatched::SerialGemm<Trans::NoTranspose, ArgTransB,
                              Algo::Gemm::Unblocked>::invoke(_alpha, aa, bb,
                                                             _beta, cc);
  }

  inline void run() {
    using value_type = typename AViewType::non_const_value_type;
    std::string name_region("KokkosBatched::Test::SerialPttrf");
    const std::string name_value_type = Test::value_type_name<value_type>();
    std::string name                  = name_region + name_value_type;
    Kokkos::RangePolicy<execution_space> policy(0, _a.extent(0));
    Kokkos::parallel_for(name.c_str(), policy, *this);
  }
};

template <typename DeviceType, typename ScalarType, typename LayoutType,
          typename AlgoTagType>
/// \brief Implementation details of batched pttrf test
///
/// \param N [in] Batch size of matrix A
void impl_test_batched_pttrf_analytical(const int N, const int BlkSize) {
  using real_type      = typename real_type<ScalarType>::type;
  using RealView2DType = Kokkos::View<real_type **, LayoutType, DeviceType>;
  using View2DType     = Kokkos::View<ScalarType **, LayoutType, DeviceType>;
  using View3DType     = Kokkos::View<ScalarType ***, LayoutType, DeviceType>;

  View3DType A("A", N, BlkSize, BlkSize),
      A_reconst("A_reconst", N, BlkSize, BlkSize);
  View3DType EL("EL", N, BlkSize, BlkSize), EU("EU", N, BlkSize, BlkSize),
      DEU("DEU", N, BlkSize, BlkSize), D("D", N, BlkSize, BlkSize),
      LD("LD", N, BlkSize, BlkSize), L("L", N, BlkSize, BlkSize),
      I("I", N, BlkSize, BlkSize);
  RealView2DType d("d", N, BlkSize),
      ones("ones", N, BlkSize);       // Diagonal components
  View2DType e("e", N, BlkSize - 1);  // Sub diagnoal components

  auto h_d    = Kokkos::create_mirror_view(d);
  auto h_e    = Kokkos::create_mirror_view(e);
  auto h_ones = Kokkos::create_mirror_view(ones);

  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      h_d(ib, i)    = 4 + ib;
      h_ones(ib, i) = 1;
    }

    for (int i = 0; i < BlkSize - 1; i++) {
      h_e(ib, i) = 1;
    }
  }

  Kokkos::fence();

  Kokkos::deep_copy(d, h_d);
  Kokkos::deep_copy(e, h_e);
  Kokkos::deep_copy(ones, h_ones);

  // Reconstruct Tridiaonal matrix A
  // A = D + EL + EU
  create_diagonal_matrix(e, EL, -1);
  create_diagonal_matrix(e, EU, 1);
  create_diagonal_matrix(d, D);

  add_matrices(D, EU, DEU);
  add_matrices(DEU, EL, A);

  // Factorize matrix A -> L * D * L.T
  // d and e are updated by pttrf
  Functor_BatchedSerialPttrf<DeviceType, RealView2DType, View2DType,
                             AlgoTagType>(d, e)
      .run();

  // Reconstruct L and D from factorized matrix
  create_diagonal_matrix(e, EL, -1);
  create_diagonal_matrix(d, D);
  create_diagonal_matrix(ones, I);

  add_matrices(EL, I, L);

  // Reconstruct A by L*D*LT
  // Gemm to compute L*D -> LD
  Functor_BatchedSerialGemm<DeviceType, ScalarType, View3DType, View3DType,
                            View3DType, Trans::NoTranspose>(1.0, L, D, 0.0, LD)
      .run();

  // Gemm to compute (L*D)*LT -> A_reconst
  Functor_BatchedSerialGemm<DeviceType, ScalarType, View3DType, View3DType,
                            View3DType, Trans::Transpose>(1.0, LD, L, 0.0,
                                                          A_reconst)
      .run();

  Kokkos::fence();

  // this eps is about 10^-14
  using ats      = typename Kokkos::ArithTraits<ScalarType>;
  using mag_type = typename ats::mag_type;
  mag_type eps   = 1.0e3 * ats::epsilon();

  auto h_A = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A);
  auto h_A_reconst =
      Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_reconst);

  // Check A = L*D*L.T
  for (int ib = 0; ib < N; ib++) {
    for (int i = 0; i < BlkSize; i++) {
      for (int j = 0; j < BlkSize; j++) {
        EXPECT_NEAR_KK(h_A_reconst(ib, i, j), h_A(ib, i, j), eps);
      }
    }
  }
}

}  // namespace Pttrf
}  // namespace Test

template <typename DeviceType, typename ScalarType, typename AlgoTagType>
int test_batched_pttrf() {
#if defined(KOKKOSKERNELS_INST_LAYOUTLEFT)
  {
    using LayoutType = Kokkos::LayoutLeft;
    for (int i = 2; i < 10; i++) {
      Test::Pttrf::impl_test_batched_pttrf_analytical<DeviceType, ScalarType,
                                                      LayoutType, AlgoTagType>(
          1, i);
      Test::Pttrf::impl_test_batched_pttrf_analytical<DeviceType, ScalarType,
                                                      LayoutType, AlgoTagType>(
          2, i);
    }
  }
#endif
#if defined(KOKKOSKERNELS_INST_LAYOUTRIGHT)
  {
    using LayoutType = Kokkos::LayoutRight;
    for (int i = 2; i < 10; i++) {
      Test::Pttrf::impl_test_batched_pttrf_analytical<DeviceType, ScalarType,
                                                      LayoutType, AlgoTagType>(
          1, i);
      Test::Pttrf::impl_test_batched_pttrf_analytical<DeviceType, ScalarType,
                                                      LayoutType, AlgoTagType>(
          2, i);
    }
  }
#endif

  return 0;
}
