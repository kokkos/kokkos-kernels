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
#ifndef TEST_UTILS_HPP
#define TEST_UTILS_HPP

#include <Kokkos_Core.hpp>
#include <iomanip>
#include "KokkosBatched_Util.hpp"

template <typename ExecutionSpace, typename AViewType, typename BViewType>
bool allclose(const AViewType& a, const BViewType& b, double rtol = 1.e-5,
              double atol = 1.e-8) {
  constexpr std::size_t rank = AViewType::rank;
  for (std::size_t i = 0; i < rank; i++) {
    assert(a.extent(i) == b.extent(i));
  }
  const auto n = a.size();

  auto* ptr_a = a.data();
  auto* ptr_b = b.data();

  int error = 0;
  Kokkos::parallel_reduce(
      Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<std::size_t>>{0, n},
      KOKKOS_LAMBDA(const int& i, int& err) {
        auto tmp_a = ptr_a[i];
        auto tmp_b = ptr_b[i];
        bool not_close =
            Kokkos::abs(tmp_a - tmp_b) > (atol + rtol * Kokkos::abs(tmp_b));
        err += static_cast<int>(not_close);
      },
      error);

  return error == 0;
}

template <typename InViewType, typename OutViewType, typename UploType>
void create_banded_triangular_matrix(InViewType& in, OutViewType& out,
                                     int k = 1, bool band_storage = true) {
  auto h_in        = Kokkos::create_mirror_view(in);
  auto h_out       = Kokkos::create_mirror_view(out);
  using value_type = typename InViewType::non_const_value_type;
  const int N = in.extent(0), BlkSize = in.extent(1);

  Kokkos::deep_copy(h_in, in);
  if (band_storage) {
    assert(out.extent(0) == in.extent(0));
    assert(out.extent(1) == k + 1);
    assert(out.extent(2) == in.extent(2));
    if constexpr (std::is_same_v<UploType, KokkosBatched::Uplo::Upper>) {
      for (int i0 = 0; i0 < N; i0++) {
        for (int i1 = 0; i1 < k + 1; i1++) {
          for (int i2 = i1; i2 < BlkSize; i2++) {
            h_out(i0, k - i1, i2) = h_in(i0, i2 - i1, i2);
          }
        }
      }
    } else {
      for (int i0 = 0; i0 < N; i0++) {
        for (int i1 = 0; i1 < k + 1; i1++) {
          for (int i2 = 0; i2 < BlkSize - i1; i2++) {
            h_out(i0, i1, i2) = h_in(i0, i2 + i1, i2);
          }
        }
      }
    }
  } else {
    for (std::size_t i = 0; i < InViewType::rank(); i++) {
      assert(out.extent(i) == in.extent(i));
    }

    if constexpr (std::is_same_v<UploType, KokkosBatched::Uplo::Upper>) {
      for (int i0 = 0; i0 < N; i0++) {
        for (int i1 = 0; i1 < BlkSize; i1++) {
          for (int i2 = i1; i2 < Kokkos::min(i1 + k + 1, BlkSize); i2++) {
            h_out(i0, i1, i2) = h_in(i0, i1, i2);
          }
        }
      }
    } else {
      for (int i0 = 0; i0 < N; i0++) {
        for (int i1 = 0; i1 < BlkSize; i1++) {
          for (int i2 = Kokkos::max(0, i1 - k); i2 <= i1; i2++) {
            h_out(i0, i1, i2) = h_in(i0, i1, i2);
          }
        }
      }
    }
  }
  Kokkos::deep_copy(out, h_out);
}

#endif