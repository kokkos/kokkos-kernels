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

#include <gtest/gtest.h>
#include <Kokkos_Core.hpp>
#include "KokkosSparse_mdf.hpp"
#include "KokkosSparse_CrsMatrix.hpp"

namespace Test {

// void foo(){

//   // const value_type four = static_cast<value_type>(4.0);

//   constexpr ordinal_type numRows  = 100;
//   constexpr ordinal_type numCols  = numRows;
//   row_map_type row_map(Kokkos::ViewAllocateWithoutInitializing("row map"),
//   numRows + 1); Kokkos::deep_copy(row_map,0);

//   constexpr value_type perc_fill = 0.3;
//   constexpr size_type targetNonZerosPerRow = numRows*perc_fill;
//   constexpr value_type num_fill_scl = 0.6;

//   Kokkos::Random_XorShift64_Pool<execution_space> random(13718 + 3);
//   Kokkos::fill_random(row_map, random,
//   size_type(targetNonZerosPerRow*num_fill_scl),
//   value_type(targetNonZerosPerRow/num_fill_scl));

//   size_type numNonZeros = 0;
//   Kokkos::parallel_scan(
//     Kokkos::RangePolicy<execution_space>(0,numRows+1),
//     KOKKOS_LAMBDA(ordinal_type i,bool is_final,size_type & runningNZ){
//       if (is_final) {
//         const auto curr_val = row_map[i];
//         row_map[i] = runningNZ;
//         if (i < numRows) runningNZ += curr_val;
//       }
//       else {
//         runningNZ += row_map[i];
//       }
//     },
//     numNonZeros
//   );

//   // constexpr size_type numNonZeros = 64;
//   // row_map_type row_map("row map", numRows + 1);
//   col_ind_type col_ind("column indices", numNonZeros);
//   values_type values("values", numNonZeros);
//   Kokkos::fill_random(values, random, value_type(1.0), value_type(10.));

// }

template <typename scalar_type, typename ordinal_type, typename size_type,
          typename device>
KokkosSparse::CrsMatrix<scalar_type, ordinal_type, device, void, size_type>
make_adv_diffusion_matrix(const scalar_type beta, const scalar_type vel_mag,
                          const size_type Nx, const size_type Ny) {
  using crs_matrix_type = KokkosSparse::CrsMatrix<scalar_type, ordinal_type,
                                                  device, void, size_type>;
  using crs_graph_type  = typename crs_matrix_type::StaticCrsGraphType;
  using row_map_type    = typename crs_graph_type::row_map_type::non_const_type;
  using col_ind_type    = typename crs_graph_type::entries_type::non_const_type;
  using values_type     = typename crs_matrix_type::values_type::non_const_type;
  using value_type      = typename crs_matrix_type::value_type;
  using execution_space = typename crs_matrix_type::execution_space;

  const ordinal_type numRows  = Nx * Ny;
  const ordinal_type& numCols = numRows;
  row_map_type row_map(Kokkos::ViewAllocateWithoutInitializing("row map"),
                       numRows + 1);

  ordinal_type numNonZeros = 0;
  Kokkos::parallel_scan(
      Kokkos::RangePolicy<execution_space>(ordinal_type(0),
                                           ordinal_type(numRows + 1)),
      KOKKOS_LAMBDA(ordinal_type i, ordinal_type & runningNZ, bool is_final) {
        const auto curr_val = (i == 0) ? 1 : 5;
        if (is_final) row_map[i] = runningNZ;
        if (i < numRows) runningNZ += curr_val;
      },
      numNonZeros);

  col_ind_type col_ind("column indices", numNonZeros);
  values_type values("values", numNonZeros);
  Kokkos::parallel_for(
      Kokkos::MDRangePolicy<Kokkos::Rank<2> >({0, 0}, {Nx, Ny}),
      KOKKOS_LAMBDA(ordinal_type iX, ordinal_type iY) {
        const ordinal_type row_XY = iX + Nx * iY;
        auto map_ind              = row_map(row_XY);
        if (row_XY == 0) {
          col_ind(map_ind) = row_XY;
          values(map_ind)  = 1.;
          return;
        }

        const ordinal_type nX = (iX + Nx - 1) % Nx;
        const ordinal_type pX = (iX + 1) % Nx;
        const ordinal_type nY = (iY + Ny - 1) % Ny;
        const ordinal_type pY = (iY + 1) % Ny;

        const ordinal_type row_pXY = pX + Nx * iY;
        const ordinal_type row_nXY = nX + Nx * iY;
        const ordinal_type row_XpY = iX + Nx * pY;
        const ordinal_type row_XnY = iX + Nx * nY;

        // Negative y dir
        col_ind(map_ind) = row_XnY;
        values(map_ind)  = beta;
        ++map_ind;
        // Negative x dir
        col_ind(map_ind) = row_nXY;
        values(map_ind)  = beta - vel_mag;
        ++map_ind;
        // Middle
        col_ind(map_ind) = row_XY;
        values(map_ind)  = -4.0 * beta + vel_mag;
        ++map_ind;
        // Positive x dir
        col_ind(map_ind) = row_pXY;
        values(map_ind)  = beta;
        ++map_ind;
        // Positive y dir
        col_ind(map_ind) = row_XpY;
        values(map_ind)  = beta;
      });

  return crs_matrix_type("A", numRows, numCols, numNonZeros, values, row_map,
                         col_ind);
}

template <typename scalar_type, typename ordinal_type, typename size_type,
          typename device>
void run_test_mdf_recr_issue() {  //

  // using execution_space = Kokkos::Serial;
  using execution_space = typename device::execution_space;

  constexpr int num_teams    = 10;
  constexpr int num_per_team = 10;
  Kokkos::View<int**, execution_space> m_data(
      Kokkos::ViewAllocateWithoutInitializing("data"), num_teams, num_per_team);
  Kokkos::View<int*, execution_space> m_num_entr(
      Kokkos::ViewAllocateWithoutInitializing("data"), num_teams);

  using team_policy_t = Kokkos::TeamPolicy<execution_space>;
  using member_t      = typename team_policy_t::member_type;

  Kokkos::parallel_for(team_policy_t(num_teams, Kokkos::AUTO, Kokkos::AUTO),
                       KOKKOS_LAMBDA(member_t team) {
                         const auto iTeam = team.league_rank();

                         // int num_added;
                         Kokkos::parallel_scan(
                             Kokkos::TeamVectorRange(team, num_per_team),
                             [&](int i, int& partial_num, bool final) {
                               if (final) m_data(iTeam, i) = partial_num;
                               partial_num += i;
                             });

                         // // Do something with num_entr ...
                         // Kokkos::single(Kokkos::PerTeam(team),[&]{
                         //   m_num_entr(iTeam) = num_added;
                         // });
                       });
}

template <typename scalar_type, typename ordinal_type, typename size_type,
          typename device>
void run_test_mdf() {  //_timing
  using crs_matrix_type = KokkosSparse::CrsMatrix<scalar_type, ordinal_type,
                                                  device, void, size_type>;
  using crs_graph_type  = typename crs_matrix_type::StaticCrsGraphType;
  using row_map_type    = typename crs_graph_type::row_map_type::non_const_type;
  using col_ind_type    = typename crs_graph_type::entries_type::non_const_type;
  using values_type     = typename crs_matrix_type::values_type::non_const_type;
  using value_type      = typename crs_matrix_type::value_type;
  using execution_space = typename crs_matrix_type::execution_space;

  const scalar_type beta    = 1.0;
  const scalar_type vel_mag = 0.5;
  const size_type Nx        = 400;
  const size_type Ny        = 400;

  crs_matrix_type A =
      make_adv_diffusion_matrix<scalar_type, ordinal_type, size_type, device>(
          beta, vel_mag, Nx, Ny);

  KokkosSparse::Experimental::MDF_handle<crs_matrix_type> handle(A);
  handle.set_verbosity(0);
  KokkosSparse::Experimental::mdf_symbolic(A, handle);
  KokkosSparse::Experimental::mdf_numeric(A, handle);
}

template <typename scalar_type, typename ordinal_type, typename size_type,
          typename device>
void run_test_mdf_real() {  //
  using crs_matrix_type = KokkosSparse::CrsMatrix<scalar_type, ordinal_type,
                                                  device, void, size_type>;
  using crs_graph_type  = typename crs_matrix_type::StaticCrsGraphType;
  using row_map_type    = typename crs_graph_type::row_map_type::non_const_type;
  using col_ind_type    = typename crs_graph_type::entries_type::non_const_type;
  using values_type     = typename crs_matrix_type::values_type::non_const_type;
  using value_type      = typename crs_matrix_type::value_type;

  const value_type four = static_cast<value_type>(4.0);

  constexpr ordinal_type numRows  = 16;
  constexpr ordinal_type numCols  = 16;
  constexpr size_type numNonZeros = 64;
  row_map_type row_map("row map", numRows + 1);
  col_ind_type col_ind("column indices", numNonZeros);
  values_type values("values", numNonZeros);

  {  // create matrix
    const size_type row_mapRaw[]    = {0,  3,  7,  11, 14, 18, 23, 28, 32,
                                    36, 41, 46, 50, 53, 57, 61, 64};
    const ordinal_type col_indRaw[] = {
        0,  1,  4, 0,  1,  2, 5,  1,  2,  3,  6,  2,  3,  7,  0,  4,
        5,  8,  1, 4,  5,  6, 9,  2,  5,  6,  7,  10, 3,  6,  7,  11,
        4,  8,  9, 12, 5,  8, 9,  10, 13, 6,  9,  10, 11, 14, 7,  10,
        11, 15, 8, 12, 13, 9, 12, 13, 14, 10, 13, 14, 15, 11, 14, 15};
    const value_type values_Raw[] = {
        4,  -1, -1, -1, 4,  -1, -1, -1, 4,  -1, -1, -1, 4,  -1, -1, 4,
        -1, -1, -1, -1, 4,  -1, -1, -1, -1, 4,  -1, -1, -1, -1, 4,  -1,
        -1, 4,  -1, -1, -1, -1, 4,  -1, -1, -1, -1, 4,  -1, -1, -1, -1,
        4,  -1, -1, 4,  -1, -1, -1, 4,  -1, -1, -1, 4,  -1, -1, -1, 4};

    typename row_map_type::HostMirror::const_type row_map_host(row_mapRaw,
                                                               numRows + 1);
    typename col_ind_type::HostMirror::const_type col_ind_host(col_indRaw,
                                                               numNonZeros);
    typename values_type::HostMirror::const_type values_host(values_Raw,
                                                             numNonZeros);

    Kokkos::deep_copy(row_map, row_map_host);
    Kokkos::deep_copy(col_ind, col_ind_host);
    Kokkos::deep_copy(values, values_host);
  }

  crs_matrix_type A = crs_matrix_type("A", numRows, numCols, numNonZeros,
                                      values, row_map, col_ind);

  KokkosSparse::Experimental::MDF_handle<crs_matrix_type> handle(A);
  handle.set_verbosity(0);
  KokkosSparse::Experimental::mdf_symbolic(A, handle);
  KokkosSparse::Experimental::mdf_numeric(A, handle);

  col_ind_type permutation = handle.get_permutation();

  bool success = true;
  typename col_ind_type::HostMirror permutation_h =
      Kokkos::create_mirror(permutation);
  Kokkos::deep_copy(permutation_h, permutation);
  const ordinal_type permutation_ref[] = {0, 3,  12, 15, 1, 2, 4, 8,
                                          7, 11, 13, 14, 5, 6, 9, 10};
  printf("MDF ordering: { ");
  for (ordinal_type idx = 0; idx < A.numRows(); ++idx) {
    printf("%d ", static_cast<int>(permutation_h(idx)));
    if (permutation_h(idx) != permutation_ref[idx]) {
      success = false;
    }
  }
  printf("}\n");
  EXPECT_TRUE(success)
      << "The permutation computed is different from the reference solution!";

  // Check the factors L and U
  handle.sort_factors();
  crs_matrix_type U = handle.getU();
  crs_matrix_type L = handle.getL();

  EXPECT_TRUE(U.numRows() == 16);
  EXPECT_TRUE(U.nnz() == 40);

  {
    auto row_map_U = Kokkos::create_mirror(U.graph.row_map);
    Kokkos::deep_copy(row_map_U, U.graph.row_map);
    auto entries_U = Kokkos::create_mirror(U.graph.entries);
    Kokkos::deep_copy(entries_U, U.graph.entries);
    auto values_U = Kokkos::create_mirror(U.values);
    Kokkos::deep_copy(values_U, U.values);

    const size_type row_map_U_ref[17]    = {0,  3,  6,  9,  12, 15, 17, 20, 22,
                                         25, 27, 30, 32, 35, 37, 39, 40};
    const ordinal_type entries_U_ref[40] = {
        0,  4,  6,  1,  5,  8,  2,  7,  10, 3,  9,  11, 4,  5,
        12, 5,  13, 6,  7,  12, 7,  14, 8,  9,  13, 9,  15, 10,
        11, 14, 11, 15, 12, 13, 14, 13, 15, 14, 15, 15};

    const scalar_type val0 = static_cast<scalar_type>(15. / 4.);
    const scalar_type val1 = static_cast<scalar_type>(val0 - 1 / val0);
    const scalar_type val2 = static_cast<scalar_type>(4 - 2 / val0);
    const scalar_type val3 =
        static_cast<scalar_type>(4 - 1 / val0 - 1 / val1 - 1 / val2);
    const scalar_type val4 = static_cast<scalar_type>(4 - 2 / val1 - 2 / val3);
    const scalar_type values_U_ref[40] = {
        4,    -1, -1,   4,  -1, -1,   4,  -1,   -1, 4,   -1,   -1, val0, -1, -1,
        val1, -1, val0, -1, -1, val1, -1, val0, -1, -1,  val1, -1, val0, -1, -1,
        val1, -1, val2, -1, -1, val3, -1, val3, -1, val4};

    for (int idx = 0; idx < 17; ++idx) {
      EXPECT_TRUE(row_map_U_ref[idx] == row_map_U(idx))
          << "rowmap_U(" << idx << ") is wrong!";
    }
    for (int idx = 0; idx < 40; ++idx) {
      EXPECT_TRUE(entries_U_ref[idx] == entries_U(idx))
          << "entries_U(" << idx << ") is wrong!";
      EXPECT_NEAR_KK(values_U_ref[idx], values_U(idx),
                     10 * Kokkos::ArithTraits<scalar_type>::eps(),
                     "An entry in U.values is wrong!");
    }

    auto row_map_L = Kokkos::create_mirror(L.graph.row_map);
    Kokkos::deep_copy(row_map_L, L.graph.row_map);
    auto entries_L = Kokkos::create_mirror(L.graph.entries);
    Kokkos::deep_copy(entries_L, L.graph.entries);
    auto values_L = Kokkos::create_mirror(L.values);
    Kokkos::deep_copy(values_L, L.values);

    const size_type row_map_L_ref[17]    = {0,  1,  2,  3,  4,  6,  9,  11, 14,
                                         16, 19, 21, 24, 27, 31, 35, 40};
    const ordinal_type entries_L_ref[40] = {
        0, 1,  2,  3, 0,  4,  1,  4, 5,  0,  6,  2, 6,  7,
        1, 8,  3,  8, 9,  2,  10, 3, 10, 11, 4,  6, 12, 5,
        8, 12, 13, 7, 10, 12, 14, 9, 11, 13, 14, 15};
    const scalar_type values_L_ref[40] = {
        1,         1,         1,         1,         -1 / four, 1,
        -1 / four, -1 / val0, 1,         -1 / four, 1,         -1 / four,
        -1 / val0, 1,         -1 / four, 1,         -1 / four, -1 / val0,
        1,         -1 / four, 1,         -1 / four, -1 / val0, 1,
        -1 / val0, -1 / val0, 1,         -1 / val1, -1 / val0, -1 / val2,
        1,         -1 / val1, -1 / val0, -1 / val2, 1,         -1 / val1,
        -1 / val1, -1 / val3, -1 / val3, 1};

    for (int idx = 0; idx < 17; ++idx) {
      EXPECT_TRUE(row_map_L_ref[idx] == row_map_L(idx))
          << "rowmap_L(" << idx << ")=" << row_map_L(idx) << " is wrong!";
    }
    for (int idx = 0; idx < 40; ++idx) {
      EXPECT_TRUE(entries_L_ref[idx] == entries_L(idx))
          << "entries_L(" << idx << ")=" << entries_L(idx)
          << " is wrong, entries_L_ref[" << idx << "]=" << entries_L_ref[idx]
          << "!";
      EXPECT_NEAR_KK(values_L_ref[idx], values_L(idx),
                     10 * Kokkos::ArithTraits<scalar_type>::eps(),
                     "An entry in L.values is wrong!");
    }
  }
}

}  // namespace Test

template <typename scalar_t, typename lno_t, typename size_type,
          typename device>
void test_mdf() {
  Test::run_test_mdf<scalar_t, lno_t, size_type, device>();
}

#define KOKKOSKERNELS_EXECUTE_TEST(SCALAR, ORDINAL, OFFSET, DEVICE)     \
  TEST_F(TestCategory,                                                  \
         sparse##_##mdf##_##SCALAR##_##ORDINAL##_##OFFSET##_##DEVICE) { \
    test_mdf<SCALAR, ORDINAL, OFFSET, DEVICE>();                        \
  }

#include <Test_Common_Test_All_Type_Combos.hpp>

#undef KOKKOSKERNELS_EXECUTE_TEST
