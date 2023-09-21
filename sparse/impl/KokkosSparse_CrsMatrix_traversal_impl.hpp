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

namespace KokkosSparse {
namespace Impl {

template <class execution_space, class matrix_type, class functor_type>
struct crsmatrix_traversal_functor {
  using size_type    = typename matrix_type::non_const_size_type;
  using ordinal_type = typename matrix_type::non_const_ordinal_type;
  using value_type   = typename matrix_type::non_const_value_type;

  using team_policy_type = Kokkos::TeamPolicy<execution_space>;
  using team_member_type = typename team_policy_type::member_type;

  matrix_type A;
  functor_type func;
  ordinal_type rows_per_team;

  crsmatrix_traversal_functor(const matrix_type& A_, const functor_type& func_,
                              const ordinal_type rows_per_team_)
      : A(A_), func(func_), rows_per_team(rows_per_team_) {}

  // RangePolicy overload
  KOKKOS_INLINE_FUNCTION void operator()(const ordinal_type rowIdx) const {
    for (size_type entryIdx = A.graph.row_map(rowIdx);
         entryIdx < A.graph.row_map(rowIdx + 1); ++entryIdx) {
      const ordinal_type colIdx = A.graph.entries(entryIdx);
      const value_type value    = A.values(entryIdx);

      func(rowIdx, entryIdx, colIdx, value);
    }
  }

  // TeamPolicy overload
  KOKKOS_INLINE_FUNCTION void operator()(const team_member_type& dev) const {
    const ordinal_type teamWork = dev.league_rank() * rows_per_team;
    Kokkos::parallel_for(
        Kokkos::TeamThreadRange(dev, rows_per_team), [&](ordinal_type loop) {
          // iRow represents a row of the matrix, so its correct type is
          // ordinal_type.
          const ordinal_type rowIdx = teamWork + loop;
          if (rowIdx >= A.numRows()) {
            return;
          }

          const ordinal_type row_length =
              A.graph.row_map(rowIdx + 1) - A.graph.row_map(rowIdx);
          Kokkos::parallel_for(
              Kokkos::ThreadVectorRange(dev, row_length),
              [&](ordinal_type rowEntryIdx) {
                const size_type entryIdx = A.graph.row_map(rowIdx) +
                                           static_cast<size_type>(rowEntryIdx);
                const ordinal_type colIdx = A.graph.entries(entryIdx);
                const value_type value    = A.values(entryIdx);

                func(rowIdx, entryIdx, colIdx, value);
              });
        });
  }
};

template <class execution_space>
int64_t crsmatrix_traversal_launch_parameters(int64_t numRows, int64_t nnz,
                                              int64_t rows_per_thread,
                                              int& team_size,
                                              int& vector_length) {
  int64_t rows_per_team;
  int64_t nnz_per_row = nnz / numRows;

  if (nnz_per_row < 1) nnz_per_row = 1;

  int max_vector_length = Kokkos::TeamPolicy<execution_space>::vector_length_max();

  if (vector_length < 1) {
    vector_length = 1;
    while (vector_length < max_vector_length && vector_length * 6 < nnz_per_row)
      vector_length *= 2;
  }

  // Determine rows per thread
  if (rows_per_thread < 1) {
    if (KokkosKernels::Impl::kk_is_gpu_exec_space<execution_space>())
      rows_per_thread = 1;
    else {
      if (nnz_per_row < 20 && nnz > 5000000) {
        rows_per_thread = 256;
      } else
        rows_per_thread = 64;
    }
  }

  if (team_size < 1) {
    if (KokkosKernels::Impl::kk_is_gpu_exec_space<execution_space>()) {
      team_size = 256 / vector_length;
    } else {
      team_size = 1;
    }
  }

  rows_per_team = rows_per_thread * team_size;

  return rows_per_team;
}

template <class execution_space, class crsmatrix_type, class functor_type>
void crsmatrix_traversal_on_host(const execution_space& space,
                                 const crsmatrix_type& A,
                                 const functor_type& func) {
  // Wrap user functor with crsmatrix_traversal_functor
  crsmatrix_traversal_functor<execution_space, crsmatrix_type, functor_type>
      traversal_func(A, func, -1);

  // Launch traversal kernel
  Kokkos::parallel_for(
      "KokkosSparse::crsmatrix_traversal",
      Kokkos::RangePolicy<execution_space>(space, 0, A.numRows()),
      traversal_func);
}

template <class execution_space, class crsmatrix_type, class functor_type>
void crsmatrix_traversal_on_gpu(const execution_space& space,
                                const crsmatrix_type& A,
                                const functor_type& func) {
  // Wrap user functor with crsmatrix_traversal_functor
  int64_t rows_per_thread = 0;
  int team_size = 0, vector_length = 0;
  const int64_t rows_per_team =
      crsmatrix_traversal_launch_parameters<execution_space>(
          A.numRows(), A.nnz(), rows_per_thread, team_size, vector_length);
  const int nteams =
      (static_cast<int>(A.numRows()) + rows_per_team - 1) / rows_per_team;
  crsmatrix_traversal_functor<execution_space, crsmatrix_type, functor_type>
      traversal_func(A, func, rows_per_team);

  // Launch traversal kernel
  Kokkos::parallel_for("KokkosSparse::crsmatrix_traversal",
                       Kokkos::TeamPolicy<execution_space>(
                           space, nteams, team_size, vector_length),
                       traversal_func);
}

}  // namespace Impl
}  // namespace KokkosSparse
