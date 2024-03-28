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

/// \file Test_Sparse_SortCrs.hpp
/// \brief Tests for sort_crs_matrix and sort_crs_graph in
/// KokkosSparse_CrsMatrix_traversal.hpp

#ifndef TEST_SPARSE_CRSMATRIX_TRAVERSAL_HPP
#define TEST_SPARSE_CRSMATRIX_TRAVERSAL_HPP

#include <Kokkos_Core.hpp>

#include "KokkosKernels_Test_Structured_Matrix.hpp"
#include "KokkosSparse_CrsMatrix_traversal.hpp"

namespace TestCrsMatrixTraversal {

template <class CrsMatrix>
struct diag_extraction {
  using diag_view    = typename CrsMatrix::values_type::non_const_type;
  using size_type    = typename CrsMatrix::non_const_size_type;
  using ordinal_type = typename CrsMatrix::non_const_ordinal_type;
  using value_type   = typename CrsMatrix::non_const_value_type;

  diag_view diag;

  diag_extraction(CrsMatrix A) {
    diag = diag_view("diag values", A.numRows());
  };

  KOKKOS_INLINE_FUNCTION void operator()(const ordinal_type rowIdx,
                                         const size_type /*entryIdx*/,
                                         const ordinal_type colIdx,
                                         const value_type value) const {
    if (rowIdx == colIdx) {
      diag(rowIdx) = value;
    }
  }
};

}  // namespace TestCrsMatrixTraversal

void testCrsMatrixTraversal(int testCase) {
  using namespace TestCrsMatrixTraversal;
  using Device =
      Kokkos::Device<TestExecSpace, typename TestExecSpace::memory_space>;
  using Matrix = KokkosSparse::CrsMatrix<default_scalar, default_lno_t, Device,
                                         void, default_size_type>;
  using Vector = Kokkos::View<default_scalar*, TestExecSpace::memory_space>;

  constexpr int nx = 4, ny = 4;
  constexpr bool leftBC = true, rightBC = false, topBC = false, botBC = false;

  Kokkos::View<int * [3], Kokkos::HostSpace> mat_structure("Matrix Structure",
                                                           2);
  mat_structure(0, 0) = nx;
  mat_structure(0, 1) = (leftBC ? 1 : 0);
  mat_structure(0, 2) = (rightBC ? 1 : 0);

  mat_structure(1, 0) = ny;
  mat_structure(1, 1) = (topBC ? 1 : 0);
  mat_structure(1, 2) = (botBC ? 1 : 0);

  Matrix A = Test::generate_structured_matrix2D<Matrix>("FD", mat_structure);

  Vector diag_ref("diag ref", A.numRows());
  auto diag_ref_h = Kokkos::create_mirror_view(diag_ref);
  diag_ref_h(0)   = 1;
  diag_ref_h(1)   = 3;
  diag_ref_h(2)   = 3;
  diag_ref_h(3)   = 2;
  diag_ref_h(4)   = 1;
  diag_ref_h(5)   = 4;
  diag_ref_h(6)   = 4;
  diag_ref_h(7)   = 3;
  diag_ref_h(8)   = 1;
  diag_ref_h(9)   = 4;
  diag_ref_h(10)  = 4;
  diag_ref_h(11)  = 3;
  diag_ref_h(12)  = 1;
  diag_ref_h(13)  = 3;
  diag_ref_h(14)  = 3;
  diag_ref_h(15)  = 2;

  // Run the diagonal extraction functor
  // using traversal function.
  diag_extraction<Matrix> func(A);
  KokkosSparse::Experimental::crsmatrix_traversal(A, func);
  Kokkos::fence();

  // Extract the diagonal view from functor
  auto diag_h = Kokkos::create_mirror_view(func.diag);
  Kokkos::deep_copy(diag_h, func.diag);

  // Check for correctness
  bool matches = true;
  for (int rowIdx = 0; rowIdx < A.numRows(); ++rowIdx) {
    if (diag_ref_h(rowIdx) != diag_h(rowIdx)) matches = false;
  }

  EXPECT_TRUE(matches)
      << "Test case " << testCase
      << ": matrix with zeros filtered out does not match reference.";
}

TEST_F(TestCategory, sparse_crsmatrix_traversal) {
  for (int testCase = 0; testCase < 1; testCase++)
    testCrsMatrixTraversal(testCase);
}

#endif  // TEST_SPARSE_CRSMATRIX_TRAVERSAL_HPP
