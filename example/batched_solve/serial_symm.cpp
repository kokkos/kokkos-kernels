// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright Contributors to the Kokkos project

#include <iostream>
#include <Kokkos_Core.hpp>
#include <KokkosBatched_Symm.hpp>

using ExecutionSpace = Kokkos::DefaultExecutionSpace;

/// \brief Example of batched symm
/// Perform a symmetric matrix-matrix multiplication, where
///        3x3 matrix
///        A: [[4, 1, 2],
///            [-1, 5, 3],
///            [7, 9, 6]]
///        B: [[1, 2],      C: [[1, 0],
///            [3, 4],          [0, 1],
///            [5, 6]]          [1, 1]]
///   alpha: 1.5, beta: 1.2
///
/// After C := alpha*A*B + beta*C
///        Lower       
///        Ref: [[55.2, 69],
///              [88.5, 109.2],
///              [97.2, 130.2]] 
///
int main(int /*argc*/, char** /*argv*/) {
  Kokkos::initialize();
  {
    using View3DType = Kokkos::View<double***, ExecutionSpace>;
    const int Nb = 10, m = 2, n = 3;

    // Matrix A
    View3DType A("A", Nb, n, n);

    // Matrix B, C and reference
    View3DType B("B", Nb, n, m), C("C", Nb, n, m), Ref("Ref", Nb, n, m);

    // Initialize A, x and y
    auto h_A   = Kokkos::create_mirror_view(A);
    auto h_B   = Kokkos::create_mirror_view(B);
    auto h_C   = Kokkos::create_mirror_view(C);
    auto h_Ref = Kokkos::create_mirror_view(Ref);

    // Upper triangular matrix
    for (int ib = 0; ib < Nb; ib++) {
      // Fill the matrix A
      h_A(ib, 0, 0) = 4;
      h_A(ib, 0, 1) = 1;
      h_A(ib, 0, 2) = 2;
      h_A(ib, 1, 0) = -1;
      h_A(ib, 1, 1) = 5;
      h_A(ib, 1, 2) = 3;
      h_A(ib, 2, 0) = 7;
      h_A(ib, 2, 1) = 9;
      h_A(ib, 2, 2) = 6;

      // Fill matrix B
      h_B(ib, 0, 0) = 1;
      h_B(ib, 0, 1) = 2;
      h_B(ib, 1, 0) = 3;
      h_B(ib, 1, 1) = 4;
      h_B(ib, 2, 0) = 5;
      h_B(ib, 2, 1) = 6;

      // Fill matrix C
      h_C(ib, 0, 0) = 1;
      h_C(ib, 0, 1) = 0;
      h_C(ib, 1, 0) = 0;
      h_C(ib, 1, 1) = 1;
      h_C(ib, 2, 0) = 1;
      h_C(ib, 2, 1) = 1;

      // Fill Reference matrix
      h_Ref(ib, 0, 0) = 55.2;
      h_Ref(ib, 0, 1) = 69;
      h_Ref(ib, 1, 0) = 88.5;
      h_Ref(ib, 1, 1) = 109.2;
      h_Ref(ib, 2, 0) = 97.2;
      h_Ref(ib, 2, 1) = 130.2;
    }
    Kokkos::deep_copy(A, h_A);
    Kokkos::deep_copy(B, h_B);
    Kokkos::deep_copy(C, h_C);

    // Compute C = alpha * A * B + beta * C
    const double alpha = 1.5, beta = 1.2;
    ExecutionSpace exec;
    using policy_type = Kokkos::RangePolicy<ExecutionSpace, Kokkos::IndexType<int>>;
    policy_type policy{exec, 0, Nb};
    Kokkos::parallel_for(
        "symm", policy, KOKKOS_LAMBDA(int ib) {
          auto sub_A = Kokkos::subview(A, ib, Kokkos::ALL, Kokkos::ALL);
          auto sub_B = Kokkos::subview(B, ib, Kokkos::ALL, Kokkos::ALL);
          auto sub_C = Kokkos::subview(C, ib, Kokkos::ALL, Kokkos::ALL);

          // C = alpha * A * B + beta * C
          KokkosBatched::SerialSymm<KokkosBatched::Side::Left, KokkosBatched::Uplo::Lower, false>::invoke(
              alpha, sub_A, sub_B, beta, sub_C);
        });

    // Confirm that the results are correct
    Kokkos::deep_copy(h_C, C);
    bool correct = true;
    double eps   = 1.0e-12;
    for (int ib = 0; ib < h_C.extent_int(0); ib++) {
      for (int i = 0; i < h_C.extent_int(1); i++) {
        for (int j = 0; j < h_C.extent_int(2); j++) {
          if (Kokkos::abs(h_C(ib, i, j) - h_Ref(ib, i, j)) > eps) correct = false;
        }
      }
    }

    if (correct) {
      std::cout << "symm works correctly!" << std::endl;
    }
  }
  Kokkos::finalize();
}
