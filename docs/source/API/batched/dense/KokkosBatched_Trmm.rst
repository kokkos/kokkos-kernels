KokkosBatched::Trmm
##################

Defined in header `KokkosBatched_Trmm_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Trmm_Decl.hpp>`_

.. code:: c++

    template <typename ArgSide, typename ArgUplo, typename ArgTrans, typename ArgDiag, typename ArgAlgo>
    struct SerialTrmm {
      template <typename ScalarType, typename AViewType, typename BViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType alpha, 
                                              const AViewType &A, 
                                              const BViewType &B);
    };

Performs batched triangular matrix-matrix multiplication (TRMM). For each triangular matrix A and general matrix B in the batch, computes:

.. math::

   B \leftarrow \alpha \cdot \text{op}(A) \cdot B

or:

.. math::

   B \leftarrow \alpha \cdot B \cdot \text{op}(A)

where:

- :math:`\text{op}(A)` can be :math:`A`, :math:`A^T` (transpose), or :math:`A^H` (Hermitian transpose)
- :math:`A` is a triangular matrix (upper or lower triangular)
- :math:`B` is a general matrix (overwritten with the result)
- :math:`\alpha` is a scalar value

The operation performs matrix-matrix multiplication where one of the matrices is triangular, which can be computed more efficiently than general matrix-matrix multiplication.

Parameters
==========

:alpha: Scalar multiplier
:A: Input view containing triangular matrices
:B: Input/output view for general matrices (overwritten with result)

Type Requirements
----------------

- ``ArgSide`` must be one of:

  - ``Side::Left`` - compute op(A)*B
  - ``Side::Right`` - compute B*op(A)

- ``ArgUplo`` must be one of:

  - ``Uplo::Upper`` - A is upper triangular
  - ``Uplo::Lower`` - A is lower triangular

- ``ArgTrans`` must be one of:

  - ``Trans::NoTranspose`` - use A as is
  - ``Trans::Transpose`` - use transpose of A
  - ``Trans::ConjTranspose`` - use conjugate transpose of A

- ``ArgDiag`` must be one of:

  - ``Diag::Unit`` - A has an implicit unit diagonal
  - ``Diag::NonUnit`` - A has a non-unit diagonal

- ``ArgAlgo`` must be one of the algorithm variants (implementation dependent)
- ``AViewType`` must be a rank-2 or rank-3 Kokkos View representing triangular matrices
- ``BViewType`` must be a rank-2 or rank-3 Kokkos View representing general matrices

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Trmm_Decl.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 1000;  // Number of matrix pairs
        int m = 4;              // Size of A (m × m)
        int n = 3;              // Columns in B (for B: m × n)
        
        // Create views for batched matrices
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, m, m),           // Triangular matrices
          B("B", batch_size, m, n),           // General matrices
          B_copy("B_copy", batch_size, m, n); // Copy for verification
        
        // Fill matrices with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th A as an upper triangular matrix
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < m; ++col) {
              if (row <= col) {
                // Upper triangular part (including diagonal)
                A(i, row, col) = 2.0;  // Simple value for verification
              } else {
                // Below diagonal (not used for upper triangular)
                A(i, row, col) = 0.0;
              }
            }
          }
          
          // Initialize B with simple pattern
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
              B(i, row, col) = 1.0;  // All ones for simple verification
              
              // Copy B for later verification
              B_copy(i, row, col) = B(i, row, col);
            }
          }
        });
        
        Kokkos::fence();
        
        // Scalar multiplier
        scalar_type alpha = 3.0;
        
        // Perform batched TRMM: B = alpha * A * B (Left side, Upper triangular)
        Kokkos::parallel_for("batch_trmm", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slices
          auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          auto B_i = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
          
          // Perform triangular matrix-matrix multiplication
          KokkosBatched::SerialTrmm<
            KokkosBatched::Side::Left,        // ArgSide (A on left)
            KokkosBatched::Uplo::Upper,       // ArgUplo (A is upper triangular)
            KokkosBatched::Trans::NoTranspose, // ArgTrans (use A as is)
            KokkosBatched::Diag::NonUnit,     // ArgDiag (A has non-unit diagonal)
            KokkosBatched::Algo::Trmm::Unblocked // ArgAlgo
          >::invoke(alpha, A_i, B_i);
        });
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(A, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto B_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(B, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto B_copy_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                              Kokkos::subview(B_copy, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        // Verify the result by computing alpha * A * B manually
        printf("TRMM verification for first matrix pair:\n");
        printf("A (triangular matrix):\n");
        for (int row = 0; row < m; ++row) {
          printf("  [");
          for (int col = 0; col < m; ++col) {
            printf("%5.1f", A_host(row, col));
            if (col < m-1) printf(", ");
          }
          printf("]\n");
        }
        
        printf("\nOriginal B:\n");
        for (int row = 0; row < m; ++row) {
          printf("  [");
          for (int col = 0; col < n; ++col) {
            printf("%5.1f", B_copy_host(row, col));
            if (col < n-1) printf(", ");
          }
          printf("]\n");
        }
        
        printf("\nResult B (after TRMM):\n");
        for (int row = 0; row < m; ++row) {
          printf("  [");
          for (int col = 0; col < n; ++col) {
            printf("%5.1f", B_host(row, col));
            if (col < n-1) printf(", ");
          }
          printf("]\n");
        }
        
        // Manual verification by computing alpha * A * B directly
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace>
          expected("expected", m, n);
        
        printf("\nManual verification (alpha * A * B):\n");
        bool correct = true;
        
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            expected(i, j) = 0.0;
            
            // Since A is upper triangular, we only need elements where col >= row
            for (int k = 0; k <= i; ++k) {
              expected(i, j) += alpha * A_host(i, k) * B_copy_host(k, j);
            }
            
            scalar_type error = std::abs(expected(i, j) - B_host(i, j));
            printf("  Expected B(%d,%d) = %5.1f, Computed = %5.1f, Error = %.6e\n",
                   i, j, expected(i, j), B_host(i, j), error);
            
            if (error > 1e-10) {
              correct = false;
            }
          }
        }
        
        if (correct) {
          printf("\nSUCCESS: TRMM result matches manual computation\n");
        } else {
          printf("\nERROR: TRMM result doesn't match manual computation\n");
        }
        
        // Demonstrate right-side multiplication (B = alpha * B * A)
        // Reset B to original values
        Kokkos::deep_copy(B, B_copy);
        
        // Perform batched TRMM: B = alpha * B * A (Right side, Upper triangular)
        Kokkos::parallel_for("batch_trmm_right", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slices
          auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          auto B_i = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
          
          // Perform triangular matrix-matrix multiplication with A on right
          KokkosBatched::SerialTrmm<
            KokkosBatched::Side::Right,       // ArgSide (A on right)
            KokkosBatched::Uplo::Upper,       // ArgUplo (A is upper triangular)
            KokkosBatched::Trans::NoTranspose, // ArgTrans (use A as is)
            KokkosBatched::Diag::NonUnit,     // ArgDiag (A has non-unit diagonal)
            KokkosBatched::Algo::Trmm::Unblocked // ArgAlgo
          >::invoke(alpha, A_i, B_i);
        });
        
        Kokkos::fence();
        
        // Copy right-side multiplication results to host
        auto B_right_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                               Kokkos::subview(B, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        printf("\nRight-side multiplication result (B = alpha * B * A):\n");
        for (int row = 0; row < m; ++row) {
          printf("  [");
          for (int col = 0; col < n; ++col) {
            printf("%5.1f", B_right_host(row, col));
            if (col < n-1) printf(", ");
          }
          printf("]\n");
        }
      }
      Kokkos::finalize();
      return 0;
    }
