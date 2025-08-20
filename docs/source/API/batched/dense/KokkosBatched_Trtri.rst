KokkosBatched::Trtri
####################

Defined in header: :code:`KokkosBatched_Trtri_Decl.hpp`

.. code:: c++

    template <typename ArgUplo, typename ArgDiag, typename ArgAlgo>
    struct SerialTrtri {
      template <typename ScalarType, typename AViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A);
    };

Computes the inverse of a triangular matrix. For each triangular matrix A in the batch, computes:

.. math::

   A \leftarrow A^{-1}

This operation computes the inverse of a triangular matrix in-place, overwriting the input matrix A with its inverse. The inverse of a triangular matrix is also triangular with the same structure.

Parameters
==========

:A: Input/output view containing triangular matrices to be inverted

Type Requirements
-----------------

- ``ArgUplo`` must be one of:

  - ``Uplo::Upper`` - A is upper triangular
  - ``Uplo::Lower`` - A is lower triangular

- ``ArgDiag`` must be one of:

  - ``Diag::Unit`` - A has an implicit unit diagonal
  - ``Diag::NonUnit`` - A has a non-unit diagonal

- ``ArgAlgo`` must be one of:

  - ``Algo::Trtri::Unblocked`` - for small matrices
  - ``Algo::Trtri::Blocked`` - for larger matrices with blocking

- ``AViewType`` must be a rank-2 or rank-3 Kokkos View representing triangular matrices

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Trtri_Decl.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 1000;  // Number of matrices
        int n = 4;              // Size of each triangular matrix
        
        // Create views for batched matrices
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, n, n),           // Triangular matrices
          A_copy("A_copy", batch_size, n, n); // Copy for verification
        
        // Fill matrices with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th upper triangular matrix
          // Use a simple pattern to ensure invertibility:
          // - Diagonal elements > sum of other elements in row
          for (int row = 0; row < n; ++row) {
            // Zero out elements below diagonal (upper triangular)
            for (int col = 0; col < row; ++col) {
              A(i, row, col) = 0.0;
            }
            
            // Set upper triangular part
            for (int col = row; col < n; ++col) {
              if (row == col) {
                A(i, row, col) = n + 1.0;  // Diagonal elements
              } else {
                A(i, row, col) = 1.0;      // Above diagonal elements
              }
            }
          }
          
          // Copy A for verification
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
              A_copy(i, row, col) = A(i, row, col);
            }
          }
        });
        
        Kokkos::fence();
        
        // Compute triangular matrix inverse
        Kokkos::parallel_for("batch_trtri", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slice
          auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          
          // Compute inverse of triangular matrix
          KokkosBatched::SerialTrtri<
            KokkosBatched::Uplo::Upper,       // ArgUplo (upper triangular)
            KokkosBatched::Diag::NonUnit,     // ArgDiag (non-unit diagonal)
            KokkosBatched::Algo::Trtri::Unblocked // ArgAlgo
          >::invoke(A_i);
        });
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto A_copy_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                              Kokkos::subview(A_copy, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto A_inv_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                             Kokkos::subview(A, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        // Verify the inverse by computing A * A^(-1) = I
        printf("Triangular matrix inverse verification:\n");
        printf("Original matrix A:\n");
        for (int row = 0; row < n; ++row) {
          printf("  [");
          for (int col = 0; col < n; ++col) {
            printf("%8.4f", A_copy_host(row, col));
            if (col < n-1) printf(", ");
          }
          printf("]\n");
        }
        
        printf("\nComputed inverse A^(-1):\n");
        for (int row = 0; row < n; ++row) {
          printf("  [");
          for (int col = 0; col < n; ++col) {
            printf("%8.4f", A_inv_host(row, col));
            if (col < n-1) printf(", ");
          }
          printf("]\n");
        }
        
        printf("\nVerification A * A^(-1) = I:\n");
        bool is_identity = true;
        
        for (int row = 0; row < n; ++row) {
          printf("  [");
          for (int col = 0; col < n; ++col) {
            // Compute dot product for this element
            scalar_type sum = 0.0;
            
            // Since A is upper triangular, we start from row
            for (int k = row; k < n; ++k) {
              sum += A_copy_host(row, k) * A_inv_host(k, col);
            }
            
            // Check if this is an identity matrix element
            scalar_type expected = (row == col) ? 1.0 : 0.0;
            scalar_type error = std::abs(sum - expected);
            
            printf("%8.4f", sum);
            if (col < n-1) printf(", ");
            
            if (error > 1e-10) {
              is_identity = false;
            }
          }
          printf("]\n");
        }
        
        if (is_identity) {
          printf("\nSUCCESS: A * A^(-1) = I verification passed\n");
        } else {
          printf("\nERROR: A * A^(-1) != I verification failed\n");
        }
        
        // Demonstrate with lower triangular matrix
        Kokkos::parallel_for("init_lower_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th lower triangular matrix
          for (int row = 0; row < n; ++row) {
            // Set lower triangular part
            for (int col = 0; col <= row; ++col) {
              if (row == col) {
                A(i, row, col) = n + 1.0;  // Diagonal elements
              } else {
                A(i, row, col) = 1.0;      // Below diagonal elements
              }
            }
            
            // Zero out elements above diagonal
            for (int col = row + 1; col < n; ++col) {
              A(i, row, col) = 0.0;
            }
          }
          
          // Copy A for verification
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
              A_copy(i, row, col) = A(i, row, col);
            }
          }
        });
        
        Kokkos::fence();
        
        // Compute lower triangular matrix inverse
        Kokkos::parallel_for("batch_lower_trtri", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slice
          auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          
          // Compute inverse of lower triangular matrix
          KokkosBatched::SerialTrtri<
            KokkosBatched::Uplo::Lower,       // ArgUplo (lower triangular)
            KokkosBatched::Diag::NonUnit,     // ArgDiag (non-unit diagonal)
            KokkosBatched::Algo::Trtri::Unblocked // ArgAlgo
          >::invoke(A_i);
        });
        
        Kokkos::fence();
        
        // Copy lower triangular results to host for verification
        auto A_lower_copy_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                                    Kokkos::subview(A_copy, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto A_lower_inv_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                                   Kokkos::subview(A, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        printf("\nLower triangular matrix example:\n");
        printf("Original lower triangular matrix:\n");
        for (int row = 0; row < n; ++row) {
          printf("  [");
          for (int col = 0; col < n; ++col) {
            printf("%8.4f", A_lower_copy_host(row, col));
            if (col < n-1) printf(", ");
          }
          printf("]\n");
        }
        
        printf("\nComputed inverse of lower triangular matrix:\n");
        for (int row = 0; row < n; ++row) {
          printf("  [");
          for (int col = 0; col < n; ++col) {
            printf("%8.4f", A_lower_inv_host(row, col));
            if (col < n-1) printf(", ");
          }
          printf("]\n");
        }
      }
      Kokkos::finalize();
      return 0;
    }
