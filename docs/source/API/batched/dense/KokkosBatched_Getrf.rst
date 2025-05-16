KokkosBatched::Getrf
###################

Defined in header `KokkosBatched_Getrf.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Getrf.hpp>`_

.. code:: c++

    template <typename ArgAlgo>
    struct SerialGetrf {
      template <typename AViewType, typename PivViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A, const PivViewType &piv);
    };

Performs batched LU factorization with partial pivoting (GETRF). For each matrix A in the batch, computes:

.. math::

   P \cdot A = L \cdot U

where:

- :math:`P` is a permutation matrix (stored as pivot indices)
- :math:`L` is a lower triangular matrix with unit diagonal
- :math:`U` is an upper triangular matrix

The factorization is performed in-place, overwriting the input matrix A with both L and U factors. The unit diagonal of L is not stored. The pivot indices are stored in the piv array.

This implementation includes numerical stability through row pivoting, unlike the basic ``KokkosBatched::LU`` which does not pivot.

Parameters
==========

:A: Input/output view for the matrix to decompose and store the LU factors
:piv: Output view to store the pivot indices

Type Requirements
----------------

- ``ArgAlgo`` must be one of:

  - ``Algo::Getrf::Unblocked`` - direct LU with pivoting
  - ``Algo::Getrf::Blocked`` - blocked algorithm for larger matrices

- ``AViewType`` must be a rank-2 or rank-3 Kokkos View representing matrices
- ``PivViewType`` must be a rank-1 or rank-2 Kokkos View to store pivot indices

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Getrf.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar and index types to use
    using scalar_type = double;
    using index_type = int;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define matrix dimensions
        int batch_size = 1000;  // Number of matrices in batch
        int m = o8;             // Rows in A
        int n = 8;              // Columns in A
        
        // Create views for batched matrices and pivots
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, m, n);  // Matrices to factorize
        
        Kokkos::View<index_type**, Kokkos::LayoutRight, device_type>
          piv("piv", batch_size, std::min(m, n));  // Pivot indices
        
        // Fill matrices with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th matrix with a pattern that will require pivoting
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
              // Make some diagonal elements small to force pivoting
              if (row == col && row % 2 == 0) {
                A(i, row, col) = 0.01;  // Small diagonal element
              } else if (row == col) {
                A(i, row, col) = 10.0;  // Large diagonal element
              } else {
                A(i, row, col) = 1.0;   // Off-diagonal elements
              }
            }
          }
          
          // Initialize pivot array (not strictly necessary)
          for (int j = 0; j < std::min(m, n); ++j) {
            piv(i, j) = 0;
          }
        });
        
        Kokkos::fence();
        
        // Perform batched LU factorization with pivoting
        Kokkos::parallel_for("batched_getrf", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slices
          auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          auto piv_i = Kokkos::subview(piv, i, Kokkos::ALL());
          
          // Perform LU factorization with pivoting
          KokkosBatched::SerialGetrf<KokkosBatched::Algo::Getrf::Unblocked>
            ::invoke(A_i, piv_i);
        });
        
        Kokkos::fence();
        
        // Copy results to host for inspection
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(A, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto piv_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                           Kokkos::subview(piv, 0, Kokkos::ALL()));
        
        // Print the LU factorization and pivots for the first matrix
        printf("LU factorization of first matrix:\n");
        for (int i = 0; i < m; ++i) {
          printf("  ");
          for (int j = 0; j < n; ++j) {
            printf("%8.4f ", A_host(i, j));
          }
          printf("\n");
        }
        
        printf("Pivot indices for first matrix:\n  ");
        for (int i = 0; i < std::min(m, n); ++i) {
          printf("%d ", piv_host(i));
        }
        printf("\n");
        
        // Extract L and U factors for illustration
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> 
          L_host("L_host", m, std::min(m, n)),
          U_host("U_host", std::min(m, n), n);
        
        // Extract L (lower triangular with unit diagonal)
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < std::min(m, n); ++j) {
            if (i > j) {
              L_host(i, j) = A_host(i, j);
            } else if (i == j) {
              L_host(i, j) = 1.0;  // Unit diagonal
            } else {
              L_host(i, j) = 0.0;
            }
          }
        }
        
        // Extract U (upper triangular)
        for (int i = 0; i < std::min(m, n); ++i) {
          for (int j = 0; j < n; ++j) {
            if (i <= j) {
              U_host(i, j) = A_host(i, j);
            } else {
              U_host(i, j) = 0.0;
            }
          }
        }
        
        printf("L factor (with unit diagonal):\n");
        for (int i = 0; i < m; ++i) {
          printf("  ");
          for (int j = 0; j < std::min(m, n); ++j) {
            printf("%8.4f ", L_host(i, j));
          }
          printf("\n");
        }
        
        printf("U factor:\n");
        for (int i = 0; i < std::min(m, n); ++i) {
          printf("  ");
          for (int j = 0; j < n; ++j) {
            printf("%8.4f ", U_host(i, j));
          }
          printf("\n");
        }
      }
      Kokkos::finalize();
      return 0;
    }
