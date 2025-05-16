KokkosBatched::UTV
#################

Defined in header `KokkosBatched_UTV_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_UTV_Decl.hpp>`_

.. code:: c++

    template <typename MemberType, typename ArgAlgo>
    struct TeamVectorUTV {
      template <typename AViewType, typename pViewType, typename UViewType, typename VViewType, typename wViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const AViewType &A, 
                                              const pViewType &p, 
                                              const UViewType &U, 
                                              const VViewType &V, 
                                              const wViewType &w,
                                              int &matrix_rank);
    };

Computes the UTV factorization of a general matrix. For each matrix A in the batch, computes:

.. math::

   UTV = AP^T

where:

- :math:`U` is a left orthogonal matrix (m × matrix_rank)
- :math:`T` is a triangular matrix (matrix_rank × matrix_rank)
- :math:`V` is a right orthogonal matrix (matrix_rank × m)
- :math:`P^T` is a permutation matrix (stored as pivot indices)
- matrix_rank is the numerical rank of A

The UTV factorization is a rank-revealing factorization that can be used for solving rank-deficient problems. When A is full rank (matrix_rank = m), the operation computes a QR factorization with column pivoting.

Parameters
==========

:member: Team execution policy instance
:A: Input matrix for factorization; on output, contains the factorized results
:p: Output view for pivot indices
:U: Output view for the left orthogonal matrix
:V: Output view for the right orthogonal matrix
:w: Workspace view for temporary calculations
:matrix_rank: Output parameter for the numerical rank of the matrix

Type Requirements
----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgAlgo`` must be algorithm variant (implementation dependent)
- ``AViewType`` must be a rank-2 or rank-3 Kokkos View representing matrices
- ``pViewType`` must be a rank-1 or rank-2 Kokkos View for pivot indices
- ``UViewType`` must be a rank-2 or rank-3 Kokkos View for left orthogonal matrices
- ``VViewType`` must be a rank-2 or rank-3 Kokkos View for right orthogonal matrices
- ``wViewType`` must be a rank-1 or rank-2 Kokkos View with sufficient workspace (at least 3*m elements)

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_UTV_Decl.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar and index types to use
    using scalar_type = double;
    using index_type = int;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 100;   // Number of matrices
        int m = 6;              // Matrix size (m × m)
        
        // Create views for batched matrices and factorization results
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, m, m),          // Input matrices (overwritten)
          A_copy("A_copy", batch_size, m, m), // Copy for verification
          U("U", batch_size, m, m),          // Left orthogonal matrices
          V("V", batch_size, m, m);          // Right orthogonal matrices
        
        Kokkos::View<index_type**, Kokkos::LayoutRight, device_type>
          p("p", batch_size, m);            // Pivot indices
        
        // Workspace (3*m elements for each matrix)
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type>
          w("w", batch_size, 3*m);          // Workspace
        
        // View to store the matrix ranks
        Kokkos::View<int*, Kokkos::LayoutRight, device_type>
          ranks("ranks", batch_size);
        
        // Fill matrices with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th matrix with a rank-deficient matrix
          // For demonstration, we'll create matrices with rank = m-2
          
          // First, set matrix to zeros
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < m; ++col) {
              A(i, row, col) = 0.0;
            }
          }
          
          // Create a matrix with rank = m-2 by setting up m-2 linearly independent rows
          for (int row = 0; row < m-2; ++row) {
            for (int col = 0; col < m; ++col) {
              // Each row has a unique pattern
              A(i, row, col) = 1.0 / (row + col + 1.0);
            }
          }
          
          // Last two rows are linear combinations of the first m-2 rows
          for (int col = 0; col < m; ++col) {
            A(i, m-2, col) = A(i, 0, col) + A(i, 1, col);
            A(i, m-1, col) = A(i, 2, col) - A(i, 3, col);
          }
          
          // Copy A for verification
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < m; ++col) {
              A_copy(i, row, col) = A(i, row, col);
            }
          }
          
          // Initialize other arrays
          for (int j = 0; j < m; ++j) {
            p(i, j) = 0;
            for (int k = 0; k < m; ++k) {
              U(i, j, k) = 0.0;
              V(i, j, k) = 0.0;
            }
          }
          
          // Initialize workspace
          for (int j = 0; j < 3*m; ++j) {
            w(i, j) = 0.0;
          }
          
          // Initialize matrix rank
          ranks(i) = 0;
        });
        
        Kokkos::fence();
        
        // Compute UTV factorization
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO, Kokkos::AUTO);
        
        Kokkos::parallel_for("batch_utv", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto p_i = Kokkos::subview(p, i, Kokkos::ALL());
            auto U_i = Kokkos::subview(U, i, Kokkos::ALL(), Kokkos::ALL());
            auto V_i = Kokkos::subview(V, i, Kokkos::ALL(), Kokkos::ALL());
            auto w_i = Kokkos::subview(w, i, Kokkos::ALL());
            
            // Reference to store the matrix rank
            int& matrix_rank = ranks(i);
            
            // Compute UTV factorization
            KokkosBatched::TeamVectorUTV<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Algo::UTV::Unblocked     // ArgAlgo
            >::invoke(member, A_i, p_i, U_i, V_i, w_i, matrix_rank);
          }
        );
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto A_copy_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                              Kokkos::subview(A_copy, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(A, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto U_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(U, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto V_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(V, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto p_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(p, 0, Kokkos::ALL()));
        auto ranks_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ranks);
        
        // Verify the factorization
        printf("UTV Factorization results for first matrix:\n");
        printf("Computed matrix rank: %d (expected %d)\n", ranks_host(0), m-2);
        
        // Verify that U is orthogonal (U^T * U = I for the first matrix_rank columns)
        printf("\nVerifying orthogonality of U (U^T * U = I):\n");
        int matrix_rank = ranks_host(0);
        
        for (int i = 0; i < matrix_rank; ++i) {
          for (int j = 0; j < matrix_rank; ++j) {
            scalar_type dot_product = 0.0;
            
            for (int k = 0; k < m; ++k) {
              dot_product += U_host(k, i) * U_host(k, j);
            }
            
            scalar_type expected = (i == j) ? 1.0 : 0.0;
            scalar_type error = std::abs(dot_product - expected);
            
            if (i <= 2 && j <= 2) {  // Print only a few entries for brevity
              printf("  U^T * U [%d,%d] = %.6f (expected %.1f, error = %.6e)\n",
                     i, j, dot_product, expected, error);
            }
          }
        }
        
        // Verify that V is orthogonal (V * V^T = I)
        printf("\nVerifying orthogonality of V (V * V^T = I):\n");
        
        for (int i = 0; i < matrix_rank; ++i) {
          for (int j = 0; j < matrix_rank; ++j) {
            scalar_type dot_product = 0.0;
            
            for (int k = 0; k < m; ++k) {
              dot_product += V_host(i, k) * V_host(j, k);
            }
            
            scalar_type expected = (i == j) ? 1.0 : 0.0;
            scalar_type error = std::abs(dot_product - expected);
            
            if (i <= 2 && j <= 2) {  // Print only a few entries for brevity
              printf("  V * V^T [%d,%d] = %.6f (expected %.1f, error = %.6e)\n",
                     i, j, dot_product, expected, error);
            }
          }
        }
        
        // Verify that UTV = A * P^T
        printf("\nVerifying UTV = A * P^T:\n");
        printf("  (Showing only top-left 3x3 submatrix for brevity)\n");
        
        // Reconstruct UTV
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace>
          UT("UT", m, matrix_rank),
          UT_V("UT_V", m, m),
          A_permuted("A_permuted", m, m);
        
        // Compute U * T (using A's upper triangular part as T)
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < matrix_rank; ++j) {
            UT(i, j) = 0.0;
            
            for (int k = 0; k <= j; ++k) {  // T is upper triangular
              UT(i, j) += U_host(i, k) * A_host(k, j);
            }
          }
        }
        
        // Compute (U * T) * V
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < m; ++j) {
            UT_V(i, j) = 0.0;
            
            for (int k = 0; k < matrix_rank; ++k) {
              UT_V(i, j) += UT(i, k) * V_host(k, j);
            }
          }
        }
        
        // Compute A * P^T (apply column permutation to A)
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < m; ++j) {
            A_permuted(i, j) = A_copy_host(i, p_host(j));
          }
        }
        
        // Compare UTV with A * P^T
        for (int i = 0; i < 3; ++i) {
          for (int j = 0; j < 3; ++j) {
            printf("  UTV[%d,%d] = %.6f, A*P^T[%d,%d] = %.6f, Diff = %.6e\n",
                   i, j, UT_V(i, j), i, j, A_permuted(i, j), 
                   std::abs(UT_V(i, j) - A_permuted(i, j)));
          }
        }
      }
      Kokkos::finalize();
      return 0;
    }
