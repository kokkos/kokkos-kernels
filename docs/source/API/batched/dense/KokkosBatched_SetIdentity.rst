KokkosBatched::SetIdentity
##########################

Defined in header `KokkosBatched_SetIdentity_Decl.hpp`

.. code:: c++

    struct SerialSetIdentity {
      template <typename AViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A);
    };

    template <typename MemberType>
    struct TeamSetIdentity {
      template <typename AViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const AViewType &A);
    };

    template <typename MemberType, typename ArgMode>
    struct SetIdentity {
      template <typename AViewType>
      KOKKOS_FORCEINLINE_FUNCTION static int invoke(const MemberType &member, 
                                                   const AViewType &A);
    };

Initializes batched matrices to identity matrices. For each matrix in the batch, sets:

.. math::

   A_{ij} = \begin{cases}
   1 & \text{if } i = j \\
   0 & \text{if } i \neq j
   \end{cases}

This operation transforms each matrix in the batch into an identity matrix, where all diagonal elements are 1 and all off-diagonal elements are 0.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:A: Input/output view for the matrices to be set to identity

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgMode`` must be one of:

  - ``Mode::Serial`` - for serial execution
  - ``Mode::Team`` - for team-based execution

- ``AViewType`` must be a rank-2 or rank-3 Kokkos View representing matrices

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_SetIdentity_Decl.hpp>

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
        int n = 5;              // Size of each square matrix
        
        // Create views for batched matrices
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, n, n);  // Matrices to be set to identity
        
        // Fill matrices with random initial values
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th matrix with non-identity values
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
              A(i, row, col) = 123.456;  // Non-identity value
            }
          }
        });
        
        Kokkos::fence();
        
        // Set matrices to identity using TeamPolicy
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO);
        
        Kokkos::parallel_for("set_identity", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slice
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            
            // Set matrix to identity using Team variant
            KokkosBatched::TeamSetIdentity<typename team_policy_type::member_type>
              ::invoke(member, A_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(A, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        // Verify that the matrix is now an identity matrix
        printf("Verifying identity matrix (first matrix in batch):\n");
        bool is_identity = true;
        
        for (int row = 0; row < n; ++row) {
          for (int col = 0; col < n; ++col) {
            double expected = (row == col) ? 1.0 : 0.0;
            double value = A_host(row, col);
            
            printf("  A(0,%d,%d) = %.1f\n", row, col, value);
            
            if (std::abs(value - expected) > 1e-10) {
              printf("  ERROR: Value at (%d,%d) should be %.1f, got %.1f\n", 
                     row, col, expected, value);
              is_identity = false;
            }
          }
        }
        
        if (is_identity) {
          printf("Verification successful: Matrix correctly set to identity\n");
        }
        
        // Alternative approach using SerialSetIdentity inside a parallel_for
        Kokkos::parallel_for("serial_set_identity", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slice
          auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          
          // Set matrix to identity using Serial variant
          KokkosBatched::SerialSetIdentity::invoke(A_i);
        });
        
        Kokkos::fence();
        
        // Check the result using the selective interface
        Kokkos::parallel_for("selective_set_identity", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // First restore the matrix to non-identity values
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            
            for (int row = 0; row < n; ++row) {
              for (int col = 0; col < n; ++col) {
                A_i(row, col) = 98.76;  // Non-identity value
              }
            }
            
            // Now set to identity using the selective interface
            KokkosBatched::SetIdentity<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Mode::Team                // ArgMode
            >::invoke(member, A_i);
          }
        );
        
        Kokkos::fence();
        
        // Verify again
        auto A2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                          Kokkos::subview(A, batch_size-1, Kokkos::ALL(), Kokkos::ALL()));
        
        printf("\nVerifying identity matrix for last matrix in batch:\n");
        is_identity = true;
        
        for (int row = 0; row < n; ++row) {
          for (int col = 0; col < n; ++col) {
            double expected = (row == col) ? 1.0 : 0.0;
            double value = A2_host(row, col);
            
            if (row < 3 && col < 3) {  // Print only a subset for clarity
              printf("  A(%d,%d,%d) = %.1f\n", batch_size-1, row, col, value);
            }
            
            if (std::abs(value - expected) > 1e-10) {
              printf("  ERROR: Value at (%d,%d) should be %.1f, got %.1f\n", 
                     row, col, expected, value);
              is_identity = false;
            }
          }
        }
        
        if (is_identity) {
          printf("Second verification successful: Matrix correctly set to identity\n");
        }
      }
      Kokkos::finalize();
      return 0;
    }
