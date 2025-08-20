KokkosBatched::SolveLU
######################

Defined in header: :code:`KokkosBatched_SolveLU_Decl.hpp`

.. code:: c++

    template <typename ArgTrans, typename ArgAlgo>
    struct SerialSolveLU {
      template <typename AViewType, typename BViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A, const BViewType &B);
    };

    template <typename MemberType, typename ArgTrans, typename ArgAlgo>
    struct TeamSolveLU {
      template <typename AViewType, typename BViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const AViewType &A, 
                                              const BViewType &B);
    };

    template <typename MemberType, typename ArgTrans, typename ArgMode, typename ArgAlgo>
    struct SolveLU {
      template <typename AViewType, typename BViewType>
      KOKKOS_FORCEINLINE_FUNCTION static int invoke(const MemberType &member, 
                                                   const AViewType &A, 
                                                   const BViewType &B);
    };

Solves a linear system using a pre-computed LU factorization. For each factorized matrix A and right-hand side B in the batch, solves:

.. math::

   \text{op}(A) X = B

where:

- :math:`\text{op}(A)` can be :math:`A` or :math:`A^T` (transpose) or :math:`A^H` (Hermitian transpose)
- :math:`A` is the LU factorization computed by a previous call to ``KokkosBatched::LU``
- :math:`B` is the right-hand side that will be overwritten with the solution :math:`X`

The solution process consists of forward and backward substitution using the L and U factors stored in A.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:A: Input view containing the LU factorization of the coefficient matrix
:B: Input/output view for the right-hand side and solution

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgTrans`` must be one of:

  - ``Trans::NoTranspose`` - solve AX = B
  - ``Trans::Transpose`` - solve A^T X = B
  - ``Trans::ConjTranspose`` - solve A^H X = B

- ``ArgMode`` must be one of:

  - ``Mode::Serial`` - for serial execution
  - ``Mode::Team`` - for team-based execution

- ``ArgAlgo`` must be one of the algorithm variants:

  - ``Algo::LU::Unblocked`` - direct LU solution
  - ``Algo::LU::Blocked`` - blocked algorithm for larger matrices

- ``AViewType`` must be a rank-2 or rank-3 Kokkos View containing the LU factorization
- ``BViewType`` must be a rank-2 or rank-3 Kokkos View for the right-hand side and solution

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_LU_Decl.hpp>
    #include <KokkosBatched_SolveLU_Decl.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define matrix dimensions
        int batch_size = 1000;  // Number of matrices in batch
        int n = 8;              // Size of each square matrix
        int nrhs = 4;           // Number of right-hand sides
        
        // Create views for batched matrices
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, n, n),            // Coefficient matrices
          A_copy("A_copy", batch_size, n, n),  // Copy for verification
          B("B", batch_size, n, nrhs);         // Right-hand sides
        
        // Fill matrices with data (diagonally dominant matrices for stability)
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th matrix in the batch as a diagonally dominant matrix
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
              if (row == col) {
                A(i, row, col) = n + 1.0; // Diagonal elements
              } else {
                A(i, row, col) = 1.0;     // Off-diagonal elements
              }
              
              // Copy A for verification later
              A_copy(i, row, col) = A(i, row, col);
            }
          }
          
          // Initialize right-hand sides
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col < nrhs; ++col) {
              // Set to column index + 1 for simplicity
              B(i, row, col) = col + 1.0;
            }
          }
        });
        
        Kokkos::fence();
        
        // Perform LU factorization of A
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_lu", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slice for matrix A
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            
            // Perform LU decomposition
            KokkosBatched::LU<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Mode::Team,               // ArgMode
              KokkosBatched::Algo::LU::Unblocked       // ArgAlgo
            >::invoke(member, A_i);
          }
        );
        
        Kokkos::fence();
        
        // Now solve the system AX = B using the LU factorization
        Kokkos::parallel_for("batched_solve", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto B_i = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
            
            // Solve the system using LU factorization
            KokkosBatched::SolveLU<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Trans::NoTranspose,       // ArgTrans
              KokkosBatched::Mode::Team,               // ArgMode
              KokkosBatched::Algo::LU::Unblocked       // ArgAlgo
            >::invoke(member, A_i, B_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto A_copy_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                             Kokkos::subview(A_copy, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto B_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                        Kokkos::subview(B, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        // Verify the solution by computing A*X and comparing with original B
        printf("Verification for first system, first right-hand side:\n");
        
        for (int row = 0; row < n; ++row) {
          scalar_type expected = 1.0; // First right-hand side was all 1's
          scalar_type computed = 0.0;
          
          for (int col = 0; col < n; ++col) {
            computed += A_copy_host(row, col) * B_host(col, 0);
          }
          
          scalar_type error = std::abs(computed - expected);
          printf("  Row %d: computed = %.6f, expected = %.6f, error = %.6e\n", 
                 row, computed, expected, error);
        }
      }
      Kokkos::finalize();
      return 0;
    }
