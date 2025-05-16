KokkosBatched::Trsm
###################

Defined in header `KokkosBatched_Trsm_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Trsm_Decl.hpp>`_

.. code:: c++

    template <typename MemberType, typename ArgSide, typename ArgUplo, typename ArgTrans, 
              typename ArgDiag, typename ArgMode, typename ArgAlgo>
    struct Trsm {
      template <typename ScalarType, typename AViewType, typename BViewType>
      KOKKOS_FORCEINLINE_FUNCTION static int invoke(const MemberType &member, 
                                                   const ScalarType alpha, 
                                                   const AViewType &A, 
                                                   const BViewType &B);
    };

Performs batched triangular solve with multiple right-hand sides. For each triangular matrix A and matrix B in the batch, solves one of the following systems:

.. math::

   \begin{align}
   \text{op}(A) X &= \alpha B \quad \text{(Left side solve)} \\
   X \text{op}(A) &= \alpha B \quad \text{(Right side solve)}
   \end{align}

where:

- :math:`\text{op}(A)` can be :math:`A`, :math:`A^T` (transpose), or :math:`A^H` (Hermitian transpose)
- :math:`A` is a triangular matrix (upper or lower triangular)
- :math:`B` and :math:`X` are general matrices
- :math:`\alpha` is a scalar value
- :math:`X` is the solution, which overwrites :math:`B`

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:alpha: Scalar multiplier for the right-hand side B
:A: Input view containing batch of triangular matrices
:B: Input/output view for the right-hand sides and solutions

Type Requirements
----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgSide`` must be one of:

  - ``Side::Left`` - solve op(A)X = αB
  - ``Side::Right`` - solve XA = αB

- ``ArgUplo`` must be one of:

  - ``Uplo::Upper`` - A is upper triangular
  - ``Uplo::Lower`` - A is lower triangular

- ``ArgTrans`` must be one of:

  - ``Trans::NoTranspose`` - use A as is
  - ``Trans::Transpose`` - use transpose of A
  - ``Trans::ConjTranspose`` - use conjugate transpose of A

- ``ArgDiag`` must be one of:

  - ``Diag::Unit`` - A has an implicit unit diagonal
  - ``Diag::NonUnit`` - A has a non-unit diagonal that must be used in the solve

- ``ArgMode`` must be one of:

  - ``Mode::Serial`` - for serial execution
  - ``Mode::Team`` - for team-based execution
  - ``Mode::TeamVector`` - for team-vector execution

- ``ArgAlgo`` must be one of:

  - ``Algo::Trsm::Unblocked`` - for small matrices
  - ``Algo::Trsm::Blocked`` - for larger matrices with blocking

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Trsm_Decl.hpp>

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
        int m = 8;              // Size of each triangular matrix
        int n = 4;              // Number of right-hand sides
        
        // Create views for batched matrices
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, m, m),   // Triangular matrices
          B("B", batch_size, m, n);   // Right-hand sides
        
        // Fill matrices with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th triangular matrix (lower triangular)
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col <= row; ++col) {
              if (row == col) {
                A(i, row, col) = 2.0; // Diagonal elements
              } else {
                A(i, row, col) = 0.5; // Below diagonal elements
              }
            }
            // Zero out elements above diagonal
            for (int col = row+1; col < m; ++col) {
              A(i, row, col) = 0.0;
            }
          }
          
          // Initialize right-hand sides
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
              B(i, row, col) = 1.0;
            }
          }
        });
        
        Kokkos::fence();
        
        // Scalar multiplier
        scalar_type alpha = 1.0;
        
        // Perform batched triangular solve using TeamPolicy
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_trsm", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto B_i = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
            
            // Perform triangular solve
            KokkosBatched::Trsm<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Side::Left,               // ArgSide
              KokkosBatched::Uplo::Lower,              // ArgUplo (lower triangular)
              KokkosBatched::Trans::NoTranspose,       // ArgTrans
              KokkosBatched::Diag::NonUnit,            // ArgDiag (non-unit diagonal)
              KokkosBatched::Mode::Team,               // ArgMode
              KokkosBatched::Algo::Trsm::Unblocked     // ArgAlgo
            >::invoke(member, alpha, A_i, B_i);
          }
        );
        
        Kokkos::fence();
        
        // B now contains the solutions to the triangular systems
        
        // Example: Copy solution from first system to host for verification
        auto B_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                          Kokkos::subview(B, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                          Kokkos::subview(A, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        // Verify the solution (for the first right-hand side)
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, Kokkos::HostSpace> 
          verify("verify", m);
        
        // Multiply A*x to verify it equals b
        for (int i = 0; i < m; ++i) {
          verify(i) = 0.0;
          for (int j = 0; j <= i; ++j) {
            verify(i) += A_host(i, j) * B_host(j, 0);
          }
          
          // verify(i) should be close to 1.0 (original right-hand side)
          // Check for accuracy
          scalar_type error = std::abs(verify(i) - 1.0);
          if (error > 1.0e-10) {
            printf("Error in solution verification: %e\n", error);
          }
        }
      }
      Kokkos::finalize();
      return 0;
    }
