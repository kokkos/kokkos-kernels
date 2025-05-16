KokkosBatched::Gesv
##################

Defined in header `KokkosBatched_Gesv.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Gesv.hpp>`_

.. code:: c++

    struct Gesv {
      struct StaticPivoting {};
      struct NoPivoting {};
      
      using Default = StaticPivoting;
    };

    template <typename ArgAlgo>
    struct SerialGesv {
      template <typename MatrixType, typename XVectorType, typename YVectorType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MatrixType A, 
                                              const XVectorType X, 
                                              const YVectorType Y,
                                              const MatrixType tmp);
    };

    template <typename MemberType, typename ArgAlgo>
    struct TeamGesv {
      template <typename MatrixType, typename VectorType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const MatrixType A, 
                                              const VectorType X, 
                                              const VectorType Y);
    };

    template <typename MemberType, typename ArgAlgo>
    struct TeamVectorGesv {
      template <typename MatrixType, typename VectorType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const MatrixType A, 
                                              const VectorType X, 
                                              const VectorType Y);
    };

Solves batched linear systems using LU decomposition. For each system in the batch, solves:

.. math::

   A_i x_i = y_i

where:

- :math:`A_i` is a square matrix
- :math:`x_i` is the solution vector to be computed
- :math:`y_i` is the right-hand side vector

The solution is computed using LU decomposition, followed by forward and backward substitution. Two pivoting strategies are available:

1. ``NoPivoting``: No pivoting is performed (faster but less stable)
2. ``StaticPivoting``: Static pivoting based on the maximum absolute value in each row and column (more stable)

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:A: Input view containing coefficient matrices
:X: Output view for the solution vectors
:Y: Input view for the right-hand side vectors
:tmp: Temporary workspace for certain algorithm variants (only for SerialGesv)

Type Requirements
----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgAlgo`` must be one of:

  - ``Gesv::NoPivoting`` - faster but less numerically stable
  - ``Gesv::StaticPivoting`` - more numerically stable with pivoting

- ``MatrixType`` must be a rank-2 or rank-3 Kokkos View representing square matrices
- ``XVectorType`` and ``YVectorType`` (or ``VectorType``) must be rank-1 or rank-2 Kokkos Views for vectors
- For ``SerialGesv``, ``tmp`` must be a matrix with dimensions at least n × (n+4) where n is the matrix size

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Gesv.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 1000;  // Number of linear systems
        int n = 4;              // Size of each square matrix/vector
        
        // Create views for batched matrices and vectors
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, n, n),           // Coefficient matrices
          A_copy("A_copy", batch_size, n, n), // Copy for verification
          tmp("tmp", batch_size, n, n+4);     // Temporary workspace
        
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type>
          X("X", batch_size, n),              // Solution vectors (output)
          Y("Y", batch_size, n);              // Right-hand side vectors
        
        // Fill matrices and vectors with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_data", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th matrix as a diagonally dominant matrix for stability
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
              if (row == col) {
                A(i, row, col) = n + 1.0;  // Diagonal elements
              } else {
                A(i, row, col) = 1.0;      // Off-diagonal elements
              }
              
              // Copy A for verification
              A_copy(i, row, col) = A(i, row, col);
            }
          }
          
          // Initialize right-hand side vectors with known values
          for (int j = 0; j < n; ++j) {
            Y(i, j) = j + 1.0;  // 1, 2, 3, 4
            X(i, j) = 0.0;      // Initialize solution to zeros
          }
        });
        
        Kokkos::fence();
        
        // Solve batched linear systems using SerialGesv with StaticPivoting
        Kokkos::parallel_for("batched_gesv", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slices
          auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          auto X_i = Kokkos::subview(X, i, Kokkos::ALL());
          auto Y_i = Kokkos::subview(Y, i, Kokkos::ALL());
          auto tmp_i = Kokkos::subview(tmp, i, Kokkos::ALL(), Kokkos::ALL());
          
          // Solve the linear system using SerialGesv with StaticPivoting
          KokkosBatched::SerialGesv<KokkosBatched::Gesv::StaticPivoting>
            ::invoke(A_i, X_i, Y_i, tmp_i);
        });
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto A_copy_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                              Kokkos::subview(A_copy, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto X_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(X, 0, Kokkos::ALL()));
        auto Y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(Y, 0, Kokkos::ALL()));
        
        // Verify the solution by computing A*X and comparing with Y
        printf("Linear system solution verification (first system):\n");
        printf("  Solution X = [");
        for (int j = 0; j < n; ++j) {
          printf("%.6f%s", X_host(j), (j < n-1) ? ", " : "");
        }
        printf("]\n");
        
        printf("  Original RHS Y = [");
        for (int j = 0; j < n; ++j) {
          printf("%.6f%s", Y_host(j), (j < n-1) ? ", " : "");
        }
        printf("]\n");
        
        printf("  Verification A*X = Y?\n");
        bool correct = true;
        
        for (int row = 0; row < n; ++row) {
          double computed = 0.0;
          
          for (int col = 0; col < n; ++col) {
            computed += A_copy_host(row, col) * X_host(col);
          }
          
          double expected = Y_host(row);
          double error = std::abs(computed - expected);
          
          printf("    Row %d: A*X = %.6f, Y = %.6f, Error = %.6e\n", 
                 row, computed, expected, error);
          
          if (error > 1e-10) {
            correct = false;
          }
        }
        
        if (correct) {
          printf("  SUCCESS: Solution X correctly solves A*X = Y\n");
        } else {
          printf("  ERROR: Solution X does not satisfy A*X = Y within tolerance\n");
        }
        
        // Now demonstrate TeamGesv with NoPivoting
        Kokkos::deep_copy(A, A_copy);  // Restore original A
        Kokkos::deep_copy(X, 0.0);     // Reset X to zeros
        
        // Create TeamPolicy
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO);
        
        Kokkos::parallel_for("team_gesv", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto X_i = Kokkos::subview(X, i, Kokkos::ALL());
            auto Y_i = Kokkos::subview(Y, i, Kokkos::ALL());
            
            // Solve the linear system using TeamGesv with NoPivoting
            KokkosBatched::TeamGesv<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Gesv::NoPivoting          // ArgAlgo
            >::invoke(member, A_i, X_i, Y_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy TeamGesv results to host
        auto X_team_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                              Kokkos::subview(X, 0, Kokkos::ALL()));
        
        printf("\nTeamGesv with NoPivoting solution (first system):\n");
        printf("  Solution X = [");
        for (int j = 0; j < n; ++j) {
          printf("%.6f%s", X_team_host(j), (j < n-1) ? ", " : "");
        }
        printf("]\n");
        
        // Verify TeamGesv solution
        printf("  Verification A*X = Y?\n");
        correct = true;
        
        for (int row = 0; row < n; ++row) {
          double computed = 0.0;
          
          for (int col = 0; col < n; ++col) {
            computed += A_copy_host(row, col) * X_team_host(col);
          }
          
          double expected = Y_host(row);
          double error = std::abs(computed - expected);
          
          printf("    Row %d: A*X = %.6f, Y = %.6f, Error = %.6e\n", 
                 row, computed, expected, error);
          
          if (error > 1e-10) {
            correct = false;
          }
        }
        
        if (correct) {
          printf("  SUCCESS: TeamGesv solution correctly solves A*X = Y\n");
        } else {
          printf("  ERROR: TeamGesv solution does not satisfy A*X = Y within tolerance\n");
        }
      }
      Kokkos::finalize();
      return 0;
    }
