KokkosBatched::Trsv
##################

Defined in header `KokkosBatched_Trsv_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Trsv_Decl.hpp>`_

.. code:: c++

    template <typename ArgUplo, typename ArgTrans, typename ArgDiag, typename ArgAlgo>
    struct SerialTrsv {
      template <typename ScalarType, typename AViewType, typename bViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType alpha, 
                                              const AViewType &A, 
                                              const bViewType &b);
    };

    template <typename MemberType, typename ArgUplo, typename ArgTrans, typename ArgDiag, typename ArgAlgo>
    struct TeamTrsv {
      template <typename ScalarType, typename AViewType, typename bViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const ScalarType alpha, 
                                              const AViewType &A, 
                                              const bViewType &b);
    };

    template <typename MemberType, typename ArgUplo, typename ArgTrans, typename ArgDiag, typename ArgAlgo>
    struct TeamVectorTrsv {
      template <typename ScalarType, typename AViewType, typename bViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const ScalarType alpha, 
                                              const AViewType &A, 
                                              const bViewType &b);
    };

    template <typename MemberType, typename ArgUplo, typename ArgTrans, typename ArgDiag, 
              typename ArgMode, typename ArgAlgo>
    struct Trsv {
      template <typename ScalarType, typename AViewType, typename bViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const ScalarType alpha, 
                                              const AViewType &A, 
                                              const bViewType &b);
    };

Performs batched triangular solve with a single right-hand side vector. For each triangular matrix A and vector b in the batch, solves:

.. math::

   \text{op}(A) x = \alpha b

where:

- :math:`\text{op}(A)` can be :math:`A`, :math:`A^T` (transpose), or :math:`A^H` (Hermitian transpose)
- :math:`A` is a triangular matrix (upper or lower triangular)
- :math:`b` is the right-hand side vector, which will be overwritten with the solution :math:`x`
- :math:`\alpha` is a scalar value

This is a specialized version of TRSM for a single right-hand side vector, which can be more efficient for that specific case.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:alpha: Scalar multiplier for the right-hand side b
:A: Input view containing batch of triangular matrices
:b: Input/output view for the right-hand side vectors and solutions

Type Requirements
----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
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

- ``ArgAlgo`` must be one of the algorithm variants (implementation dependent)
- ``AViewType`` must be a rank-2 or rank-3 Kokkos View representing triangular matrices
- ``bViewType`` must be a rank-1 or rank-2 Kokkos View for vectors

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Trsv_Decl.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 1000;  // Number of triangular solves
        int n = 4;              // Size of each triangular matrix/vector
        
        // Create views for batched matrices and vectors
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, n, n),           // Triangular matrices
          A_copy("A_copy", batch_size, n, n); // Copy for verification
        
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type>
          b("b", batch_size, n);              // Right-hand side vectors (will be overwritten with solution)
        
        // Fill matrices and vectors with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_data", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th lower triangular matrix
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col <= row; ++col) {  // Lower triangular part
              if (row == col) {
                A(i, row, col) = 2.0;  // Diagonal elements
              } else {
                A(i, row, col) = 1.0;  // Below diagonal elements
              }
            }
            
            // Zero out elements above diagonal
            for (int col = row+1; col < n; ++col) {
              A(i, row, col) = 0.0;
            }
          }
          
          // Copy A for verification
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
              A_copy(i, row, col) = A(i, row, col);
            }
          }
          
          // Initialize right-hand side vectors
          for (int j = 0; j < n; ++j) {
            b(i, j) = j + 1.0;  // 1, 2, 3, 4
          }
        });
        
        Kokkos::fence();
        
        // Save original right-hand side for verification
        auto b_orig = Kokkos::create_mirror_view(b);
        Kokkos::deep_copy(b_orig, b);
        
        // Scalar multiplier (typically 1.0 for solving A*x = b)
        scalar_type alpha = 1.0;
        
        // Solve batched triangular systems using SerialTrsv
        Kokkos::parallel_for("batched_trsv", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slices
          auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          auto b_i = Kokkos::subview(b, i, Kokkos::ALL());
          
          // Solve the triangular system using SerialTrsv
          KokkosBatched::SerialTrsv<
            KokkosBatched::Uplo::Lower,         // ArgUplo (lower triangular)
            KokkosBatched::Trans::NoTranspose,  // ArgTrans
            KokkosBatched::Diag::NonUnit,       // ArgDiag (non-unit diagonal)
            KokkosBatched::Algo::Trsv::Unblocked // ArgAlgo
          >::invoke(alpha, A_i, b_i);
        });
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto A_copy_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                              Kokkos::subview(A_copy, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto b_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(b, 0, Kokkos::ALL()));
        auto b_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                              Kokkos::subview(b_orig, 0, Kokkos::ALL()));
        
        // Verify the solution by computing A*x and comparing with original b
        printf("Triangular solve verification (first system):\n");
        printf("  Solution x = [");
        for (int j = 0; j < n; ++j) {
          printf("%.6f%s", b_host(j), (j < n-1) ? ", " : "");
        }
        printf("]\n");
        
        printf("  Original RHS b = [");
        for (int j = 0; j < n; ++j) {
          printf("%.6f%s", b_orig_host(j), (j < n-1) ? ", " : "");
        }
        printf("]\n");
        
        printf("  Verification A*x = b?\n");
        bool correct = true;
        
        for (int row = 0; row < n; ++row) {
          double computed = 0.0;
          
          // Since A is lower triangular, we only need to compute up to the diagonal
          for (int col = 0; col <= row; ++col) {
            computed += A_copy_host(row, col) * b_host(col);
          }
          
          double expected = b_orig_host(row);
          double error = std::abs(computed - expected);
          
          printf("    Row %d: A*x = %.6f, b = %.6f, Error = %.6e\n", 
                 row, computed, expected, error);
          
          if (error > 1e-10) {
            correct = false;
          }
        }
        
        if (correct) {
          printf("  SUCCESS: Solution x correctly solves A*x = b\n");
        } else {
          printf("  ERROR: Solution x does not satisfy A*x = b within tolerance\n");
        }
        
        // Now demonstrate TeamTrsv with upper triangular matrix
        // Create upper triangular matrices
        Kokkos::parallel_for("init_upper_data", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th upper triangular matrix
          for (int row = 0; row < n; ++row) {
            // Zero out elements below diagonal
            for (int col = 0; col < row; ++col) {
              A(i, row, col) = 0.0;
            }
            
            // Set upper triangular part
            for (int col = row; col < n; ++col) {
              if (row == col) {
                A(i, row, col) = 2.0;  // Diagonal elements
              } else {
                A(i, row, col) = 1.0;  // Above diagonal elements
              }
            }
          }
          
          // Copy A for verification
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
              A_copy(i, row, col) = A(i, row, col);
            }
          }
          
          // Reset right-hand side vectors
          for (int j = 0; j < n; ++j) {
            b(i, j) = j + 1.0;  // 1, 2, 3, 4
          }
        });
        
        Kokkos::fence();
        
        // Update original right-hand side for verification
        Kokkos::deep_copy(b_orig, b);
        
        // Create TeamPolicy
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO);
        
        // Solve batched upper triangular systems using TeamTrsv
        Kokkos::parallel_for("team_trsv", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto b_i = Kokkos::subview(b, i, Kokkos::ALL());
            
            // Solve the triangular system using TeamTrsv
            KokkosBatched::TeamTrsv<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Uplo::Upper,              // ArgUplo (upper triangular)
              KokkosBatched::Trans::NoTranspose,       // ArgTrans
              KokkosBatched::Diag::NonUnit,            // ArgDiag (non-unit diagonal)
              KokkosBatched::Algo::Trsv::Unblocked     // ArgAlgo
            >::invoke(member, alpha, A_i, b_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy upper triangular results to host for verification
        auto A_upper_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                               Kokkos::subview(A_copy, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto b_upper_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                               Kokkos::subview(b, 0, Kokkos::ALL()));
        auto b_upper_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                                    Kokkos::subview(b_orig, 0, Kokkos::ALL()));
        
        printf("\nUpper triangular solve verification (first system):\n");
        printf("  Solution x = [");
        for (int j = 0; j < n; ++j) {
          printf("%.6f%s", b_upper_host(j), (j < n-1) ? ", " : "");
        }
        printf("]\n");
        
        printf("  Verification A*x = b?\n");
        correct = true;
        
        for (int row = 0; row < n; ++row) {
          double computed = 0.0;
          
          // Since A is upper triangular, we compute from the diagonal to the end
          for (int col = row; col < n; ++col) {
            computed += A_upper_host(row, col) * b_upper_host(col);
          }
          
          double expected = b_upper_orig_host(row);
          double error = std::abs(computed - expected);
          
          printf("    Row %d: A*x = %.6f, b = %.6f, Error = %.6e\n", 
                 row, computed, expected, error);
          
          if (error > 1e-10) {
            correct = false;
          }
        }
        
        if (correct) {
          printf("  SUCCESS: Upper triangular solution correctly solves A*x = b\n");
        } else {
          printf("  ERROR: Upper triangular solution does not satisfy A*x = b\n");
        }
      }
      Kokkos::finalize();
      return 0;
    }
