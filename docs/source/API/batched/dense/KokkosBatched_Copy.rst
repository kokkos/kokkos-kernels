KokkosBatched::Copy
#################

Defined in header `KokkosBatched_Copy_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Copy_Decl.hpp>`_

.. code:: c++

    template <typename ArgTrans = Trans::NoTranspose, int rank = 2>
    struct SerialCopy {
      template <typename AViewType, typename BViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A, const BViewType &B);
    };

    template <typename MemberType, typename ArgTrans = Trans::NoTranspose, int rank = 2>
    struct TeamCopy {
      template <typename AViewType, typename BViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const AViewType &A, 
                                              const BViewType &B);
    };

    template <typename MemberType, typename ArgTrans = Trans::NoTranspose, int rank = 2>
    struct TeamVectorCopy {
      template <typename AViewType, typename BViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const AViewType &A, 
                                              const BViewType &B);
    };

    template <typename MemberType, typename ArgTrans, typename ArgMode, int rank = 2>
    struct Copy {
      template <typename AViewType, typename BViewType>
      KOKKOS_FORCEINLINE_FUNCTION static int invoke(const MemberType &member, 
                                                   const AViewType &A, 
                                                   const BViewType &B);
    };

Performs batched matrix or vector copying from source to destination. For each pair of matrices or vectors in the batch, copies:

.. math::

   B = \text{op}(A)

where:

- :math:`\text{op}(A)` can be :math:`A` or :math:`A^T` (transpose)
- :math:`A` is the source matrix or vector
- :math:`B` is the destination matrix or vector

This operation supports both rank-1 (vector) and rank-2 (matrix) views, controlled by the ``rank`` template parameter.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:A: Input view containing source matrices or vectors
:B: Output view for destination matrices or vectors

Type Requirements
----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgTrans`` must be one of:

  - ``Trans::NoTranspose`` - copy A to B (default)
  - ``Trans::Transpose`` - copy transpose of A to B

- ``ArgMode`` must be one of:

  - ``Mode::Serial`` - for serial execution
  - ``Mode::Team`` - for team-based execution
  - ``Mode::TeamVector`` - for team-vector execution

- ``rank`` must be either 1 (for vectors) or 2 (for matrices, default)
- ``AViewType`` and ``BViewType`` must be Kokkos Views with compatible dimensions:

  - For rank=1: A(n) → B(n) for NoTranspose
  - For rank=2: A(m,n) → B(m,n) for NoTranspose, or A(m,n) → B(n,m) for Transpose

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Copy_Decl.hpp>

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
        int m = 8;              // Rows
        int n = 6;              // Columns
        
        // Create views for batched matrices
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, m, n),   // Source matrices
          B("B", batch_size, m, n),   // Destination for direct copy
          C("C", batch_size, n, m);   // Destination for transposed copy
        
        // Initialize source matrices
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th source matrix with index-based values
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
              A(i, row, col) = 10.0 * row + col + 1.0;
            }
          }
          
          // Initialize destination matrices to zero
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
              B(i, row, col) = 0.0;
            }
          }
          
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col < m; ++col) {
              C(i, row, col) = 0.0;
            }
          }
        });
        
        Kokkos::fence();
        
        // Perform batched direct copy using TeamPolicy
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_direct_copy", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto B_i = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
            
            // Perform direct copy (A → B)
            KokkosBatched::TeamCopy<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Trans::NoTranspose,       // ArgTrans
              2                                        // rank
            >::invoke(member, A_i, B_i);
          }
        );
        
        Kokkos::fence();
        
        // Perform batched transposed copy using TeamVectorPolicy
        team_policy_type policy_team_vector(batch_size, Kokkos::AUTO, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_transpose_copy", policy_team_vector, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto C_i = Kokkos::subview(C, i, Kokkos::ALL(), Kokkos::ALL());
            
            // Perform transposed copy (A^T → C)
            KokkosBatched::TeamVectorCopy<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Trans::Transpose,         // ArgTrans
              2                                        // rank
            >::invoke(member, A_i, C_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(A, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto B_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(B, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto C_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(C, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        // Verify the direct copy (A → B)
        printf("Verifying direct copy (first few elements):\n");
        for (int row = 0; row < std::min(3, m); ++row) {
          for (int col = 0; col < std::min(3, n); ++col) {
            printf("  A(%d,%d) = %.1f, B(%d,%d) = %.1f\n", 
                   row, col, A_host(row, col), row, col, B_host(row, col));
            
            // Check for errors
            if (std::abs(A_host(row, col) - B_host(row, col)) > 1e-10) {
              printf("  ERROR: Direct copy mismatch at (%d,%d)\n", row, col);
            }
          }
        }
        
        // Verify the transposed copy (A^T → C)
        printf("\nVerifying transposed copy (first few elements):\n");
        for (int row = 0; row < std::min(3, n); ++row) {
          for (int col = 0; col < std::min(3, m); ++col) {
            printf("  A(%d,%d) = %.1f, C(%d,%d) = %.1f\n", 
                   col, row, A_host(col, row), row, col, C_host(row, col));
            
            // Check for errors
            if (std::abs(A_host(col, row) - C_host(row, col)) > 1e-10) {
              printf("  ERROR: Transposed copy mismatch at A(%d,%d) vs C(%d,%d)\n", 
                     col, row, row, col);
            }
          }
        }
        
        // Demonstrate vector copying
        int vec_length = 10;
        
        // Create views for batched vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type> 
          X("X", batch_size, vec_length),   // Source vectors
          Y("Y", batch_size, vec_length);   // Destination vectors
        
        // Initialize source vectors
        Kokkos::parallel_for("init_vectors", policy, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < vec_length; ++j) {
            X(i, j) = j + 1.0;
            Y(i, j) = 0.0;
          }
        });
        
        Kokkos::fence();
        
        // Perform batched vector copy using SerialCopy inside a parallel_for
        Kokkos::parallel_for("batched_vector_copy", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slices
          auto X_i = Kokkos::subview(X, i, Kokkos::ALL());
          auto Y_i = Kokkos::subview(Y, i, Kokkos::ALL());
          
          // Perform vector copy (X → Y)
          KokkosBatched::SerialCopy<
            KokkosBatched::Trans::NoTranspose,  // ArgTrans
            1                                   // rank = 1 for vectors
          >::invoke(X_i, Y_i);
        });
        
        Kokkos::fence();
        
        // Copy vector results to host for verification
        auto X_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(X, 0, Kokkos::ALL()));
        auto Y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(Y, 0, Kokkos::ALL()));
        
        // Verify the vector copy
        printf("\nVerifying vector copy (first few elements):\n");
        for (int j = 0; j < std::min(5, vec_length); ++j) {
          printf("  X(%d) = %.1f, Y(%d) = %.1f\n", j, X_host(j), j, Y_host(j));
          
          // Check for errors
          if (std::abs(X_host(j) - Y_host(j)) > 1e-10) {
            printf("  ERROR: Vector copy mismatch at element %d\n", j);
          }
        }
      }
      Kokkos::finalize();
      return 0;
    }
