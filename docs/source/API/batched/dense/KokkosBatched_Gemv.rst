KokkosBatched::Gemv
###################

Defined in header `KokkosBatched_Gemv_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Gemv_Decl.hpp>`_

.. code:: c++

    template <typename ArgTrans, typename ArgAlgo>
    struct SerialGemv {
      template <typename ScalarType, typename AViewType, typename xViewType, typename yViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType alpha, const AViewType &A, 
                                              const xViewType &x, const ScalarType beta, 
                                              const yViewType &y);
    };

    template <typename MemberType, typename ArgTrans, typename ArgAlgo>
    struct TeamGemv {
      template <typename ScalarType, typename AViewType, typename xViewType, typename yViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, 
                                              const AViewType &A, const xViewType &x, 
                                              const ScalarType beta, const yViewType &y);
    };

    template <typename MemberType, typename ArgTrans, typename ArgAlgo>
    struct TeamVectorGemv {
      template <typename ScalarType, typename AViewType, typename xViewType, typename yViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, 
                                              const AViewType &A, const xViewType &x, 
                                              const ScalarType beta, const yViewType &y);
    };

Performs batched general matrix-vector multiplication (GEMV). For each matrix-vector pair in the batch, computes:

.. math::

   y = \alpha \cdot \text{op}(A) \cdot x + \beta \cdot y

where:

- :math:`\text{op}(A)` can be :math:`A` or :math:`A^T` (transpose) or :math:`A^H` (Hermitian transpose)
- :math:`\alpha` and :math:`\beta` are scalar values
- :math:`A` is a matrix
- :math:`x` and :math:`y` are vectors
- The operation updates :math:`y` in-place

This is a fundamental BLAS Level 2 operation implemented for batched execution in parallel computing environments.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:alpha: Scalar multiplier for the matrix-vector product
:A: Input view containing batch of matrices
:x: Input view containing batch of vectors
:beta: Scalar multiplier for the y vector
:y: Input/output view for vectors that will be updated

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgTrans`` must be one of:

  - ``Trans::NoTranspose`` - use A as is
  - ``Trans::Transpose`` - use transpose of A
  - ``Trans::ConjTranspose`` - use conjugate transpose of A

- ``ArgAlgo`` must be one of algorithm variants (implementation dependent)
- ``AViewType`` must be a rank-2 or rank-3 Kokkos View representing matrices
- ``xViewType`` and ``yViewType`` must be rank-1 or rank-2 Kokkos Views representing vectors with compatible dimensions:

  - For NoTranspose: A(m,n), x(n), y(m)
  - For Transpose: A(m,n), x(m), y(n)

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Gemv_Decl.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 1000;   // Number of matrix-vector operations
        int m = 8;               // Rows in matrix
        int n = 6;               // Columns in matrix
        
        // Create views for batched matrices and vectors
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, m, n);  // Matrices
        
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type>
          x("x", batch_size, n),     // Input vectors for NoTranspose case
          y("y", batch_size, m);     // Output vectors
        
        // Fill matrices and vectors with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_data", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th matrix with a simple pattern
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
              A(i, row, col) = 1.0; // Simple matrix with all ones
            }
          }
          
          // Initialize vectors
          for (int j = 0; j < n; ++j) {
            x(i, j) = 1.0;  // Vector of ones
          }
          
          for (int j = 0; j < m; ++j) {
            y(i, j) = 0.0;  // Initialize y to zeros
          }
        });
        
        Kokkos::fence();
        
        // Define scalar multipliers
        scalar_type alpha = 2.0;  // Multiplier for A*x
        scalar_type beta = 1.0;   // Multiplier for y
        
        // Perform batched GEMV using TeamPolicy
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_gemv", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto x_i = Kokkos::subview(x, i, Kokkos::ALL());
            auto y_i = Kokkos::subview(y, i, Kokkos::ALL());
            
            // Perform GEMV using Team variant
            KokkosBatched::TeamGemv<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Trans::NoTranspose,       // ArgTrans
              KokkosBatched::Algo::Gemv::Unblocked     // ArgAlgo
            >::invoke(member, alpha, A_i, x_i, beta, y_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(y, 0, Kokkos::ALL()));
        
        // Verify results for first batch
        // Expected: y = alpha*A*x + beta*y = 2.0*(matrix of ones)*(vector of ones) + 1.0*(vector of zeros)
        // Since each row of A has n elements = 6, and x is all ones, each element of y should be 2.0*6.0 = 12.0
        printf("GEMV result verification (first few elements):\n");
        const double expected_value = 2.0 * n;  // alpha * (dot product of row of ones with x of ones)
        
        for (int j = 0; j < std::min(5, m); ++j) {
          printf("  y(%d) = %.1f (expected %.1f)\n", j, y_host(j), expected_value);
          
          if (std::abs(y_host(j) - expected_value) > 1e-10) {
            printf("  ERROR: Value mismatch at element %d\n", j);
          }
        }
        
        // Now demonstrate transpose version
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type>
          x_trans("x_trans", batch_size, m),  // Input vectors for Transpose case
          y_trans("y_trans", batch_size, n);  // Output vectors
        
        // Initialize vectors for transpose case
        Kokkos::parallel_for("init_trans_data", policy, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < m; ++j) {
            x_trans(i, j) = 1.0;  // Vector of ones
          }
          
          for (int j = 0; j < n; ++j) {
            y_trans(i, j) = 0.0;  // Initialize y to zeros
          }
        });
        
        Kokkos::fence();
        
        // Perform batched transpose GEMV (A^T * x)
        Kokkos::parallel_for("batched_gemv_trans", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto x_i = Kokkos::subview(x_trans, i, Kokkos::ALL());
            auto y_i = Kokkos::subview(y_trans, i, Kokkos::ALL());
            
            // Perform transpose GEMV using Team variant
            KokkosBatched::TeamGemv<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Trans::Transpose,         // ArgTrans
              KokkosBatched::Algo::Gemv::Unblocked     // ArgAlgo
            >::invoke(member, alpha, A_i, x_i, beta, y_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy transpose results to host for verification
        auto y_trans_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                              Kokkos::subview(y_trans, 0, Kokkos::ALL()));
        
        // Verify transpose results for first batch
        // Expected: y = alpha*A^T*x + beta*y = 2.0*(transpose of ones matrix)*(vector of ones) + 1.0*(vector of zeros)
        // Since each column of A has m elements = 8, and x is all ones, each element of y should be 2.0*8.0 = 16.0
        printf("\nTranspose GEMV result verification (first few elements):\n");
        const double expected_trans_value = 2.0 * m;  // alpha * (dot product of column of ones with x of ones)
        
        for (int j = 0; j < std::min(5, n); ++j) {
          printf("  y_trans(%d) = %.1f (expected %.1f)\n", j, y_trans_host(j), expected_trans_value);
          
          if (std::abs(y_trans_host(j) - expected_trans_value) > 1e-10) {
            printf("  ERROR: Value mismatch at element %d\n", j);
          }
        }
      }
      Kokkos::finalize();
      return 0;
    }
