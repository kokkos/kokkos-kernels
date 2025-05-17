KokkosBatched::HadamardProduct
##############################

Defined in header: :code:`KokkosBatched_HadamardProduct.hpp`

.. code:: c++

    struct SerialHadamardProduct {
      template <typename XViewType, typename YViewType, typename VViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const XViewType &X, 
                                               const YViewType &Y, 
                                               const VViewType &V);
    };

    template <typename MemberType>
    struct TeamHadamardProduct {
      template <typename XViewType, typename YViewType, typename VViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                               const XViewType &X, 
                                               const YViewType &Y, 
                                               const VViewType &V);
    };

    template <typename MemberType>
    struct TeamVectorHadamardProduct {
      template <typename XViewType, typename YViewType, typename VViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                               const XViewType &X, 
                                               const YViewType &Y, 
                                               const VViewType &V);
    };

    template <typename MemberType, typename ArgMode>
    struct HadamardProduct {
      template <typename XViewType, typename YViewType, typename VViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                               const XViewType &X, 
                                               const YViewType &Y, 
                                               const VViewType &V);
    };

Performs element-wise multiplication (Hadamard product) of batched matrices or vectors. For each triplet of matrices or vectors in the batch, computes:

.. math::

   V_{ij} = X_{ij} \cdot Y_{ij}

where:

- :math:`X_{ij}` and :math:`Y_{ij}` are elements of the input matrices or vectors
- :math:`V_{ij}` is the element of the output matrix or vector

The Hadamard product performs element-wise multiplication rather than the standard matrix multiplication, producing a matrix of the same dimensions as the inputs.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:X: Input view containing first batch of matrices or vectors
:Y: Input view containing second batch of matrices or vectors
:V: Output view for the element-wise product results

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgMode`` must be one of:

  - ``Mode::Serial`` - for serial execution
  - ``Mode::Team`` - for team-based execution
  - ``Mode::TeamVector`` - for team-vector execution

- ``XViewType``, ``YViewType``, and ``VViewType`` must be:

  - Rank-2 Kokkos Views with compatible dimensions (batch_size, vector_length) or (batch_size, m, n)
  - Value types that support multiplication

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_HadamardProduct.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 1000;    // Number of matrix triplets
        int m = 4;                // Rows in each matrix
        int n = 5;                // Columns in each matrix
        
        // Create views for batched matrices
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          X("X", batch_size, m, n),  // First input matrices
          Y("Y", batch_size, m, n),  // Second input matrices
          V("V", batch_size, m, n);  // Output matrices
        
        // Fill matrices with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th matrix triplet
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
              X(i, row, col) = 2.0;                  // All elements = 2.0
              Y(i, row, col) = static_cast<double>(row + col + 1);  // Varying values
              V(i, row, col) = 0.0;                  // Initialize output to zero
            }
          }
        });
        
        Kokkos::fence();
        
        // Perform batched Hadamard product using TeamPolicy
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_hadamard", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto X_i = Kokkos::subview(X, i, Kokkos::ALL(), Kokkos::ALL());
            auto Y_i = Kokkos::subview(Y, i, Kokkos::ALL(), Kokkos::ALL());
            auto V_i = Kokkos::subview(V, i, Kokkos::ALL(), Kokkos::ALL());
            
            // Perform Hadamard product using Team variant
            KokkosBatched::TeamHadamardProduct<typename team_policy_type::member_type>
              ::invoke(member, X_i, Y_i, V_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto X_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(X, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto Y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(Y, 0, Kokkos::ALL(), Kokkos::ALL()));
        auto V_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(V, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        // Verify the Hadamard product results for first matrix
        printf("Hadamard product verification (first matrix):\n");
        bool correct = true;
        
        for (int row = 0; row < m; ++row) {
          for (int col = 0; col < n; ++col) {
            double expected = X_host(row, col) * Y_host(row, col);
            double computed = V_host(row, col);
            
            if (row < 2 && col < 3) {  // Print only a few elements
              printf("  V(%d,%d) = X(%d,%d) * Y(%d,%d) = %.1f * %.1f = %.1f\n",
                     row, col, row, col, row, col,
                     X_host(row, col), Y_host(row, col), V_host(row, col));
            }
            
            if (std::abs(computed - expected) > 1e-10) {
              printf("  ERROR: Value mismatch at (%d,%d): computed = %.1f, expected = %.1f\n",
                     row, col, computed, expected);
              correct = false;
            }
          }
        }
        
        if (correct) {
          printf("Verification successful: Element-wise product correctly computed\n");
        }
        
        // Now demonstrate with vectors (1D arrays)
        int vector_length = 10;
        
        // Create views for batched vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type> 
          X_vec("X_vec", batch_size, vector_length),  // First input vectors
          Y_vec("Y_vec", batch_size, vector_length),  // Second input vectors
          V_vec("V_vec", batch_size, vector_length);  // Output vectors
        
        // Fill vectors with data
        Kokkos::parallel_for("init_vectors", policy, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < vector_length; ++j) {
            X_vec(i, j) = 3.0;                   // All elements = 3.0
            Y_vec(i, j) = static_cast<double>(j + 1);  // Increasing values
            V_vec(i, j) = 0.0;                   // Initialize output to zero
          }
        });
        
        Kokkos::fence();
        
        // Perform batched vector Hadamard product using TeamVectorPolicy
        team_policy_type policy_team_vector(batch_size, Kokkos::AUTO, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_vector_hadamard", policy_team_vector, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto X_i = Kokkos::subview(X_vec, i, Kokkos::ALL());
            auto Y_i = Kokkos::subview(Y_vec, i, Kokkos::ALL());
            auto V_i = Kokkos::subview(V_vec, i, Kokkos::ALL());
            
            // Perform Hadamard product using TeamVector variant
            KokkosBatched::TeamVectorHadamardProduct<typename team_policy_type::member_type>
              ::invoke(member, X_i, Y_i, V_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy vector results to host for verification
        auto X_vec_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                             Kokkos::subview(X_vec, 0, Kokkos::ALL()));
        auto Y_vec_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                             Kokkos::subview(Y_vec, 0, Kokkos::ALL()));
        auto V_vec_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                             Kokkos::subview(V_vec, 0, Kokkos::ALL()));
        
        // Verify the vector Hadamard product results
        printf("\nVector Hadamard product verification (first few elements):\n");
        correct = true;
        
        for (int j = 0; j < std::min(5, vector_length); ++j) {
          double expected = X_vec_host(j) * Y_vec_host(j);
          double computed = V_vec_host(j);
          
          printf("  V(%d) = X(%d) * Y(%d) = %.1f * %.1f = %.1f\n",
                 j, j, j, X_vec_host(j), Y_vec_host(j), V_vec_host(j));
          
          if (std::abs(computed - expected) > 1e-10) {
            printf("  ERROR: Value mismatch at element %d\n", j);
            correct = false;
          }
        }
        
        if (correct) {
          printf("Vector verification successful: Element-wise product correctly computed\n");
        }
      }
      Kokkos::finalize();
      return 0;
    }
