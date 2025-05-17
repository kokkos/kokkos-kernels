KokkosBatched::Dot
##################

Defined in header: :code:`KokkosBatched_Dot.hpp`

.. code:: c++

    template <typename ArgTrans = Trans::NoTranspose>
    struct SerialDot {
      template <typename XViewType, typename YViewType, typename DotViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const XViewType &X, 
                                              const YViewType &Y, 
                                              const DotViewType &dot);
    };

    template <typename MemberType, typename ArgTrans = Trans::NoTranspose>
    struct TeamDot {
      template <typename XViewType, typename YViewType, typename DotViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const XViewType &X, 
                                              const YViewType &Y, 
                                              const DotViewType &dot);
    };

    template <typename MemberType, typename ArgTrans = Trans::NoTranspose>
    struct TeamVectorDot {
      template <typename XViewType, typename YViewType, typename DotViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const XViewType &X, 
                                              const YViewType &Y, 
                                              const DotViewType &dot);
    };

Computes the dot product of batched vectors. For each pair of vectors in the batch, the operation calculates:

.. math::

   \text{dot}_i = \sum_{j} x_{ij} \cdot y_{ij}

Depending on the ``ArgTrans`` template parameter, the operation computes:

- For ``Trans::NoTranspose``: Dot product of corresponding rows in X and Y
- For ``Trans::Transpose`` or ``Trans::ConjTranspose``: Dot product of corresponding columns in X and Y

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:X: Input view containing batch of vectors
:Y: Input view containing batch of vectors
:dot: Output view to store the dot product results

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgTrans`` must be one of:

  - ``Trans::NoTranspose`` - use rows of matrices (default)
  - ``Trans::Transpose`` or ``Trans::ConjTranspose`` - use columns of matrices

- ``XViewType`` and ``YViewType`` must be rank-2 Kokkos Views with compatible dimensions
- ``DotViewType`` must be a rank-1 Kokkos View to store the results

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Dot.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define matrix dimensions
        int batch_size = 1000;    // Number of vector batches
        int vector_length = 128;  // Length of each vector
        
        // Create views for batched vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type> 
          X("X", batch_size, vector_length),
          Y("Y", batch_size, vector_length);
        
        // Create view for dot product results
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, device_type>
          dot_result("dot_result", batch_size);
        
        // Fill vectors with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_vectors", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th vector pair
          for (int j = 0; j < vector_length; ++j) {
            X(i, j) = 1.0;              // All ones
            Y(i, j) = static_cast<double>(j + 1); // Increasing values
          }
        });
        
        Kokkos::fence();
        
        // Compute batched dot products using different execution modes
        
        // 1. Serial mode (inside a parallel_for)
        Kokkos::parallel_for("serial_dot", policy, KOKKOS_LAMBDA(const int i) {
          // Extract the i-th vectors
          auto X_i = Kokkos::subview(X, i, Kokkos::ALL());
          auto Y_i = Kokkos::subview(Y, i, Kokkos::ALL());
          auto dot_i = Kokkos::subview(dot_result, i);
          
          // Compute dot product in serial mode
          KokkosBatched::SerialDot<>::invoke(X_i, Y_i, dot_i);
        });
        
        Kokkos::fence();
        
        // 2. Team mode
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO);
        
        Kokkos::parallel_for("team_dot", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract the i-th vectors
            auto X_i = Kokkos::subview(X, i, Kokkos::ALL());
            auto Y_i = Kokkos::subview(Y, i, Kokkos::ALL());
            auto dot_i = Kokkos::subview(dot_result, i);
            
            // Compute dot product using Team mode
            KokkosBatched::TeamDot<typename team_policy_type::member_type>::invoke(
              member, X_i, Y_i, dot_i);
          }
        );
        
        Kokkos::fence();
        
        // 3. TeamVector mode
        team_policy_type policy_team_vector(batch_size, Kokkos::AUTO, Kokkos::AUTO);
        
        Kokkos::parallel_for("teamvector_dot", policy_team_vector, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract the i-th vectors
            auto X_i = Kokkos::subview(X, i, Kokkos::ALL());
            auto Y_i = Kokkos::subview(Y, i, Kokkos::ALL());
            auto dot_i = Kokkos::subview(dot_result, i);
            
            // Compute dot product using TeamVector mode
            KokkosBatched::TeamVectorDot<typename team_policy_type::member_type>::invoke(
              member, X_i, Y_i, dot_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto dot_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), dot_result);
        
        // For this example, the expected dot product for each vector pair is:
        // Sum of 1 * (j+1) for j=0 to vector_length-1, which equals:
        // vector_length * (vector_length + 1) / 2
        double expected = static_cast<double>(vector_length) * (vector_length + 1) / 2;
        
        // Verify the first few results
        for (int i = 0; i < std::min(5, batch_size); ++i) {
          printf("Batch %d: Dot product = %.1f (expected %.1f)\n", 
                 i, dot_host(i), expected);
        }
      }
      Kokkos::finalize();
      return 0;
    }
