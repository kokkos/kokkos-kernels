KokkosBatched::Xpay
###################

Defined in header: :code:`KokkosBatched_Xpay.hpp`

.. code:: c++

    struct SerialXpay {
      template <typename ViewType, typename alphaViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const alphaViewType &alpha, 
                                               const ViewType &X, 
                                               const ViewType &Y);
    };

    template <typename MemberType>
    struct TeamXpay {
      template <typename ViewType, typename alphaViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                               const alphaViewType &alpha, 
                                               const ViewType &X, 
                                               const ViewType &Y);
    };

    template <typename MemberType>
    struct TeamVectorXpay {
      template <typename ViewType, typename alphaViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                               const alphaViewType &alpha, 
                                               const ViewType &X, 
                                               const ViewType &Y);
    };

Performs batched XPAY operations on sets of vectors. For each vector pair in the batch, computes:

.. math::

   y_i = x_i + \alpha_i \cdot y_i

where:

- :math:`x_i` and :math:`y_i` are vectors in the i-th batch
- :math:`\alpha_i` is a scalar value for the i-th operation
- The operation updates :math:`y_i` in-place

This operation is similar to AXPY but with the scalar multiplier applied to Y instead of X. It is a variation of the BLAS Level 1 operation implemented for batched execution.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:alpha: Input view containing scalar coefficients
:X: Input view containing batch of vectors to be added
:Y: Input/output view containing batch of vectors to be scaled and updated

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ViewType`` must be:

  - Rank-2 Kokkos View with dimensions (batch_size, vector_length)
  - Value type that supports multiplication and addition

- ``alphaViewType`` must be:

  - Rank-1 Kokkos View with dimension (batch_size)
  - Value type compatible with ViewType elements for multiplication

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Xpay.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 1000;    // Number of vector pairs
        int vector_length = 128;  // Length of each vector
        
        // Create views for batched vectors and alpha values
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type> 
          X("X", batch_size, vector_length),
          Y("Y", batch_size, vector_length);
        
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, device_type>
          alpha("alpha", batch_size);
        
        // Fill vectors with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_data", policy, KOKKOS_LAMBDA(const int i) {
          // Set alpha value for this batch
          alpha(i) = 3.0;
          
          // Initialize the i-th vector pair
          for (int j = 0; j < vector_length; ++j) {
            X(i, j) = 2.0;
            Y(i, j) = 4.0;
          }
        });
        
        Kokkos::fence();
        
        // Perform batched XPAY using TeamPolicy with TeamVector
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_xpay", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto X_i = Kokkos::subview(X, i, Kokkos::ALL());
            auto Y_i = Kokkos::subview(Y, i, Kokkos::ALL());
            auto alpha_i = Kokkos::subview(alpha, i);
            
            // Perform XPAY using TeamVector variant
            KokkosBatched::TeamVectorXpay<typename team_policy_type::member_type>
              ::invoke(member, alpha_i, X_i, Y_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto Y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Y);
        
        // Verify the first vector's results
        // Expected: Y = X + alpha*Y = 2.0 + 3.0*4.0 = 2.0 + 12.0 = 14.0
        const double expected_value = 14.0;
        bool correct = true;
        
        printf("Verifying XPAY results:\n");
        for (int j = 0; j < std::min(5, vector_length); ++j) {
          printf("  Y(0,%d) = %.1f\n", j, Y_host(0, j));
          
          if (std::abs(Y_host(0, j) - expected_value) > 1e-10) {
            printf("  ERROR: Value mismatch at element %d\n", j);
            correct = false;
          }
        }
        
        if (correct) {
          printf("Verification successful: Y = X + alpha*Y correctly computed\n");
        }
        
        // Compare with AXPY (y = alpha*x + y) for educational purposes
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type> 
          X2("X2", batch_size, vector_length),
          Y2("Y2", batch_size, vector_length);
          
        // Initialize vectors for AXPY
        Kokkos::parallel_for("init_axpy_data", policy, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < vector_length; ++j) {
            X2(i, j) = 4.0;  // Same as Y in XPAY example
            Y2(i, j) = 2.0;  // Same as X in XPAY example
          }
        });
        
        Kokkos::fence();
        
        // Perform "mock" AXPY manually (just to show the difference)
        Kokkos::parallel_for("manual_axpy", policy, KOKKOS_LAMBDA(const int i) {
          for (int j = 0; j < vector_length; ++j) {
            Y2(i, j) = alpha(i) * X2(i, j) + Y2(i, j);
            // Result: Y2 = alpha*X2 + Y2 = 3.0*4.0 + 2.0 = 12.0 + 2.0 = 14.0
          }
        });
        
        Kokkos::fence();
        
        // Copy AXPY results to host for comparison
        auto Y2_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Y2);
        
        printf("\nComparing with AXPY results:\n");
        printf("  XPAY: Y = X + alpha*Y = 2.0 + 3.0*4.0 = 14.0\n");
        printf("  AXPY: Y = alpha*X + Y = 3.0*4.0 + 2.0 = 14.0\n");
        printf("  Same result with parameters swapped\n");
      }
      Kokkos::finalize();
      return 0;
    }
