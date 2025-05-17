KokkosBatched::Axpy
###################

Defined in header `KokkosBatched_Axpy.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Axpy.hpp>`_

.. code:: c++

    struct SerialAxpy {
      template <typename XViewType, typename YViewType, typename alphaViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const alphaViewType &alpha, 
                                               const XViewType &X, 
                                               const YViewType &Y);
    };

    template <typename MemberType>
    struct TeamAxpy {
      template <typename XViewType, typename YViewType, typename alphaViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                               const alphaViewType &alpha, 
                                               const XViewType &X, 
                                               const YViewType &Y);
    };

    template <typename MemberType>
    struct TeamVectorAxpy {
      template <typename XViewType, typename YViewType, typename alphaViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                               const alphaViewType &alpha, 
                                               const XViewType &X, 
                                               const YViewType &Y);
    };

Performs batched AXPY operations on sets of vectors. For each vector pair in the batch, computes:

.. math::

   y_i = \alpha_i \cdot x_i + y_i

where:

- :math:`x_i` and :math:`y_i` are vectors in the i-th batch
- :math:`\alpha_i` is a scalar value for the i-th operation
- The operation updates :math:`y_i` in-place

This is a fundamental BLAS Level 1 operation implemented for batched execution in parallel computing environments.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:alpha: Input view containing scalar coefficients
:X: Input view containing batch of vectors to be scaled
:Y: Input/output view containing batch of vectors to be updated

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``XViewType`` and ``YViewType`` must be:

  - Rank-2 Kokkos Views with dimensions (batch_size, vector_length)
  - Compatible value types that support multiplication and addition
  - Compatible dimensions (same number of vectors with same lengths)

- ``alphaViewType`` must be:

  - Rank-1 Kokkos View with dimension (batch_size)
  - Value type compatible with XViewType elements for multiplication

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Axpy.hpp>

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
          alpha(i) = 2.0;
          
          // Initialize the i-th vector pair
          for (int j = 0; j < vector_length; ++j) {
            X(i, j) = 1.0;
            Y(i, j) = 3.0;
          }
        });
        
        Kokkos::fence();
        
        // Perform batched AXPY using TeamPolicy with TeamVector
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_axpy", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto X_i = Kokkos::subview(X, i, Kokkos::ALL());
            auto Y_i = Kokkos::subview(Y, i, Kokkos::ALL());
            auto alpha_i = Kokkos::subview(alpha, i);
            
            // Perform AXPY using TeamVector variant
            KokkosBatched::TeamVectorAxpy<typename team_policy_type::member_type>
              ::invoke(member, alpha_i, X_i, Y_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto Y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), Y);
        
        // Verify the first vector's results
        // Expected: Y = alpha*X + Y = 2.0*1.0 + 3.0 = 5.0
        const double expected_value = 5.0;
        bool correct = true;
        
        for (int j = 0; j < std::min(5, vector_length); ++j) {
          if (std::abs(Y_host(0, j) - expected_value) > 1e-10) {
            printf("Error at element %d: got %f, expected %f\n", 
                   j, Y_host(0, j), expected_value);
            correct = false;
          }
        }
        
        if (correct) {
          printf("Verification successful: Y = alpha*X + Y correctly computed\n");
        }
      }
      Kokkos::finalize();
      return 0;
    }
