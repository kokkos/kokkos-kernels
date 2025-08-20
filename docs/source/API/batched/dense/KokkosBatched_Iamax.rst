KokkosBatched::Iamax
####################

Defined in header: :code:`KokkosBatched_Iamax.hpp`

.. code:: c++

    struct SerialIamax {
      template <typename XViewType>
      KOKKOS_INLINE_FUNCTION static typename XViewType::size_type invoke(const XViewType &x);
    };

Finds the index of the first element having maximum absolute value in a vector. This is a batched implementation of the BLAS Level 1 ``IAMAX`` operation, which returns:

.. math::

   \text{index of } \max_{i}(|x_i|)

For complex numbers, the absolute value is defined as the magnitude of the complex number.

Parameters
==========

:x: Input view containing the vector to search

Type Requirements
-----------------

- ``XViewType`` must be a rank-1 Kokkos View representing a vector

Return Value
------------

- Returns the index of the first element having maximum absolute value
- Returns 0 for an empty vector (consistent with BLAS)

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Iamax.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    using size_type = int;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 1000;     // Number of vector operations
        int vector_length = 8;     // Length of each vector
        
        // Create views for batched vectors and results
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type> 
          X("X", batch_size, vector_length);  // Input vectors
        
        Kokkos::View<size_type*, Kokkos::LayoutRight, device_type>
          max_index("max_index", batch_size);  // Results: index of max abs value
        
        // Fill vectors with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_vectors", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th vector with a known pattern
          // Make the maximum absolute value at different positions for testing
          for (int j = 0; j < vector_length; ++j) {
            // Base pattern: alternating positive and negative values
            X(i, j) = (j % 2 == 0) ? (j + 1.0) : -(j + 1.0);
          }
          
          // Set a specific element to have the maximum absolute value
          // Use a different position for each batch to test robustness
          int max_pos = i % vector_length;
          X(i, max_pos) = (i % 2 == 0) ? 10.0 : -10.0;  // Max abs value = 10.0
        });
        
        Kokkos::fence();
        
        // Find the index of max absolute value for each vector
        Kokkos::parallel_for("batched_iamax", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slice
          auto X_i = Kokkos::subview(X, i, Kokkos::ALL());
          
          // Find the index of maximum absolute value
          max_index(i) = KokkosBatched::SerialIamax::invoke(X_i);
        });
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto X_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X);
        auto max_index_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), max_index);
        
        // Verify the results for a few batches
        printf("IAMAX results verification:\n");
        
        for (int i = 0; i < std::min(5, batch_size); ++i) {
          printf("Batch %d - Vector: [", i);
          for (int j = 0; j < vector_length; ++j) {
            printf("%.1f%s", X_host(i, j), (j < vector_length-1) ? ", " : "");
          }
          printf("]\n");
          
          int computed_index = max_index_host(i);
          printf("  Computed index of max abs value: %d\n", computed_index);
          
          // Verify by manually finding the max abs value
          scalar_type max_abs = 0.0;
          int expected_index = 0;
          
          for (int j = 0; j < vector_length; ++j) {
            scalar_type abs_val = std::abs(X_host(i, j));
            if (abs_val > max_abs) {
              max_abs = abs_val;
              expected_index = j;
            }
          }
          
          printf("  Expected index: %d, value: %.1f\n", expected_index, X_host(i, expected_index));
          
          if (computed_index == expected_index) {
            printf("  CORRECT: Indices match\n");
          } else {
            printf("  ERROR: Indices don't match\n");
          }
          printf("\n");
        }
        
        // Special cases demonstration
        printf("Special cases demonstration:\n");
        
        // Case 1: Empty vector (should return 0)
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, Kokkos::HostSpace> 
          empty_vec("empty", 0);
        
        int empty_result = KokkosBatched::SerialIamax::invoke(empty_vec);
        printf("  Empty vector result: %d (expected 0)\n", empty_result);
        
        // Case 2: Vector with all same absolute values (should return first occurrence)
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, Kokkos::HostSpace> 
          same_vec("same", 5);
        
        for (int i = 0; i < 5; ++i) {
          same_vec(i) = (i % 2 == 0) ? 5.0 : -5.0;  // Same absolute value
        }
        
        int same_result = KokkosBatched::SerialIamax::invoke(same_vec);
        printf("  Vector with all same absolute values: [5.0, -5.0, 5.0, -5.0, 5.0]\n");
        printf("  Result: %d (expected 0, the first occurrence)\n", same_result);
      }
      Kokkos::finalize();
      return 0;
    }
