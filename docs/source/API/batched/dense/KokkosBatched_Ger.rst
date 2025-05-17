KokkosBatched::Ger
##################

Defined in header: :code:`KokkosBatched_Ger.hpp`

.. code:: c++

    template <typename ArgTrans>
    struct SerialGer {
      template <typename ScalarType, typename XViewType, typename YViewType, typename AViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType alpha, 
                                              const XViewType &x, 
                                              const YViewType &y, 
                                              const AViewType &a);
    };

Performs batched general rank-1 update (GER). For each set of vectors x, y and matrix A in the batch, computes:

.. math::

   A = \alpha \cdot x \cdot y^T + A

or for complex vectors:

.. math::

   A = \alpha \cdot x \cdot y^H + A

where:

- :math:`\alpha` is a scalar value
- :math:`x` is a vector of length m
- :math:`y` is a vector of length n
- :math:`A` is a m × n matrix
- :math:`y^T` is the transpose of y
- :math:`y^H` is the conjugate transpose of y
- The operation updates :math:`A` in-place

This is a fundamental BLAS Level 2 operation that performs a rank-1 update to a matrix based on the outer product of two vectors.

Parameters
==========

:alpha: Scalar multiplier for the outer product
:x: Input view containing batch of vectors of length m
:y: Input view containing batch of vectors of length n
:a: Input/output view for matrices that will be updated

Type Requirements
-----------------

- ``ArgTrans`` must be one of:

  - ``Trans::Transpose`` - use transpose of y (regular GER operation)
  - ``Trans::ConjTranspose`` - use conjugate transpose of y (for complex vectors)

- ``ScalarType`` must be a scalar type compatible with multiplication operations
- ``XViewType`` must be a rank-1 or rank-2 Kokkos View of length m
- ``YViewType`` must be a rank-1 or rank-2 Kokkos View of length n
- ``AViewType`` must be a rank-2 or rank-3 Kokkos View of size m × n

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Ger.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 1000;  // Number of operations
        int m = 4;              // Length of x vectors
        int n = 5;              // Length of y vectors
        
        // Create views for batched vectors and matrices
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type> 
          x("x", batch_size, m),  // x vectors
          y("y", batch_size, n);  // y vectors
        
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type>
          A("A", batch_size, m, n);  // Matrices
        
        // Fill vectors and matrices with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_data", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize x vectors with ascending values
          for (int j = 0; j < m; ++j) {
            x(i, j) = static_cast<double>(j + 1);
          }
          
          // Initialize y vectors with descending values
          for (int j = 0; j < n; ++j) {
            y(i, j) = static_cast<double>(n - j);
          }
          
          // Initialize matrices with zeros
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
              A(i, row, col) = 1.0;  // Start with ones for easier verification
            }
          }
        });
        
        Kokkos::fence();
        
        // Define scalar multiplier
        scalar_type alpha = 2.0;
        
        // Perform batched GER operations
        Kokkos::parallel_for("batched_ger", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slices
          auto x_i = Kokkos::subview(x, i, Kokkos::ALL());
          auto y_i = Kokkos::subview(y, i, Kokkos::ALL());
          auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
          
          // Perform rank-1 update (GER) using Serial variant
          KokkosBatched::SerialGer<KokkosBatched::Trans::Transpose>
            ::invoke(alpha, x_i, y_i, A_i);
        });
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto x_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(x, 0, Kokkos::ALL()));
        auto y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(y, 0, Kokkos::ALL()));
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(A, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        // Verify the GER result for the first set
        printf("GER operation verification (first batch):\n");
        printf("  x = [");
        for (int j = 0; j < m; ++j) {
          printf("%.1f%s", x_host(j), (j < m-1) ? ", " : "");
        }
        printf("]\n");
        
        printf("  y = [");
        for (int j = 0; j < n; ++j) {
          printf("%.1f%s", y_host(j), (j < n-1) ? ", " : "");
        }
        printf("]\n");
        
        printf("  Result matrix A after alpha*x*y^T + A:\n");
        for (int row = 0; row < m; ++row) {
          printf("    [");
          for (int col = 0; col < n; ++col) {
            printf("%.1f%s", A_host(row, col), (col < n-1) ? ", " : "");
          }
          printf("]\n");
        }
        
        // Validate against expected computation
        bool correct = true;
        printf("\nValidation against manual calculation:\n");
        
        for (int row = 0; row < m; ++row) {
          for (int col = 0; col < n; ++col) {
            // Expected: A = alpha*x*y^T + initial_A
            double expected = alpha * x_host(row) * y_host(col) + 1.0; // Initial A was 1.0
            double computed = A_host(row, col);
            
            if (std::abs(computed - expected) > 1e-10) {
              printf("  ERROR: A(%d,%d) expected %.1f, got %.1f\n", 
                     row, col, expected, computed);
              correct = false;
            }
          }
        }
        
        if (correct) {
          printf("  All elements match expected values!\n");
        }
        
        // Demonstrate the GER operation for complex numbers
        // Here we'll simulate complex operations using double values
        printf("\nDemonstration of how a complex GER would differ:\n");
        printf("  For complex values, regular GER uses Trans::Transpose (y^T)\n");
        printf("  For complex conjugate GER, use Trans::ConjTranspose (y^H)\n");
        printf("  The difference affects only complex data types\n");
      }
      Kokkos::finalize();
      return 0;
    }
