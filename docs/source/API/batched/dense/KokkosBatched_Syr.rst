KokkosBatched::Syr
##################

Defined in header: :code:`KokkosBatched_Syr.hpp`

.. code-block:: c++

    template <typename ArgUplo, typename ArgTrans>
    struct SerialSyr {
      template <typename ScalarType, typename XViewType, typename AViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const ScalarType alpha,
             const XViewType& x,
             const AViewType& a);
    };

The ``Syr`` function performs a symmetric rank-1 update of a matrix. This operation is equivalent to the BLAS routines ``DSYR`` for real matrices or ``CHER`` for complex matrices.

Mathematically, it performs:

.. math::

    A := \alpha \cdot x \cdot x^T + A

for real matrices, or

.. math::

    A := \alpha \cdot x \cdot x^H + A

for complex matrices, where:

- :math:`A` is a symmetric or Hermitian matrix
- :math:`x` is a vector
- :math:`\alpha` is a scalar
- :math:`x^T` is the transpose of :math:`x`
- :math:`x^H` is the conjugate transpose of :math:`x`

Only the upper or lower triangular part of A (as specified by ArgUplo) is updated.

Parameters
==========

:alpha: Scalar value
:x: Input view containing the vector x
:a: Input/output view containing the symmetric/Hermitian matrix A

Type Requirements
-----------------

- ``ArgUplo`` must be one of the following:
   - ``KokkosBatched::Uplo::Upper`` to update the upper triangular part of A
   - ``KokkosBatched::Uplo::Lower`` to update the lower triangular part of A

- ``ArgTrans`` must be one of the following:
   - ``KokkosBatched::Trans::Transpose`` for {s,d,c,z}syr operations (x*x^T)
   - ``KokkosBatched::Trans::ConjTranspose`` for {c,z}her operations (x*x^H)

- ``ScalarType`` must be a scalar type compatible with the element type of the views
- ``XViewType`` must be a rank-1 view containing the vector
- ``AViewType`` must be a rank-2 view representing the symmetric/Hermitian matrix
- All views must be accessible in the execution space

Examples
========

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Syr.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimension
        int n = 5;
        
        // Create matrix and vector
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A("A", n, n);
        Kokkos::View<scalar_type*, memory_space> x("x", n);
        
        // Initialize matrix and vector on host
        auto A_host = Kokkos::create_mirror_view(A);
        auto x_host = Kokkos::create_mirror_view(x);
        
        // Initialize matrix with identity
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            A_host(i, j) = (i == j) ? 1.0 : 0.0;
          }
        }
        
        // Initialize vector with known values
        for (int i = 0; i < n; ++i) {
          x_host(i) = i + 1.0;
        }
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(x, x_host);
        
        // Scalar value for the update
        scalar_type alpha = 2.0;
        
        // Perform symmetric rank-1 update (upper triangular)
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialSyr<KokkosBatched::Uplo::Upper, 
                                 KokkosBatched::Trans::Transpose>::invoke(alpha, x, A);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify results by explicitly computing alpha*x*x^T + A
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> 
          A_expected("A_expected", n, n);
        
        // Start with identity
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            A_expected(i, j) = (i == j) ? 1.0 : 0.0;
          }
        }
        
        // Add alpha*x*x^T to upper triangular part
        for (int i = 0; i < n; ++i) {
          for (int j = i; j < n; ++j) { // Upper triangular part only
            A_expected(i, j) += alpha * x_host(i) * x_host(j);
          }
        }
        
        // Check results
        bool test_passed = true;
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            if (j >= i) { // Only check updated part
              if (std::abs(A_host(i, j) - A_expected(i, j)) > 1e-10) {
                test_passed = false;
                std::cout << "Mismatch at (" << i << ", " << j << "): " 
                          << A_host(i, j) << " vs expected " << A_expected(i, j) << std::endl;
              }
            } else {
              // Lower triangular part should remain unchanged
              if (std::abs(A_host(i, j) - ((i == j) ? 1.0 : 0.0)) > 1e-10) {
                test_passed = false;
                std::cout << "Lower triangular part changed at (" << i << ", " << j << "): " 
                          << A_host(i, j) << " vs expected " << ((i == j) ? 1.0 : 0.0) << std::endl;
              }
            }
          }
        }
        
        if (test_passed) {
          std::cout << "Syr test: PASSED" << std::endl;
        } else {
          std::cout << "Syr test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Complex Example
---------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Syr.hpp>
    #include <complex>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Complex scalar type
    using scalar_type = Kokkos::complex<double>;
    using real_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimension
        int n = 4;
        
        // Create matrix and vector
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A("A", n, n);
        Kokkos::View<scalar_type*, memory_space> x("x", n);
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror_view(A);
        auto x_host = Kokkos::create_mirror_view(x);
        
        // Initialize matrix with identity
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            A_host(i, j) = (i == j) ? scalar_type(1.0, 0.0) : scalar_type(0.0, 0.0);
          }
        }
        
        // Initialize vector with complex values
        for (int i = 0; i < n; ++i) {
          x_host(i) = scalar_type(i + 1.0, i * 0.5);
        }
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(x, x_host);
        
        // Real scalar for Hermitian update (alpha must be real for Hermitian matrices)
        real_type alpha = 1.0;
        
        // Perform Hermitian rank-1 update (lower triangular)
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialSyr<KokkosBatched::Uplo::Lower, 
                                 KokkosBatched::Trans::ConjTranspose>::invoke(alpha, x, A);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify results by explicitly computing alpha*x*x^H + A
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> 
          A_expected("A_expected", n, n);
        
        // Start with identity
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            A_expected(i, j) = (i == j) ? scalar_type(1.0, 0.0) : scalar_type(0.0, 0.0);
          }
        }
        
        // Add alpha*x*x^H to lower triangular part
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j <= i; ++j) { // Lower triangular part only
            A_expected(i, j) += alpha * x_host(i) * Kokkos::conj(x_host(j));
          }
        }
        
        // Check results
        bool test_passed = true;
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            if (j <= i) { // Only check updated part
              if (std::abs(A_host(i, j).real() - A_expected(i, j).real()) > 1e-10 ||
                  std::abs(A_host(i, j).imag() - A_expected(i, j).imag()) > 1e-10) {
                test_passed = false;
                std::cout << "Mismatch at (" << i << ", " << j << "): " 
                          << A_host(i, j) << " vs expected " << A_expected(i, j) << std::endl;
              }
            } else {
              // Upper triangular part should remain unchanged
              if (std::abs(A_host(i, j).real() - ((i == j) ? 1.0 : 0.0)) > 1e-10 ||
                  std::abs(A_host(i, j).imag()) > 1e-10) {
                test_passed = false;
                std::cout << "Upper triangular part changed at (" << i << ", " << j << "): " 
                          << A_host(i, j) << " vs expected " << ((i == j) ? 1.0 : 0.0) << std::endl;
              }
            }
          }
        }
        
        if (test_passed) {
          std::cout << "Syr (Her) complex test: PASSED" << std::endl;
        } else {
          std::cout << "Syr (Her) complex test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Batched Example
---------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Syr.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Batch and matrix dimensions
        int batch_size = 50; // Number of matrices
        int n = 5;           // Matrix dimension
        
        // Create batched views
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A("A", batch_size, n, n);
        Kokkos::View<scalar_type**, memory_space> 
          x("x", batch_size, n);
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror_view(A);
        auto x_host = Kokkos::create_mirror_view(x);
        
        for (int b = 0; b < batch_size; ++b) {
          // Initialize each matrix with identity
          for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
              A_host(b, i, j) = (i == j) ? 1.0 : 0.0;
            }
          }
          
          // Initialize each vector with unique values
          for (int i = 0; i < n; ++i) {
            x_host(b, i) = (i + 1.0) * (b + 1.0) * 0.1;
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(x, x_host);
        
        // Scalar values for the updates (one per batch)
        Kokkos::View<scalar_type*, memory_space> alpha("alpha", batch_size);
        auto alpha_host = Kokkos::create_mirror_view(alpha);
        
        for (int b = 0; b < batch_size; ++b) {
          alpha_host(b) = 2.0 + 0.1 * b;
        }
        
        Kokkos::deep_copy(alpha, alpha_host);
        
        // Perform batched symmetric rank-1 updates
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto A_b = Kokkos::subview(A, b, Kokkos::ALL(), Kokkos::ALL());
          auto x_b = Kokkos::subview(x, b, Kokkos::ALL());
          
          KokkosBatched::SerialSyr<KokkosBatched::Uplo::Upper, 
                                 KokkosBatched::Trans::Transpose>::invoke(alpha(b), x_b, A_b);
        });
        
        // Results are now in A
        // Each A(b, :, :) contains the updated matrix
      }
      Kokkos::finalize();
      return 0;
    }
