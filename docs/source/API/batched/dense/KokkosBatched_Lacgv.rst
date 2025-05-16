KokkosBatched::Lacgv
##################

Defined in header `KokkosBatched_Lacgv.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/src/batched/KokkosBatched_Lacgv.hpp>`_

.. code-block:: c++

    struct SerialLacgv {
      template <typename XViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const XViewType& x);
    };

The ``Lacgv`` function conjugates the elements of a complex vector. For a real vector, no operation is performed. This operation is equivalent to the BLAS routine ``ZLACGV`` for complex vectors.

Mathematically, for a complex vector :math:`x` with elements :math:`x_i`, the operation performs:

.. math::

    x_i := \overline{x_i} \quad \text{for all } i

where :math:`\overline{x_i}` denotes the complex conjugate of :math:`x_i`.

Parameters
==========

:x: Input/output view containing the vector to be conjugated

Type Requirements
----------------

- ``XViewType`` must be a rank-1 view containing the vector
- If the elements of ``XViewType`` are complex values, the view must support complex conjugation
- For real valued ``XViewType``, this function is a no-op
- The view must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Lacgv.hpp>
    #include <complex>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type
    using scalar_type = Kokkos::complex<double>;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Vector dimension
        int n = 10;
        
        // Create vector
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, memory_space> x("x", n);
        
        // Initialize vector on host
        auto x_host = Kokkos::create_mirror_view(x);
        
        for (int i = 0; i < n; ++i) {
          // Initialize with complex values
          x_host(i) = scalar_type(i + 1.0, i * 0.5);
        }
        
        // Save a copy of the original vector for verification
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, memory_space> x_orig("x_orig", n);
        auto x_orig_host = Kokkos::create_mirror_view(x_orig);
        Kokkos::deep_copy(x_orig_host, x_host);
        
        // Copy initialized data to device
        Kokkos::deep_copy(x, x_host);
        Kokkos::deep_copy(x_orig, x_orig_host);
        
        // Apply conjugation
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialLacgv::invoke(x);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(x_host, x);
        
        // Verify that each element is conjugated
        bool test_passed = true;
        for (int i = 0; i < n; ++i) {
          scalar_type expected = Kokkos::conj(x_orig_host(i));
          if (std::abs(x_host(i).real() - expected.real()) > 1e-10 || 
              std::abs(x_host(i).imag() - expected.imag()) > 1e-10) {
            test_passed = false;
            std::cout << "Mismatch at index " << i << ": " 
                      << x_host(i) << " vs expected " << expected << std::endl;
          }
        }
        
        // Apply conjugation again - should restore the original vector
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialLacgv::invoke(x);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(x_host, x);
        
        // Verify that we've returned to the original values
        for (int i = 0; i < n; ++i) {
          if (std::abs(x_host(i).real() - x_orig_host(i).real()) > 1e-10 || 
              std::abs(x_host(i).imag() - x_orig_host(i).imag()) > 1e-10) {
            test_passed = false;
            std::cout << "Double conjugation failed at index " << i << ": " 
                      << x_host(i) << " vs original " << x_orig_host(i) << std::endl;
          }
        }
        
        if (test_passed) {
          std::cout << "Lacgv test: PASSED" << std::endl;
        } else {
          std::cout << "Lacgv test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Batched Example
--------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Lacgv.hpp>
    #include <complex>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type
    using scalar_type = Kokkos::complex<double>;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Batch and vector dimensions
        int batch_size = 100; // Number of vectors
        int n = 10;           // Vector length
        
        // Create batched vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          x("x", batch_size, n);
        
        // Initialize on host
        auto x_host = Kokkos::create_mirror_view(x);
        
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            // Initialize with complex values
            x_host(b, i) = scalar_type(i + 1.0 + 0.1 * b, i * 0.5 + 0.05 * b);
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(x, x_host);
        
        // Save original for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          x_orig("x_orig", batch_size, n);
        Kokkos::deep_copy(x_orig, x);
        
        // Apply conjugation to all vectors in batch
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto x_b = Kokkos::subview(x, b, Kokkos::ALL());
          
          KokkosBatched::SerialLacgv::invoke(x_b);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(x_host, x);
        
        // Verify conjugation for all batches
        auto x_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), x_orig);
        
        bool test_passed = true;
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            scalar_type expected = Kokkos::conj(x_orig_host(b, i));
            if (std::abs(x_host(b, i).real() - expected.real()) > 1e-10 || 
                std::abs(x_host(b, i).imag() - expected.imag()) > 1e-10) {
              test_passed = false;
              std::cout << "Batch " << b << " mismatch at index " << i << ": " 
                        << x_host(b, i) << " vs expected " << expected << std::endl;
              break;
            }
          }
          if (!test_passed) break;
        }
        
        if (test_passed) {
          std::cout << "Batched Lacgv test: PASSED" << std::endl;
        } else {
          std::cout << "Batched Lacgv test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
