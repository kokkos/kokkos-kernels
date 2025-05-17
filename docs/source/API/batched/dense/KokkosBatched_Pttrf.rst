KokkosBatched::Pttrf
####################

Defined in header: :code:`KokkosBatched_Pttrf.hpp`

.. code-block:: c++

    template <typename ArgAlgo>
    struct SerialPttrf {
      template <typename DViewType, typename EViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const DViewType& d,
             const EViewType& e);
    };

The ``Pttrf`` function computes the L*D*L^T factorization of a symmetric positive definite tridiagonal matrix A. This operation is equivalent to the LAPACK routine ``DPTTRF`` for real matrices or ``ZPTTRF`` for complex matrices.

The factorization has the form:

.. math::

    A = L \cdot D \cdot L^T

where:

- :math:`L` is a unit lower bidiagonal matrix with subdiagonals stored in ``e``
- :math:`D` is a diagonal matrix with diagonal elements stored in ``d``

Parameters
==========

:d: Input/output view containing diagonal elements of the matrix. On exit, contains the diagonal elements of D.
:e: Input/output view containing subdiagonal elements of the matrix. On exit, contains the subdiagonal elements of L.

Type Requirements
-----------------

- ``ArgAlgo`` must be ``KokkosBatched::Algo::Pttrf::Unblocked`` for the unblocked algorithm
- ``DViewType`` must be a rank-1 view containing the diagonal elements (length n)
- ``EViewType`` must be a rank-1 view containing the subdiagonal elements (length n-1)
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Pttrf.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimension
        int n = 10;
        
        // Create diagonal and off-diagonal vectors
        Kokkos::View<scalar_type*, memory_space> d("d", n);      // Diagonal elements
        Kokkos::View<scalar_type*, memory_space> e("e", n-1);    // Subdiagonal elements
        
        // Initialize vectors on host
        auto d_host = Kokkos::create_mirror_view(d);
        auto e_host = Kokkos::create_mirror_view(e);
        
        // Fill with a symmetric positive definite tridiagonal matrix
        // Using a simple model problem (1D Poisson equation discretization)
        for (int i = 0; i < n; ++i) {
          d_host(i) = 2.0;  // Diagonal
        }
        for (int i = 0; i < n-1; ++i) {
          e_host(i) = -1.0; // Subdiagonal
        }
        
        // Copy to device
        Kokkos::deep_copy(d, d_host);
        Kokkos::deep_copy(e, e_host);
        
        // Save original values for verification
        Kokkos::View<scalar_type*, memory_space> d_orig("d_orig", n);
        Kokkos::View<scalar_type*, memory_space> e_orig("e_orig", n-1);
        
        Kokkos::deep_copy(d_orig, d);
        Kokkos::deep_copy(e_orig, e);
        
        // Compute the factorization
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialPttrf<KokkosBatched::Algo::Pttrf::Unblocked>::invoke(d, e);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(d_host, d);
        Kokkos::deep_copy(e_host, e);
        
        // Verify the factorization by reconstructing A = L*D*L^T
        auto d_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_orig);
        auto e_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), e_orig);
        
        // Create full matrices for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> 
          A_orig("A_orig", n, n),
          L("L", n, n),
          D("D", n, n),
          LDLT("LDLT", n, n);
        
        // Construct original A in full storage
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            A_orig(i, j) = 0.0;
          }
          A_orig(i, i) = d_orig_host(i);
        }
        
        for (int i = 0; i < n-1; ++i) {
          A_orig(i+1, i) = e_orig_host(i);
          A_orig(i, i+1) = e_orig_host(i); // Symmetric
        }
        
        // Construct L and D from factorization
        // L is unit lower bidiagonal
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            L(i, j) = 0.0;
            D(i, j) = 0.0;
          }
          L(i, i) = 1.0;     // Unit diagonal
          D(i, i) = d_host(i); // Diagonal matrix D
        }
        
        for (int i = 0; i < n-1; ++i) {
          L(i+1, i) = e_host(i); // Subdiagonal of L
        }
        
        // Compute L*D*L^T
        // First, L*D
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> LD("LD", n, n);
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            LD(i, j) = 0.0;
            for (int k = 0; k < n; ++k) {
              LD(i, j) += L(i, k) * D(k, j);
            }
          }
        }
        
        // Then, (L*D)*L^T
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            LDLT(i, j) = 0.0;
            for (int k = 0; k < n; ++k) {
              LDLT(i, j) += LD(i, k) * L(j, k); // Note: L^T(k,j) = L(j,k)
            }
          }
        }
        
        // Verify A_orig ≈ LDLT
        bool test_passed = true;
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            if (std::abs(LDLT(i, j) - A_orig(i, j)) > 1e-10) {
              test_passed = false;
              std::cout << "Mismatch at (" << i << ", " << j << "): " 
                        << LDLT(i, j) << " vs " << A_orig(i, j) << std::endl;
            }
          }
        }
        
        if (test_passed) {
          std::cout << "Pttrf test: PASSED" << std::endl;
        } else {
          std::cout << "Pttrf test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Batched Example
--------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Pttrf.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Batch and matrix dimensions
        int batch_size = 50; // Number of matrices
        int n = 10;          // Matrix dimension
        
        // Create batched views
        Kokkos::View<scalar_type**, memory_space> d("d", batch_size, n);       // Diagonal elements
        Kokkos::View<scalar_type**, memory_space> e("e", batch_size, n-1);     // Subdiagonal elements
        
        // Initialize on host
        auto d_host = Kokkos::create_mirror_view(d);
        auto e_host = Kokkos::create_mirror_view(e);
        
        for (int b = 0; b < batch_size; ++b) {
          // Fill with a symmetric positive definite tridiagonal matrix
          // Slightly different for each batch
          for (int i = 0; i < n; ++i) {
            d_host(b, i) = 2.0 + 0.1 * b;  // Diagonal
          }
          for (int i = 0; i < n-1; ++i) {
            e_host(b, i) = -1.0 - 0.01 * b; // Subdiagonal
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(d, d_host);
        Kokkos::deep_copy(e, e_host);
        
        // Save original for verification
        Kokkos::View<scalar_type**, memory_space> d_orig("d_orig", batch_size, n);
        Kokkos::View<scalar_type**, memory_space> e_orig("e_orig", batch_size, n-1);
        
        Kokkos::deep_copy(d_orig, d);
        Kokkos::deep_copy(e_orig, e);
        
        // Perform batched factorization
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto d_b = Kokkos::subview(d, b, Kokkos::ALL());
          auto e_b = Kokkos::subview(e, b, Kokkos::ALL());
          
          KokkosBatched::SerialPttrf<KokkosBatched::Algo::Pttrf::Unblocked>::invoke(d_b, e_b);
        });
        
        // Factorizations are now in d and e
        // Each pair (d(b, :), e(b, :)) contains the factors for matrix b
      }
      Kokkos::finalize();
      return 0;
    }
