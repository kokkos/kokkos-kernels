KokkosBatched::Pttrs
####################

Defined in header: :code:`KokkosBatched_Pttrs.hpp`

.. code-block:: c++

    template <typename ArgUplo, typename ArgAlgo>
    struct SerialPttrs {
      template <typename DViewType, typename EViewType, typename BViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const DViewType& d,
             const EViewType& e,
             const BViewType& b);
    };

The ``Pttrs`` function solves a system of linear equations with a symmetric positive definite tridiagonal matrix using the L*D*L^T factorization computed by ``Pttrf``. This operation is equivalent to the LAPACK routine ``DPTTRS`` for real matrices or ``ZPTTRS`` for complex matrices.

Given the L*D*L^T factorization of a symmetric positive definite tridiagonal matrix A:

.. math::

    A = L \cdot D \cdot L^T

the function solves the system of equations :math:`A \cdot X = B` for X.

Parameters
==========

:d: Input view containing the diagonal elements of D from the factorization
:e: Input view containing the subdiagonal elements of L from the factorization
:b: Input/output view containing the right-hand side(s) on input and the solution(s) on output

Type Requirements
-----------------

- ``ArgUplo`` must be one of the following:
   - ``KokkosBatched::Uplo::Upper`` if vector e specifies the superdiagonal of a unit bidiagonal matrix U
   - ``KokkosBatched::Uplo::Lower`` if vector e specifies the subdiagonal of a unit bidiagonal matrix L
   - This parameter is primarily used for complex matrices

- ``ArgAlgo`` must be ``KokkosBatched::Algo::Pttrs::Unblocked`` for the unblocked algorithm
- ``DViewType`` must be a rank-1 view containing the diagonal elements (length n)
- ``EViewType`` must be a rank-1 view containing the subdiagonal elements (length n-1)
- ``BViewType`` must be a rank-1 view for a single right-hand side, or a rank-2 view for multiple right-hand sides
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Pttrf.hpp>
    #include <KokkosBatched_Pttrs.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimension and number of right-hand sides
        int n = 10;
        int nrhs = 2;
        
        // Create diagonal, off-diagonal, and right-hand side vectors
        Kokkos::View<scalar_type*, memory_space> d("d", n);      // Diagonal elements
        Kokkos::View<scalar_type*, memory_space> e("e", n-1);    // Subdiagonal elements
        Kokkos::View<scalar_type**, memory_space> B("B", n, nrhs); // Right-hand sides
        
        // Initialize vectors on host
        auto d_host = Kokkos::create_mirror_view(d);
        auto e_host = Kokkos::create_mirror_view(e);
        auto B_host = Kokkos::create_mirror_view(B);
        
        // Fill with a symmetric positive definite tridiagonal matrix
        // Using a simple model problem (1D Poisson equation discretization)
        for (int i = 0; i < n; ++i) {
          d_host(i) = 2.0;  // Diagonal
        }
        for (int i = 0; i < n-1; ++i) {
          e_host(i) = -1.0; // Subdiagonal
        }
        
        // Initialize right-hand sides
        for (int j = 0; j < nrhs; ++j) {
          for (int i = 0; i < n; ++i) {
            B_host(i, j) = 1.0 + i + j*n;
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(d, d_host);
        Kokkos::deep_copy(e, e_host);
        Kokkos::deep_copy(B, B_host);
        
        // Save original values for verification
        Kokkos::View<scalar_type*, memory_space> d_orig("d_orig", n);
        Kokkos::View<scalar_type*, memory_space> e_orig("e_orig", n-1);
        Kokkos::View<scalar_type**, memory_space> B_orig("B_orig", n, nrhs);
        
        Kokkos::deep_copy(d_orig, d);
        Kokkos::deep_copy(e_orig, e);
        Kokkos::deep_copy(B_orig, B);
        
        // Compute the factorization
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialPttrf<KokkosBatched::Algo::Pttrf::Unblocked>::invoke(d, e);
        });
        
        // Solve the system using the factorization
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialPttrs<KokkosBatched::Uplo::Lower, 
                                    KokkosBatched::Algo::Pttrs::Unblocked>::invoke(d, e, B);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(B_host, B);
        
        // Verify solution by checking A*X ≈ B_orig
        auto d_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), d_orig);
        auto e_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), e_orig);
        auto B_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B_orig);
        
        // Create full matrix A for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> A("A", n, n);
        
        // Construct original A in full storage
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            A(i, j) = 0.0;
          }
          A(i, i) = d_orig_host(i);
        }
        
        for (int i = 0; i < n-1; ++i) {
          A(i+1, i) = e_orig_host(i);
          A(i, i+1) = e_orig_host(i); // Symmetric
        }
        
        // Check A*X ≈ B_orig
        bool test_passed = true;
        for (int j = 0; j < nrhs; ++j) {
          for (int i = 0; i < n; ++i) {
            scalar_type sum = 0.0;
            
            // Compute row i of A * column j of X
            for (int k = 0; k < n; ++k) {
              sum += A(i, k) * B_host(k, j);
            }
            
            // Check against original right-hand side
            if (std::abs(sum - B_orig_host(i, j)) > 1e-10) {
              test_passed = false;
              std::cout << "Mismatch at (" << i << ", " << j << "): " 
                        << sum << " vs " << B_orig_host(i, j) << std::endl;
            }
          }
        }
        
        if (test_passed) {
          std::cout << "Pttrs test: PASSED" << std::endl;
        } else {
          std::cout << "Pttrs test: FAILED" << std::endl;
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
    #include <KokkosBatched_Pttrs.hpp>
    
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
        int nrhs = 2;        // Number of right-hand sides
        
        // Create batched views
        Kokkos::View<scalar_type**, memory_space> d("d", batch_size, n);       // Diagonal elements
        Kokkos::View<scalar_type**, memory_space> e("e", batch_size, n-1);     // Subdiagonal elements
        Kokkos::View<scalar_type***, memory_space> B("B", batch_size, n, nrhs); // Right-hand sides
        
        // Initialize on host
        auto d_host = Kokkos::create_mirror_view(d);
        auto e_host = Kokkos::create_mirror_view(e);
        auto B_host = Kokkos::create_mirror_view(B);
        
        for (int b = 0; b < batch_size; ++b) {
          // Fill with a symmetric positive definite tridiagonal matrix
          // Slightly different for each batch
          for (int i = 0; i < n; ++i) {
            d_host(b, i) = 2.0 + 0.1 * b;  // Diagonal
          }
          for (int i = 0; i < n-1; ++i) {
            e_host(b, i) = -1.0 - 0.01 * b; // Subdiagonal
          }
          
          // Initialize right-hand sides
          for (int j = 0; j < nrhs; ++j) {
            for (int i = 0; i < n; ++i) {
              B_host(b, i, j) = 1.0 + i + j*n + b*0.1;
            }
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(d, d_host);
        Kokkos::deep_copy(e, e_host);
        Kokkos::deep_copy(B, B_host);
        
        // Save original for verification
        Kokkos::View<scalar_type**, memory_space> d_orig("d_orig", batch_size, n);
        Kokkos::View<scalar_type**, memory_space> e_orig("e_orig", batch_size, n-1);
        Kokkos::View<scalar_type***, memory_space> B_orig("B_orig", batch_size, n, nrhs);
        
        Kokkos::deep_copy(d_orig, d);
        Kokkos::deep_copy(e_orig, e);
        Kokkos::deep_copy(B_orig, B);
        
        // Perform batched factorization
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto d_b = Kokkos::subview(d, b, Kokkos::ALL());
          auto e_b = Kokkos::subview(e, b, Kokkos::ALL());
          
          KokkosBatched::SerialPttrf<KokkosBatched::Algo::Pttrf::Unblocked>::invoke(d_b, e_b);
        });
        
        // Solve batched linear systems
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto d_b = Kokkos::subview(d, b, Kokkos::ALL());
          auto e_b = Kokkos::subview(e, b, Kokkos::ALL());
          auto B_b = Kokkos::subview(B, b, Kokkos::ALL(), Kokkos::ALL());
          
          KokkosBatched::SerialPttrs<KokkosBatched::Uplo::Lower, 
                                    KokkosBatched::Algo::Pttrs::Unblocked>::invoke(d_b, e_b, B_b);
        });
        
        // Solutions are now in B
        // Each B(b, :, :) contains the solution for the corresponding system
      }
      Kokkos::finalize();
      return 0;
    }
