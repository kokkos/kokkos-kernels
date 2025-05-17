KokkosBatched::Pbtrs
####################

Defined in header: :code:`KokkosBatched_Pbtrs.hpp`

.. code-block:: c++

    template <typename ArgUplo, typename ArgAlgo>
    struct SerialPbtrs {
      template <typename ABViewType, typename BViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const ABViewType& ab,
             const BViewType& b);
    };

The ``Pbtrs`` function solves a system of linear equations with a symmetric positive definite band matrix using the Cholesky factorization computed by ``Pbtrf``. This operation is equivalent to the LAPACK routine ``DPBTRS`` for real matrices or ``ZPBTRS`` for complex matrices.

Given a Cholesky factorization of a symmetric positive definite band matrix A:

.. math::

    A = U^T U \quad \text{if ArgUplo is Uplo::Upper}

or

.. math::

    A = L L^T \quad \text{if ArgUplo is Uplo::Lower}

the function solves the system of equations :math:`A \cdot X = B` for X.

Parameters
==========

:ab: Input view containing the Cholesky factorization of the band matrix in compact band storage format
:b: Input/output view containing the right-hand side(s) on input and the solution(s) on output

Type Requirements
-----------------

- ``ArgUplo`` must be one of the following:
   - ``KokkosBatched::Uplo::Upper`` for upper triangular factorization
   - ``KokkosBatched::Uplo::Lower`` for lower triangular factorization

- ``ArgAlgo`` must be ``KokkosBatched::Algo::Pbtrs::Unblocked`` for the unblocked algorithm
- ``ABViewType`` must be a rank-2 view containing the band matrix in the appropriate format
- ``BViewType`` must be a rank-1 view for a single right-hand side, or a rank-2 view for multiple right-hand sides
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Pbtrf.hpp>
    #include <KokkosBatched_Pbtrs.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int n = 10;           // Matrix dimension 
        int nrhs = 2;         // Number of right-hand sides
        int kd = 2;           // Number of superdiagonals
        int ldab = kd + 1;    // Leading dimension of band matrix
        
        // Create banded matrix (upper triangular band format)
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> ab("ab", ldab, n);
        
        // Create right-hand sides / solution
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> B("B", n, nrhs);
        
        // Initialize band matrix on host with a positive definite matrix
        auto ab_host = Kokkos::create_mirror_view(ab);
        
        // Clear matrix first
        for (int j = 0; j < n; ++j) {
          for (int i = 0; i < ldab; ++i) {
            ab_host(i, j) = 0.0;
          }
        }
        
        // Fill band matrix with SPD pattern (diagonally dominant)
        for (int j = 0; j < n; ++j) {
          // Diagonal entries (stored at row kd)
          ab_host(kd, j) = 4.0;
          
          // Superdiagonal entries (if within band)
          if (j < n-1) ab_host(kd-1, j+1) = -1.0;
          if (j < n-2) ab_host(kd-2, j+2) = -0.5;
        }
        
        // Initialize right-hand sides on host
        auto B_host = Kokkos::create_mirror_view(B);
        for (int j = 0; j < nrhs; ++j) {
          for (int i = 0; i < n; ++i) {
            B_host(i, j) = 1.0 + i + j*n;
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(ab, ab_host);
        Kokkos::deep_copy(B, B_host);
        
        // Save a copy of the original matrix and right-hand sides for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> ab_orig("ab_orig", ldab, n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> B_orig("B_orig", n, nrhs);
        
        Kokkos::deep_copy(ab_orig, ab);
        Kokkos::deep_copy(B_orig, B);
        
        // Perform Cholesky factorization
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialPbtrf<KokkosBatched::Uplo::Upper, 
                                    KokkosBatched::Algo::Pbtrf::Unblocked>::invoke(ab);
        });
        
        // Solve the system using the factorization
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialPbtrs<KokkosBatched::Uplo::Upper, 
                                    KokkosBatched::Algo::Pbtrs::Unblocked>::invoke(ab, B);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(B_host, B);
        
        // Verify solution by checking A*X ≈ B_orig
        // For verification, extract full matrix A from band storage
        auto ab_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ab_orig);
        auto B_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B_orig);
        
        // Create full matrix A for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> A_full("A_full", n, n);
        
        // Extract band matrix to full storage
        for (int j = 0; j < n; ++j) {
          for (int i = std::max(0, j-kd); i <= j; ++i) {
            int ab_row = kd + i - j;
            A_full(i, j) = ab_orig_host(ab_row, j);
            A_full(j, i) = ab_orig_host(ab_row, j); // Symmetric
          }
        }
        
        // Check A*X ≈ B_orig
        bool test_passed = true;
        for (int j = 0; j < nrhs; ++j) {
          for (int i = 0; i < n; ++i) {
            scalar_type sum = 0.0;
            
            // Compute row i of A * column j of X
            for (int k = 0; k < n; ++k) {
              sum += A_full(i, k) * B_host(k, j);
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
          std::cout << "Pbtrs test: PASSED" << std::endl;
        } else {
          std::cout << "Pbtrs test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Batched Example
--------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Pbtrf.hpp>
    #include <KokkosBatched_Pbtrs.hpp>
    
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
        int kd = 2;          // Number of superdiagonals
        int ldab = kd + 1;   // Leading dimension of band matrix
        
        // Create batched views
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          ab("ab", batch_size, ldab, n);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          B("B", batch_size, n, nrhs);
        
        // Initialize on host
        auto ab_host = Kokkos::create_mirror_view(ab);
        auto B_host = Kokkos::create_mirror_view(B);
        
        for (int b = 0; b < batch_size; ++b) {
          // Clear matrix first
          for (int j = 0; j < n; ++j) {
            for (int i = 0; i < ldab; ++i) {
              ab_host(b, i, j) = 0.0;
            }
          }
          
          // Fill band matrix with SPD pattern (diagonally dominant)
          for (int j = 0; j < n; ++j) {
            // Diagonal entries (stored at row kd)
            ab_host(b, kd, j) = 4.0 + 0.1 * b;
            
            // Superdiagonal entries (if within band)
            if (j < n-1) ab_host(b, kd-1, j+1) = -1.0 - 0.01 * b;
            if (j < n-2) ab_host(b, kd-2, j+2) = -0.5 - 0.005 * b;
          }
          
          // Initialize right-hand sides
          for (int j = 0; j < nrhs; ++j) {
            for (int i = 0; i < n; ++i) {
              B_host(b, i, j) = 1.0 + i + j*n + b*0.1;
            }
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(ab, ab_host);
        Kokkos::deep_copy(B, B_host);
        
        // Save original for verification
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          ab_orig("ab_orig", batch_size, ldab, n);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          B_orig("B_orig", batch_size, n, nrhs);
        
        Kokkos::deep_copy(ab_orig, ab);
        Kokkos::deep_copy(B_orig, B);
        
        // Perform batched Cholesky factorization
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto ab_b = Kokkos::subview(ab, b, Kokkos::ALL(), Kokkos::ALL());
          
          KokkosBatched::SerialPbtrf<KokkosBatched::Uplo::Upper, 
                                    KokkosBatched::Algo::Pbtrf::Unblocked>::invoke(ab_b);
        });
        
        // Solve batched linear systems
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto ab_b = Kokkos::subview(ab, b, Kokkos::ALL(), Kokkos::ALL());
          auto B_b = Kokkos::subview(B, b, Kokkos::ALL(), Kokkos::ALL());
          
          KokkosBatched::SerialPbtrs<KokkosBatched::Uplo::Upper, 
                                    KokkosBatched::Algo::Pbtrs::Unblocked>::invoke(ab_b, B_b);
        });
        
        // Solutions are now in B
        // Each B(b, :, :) contains the solution for the corresponding system
      }
      Kokkos::finalize();
      return 0;
    }
