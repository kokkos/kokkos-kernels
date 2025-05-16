KokkosBatched::Pbtrf
##################

Defined in header `KokkosBatched_Pbtrf.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/src/batched/KokkosBatched_Pbtrf.hpp>`_

.. code-block:: c++

    template <typename ArgUplo, typename ArgAlgo>
    struct SerialPbtrf {
      template <typename ABViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const ABViewType& ab);
    };

The ``Pbtrf`` function computes the Cholesky factorization of a real symmetric positive definite band matrix or a complex Hermitian positive definite band matrix. This operation is equivalent to the LAPACK routine ``DPBTRF`` for real matrices or ``ZPBTRF`` for complex matrices.

The factorization has the form:

.. math::

    A = U^T U \quad \text{if ArgUplo is Uplo::Upper}

or

.. math::

    A = L L^T \quad \text{if ArgUplo is Uplo::Lower}

where :math:`U` is an upper triangular band matrix and :math:`L` is a lower triangular band matrix. For complex matrices, :math:`U^T` and :math:`L^T` represent the conjugate transpose.

Parameters
==========

:ab: Input/output view containing the band matrix in compact band storage format. On exit, contains the Cholesky factorization.

Type Requirements
----------------

- ``ArgUplo`` must be one of the following:
   - ``KokkosBatched::Uplo::Upper`` for upper triangular factorization
   - ``KokkosBatched::Uplo::Lower`` for lower triangular factorization

- ``ArgAlgo`` must be ``KokkosBatched::Algo::Pbtrf::Unblocked`` for the unblocked algorithm
- ``ABViewType`` must be a rank-2 view containing the band matrix in the appropriate format
- The view must be accessible in the execution space

Band Storage Format
------------------

In the band storage format:

- If ``Uplo::Upper``: Element A(i,j) is stored in ab(ku+i-j,j) for max(0,j-ku) <= i <= j, where ku is the number of superdiagonals.
- If ``Uplo::Lower``: Element A(i,j) is stored in ab(i-j,j) for j <= i <= min(n-1,j+kl), where kl is the number of subdiagonals.

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Pbtrf.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions and band parameters
        int n = 10;           // Matrix dimension 
        int kd = 2;           // Number of (super/sub)diagonals
        int ldab = kd + 1;    // Leading dimension of band matrix
        
        // Create banded matrix (upper triangular band format)
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> ab("ab", ldab, n);
        
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
          
          // Create symmetric entries (not stored directly in upper format)
        }
        
        // Copy to device
        Kokkos::deep_copy(ab, ab_host);
        
        // Save a copy of the original matrix for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> ab_orig("ab_orig", ldab, n);
        Kokkos::deep_copy(ab_orig, ab);
        
        // Perform Cholesky factorization
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialPbtrf<KokkosBatched::Uplo::Upper, 
                                    KokkosBatched::Algo::Pbtrf::Unblocked>::invoke(ab);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(ab_host, ab);
        
        // At this point, ab_host contains the Cholesky factor U in band format
        // We can verify by reconstructing A = U^T * U and comparing with original
        
        // Create full matrices for verification
        // (In a real application, you would work directly with the banded format)
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> 
          A_full("A_full", n, n),
          U_full("U_full", n, n),
          UtU("UtU", n, n);
        
        // Extract original matrix A to full storage
        auto ab_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ab_orig);
        for (int j = 0; j < n; ++j) {
          for (int i = std::max(0, j-kd); i <= j; ++i) {
            int ab_row = kd + i - j;
            A_full(i, j) = ab_orig_host(ab_row, j);
            A_full(j, i) = ab_orig_host(ab_row, j); // Symmetric
          }
        }
        
        // Extract U to full storage
        for (int j = 0; j < n; ++j) {
          for (int i = std::max(0, j-kd); i <= j; ++i) {
            int ab_row = kd + i - j;
            U_full(i, j) = ab_host(ab_row, j);
          }
        }
        
        // Compute U^T * U
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            UtU(i, j) = 0.0;
            for (int k = 0; k < n; ++k) {
              UtU(i, j) += U_full(k, i) * U_full(k, j);
            }
          }
        }
        
        // Verify U^T * U ≈ A
        bool test_passed = true;
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            if (std::abs(UtU(i, j) - A_full(i, j)) > 1e-10) {
              test_passed = false;
              std::cout << "Mismatch at (" << i << ", " << j << "): " 
                        << UtU(i, j) << " vs " << A_full(i, j) << std::endl;
            }
          }
        }
        
        if (test_passed) {
          std::cout << "Pbtrf test: PASSED" << std::endl;
        } else {
          std::cout << "Pbtrf test: FAILED" << std::endl;
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
        int kd = 2;          // Number of (super/sub)diagonals
        int ldab = kd + 1;   // Leading dimension of band matrix
        
        // Create batched views for band matrices
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          ab("ab", batch_size, ldab, n);
        
        // Initialize on host
        auto ab_host = Kokkos::create_mirror_view(ab);
        
        for (int b = 0; b < batch_size; ++b) {
          // Clear matrix first
          for (int j = 0; j < n; ++j) {
            for (int i = 0; i < ldab; ++i) {
              ab_host(b, i, j) = 0.0;
            }
          }
          
          // Fill band matrix with SPD pattern (diagonally dominant)
          // Each batch gets slightly different values
          for (int j = 0; j < n; ++j) {
            // Diagonal entries (stored at row kd)
            ab_host(b, kd, j) = 4.0 + 0.1 * b;
            
            // Superdiagonal entries (if within band)
            if (j < n-1) ab_host(b, kd-1, j+1) = -1.0 - 0.01 * b;
            if (j < n-2) ab_host(b, kd-2, j+2) = -0.5 - 0.005 * b;
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(ab, ab_host);
        
        // Perform batch of Cholesky factorizations
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto ab_b = Kokkos::subview(ab, b, Kokkos::ALL(), Kokkos::ALL());
          
          KokkosBatched::SerialPbtrf<KokkosBatched::Uplo::Upper, 
                                    KokkosBatched::Algo::Pbtrf::Unblocked>::invoke(ab_b);
        });
        
        // Results are now in ab
        // Each ab(b, :, :) contains a Cholesky factorization
      }
      Kokkos::finalize();
      return 0;
    }
