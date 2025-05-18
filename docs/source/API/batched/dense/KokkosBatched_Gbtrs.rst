KokkosBatched::Gbtrs
####################

Defined in header: :code:`KokkosBatched_Gbtrs.hpp`

.. code-block:: c++

    template <typename ArgTrans, typename ArgAlgo>
    struct SerialGbtrs {
      template <typename AViewType, typename PivViewType, typename BViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const AViewType& A,
             const PivViewType& piv,
             const BViewType& b,
             const int kl,
             const int ku);
    };

The ``Gbtrs`` function solves a system of linear equations with a general band matrix A using the LU factorization computed by ``Gbtrf``. The function can solve the following systems:

1. :math:`A \cdot X = B` (Trans::NoTranspose)
2. :math:`A^T \cdot X = B` (Trans::Transpose)
3. :math:`A^H \cdot X = B` (Trans::ConjTranspose)

where A is an N-by-N band matrix with KL subdiagonals and KU superdiagonals, and B is a matrix with multiple right-hand sides.

Parameters
==========

:A: Input banded matrix view containing the LU factorization from Gbtrf
:piv: Input view containing the pivot indices from Gbtrf
:b: Input/output view containing right-hand sides on input and solutions on output
:kl: Number of subdiagonals within the band of A (kl ≥ 0)
:ku: Number of superdiagonals within the band of A (ku ≥ 0)

Type Requirements
-----------------

- ``ArgTrans`` must be one of the following:
   - ``KokkosBatched::Trans::NoTranspose`` to solve :math:`A \cdot X = B`
   - ``KokkosBatched::Trans::Transpose`` to solve :math:`A^T \cdot X = B`
   - ``KokkosBatched::Trans::ConjTranspose`` to solve :math:`A^H \cdot X = B`

- ``ArgAlgo`` must be ``KokkosBatched::Algo::Gbtrs::Unblocked`` for the unblocked algorithm
- ``AViewType`` must be a rank-2 view containing the banded matrix in the appropriate format with LU factorization
- ``PivViewType`` must be a rank-1 view containing the pivot indices
- ``BViewType`` must be a rank-1 view for a single right-hand side, or a rank-2 view for multiple right-hand sides
- All views must be accessible in the execution space

Examples
========

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Gbtrf.hpp>
    #include <KokkosBatched_Gbtrs.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions and band parameters
        int n = 10;          // Matrix dimension
        int nrhs = 2;        // Number of right-hand sides
        int kl = 2;          // Number of subdiagonals
        int ku = 1;          // Number of superdiagonals
        int ldab = 2*kl+ku+1; // Leading dimension of band matrix
        
        // Create banded matrix, pivot vector, and right-hand sides
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> Ab("Ab", ldab, n);
        Kokkos::View<int*, memory_space> piv("piv", n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> B("B", n, nrhs);
        
        // Initialize banded matrix on host
        auto Ab_host = Kokkos::create_mirror_view(Ab);
        
        // Create a diagonally dominant matrix for stability
        for (int j = 0; j < n; ++j) {
          for (int i = std::max(0, j-ku); i <= std::min(n-1, j+kl); ++i) {
            int band_row = ku + i - j;
            
            if (i == j) {
              // Diagonal - make it dominant
              Ab_host(band_row, j) = 10.0;
            } else {
              // Off-diagonal
              Ab_host(band_row, j) = -1.0;
            }
          }
        }
        
        // Initialize right-hand sides on host
        auto B_host = Kokkos::create_mirror_view(B);
        for (int j = 0; j < nrhs; ++j) {
          for (int i = 0; i < n; ++i) {
            B_host(i, j) = 1.0 + i + j*n;
          }
        }
        
        // Save a copy of the original matrix and right-hand sides for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> Ab_orig("Ab_orig", ldab, n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> B_orig("B_orig", n, nrhs);
        
        auto Ab_orig_host = Kokkos::create_mirror_view(Ab_orig);
        auto B_orig_host = Kokkos::create_mirror_view(B_orig);
        
        Kokkos::deep_copy(Ab_orig_host, Ab_host);
        Kokkos::deep_copy(B_orig_host, B_host);
        
        // Copy initialized data to device
        Kokkos::deep_copy(Ab, Ab_host);
        Kokkos::deep_copy(B, B_host);
        Kokkos::deep_copy(Ab_orig, Ab_orig_host);
        Kokkos::deep_copy(B_orig, B_orig_host);
        
        // Perform LU factorization
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialGbtrf<KokkosBatched::Algo::Gbtrf::Unblocked>::invoke(Ab, piv, kl, ku);
        });
        
        // Solve the linear system
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialGbtrs<KokkosBatched::Trans::NoTranspose, 
                                    KokkosBatched::Algo::Gbtrs::Unblocked>::invoke(Ab, piv, B, kl, ku);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(B_host, B);
        
        // Verify the solution by checking A*X ≈ B_orig
        // For a band matrix, this involves manually computing the matrix-vector product
        // using the band structure
        
        bool test_passed = true;
        for (int j = 0; j < nrhs; ++j) {
          for (int i = 0; i < n; ++i) {
            scalar_type sum = 0.0;
            
            // Compute row i of A * column j of X
            for (int k = std::max(0, i-kl); k <= std::min(n-1, i+ku); ++k) {
              int band_row = ku + i - k;
              sum += Ab_orig_host(band_row, k) * B_host(k, j);
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
          std::cout << "Gbtrs test: PASSED" << std::endl;
        } else {
          std::cout << "Gbtrs test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Batched Example
---------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Gbtrf.hpp>
    #include <KokkosBatched_Gbtrs.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Batch and matrix dimensions
        int batch_size = 100; // Number of matrices
        int n = 10;           // Matrix dimension
        int nrhs = 2;         // Number of right-hand sides
        int kl = 2;           // Number of subdiagonals
        int ku = 1;           // Number of superdiagonals
        int ldab = 2*kl+ku+1; // Leading dimension of band matrix
        
        // Create batched views
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          Ab("Ab", batch_size, ldab, n);
        Kokkos::View<int**, memory_space> piv("piv", batch_size, n);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          B("B", batch_size, n, nrhs);
        
        // Initialize on host
        auto Ab_host = Kokkos::create_mirror_view(Ab);
        auto B_host = Kokkos::create_mirror_view(B);
        
        for (int b = 0; b < batch_size; ++b) {
          // Create a diagonally dominant matrix for stability
          for (int j = 0; j < n; ++j) {
            for (int i = std::max(0, j-ku); i <= std::min(n-1, j+kl); ++i) {
              int band_row = ku + i - j;
              
              if (i == j) {
                // Diagonal - make it dominant
                Ab_host(b, band_row, j) = 10.0 + 0.1 * b;
              } else {
                // Off-diagonal
                Ab_host(b, band_row, j) = -1.0 - 0.01 * b;
              }
            }
          }
          
          // Initialize right-hand sides
          for (int j = 0; j < nrhs; ++j) {
            for (int i = 0; i < n; ++i) {
              B_host(b, i, j) = 1.0 + i + j*n + b*0.1;
            }
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(Ab, Ab_host);
        Kokkos::deep_copy(B, B_host);
        
        // Save original for verification
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          Ab_orig("Ab_orig", batch_size, ldab, n);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          B_orig("B_orig", batch_size, n, nrhs);
        
        Kokkos::deep_copy(Ab_orig, Ab);
        Kokkos::deep_copy(B_orig, B);
        
        // Perform batched LU factorization
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto Ab_b = Kokkos::subview(Ab, b, Kokkos::ALL(), Kokkos::ALL());
          auto piv_b = Kokkos::subview(piv, b, Kokkos::ALL());
          
          KokkosBatched::SerialGbtrf<KokkosBatched::Algo::Gbtrf::Unblocked>::invoke(Ab_b, piv_b, kl, ku);
        });
        
        // Solve batched linear systems
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto Ab_b = Kokkos::subview(Ab, b, Kokkos::ALL(), Kokkos::ALL());
          auto piv_b = Kokkos::subview(piv, b, Kokkos::ALL());
          auto B_b = Kokkos::subview(B, b, Kokkos::ALL(), Kokkos::ALL());
          
          KokkosBatched::SerialGbtrs<KokkosBatched::Trans::NoTranspose, 
                                    KokkosBatched::Algo::Gbtrs::Unblocked>::invoke(Ab_b, piv_b, B_b, kl, ku);
        });
        
        // Solutions are now in B
        // Each B(b, :, :) contains the solution for the corresponding system
      }
      Kokkos::finalize();
      return 0;
    }
