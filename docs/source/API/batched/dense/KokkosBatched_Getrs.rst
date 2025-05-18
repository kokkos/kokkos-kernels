KokkosBatched::Getrs
####################

Defined in header: :code:`KokkosBatched_Getrs.hpp`

.. code-block:: c++

    template <typename ArgTrans, typename ArgAlgo>
    struct SerialGetrs {
      template <typename AViewType, typename PivViewType, typename BViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const AViewType& A,
             const PivViewType& piv,
             const BViewType& b);
    };

The ``Getrs`` function solves a system of linear equations with a general N-by-N matrix A using the LU factorization computed by ``Getrf``. The function can solve the following systems:

1. :math:`A \cdot X = B` (Trans::NoTranspose)
2. :math:`A^T \cdot X = B` (Trans::Transpose)

where A is a general square matrix that has been factorized using LU decomposition, and B is a matrix with one or more right-hand sides.

Parameters
==========

:A: Input matrix view containing the LU factorization from Getrf
:piv: Input view containing the pivot indices from Getrf
:b: Input/output view containing right-hand sides on input and solutions on output

Type Requirements
-----------------

- ``ArgTrans`` must be one of the following:
   - ``KokkosBatched::Trans::NoTranspose`` to solve :math:`A \cdot X = B`
   - ``KokkosBatched::Trans::Transpose`` to solve :math:`A^T \cdot X = B`

- ``ArgAlgo`` must specify the algorithm to be used
- ``AViewType`` must be a rank-2 view containing the matrix with LU factorization
- ``PivViewType`` must be a rank-1 view containing the pivot indices
- ``BViewType`` must be a rank-1 view for a single right-hand side, or a rank-2 view for multiple right-hand sides
- All views must be accessible in the execution space

Examples
========

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Getrf.hpp>
    #include <KokkosBatched_Getrs.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int n = 10;      // Matrix dimension
        int nrhs = 2;    // Number of right-hand sides
        
        // Create matrix, pivot vector, and right-hand sides
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A("A", n, n);
        Kokkos::View<int*, memory_space> piv("piv", n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> B("B", n, nrhs);
        
        // Initialize matrix on host
        auto A_host = Kokkos::create_mirror_view(A);
        
        // Create a diagonally dominant matrix for stability
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            if (i == j) {
              // Diagonal - make it dominant
              A_host(i, j) = n + 1.0;
            } else {
              // Off-diagonal
              A_host(i, j) = 1.0;
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
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A_orig("A_orig", n, n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> B_orig("B_orig", n, nrhs);
        
        auto A_orig_host = Kokkos::create_mirror_view(A_orig);
        auto B_orig_host = Kokkos::create_mirror_view(B_orig);
        
        Kokkos::deep_copy(A_orig_host, A_host);
        Kokkos::deep_copy(B_orig_host, B_host);
        
        // Copy initialized data to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(B, B_host);
        Kokkos::deep_copy(A_orig, A_orig_host);
        Kokkos::deep_copy(B_orig, B_orig_host);
        
        // Perform LU factorization
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialGetrf<KokkosBatched::Algo::Getrf::Unblocked>::invoke(A, piv);
        });
        
        // Solve the linear system
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialGetrs<KokkosBatched::Trans::NoTranspose, 
                                    KokkosBatched::Algo::Getrs::Unblocked>::invoke(A, piv, B);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(B_host, B);
        
        // Verify the solution by checking A*X ≈ B_orig
        bool test_passed = true;
        for (int j = 0; j < nrhs; ++j) {
          for (int i = 0; i < n; ++i) {
            scalar_type sum = 0.0;
            
            // Compute row i of A * column j of X
            for (int k = 0; k < n; ++k) {
              sum += A_orig_host(i, k) * B_host(k, j);
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
          std::cout << "Getrs test: PASSED" << std::endl;
        } else {
          std::cout << "Getrs test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Batched Example
---------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Getrf.hpp>
    #include <KokkosBatched_Getrs.hpp>
    
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
        
        // Create batched views
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A("A", batch_size, n, n);
        Kokkos::View<int**, memory_space> piv("piv", batch_size, n);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          B("B", batch_size, n, nrhs);
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror_view(A);
        auto B_host = Kokkos::create_mirror_view(B);
        
        for (int b = 0; b < batch_size; ++b) {
          // Create a diagonally dominant matrix for stability
          for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
              if (i == j) {
                // Diagonal - make it dominant
                A_host(b, i, j) = n + 1.0 + 0.1 * b;
              } else {
                // Off-diagonal
                A_host(b, i, j) = 1.0 + 0.01 * b;
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
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(B, B_host);
        
        // Save original for verification
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A_orig("A_orig", batch_size, n, n);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          B_orig("B_orig", batch_size, n, nrhs);
        
        Kokkos::deep_copy(A_orig, A);
        Kokkos::deep_copy(B_orig, B);
        
        // Perform batched LU factorization
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto A_b = Kokkos::subview(A, b, Kokkos::ALL(), Kokkos::ALL());
          auto piv_b = Kokkos::subview(piv, b, Kokkos::ALL());
          
          KokkosBatched::SerialGetrf<KokkosBatched::Algo::Getrf::Unblocked>::invoke(A_b, piv_b);
        });
        
        // Solve batched linear systems
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto A_b = Kokkos::subview(A, b, Kokkos::ALL(), Kokkos::ALL());
          auto piv_b = Kokkos::subview(piv, b, Kokkos::ALL());
          auto B_b = Kokkos::subview(B, b, Kokkos::ALL(), Kokkos::ALL());
          
          KokkosBatched::SerialGetrs<KokkosBatched::Trans::NoTranspose, 
                                    KokkosBatched::Algo::Getrs::Unblocked>::invoke(A_b, piv_b, B_b);
        });
        
        // Solutions are now in B
        // Each B(b, :, :) contains the solution for the corresponding system
      }
      Kokkos::finalize();
      return 0;
    }
