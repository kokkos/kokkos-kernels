KokkosBatched::Laswp
####################

Defined in header: :code:`KokkosBatched_Laswp.hpp`

.. code-block:: c++

    template <typename ArgDirect>
    struct SerialLaswp {
      template <typename PivViewType, typename AViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const PivViewType& piv,
             const AViewType& A);
    };

The ``Laswp`` function performs a series of row interchanges on a matrix A. One row interchange is initiated for each row of A according to the pivot indices provided. This operation is equivalent to the LAPACK routine ``LASWP``.

Mathematically, for each row index i, the operation swaps row i with row piv(i) in the matrix A.

Parameters
==========

:piv: Input view containing the pivot indices
:A: Input/output matrix view to which the row interchanges are applied

Type Requirements
-----------------

- ``ArgDirect`` must specify the direction of the permutation:
   - ``KokkosBatched::Direct::Forward`` to apply pivots from first to last
   - ``KokkosBatched::Direct::Backward`` to apply pivots from last to first

- ``PivViewType`` must be a rank-1 view containing the pivot indices
- ``AViewType`` must be a rank-2 view representing the matrix to be permuted
- All views must be accessible in the execution space

Examples
========

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Laswp.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int m = 6;  // Number of rows
        int n = 4;  // Number of columns
        
        // Create matrix and pivot indices
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A("A", m, n);
        Kokkos::View<int*, memory_space> piv("piv", m);
        
        // Initialize matrix on host
        auto A_host = Kokkos::create_mirror_view(A);
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            // Initialize with easily identifiable patterns
            A_host(i, j) = 10 * (i + 1) + (j + 1);
          }
        }
        
        // Initialize pivot indices on host
        auto piv_host = Kokkos::create_mirror_view(piv);
        // Example pivot pattern: swap row 0 with 3, row 1 with 4, row 2 with 5
        piv_host(0) = 3;
        piv_host(1) = 4;
        piv_host(2) = 5;
        piv_host(3) = 0;
        piv_host(4) = 1;
        piv_host(5) = 2;
        
        // Copy initialized data to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(piv, piv_host);
        
        // Save a copy of the original matrix for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A_orig("A_orig", m, n);
        Kokkos::deep_copy(A_orig, A);
        auto A_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_orig);
        
        // Apply row interchanges
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialLaswp<KokkosBatched::Direct::Forward>::invoke(piv, A);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify that rows have been swapped correctly
        bool test_passed = true;
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            // Check if row i now has data from row piv_host(i)
            if (std::abs(A_host(i, j) - A_orig_host(piv_host(i), j)) > 1e-10) {
              test_passed = false;
              std::cout << "Mismatch at (" << i << ", " << j << "): " 
                        << A_host(i, j) << " vs expected " << A_orig_host(piv_host(i), j) << std::endl;
            }
          }
        }
        
        // Apply inverse permutation to restore original matrix
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialLaswp<KokkosBatched::Direct::Backward>::invoke(piv, A);
        });
        
        // Verify original matrix is restored
        Kokkos::deep_copy(A_host, A);
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            if (std::abs(A_host(i, j) - A_orig_host(i, j)) > 1e-10) {
              test_passed = false;
              std::cout << "Inverse permutation failed at (" << i << ", " << j << "): " 
                        << A_host(i, j) << " vs original " << A_orig_host(i, j) << std::endl;
            }
          }
        }
        
        if (test_passed) {
          std::cout << "Laswp test: PASSED" << std::endl;
        } else {
          std::cout << "Laswp test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Batched Example
---------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Laswp.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Batch and matrix dimensions
        int batch_size = 50; // Number of matrices
        int m = 6;           // Number of rows
        int n = 4;           // Number of columns
        
        // Create batched views
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A("A", batch_size, m, n);
        Kokkos::View<int**, memory_space> 
          piv("piv", batch_size, m);
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror_view(A);
        auto piv_host = Kokkos::create_mirror_view(piv);
        
        for (int b = 0; b < batch_size; ++b) {
          // Initialize matrix with unique values per batch
          for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
              A_host(b, i, j) = 100 * (b + 1) + 10 * (i + 1) + (j + 1);
            }
          }
          
          // Create pivot indices - custom pattern per batch
          // Here we're using a simple pattern: reverse the rows
          for (int i = 0; i < m; ++i) {
            piv_host(b, i) = m - i - 1;
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(piv, piv_host);
        
        // Save original for verification
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A_orig("A_orig", batch_size, m, n);
        Kokkos::deep_copy(A_orig, A);
        
        // Apply row interchanges for each batch
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto A_b = Kokkos::subview(A, b, Kokkos::ALL(), Kokkos::ALL());
          auto piv_b = Kokkos::subview(piv, b, Kokkos::ALL());
          
          KokkosBatched::SerialLaswp<KokkosBatched::Direct::Forward>::invoke(piv_b, A_b);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify for each batch
        auto A_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_orig);
        
        bool test_passed = true;
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
              // Check if row i now has data from row piv_host(b, i)
              if (std::abs(A_host(b, i, j) - A_orig_host(b, piv_host(b, i), j)) > 1e-10) {
                test_passed = false;
                std::cout << "Batch " << b << " mismatch at (" << i << ", " << j << "): " 
                          << A_host(b, i, j) << " vs expected " 
                          << A_orig_host(b, piv_host(b, i), j) << std::endl;
                break;
              }
            }
            if (!test_passed) break;
          }
          if (!test_passed) break;
        }
        
        if (test_passed) {
          std::cout << "Batched Laswp test: PASSED" << std::endl;
        } else {
          std::cout << "Batched Laswp test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
