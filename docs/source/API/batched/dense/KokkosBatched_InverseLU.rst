KokkosBatched::InverseLU
##################

Defined in header `KokkosBatched_InverseLU_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/src/batched/KokkosBatched_InverseLU_Decl.hpp>`_

.. code-block:: c++

    template <typename ArgAlgo>
    struct SerialInverseLU {
      template <typename AViewType, typename wViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const AViewType& A,
             const wViewType& w);
    };
    
    template <typename MemberType, typename ArgAlgo>
    struct TeamInverseLU {
      template <typename AViewType, typename wViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType& member,
             const AViewType& A,
             const wViewType& w);
    };

The ``InverseLU`` function computes the inverse of a matrix using its LU factorization. It assumes that the input matrix ``A`` already contains the LU factorization (as computed by ``Getrf`` or similar function). The function returns the inverse of the original matrix in the ``A`` view.

The algorithm performs the following steps:
1. Copies the LU factorization from A to workspace w
2. Sets A to the identity matrix
3. Solves the system (LU) * A = I

Mathematically, given a matrix A with its LU factorization A = P*L*U (where P is a permutation matrix, L is lower triangular with unit diagonal, and U is upper triangular), this function computes A⁻¹.

Parameters
==========

:member: Team execution policy instance (only for team version)
:A: Input/output matrix view containing LU factorization on input and matrix inverse on output
:w: Workspace view with enough space to hold a copy of A

Type Requirements
----------------

- ``ArgAlgo`` specifies the algorithm to be used for the SolveLU operation
- ``MemberType`` must be a Kokkos TeamPolicy member type (only for team version)
- ``AViewType`` must be a rank-2 view containing the LU factorization of the matrix
- ``wViewType`` must be a rank-1 view with enough space to reinterpret as a matrix of the same dimensions as A
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Getrf.hpp>
    #include <KokkosBatched_InverseLU_Decl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int n = 5;  // Matrix dimension
        
        // Create matrix and workspace
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A("A", n, n);
        Kokkos::View<scalar_type*, memory_space> w("w", n * n);
        
        // Initialize matrix on host
        auto A_host = Kokkos::create_mirror_view(A);
        
        // Create a well-conditioned matrix for stability
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            if (i == j) {
              // Diagonal
              A_host(i, j) = 10.0;
            } else {
              // Off-diagonal
              A_host(i, j) = 1.0;
            }
          }
        }
        
        // Save a copy of the original matrix for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A_orig("A_orig", n, n);
        auto A_orig_host = Kokkos::create_mirror_view(A_orig);
        Kokkos::deep_copy(A_orig_host, A_host);
        
        // Copy initialized data to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(A_orig, A_orig_host);
        
        // Create pivot array for LU factorization
        Kokkos::View<int*, memory_space> piv("piv", n);
        
        // Perform LU factorization in-place
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialGetrf<KokkosBatched::Algo::Getrf::Unblocked>::invoke(A, piv);
        });
        
        // Compute matrix inverse using InverseLU
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialInverseLU<KokkosBatched::Algo::SolveLU::Unblocked>::invoke(A, w);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify the inverse by checking A_orig * A_inv ≈ I
        bool test_passed = true;
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            scalar_type sum = 0.0;
            
            // Compute element (i,j) of A_orig * A_inv
            for (int k = 0; k < n; ++k) {
              sum += A_orig_host(i, k) * A_host(k, j);
            }
            
            // Check against identity matrix
            scalar_type expected = (i == j) ? 1.0 : 0.0;
            if (std::abs(sum - expected) > 1e-10) {
              test_passed = false;
              std::cout << "Mismatch at (" << i << ", " << j << "): " 
                        << sum << " vs " << expected << std::endl;
            }
          }
        }
        
        if (test_passed) {
          std::cout << "InverseLU test: PASSED" << std::endl;
        } else {
          std::cout << "InverseLU test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Team Version Example
------------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Getrf.hpp>
    #include <KokkosBatched_InverseLU_Decl.hpp>
    
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
          w("w", batch_size, n * n);
        Kokkos::View<int**, memory_space> 
          piv("piv", batch_size, n);
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror_view(A);
        
        for (int b = 0; b < batch_size; ++b) {
          // Create a well-conditioned matrix for stability
          for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
              if (i == j) {
                // Diagonal
                A_host(b, i, j) = 10.0 + 0.1 * b;
              } else {
                // Off-diagonal
                A_host(b, i, j) = 1.0 + 0.01 * b;
              }
            }
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        
        // Save original for verification
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A_orig("A_orig", batch_size, n, n);
        Kokkos::deep_copy(A_orig, A);
        
        // Perform batched LU factorization
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto A_b = Kokkos::subview(A, b, Kokkos::ALL(), Kokkos::ALL());
          auto piv_b = Kokkos::subview(piv, b, Kokkos::ALL());
          
          KokkosBatched::SerialGetrf<KokkosBatched::Algo::Getrf::Unblocked>::invoke(A_b, piv_b);
        });
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        policy_type policy(batch_size, Kokkos::AUTO);
        
        // Compute batched matrix inverses using TeamInverseLU
        Kokkos::parallel_for("InverseLU", policy, 
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            auto A_b = Kokkos::subview(A, b, Kokkos::ALL(), Kokkos::ALL());
            auto w_b = Kokkos::subview(w, b, Kokkos::ALL());
            
            KokkosBatched::TeamInverseLU<typename policy_type::member_type, 
                                        KokkosBatched::Algo::SolveLU::Unblocked>
              ::invoke(member, A_b, w_b);
          }
        );
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify the inverse by checking A_orig * A_inv ≈ I for each batch
        auto A_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_orig);
        
        bool test_passed = true;
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
              scalar_type sum = 0.0;
              
              // Compute element (i,j) of A_orig * A_inv
              for (int k = 0; k < n; ++k) {
                sum += A_orig_host(b, i, k) * A_host(b, k, j);
              }
              
              // Check against identity matrix
              scalar_type expected = (i == j) ? 1.0 : 0.0;
              if (std::abs(sum - expected) > 1e-10) {
                test_passed = false;
                std::cout << "Batch " << b << " mismatch at (" << i << ", " << j << "): " 
                          << sum << " vs " << expected << std::endl;
                break;
              }
            }
            if (!test_passed) break;
          }
          if (!test_passed) break;
        }
        
        if (test_passed) {
          std::cout << "Batched TeamInverseLU test: PASSED" << std::endl;
        } else {
          std::cout << "Batched TeamInverseLU test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
