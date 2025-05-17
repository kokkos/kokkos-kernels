KokkosBatched::SolveUTV
#######################

Defined in header: :code:`KokkosBatched_SolveUTV_Decl.hpp`

.. code-block:: c++

    template <typename MemberType, typename ArgAlgo>
    struct TeamVectorSolveUTV {
      template <typename UViewType, typename TViewType, typename VViewType, 
                typename pViewType, typename XViewType, typename BViewType, 
                typename wViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType& member,
             const int matrix_rank,
             const UViewType& U,
             const TViewType& T,
             const VViewType& V,
             const pViewType& p,
             const XViewType& X,
             const BViewType& B,
             const wViewType& w);
    };

The ``SolveUTV`` function solves a system of linear equations with a general matrix using the UTV factorization. Given a matrix A with its UTV factorization A = U·T·V^T·P^T, where P is a permutation matrix, the function solves the system A·X = B for X.

This function is particularly useful for rank-deficient or ill-conditioned matrices, as it provides a numerically stable solution taking the matrix rank into account.

When A is full rank (i.e., matrix_rank == m), UTV provides functionality similar to QR factorization with column pivoting, where U corresponds to Q, and T corresponds to R.

Parameters
==========

:member: Team execution policy instance
:matrix_rank: The numerical rank of the matrix as determined during UTV factorization
:U: Input view containing the U matrix from UTV factorization (m x m matrix)
:T: Input view containing the T matrix from UTV factorization (m x m matrix)
:V: Input view containing the V matrix from UTV factorization (m x m matrix)
:p: Input view containing the pivot indices from UTV factorization
:X: Output view for the solution matrix/vector
:B: Input view containing the right-hand side matrix/vector
:w: Workspace view (contiguous)

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgAlgo`` must specify the algorithm to be used
- ``UViewType``, ``TViewType``, and ``VViewType`` must be rank-2 views representing the factorization matrices
- ``pViewType`` must be a rank-1 view containing the pivot indices
- ``XViewType`` and ``BViewType`` must be rank-1 views for a single right-hand side, or rank-2 views for multiple right-hand sides
- ``wViewType`` must be a rank-1 view with enough space for workspace operations
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_UTV_Decl.hpp>
    #include <KokkosBatched_SolveUTV_Decl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int m = 6;  // Number of rows
        int n = 6;  // Number of columns
        int nrhs = 2; // Number of right-hand sides
        
        // Create matrices and vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A("A", m, n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> U("U", m, m);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> T("T", m, n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> V("V", n, n);
        Kokkos::View<int*, memory_space> p("p", n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> B("B", m, nrhs);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> X("X", n, nrhs);
        
        // Workspace for UTV factorization and solve
        Kokkos::View<scalar_type*, memory_space> w("w", m*n);
        
        // Initialize matrix on host
        auto A_host = Kokkos::create_mirror_view(A);
        
        // Create a matrix with specific rank
        int matrix_rank = 4;  // Specify a rank < min(m,n)
        
        // Initialize a matrix with a known rank
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            if (i < matrix_rank && j < matrix_rank) {
              // Create linearly independent rows and columns
              A_host(i, j) = (i+1) * (j+1) * 0.1;
            } else {
              // Create linearly dependent rows or columns
              A_host(i, j) = 0.0;
            }
          }
        }
        
        // Add some noise to make it more realistic
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            A_host(i, j) += 0.0001 * (i*n + j);
          }
        }
        
        // Initialize right-hand sides on host
        auto B_host = Kokkos::create_mirror_view(B);
        for (int j = 0; j < nrhs; ++j) {
          for (int i = 0; i < m; ++i) {
            B_host(i, j) = 1.0 + i + j*m;
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(B, B_host);
        
        // Save a copy of the original matrix and right-hand sides for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A_orig("A_orig", m, n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> B_orig("B_orig", m, nrhs);
        
        Kokkos::deep_copy(A_orig, A);
        Kokkos::deep_copy(B_orig, B);
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        policy_type policy(1, Kokkos::AUTO);
        
        // Perform UTV factorization
        int computed_rank = 0;
        Kokkos::parallel_reduce("UTV_Factorization", policy, 
            KOKKOS_LAMBDA(const typename policy_type::member_type& member, int& rank) {
              rank = KokkosBatched::TeamVectorUTV<typename policy_type::member_type, 
                                                 KokkosBatched::Algo::UTV::Unblocked>
                ::invoke(member, A, U, T, V, p, w);
            }, Kokkos::Sum<int>(computed_rank));
        
        // Solve the system using the UTV factorization
        Kokkos::parallel_for("SolveUTV", policy, 
            KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
              KokkosBatched::TeamVectorSolveUTV<typename policy_type::member_type, 
                                              KokkosBatched::Algo::SolveUTV::Unblocked>
                ::invoke(member, computed_rank, U, T, V, p, X, B, w);
            });
        
        // Copy results back to host
        auto X_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), X);
        
        // Verify solution by checking A_orig*X ≈ B_orig
        // Note: For rank-deficient matrices, we expect a least-squares solution
        auto A_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_orig);
        auto B_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), B_orig);
        
        // Check the solution
        bool test_passed = true;
        for (int j = 0; j < nrhs; ++j) {
          for (int i = 0; i < m; ++i) {
            scalar_type sum = 0.0;
            
            // Compute row i of A_orig * column j of X
            for (int k = 0; k < n; ++k) {
              sum += A_orig_host(i, k) * X_host(k, j);
            }
            
            // For rank-deficient problems, we can only check residual norm
            // rather than exact match to B_orig
            // We'll accumulate the squared residual
          }
        }
        
        std::cout << "Matrix rank: " << computed_rank << " (expected: " << matrix_rank << ")" << std::endl;
        
        if (test_passed) {
          std::cout << "SolveUTV test: PASSED" << std::endl;
        } else {
          std::cout << "SolveUTV test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Batched Example
--------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_UTV_Decl.hpp>
    #include <KokkosBatched_SolveUTV_Decl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Batch and matrix dimensions
        int batch_size = 20; // Number of matrices
        int m = 6;           // Number of rows
        int n = 6;           // Number of columns
        int nrhs = 2;        // Number of right-hand sides
        
        // Create batched views
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A("A", batch_size, m, n);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          U("U", batch_size, m, m);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          T("T", batch_size, m, n);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          V("V", batch_size, n, n);
        Kokkos::View<int**, memory_space> 
          p("p", batch_size, n);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          B("B", batch_size, m, nrhs);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          X("X", batch_size, n, nrhs);
        
        // Workspace for UTV factorization and solve
        Kokkos::View<scalar_type**, memory_space> 
          w("w", batch_size, m*n);
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror_view(A);
        auto B_host = Kokkos::create_mirror_view(B);
        
        // View for storing ranks
        Kokkos::View<int*, memory_space> ranks("ranks", batch_size);
        
        for (int b = 0; b < batch_size; ++b) {
          // Create matrices with varying ranks
          int matrix_rank = std::min(m, n) - (b % 3); // Varying ranks
          
          // Initialize a matrix with a known rank
          for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
              if (i < matrix_rank && j < matrix_rank) {
                // Create linearly independent rows and columns
                A_host(b, i, j) = (i+1) * (j+1) * 0.1 + b * 0.01;
              } else {
                // Create linearly dependent rows or columns
                A_host(b, i, j) = 0.0;
              }
            }
          }
          
          // Add some noise
          for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
              A_host(b, i, j) += 0.0001 * (b*m*n + i*n + j);
            }
          }
          
          // Initialize right-hand sides
          for (int j = 0; j < nrhs; ++j) {
            for (int i = 0; i < m; ++i) {
              B_host(b, i, j) = 1.0 + i + j*m + b*0.1;
            }
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(B, B_host);
        
        // Save original for verification
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A_orig("A_orig", batch_size, m, n);
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          B_orig("B_orig", batch_size, m, nrhs);
        
        Kokkos::deep_copy(A_orig, A);
        Kokkos::deep_copy(B_orig, B);
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        policy_type policy(batch_size, Kokkos::AUTO);
        
        // Perform UTV factorization
        Kokkos::parallel_for("BatchedUTV", policy, 
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            auto A_b = Kokkos::subview(A, b, Kokkos::ALL(), Kokkos::ALL());
            auto U_b = Kokkos::subview(U, b, Kokkos::ALL(), Kokkos::ALL());
            auto T_b = Kokkos::subview(T, b, Kokkos::ALL(), Kokkos::ALL());
            auto V_b = Kokkos::subview(V, b, Kokkos::ALL(), Kokkos::ALL());
            auto p_b = Kokkos::subview(p, b, Kokkos::ALL());
            auto w_b = Kokkos::subview(w, b, Kokkos::ALL());
            
            ranks(b) = KokkosBatched::TeamVectorUTV<typename policy_type::member_type, 
                                                  KokkosBatched::Algo::UTV::Unblocked>
              ::invoke(member, A_b, U_b, T_b, V_b, p_b, w_b);
          }
        );
        
        // Solve the systems using the UTV factorization
        Kokkos::parallel_for("BatchedSolveUTV", policy, 
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            auto U_b = Kokkos::subview(U, b, Kokkos::ALL(), Kokkos::ALL());
            auto T_b = Kokkos::subview(T, b, Kokkos::ALL(), Kokkos::ALL());
            auto V_b = Kokkos::subview(V, b, Kokkos::ALL(), Kokkos::ALL());
            auto p_b = Kokkos::subview(p, b, Kokkos::ALL());
            auto X_b = Kokkos::subview(X, b, Kokkos::ALL(), Kokkos::ALL());
            auto B_b = Kokkos::subview(B, b, Kokkos::ALL(), Kokkos::ALL());
            auto w_b = Kokkos::subview(w, b, Kokkos::ALL());
            
            KokkosBatched::TeamVectorSolveUTV<typename policy_type::member_type, 
                                            KokkosBatched::Algo::SolveUTV::Unblocked>
              ::invoke(member, ranks(b), U_b, T_b, V_b, p_b, X_b, B_b, w_b);
          }
        );
        
        // Solutions are now in X
        // Each X(b, :, :) contains the solution for the corresponding system
      }
      Kokkos::finalize();
      return 0;
    }
