KokkosBatched::ApplyPivot
#########################

Defined in header: :code:`KokkosBatched_ApplyPivot_Decl.hpp`

.. code-block:: c++

    template <typename MemberType, typename ArgSide, typename ArgDirect>
    struct TeamVectorApplyPivot {
      // Single pivot index version
      template <typename AViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType& member,
             const int piv,
             const AViewType& A);
      
      // Pivot array version
      template <typename PivViewType, typename AViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType& member,
             const PivViewType& piv,
             const AViewType& A);
    };

The ``ApplyPivot`` operation performs row or column interchanges on a matrix. It applies pivoting based on a single pivot index or an array of pivot indices, effectively implementing permutation matrices without explicitly forming them.

When applied from the left side, it performs row pivoting; when applied from the right side, it performs column pivoting.

Mathematically, for row pivoting with a single pivot index (applied from the left):

.. math::

    \text{row}_i \leftrightarrow \text{row}_{\text{piv}}

For row pivoting with a pivot array (applied from the left):

.. math::

    P \cdot A

where :math:`P` is the permutation matrix corresponding to the pivot indices.

For column pivoting with a pivot array (applied from the right):

.. math::

    A \cdot P^T

where :math:`P^T` is the transpose of the permutation matrix corresponding to the pivot indices.

Parameters
==========

:member: Team execution policy instance
:piv: Single pivot index or view containing the pivot indices
:A: Input/output matrix view to which the pivoting is applied

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgSide`` must be one of:
   - ``KokkosBatched::Side::Left`` to apply row pivoting
   - ``KokkosBatched::Side::Right`` to apply column pivoting
- ``ArgDirect`` must be one of:
   - ``KokkosBatched::Direct::Forward`` to apply pivots from first to last
   - ``KokkosBatched::Direct::Backward`` to apply pivots from last to first
- ``PivViewType`` must be a rank-1 view containing the pivot indices
- ``AViewType`` must be a rank-2 view representing the matrix
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_ApplyPivot_Impl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int m = 5;  // Number of rows
        int n = 4;  // Number of columns
        
        // Create matrix and pivot array
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A("A", m, n);
        Kokkos::View<int*, memory_space> piv("piv", m);
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror_view(A);
        auto piv_host = Kokkos::create_mirror_view(piv);
        
        // Initialize A with recognizable pattern
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            A_host(i, j) = (i + 1) * 10 + (j + 1);
          }
        }
        
        // Define pivot indices: swap rows 0 and 2, 1 and 3, leave row 4 alone
        piv_host(0) = 2;
        piv_host(1) = 3;
        piv_host(2) = 0;
        piv_host(3) = 1;
        piv_host(4) = 4;
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(piv, piv_host);
        
        // Save a copy of the original matrix for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A_orig("A_orig", m, n);
        Kokkos::deep_copy(A_orig, A);
        
        // Create team policy with single team
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        policy_type policy(1, Kokkos::AUTO);
        
        // Apply row pivoting
        Kokkos::parallel_for("ApplyPivot", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            KokkosBatched::TeamVectorApplyPivot<typename policy_type::member_type,
                                              KokkosBatched::Side::Left,
                                              KokkosBatched::Direct::Forward>
              ::invoke(member, piv, A);
          }
        );
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify results
        auto A_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_orig);
        
        bool test_passed = true;
        for (int i = 0; i < m; ++i) {
          int source_row = piv_host(i);
          for (int j = 0; j < n; ++j) {
            // Check if row i now contains what was in row piv_host(i)
            if (std::abs(A_host(i, j) - A_orig_host(source_row, j)) > 1e-12) {
              test_passed = false;
              std::cout << "Mismatch at (" << i << ", " << j << "): " 
                        << A_host(i, j) << " vs expected " << A_orig_host(source_row, j) << std::endl;
            }
          }
        }
        
        if (test_passed) {
          std::cout << "ApplyPivot row pivoting test: PASSED" << std::endl;
        } else {
          std::cout << "ApplyPivot row pivoting test: FAILED" << std::endl;
        }
        
        // Now test applying pivoting in reverse to get back the original matrix
        Kokkos::parallel_for("ApplyPivotReverse", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            KokkosBatched::TeamVectorApplyPivot<typename policy_type::member_type,
                                              KokkosBatched::Side::Left,
                                              KokkosBatched::Direct::Backward>
              ::invoke(member, piv, A);
          }
        );
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify we're back to the original
        test_passed = true;
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            if (std::abs(A_host(i, j) - A_orig_host(i, j)) > 1e-12) {
              test_passed = false;
              std::cout << "Reverse pivoting failed at (" << i << ", " << j << "): " 
                        << A_host(i, j) << " vs original " << A_orig_host(i, j) << std::endl;
            }
          }
        }
        
        if (test_passed) {
          std::cout << "ApplyPivot reverse test: PASSED" << std::endl;
        } else {
          std::cout << "ApplyPivot reverse test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Column Pivoting Example
---------------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_ApplyPivot_Impl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int m = 4;  // Number of rows
        int n = 5;  // Number of columns
        
        // Create matrix and pivot array
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A("A", m, n);
        Kokkos::View<int*, memory_space> piv("piv", n);
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror_view(A);
        auto piv_host = Kokkos::create_mirror_view(piv);
        
        // Initialize A with recognizable pattern
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            A_host(i, j) = (i + 1) * 10 + (j + 1);
          }
        }
        
        // Define pivot indices: swap columns 0 and 2, 1 and 3, leave column 4 alone
        piv_host(0) = 2;
        piv_host(1) = 3;
        piv_host(2) = 0;
        piv_host(3) = 1;
        piv_host(4) = 4;
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(piv, piv_host);
        
        // Save a copy of the original matrix for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A_orig("A_orig", m, n);
        Kokkos::deep_copy(A_orig, A);
        
        // Create team policy with single team
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        policy_type policy(1, Kokkos::AUTO);
        
        // Apply column pivoting
        Kokkos::parallel_for("ApplyPivotColumn", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            KokkosBatched::TeamVectorApplyPivot<typename policy_type::member_type,
                                              KokkosBatched::Side::Right,
                                              KokkosBatched::Direct::Forward>
              ::invoke(member, piv, A);
          }
        );
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify results
        auto A_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_orig);
        
        bool test_passed = true;
        for (int j = 0; j < n; ++j) {
          int source_col = piv_host(j);
          for (int i = 0; i < m; ++i) {
            // Check if column j now contains what was in column piv_host(j)
            if (std::abs(A_host(i, j) - A_orig_host(i, source_col)) > 1e-12) {
              test_passed = false;
              std::cout << "Mismatch at (" << i << ", " << j << "): " 
                        << A_host(i, j) << " vs expected " << A_orig_host(i, source_col) << std::endl;
            }
          }
        }
        
        if (test_passed) {
          std::cout << "ApplyPivot column pivoting test: PASSED" << std::endl;
        } else {
          std::cout << "ApplyPivot column pivoting test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Batched Example
------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_ApplyPivot_Impl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Batch and matrix dimensions
        int batch_size = 5;  // Number of matrices
        int m = 4;           // Number of rows
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
          // Initialize each matrix with a unique pattern
          for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
              A_host(b, i, j) = (b + 1) * 100 + (i + 1) * 10 + (j + 1);
            }
          }
          
          // Set up different pivots for each batch
          // Simple pattern: reverse the rows
          for (int i = 0; i < m; ++i) {
            piv_host(b, i) = m - 1 - i;
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(piv, piv_host);
        
        // Save original for verification
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A_orig("A_orig", batch_size, m, n);
        Kokkos::deep_copy(A_orig, A);
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        policy_type policy(batch_size, Kokkos::AUTO);
        
        // Apply row pivoting to each matrix
        Kokkos::parallel_for("BatchedApplyPivot", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            auto A_b = Kokkos::subview(A, b, Kokkos::ALL(), Kokkos::ALL());
            auto piv_b = Kokkos::subview(piv, b, Kokkos::ALL());
            
            KokkosBatched::TeamVectorApplyPivot<typename policy_type::member_type,
                                              KokkosBatched::Side::Left,
                                              KokkosBatched::Direct::Forward>
              ::invoke(member, piv_b, A_b);
          }
        );
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify for each batch
        auto A_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_orig);
        
        bool test_passed = true;
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < m; ++i) {
            int source_row = piv_host(b, i);
            for (int j = 0; j < n; ++j) {
              // Check if row i now contains what was in row piv_host(b, i)
              if (std::abs(A_host(b, i, j) - A_orig_host(b, source_row, j)) > 1e-12) {
                test_passed = false;
                std::cout << "Batch " << b << " mismatch at (" << i << ", " << j << "): " 
                          << A_host(b, i, j) << " vs expected " 
                          << A_orig_host(b, source_row, j) << std::endl;
                break;
              }
            }
            if (!test_passed) break;
          }
          if (!test_passed) break;
        }
        
        if (test_passed) {
          std::cout << "Batched ApplyPivot test: PASSED" << std::endl;
        } else {
          std::cout << "Batched ApplyPivot test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
