KokkosBatched::ApplyHouseholder
###############################

Defined in header `KokkosBatched_ApplyHouseholder_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/src/batched/KokkosBatched_ApplyHouseholder_Decl.hpp>`_

.. code-block:: c++

    // Serial version
    template <typename ArgSide>
    struct SerialApplyHouseholder {
      template <typename uViewType, typename tauViewType, typename AViewType, typename wViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const uViewType& u2,
             const tauViewType& tau,
             const AViewType& A,
             const wViewType& w);
    };
    
    // Team Vector version
    template <typename MemberType, typename ArgSide>
    struct TeamVectorApplyHouseholder {
      template <typename uViewType, typename tauViewType, typename AViewType, typename wViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType& member,
             const uViewType& u2,
             const tauViewType& tau,
             const AViewType& A,
             const wViewType& w);
    };

The ``ApplyHouseholder`` operation applies a Householder transformation to a matrix. This is a fundamental building block for many linear algebra operations, including QR factorization.

A Householder transformation is defined by a Householder vector ``u`` and a scalar ``tau``. It applies the transformation matrix :math:`H = I - \tau u u^T` (for real matrices) or :math:`H = I - \tau u u^H` (for complex matrices) to a target matrix.

Mathematically, when applied from the left side:

.. math::

    A := H \cdot A = (I - \tau u u^T) \cdot A

When applied from the right side:

.. math::

    A := A \cdot H = A \cdot (I - \tau u u^T)

where:

- :math:`I` is the identity matrix
- :math:`u` is the Householder vector
- :math:`\tau` is the Householder scalar
- :math:`u^T` is the transpose of :math:`u`
- :math:`u^H` is the conjugate transpose of :math:`u` (for complex matrices)

Parameters
==========

:member: Team execution policy instance (only for team version)
:u2: View containing the Householder vector
:tau: View containing the Householder scalar
:A: Input/output matrix view to which the transformation is applied
:w: Workspace view for temporary calculations

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgSide`` must be one of:
   - ``KokkosBatched::Side::Left`` to apply the transformation from the left
   - ``KokkosBatched::Side::Right`` to apply the transformation from the right
- ``uViewType`` must be a rank-1 view containing the Householder vector
- ``tauViewType`` must be a scalar or a rank-0 view containing the Householder scalar
- ``AViewType`` must be a rank-2 view representing the matrix to transform
- ``wViewType`` must be a rank-1 view with sufficient workspace for the computation
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_ApplyHouseholder_Decl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int m = 5;  // Number of rows
        int n = 3;  // Number of columns
        
        // Create matrices and vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A("A", m, n);
        Kokkos::View<scalar_type*, memory_space> u("u", m);   // Householder vector
        Kokkos::View<scalar_type, memory_space> tau("tau");   // Householder scalar
        Kokkos::View<scalar_type*, memory_space> w("w", n);   // Workspace
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror_view(A);
        auto u_host = Kokkos::create_mirror_view(u);
        auto tau_host = Kokkos::create_mirror_view(tau);
        
        // Initialize A with recognizable pattern
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            A_host(i, j) = (i + 1) * 10 + (j + 1);
          }
        }
        
        // Initialize Householder vector (first element is 1.0, rest are zeros by convention)
        u_host(0) = 1.0;
        for (int i = 1; i < m; ++i) {
          u_host(i) = 0.5 * i;
        }
        
        // Set tau
        tau_host() = 0.5;
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(u, u_host);
        Kokkos::deep_copy(tau, tau_host);
        
        // Save a copy of the original matrix for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A_orig("A_orig", m, n);
        Kokkos::deep_copy(A_orig, A);
        
        // Apply Householder transformation from the left
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialApplyHouseholder<KokkosBatched::Side::Left>
            ::invoke(u, tau, A, w);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify results: Manually compute H*A and compare
        auto A_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_orig);
        
        // Calculate expected result (H*A) on host
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> 
          A_expected("A_expected", m, n);
        
        // First compute v = u^T * A_orig (a row vector)
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, Kokkos::HostSpace> 
          v("v", n);
        
        for (int j = 0; j < n; ++j) {
          v(j) = 0.0;
          for (int i = 0; i < m; ++i) {
            v(j) += u_host(i) * A_orig_host(i, j);
          }
        }
        
        // Now compute A_expected = A_orig - tau * u * v
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            A_expected(i, j) = A_orig_host(i, j) - tau_host() * u_host(i) * v(j);
          }
        }
        
        // Compare results
        bool test_passed = true;
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            if (std::abs(A_host(i, j) - A_expected(i, j)) > 1e-12) {
              test_passed = false;
              std::cout << "Mismatch at (" << i << ", " << j << "): " 
                        << A_host(i, j) << " vs expected " << A_expected(i, j) << std::endl;
            }
          }
        }
        
        if (test_passed) {
          std::cout << "ApplyHouseholder left side test: PASSED" << std::endl;
        } else {
          std::cout << "ApplyHouseholder left side test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Team Vector Version Example
--------------------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_ApplyHouseholder_Decl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Batch and matrix dimensions
        int batch_size = 5;  // Number of matrices
        int m = 5;           // Number of rows
        int n = 3;           // Number of columns
        
        // Create batched views
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A("A", batch_size, m, n);
        Kokkos::View<scalar_type**, memory_space> 
          u("u", batch_size, m);  // Householder vectors
        Kokkos::View<scalar_type*, memory_space> 
          tau("tau", batch_size); // Householder scalars
        Kokkos::View<scalar_type**, memory_space> 
          w("w", batch_size, n);  // Workspaces
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror_view(A);
        auto u_host = Kokkos::create_mirror_view(u);
        auto tau_host = Kokkos::create_mirror_view(tau);
        
        for (int b = 0; b < batch_size; ++b) {
          // Initialize A with recognizable pattern
          for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
              A_host(b, i, j) = (b + 1) * 100 + (i + 1) * 10 + (j + 1);
            }
          }
          
          // Initialize Householder vector
          u_host(b, 0) = 1.0;
          for (int i = 1; i < m; ++i) {
            u_host(b, i) = 0.5 * i * (b + 1);
          }
          
          // Set tau
          tau_host(b) = 0.5 * (b + 1);
        }
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(u, u_host);
        Kokkos::deep_copy(tau, tau_host);
        
        // Save original for verification
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A_orig("A_orig", batch_size, m, n);
        Kokkos::deep_copy(A_orig, A);
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        policy_type policy(batch_size, Kokkos::AUTO);
        
        // Apply Householder transformations using team parallelism
        Kokkos::parallel_for("BatchedApplyHouseholder", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            auto A_b = Kokkos::subview(A, b, Kokkos::ALL(), Kokkos::ALL());
            auto u_b = Kokkos::subview(u, b, Kokkos::ALL());
            auto w_b = Kokkos::subview(w, b, Kokkos::ALL());
            
            KokkosBatched::TeamVectorApplyHouseholder<typename policy_type::member_type, 
                                                    KokkosBatched::Side::Left>
              ::invoke(member, u_b, tau(b), A_b, w_b);
          }
        );
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Verify results for a few batches
        auto A_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), A_orig);
        
        bool test_passed = true;
        for (int b = 0; b < 1; ++b) { // Just check first batch for simplicity
          // Calculate expected result manually
          Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> 
            A_expected("A_expected", m, n);
          
          // First compute v = u^T * A_orig (a row vector)
          Kokkos::View<scalar_type*, Kokkos::LayoutRight, Kokkos::HostSpace> 
            v("v", n);
          
          for (int j = 0; j < n; ++j) {
            v(j) = 0.0;
            for (int i = 0; i < m; ++i) {
              v(j) += u_host(b, i) * A_orig_host(b, i, j);
            }
          }
          
          // Now compute A_expected = A_orig - tau * u * v
          for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
              A_expected(i, j) = A_orig_host(b, i, j) - tau_host(b) * u_host(b, i) * v(j);
            }
          }
          
          // Compare results
          for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
              if (std::abs(A_host(b, i, j) - A_expected(i, j)) > 1e-12) {
                test_passed = false;
                std::cout << "Batch " << b << " mismatch at (" << i << ", " << j << "): " 
                          << A_host(b, i, j) << " vs expected " << A_expected(i, j) << std::endl;
              }
            }
          }
        }
        
        if (test_passed) {
          std::cout << "Batched TeamVectorApplyHouseholder test: PASSED" << std::endl;
        } else {
          std::cout << "Batched TeamVectorApplyHouseholder test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
