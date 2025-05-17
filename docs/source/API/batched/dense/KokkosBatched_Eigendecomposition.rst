KokkosBatched::Eigendecomposition
#################################

Defined in header `KokkosBatched_Eigendecomposition_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Eigendecomposition_Decl.hpp>`_

.. code:: c++

    struct SerialEigendecomposition {
      template <typename AViewType, typename EViewType, typename UViewType, typename WViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const AViewType &A, 
                                              const EViewType &er, 
                                              const EViewType &ei,
                                              const UViewType &UL, 
                                              const UViewType &UR, 
                                              const WViewType &W);
    };

    template <typename MemberType>
    struct TeamVectorEigendecomposition {
      template <typename AViewType, typename EViewType, typename UViewType, typename WViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const AViewType &A, 
                                              const EViewType &er, 
                                              const EViewType &ei,
                                              const UViewType &UL, 
                                              const UViewType &UR, 
                                              const WViewType &W);
    };

Performs eigendecomposition of a general non-symmetric matrix. For each matrix A in the batch, computes:

.. math::

   A = UL \cdot \Lambda \cdot UR

where:

- :math:`UL` contains the left eigenvectors
- :math:`UR` contains the right eigenvectors
- :math:`\Lambda` is a diagonal matrix of eigenvalues (stored in vectors er and ei)

The implementation first reduces the matrix to upper Hessenberg form, then applies the Francis double shift QR algorithm to compute the Schur form. The eigenvectors are then computed from the Schur form.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:A: Input matrix to decompose; on exit, contains the Schur form
:er: Output view for the real parts of eigenvalues
:ei: Output view for the imaginary parts of eigenvalues
:UL: Output view for left eigenvectors
:UR: Output view for right eigenvectors
:W: Workspace view for temporary calculations

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``AViewType`` must be a rank-2 or rank-3 Kokkos View representing square matrices
- ``EViewType`` must be a rank-1 or rank-2 Kokkos View to store eigenvalues
- ``UViewType`` must be a rank-2 or rank-3 Kokkos View to store eigenvectors
- ``WViewType`` must be a rank-1 or rank-2 Kokkos View with sufficient workspace:
  
  - At least (2*m*m+5*m) where m is the dimension of matrix A

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Eigendecomposition_Decl.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define matrix dimensions
        int batch_size = 100;  // Number of matrices in batch
        int n = 4;             // Size of each square matrix
        
        // Create views for input matrices and results
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, n, n),        // Input matrices
          UL("UL", batch_size, n, n),      // Left eigenvectors
          UR("UR", batch_size, n, n);      // Right eigenvectors
        
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type>
          er("er", batch_size, n),        // Real parts of eigenvalues
          ei("ei", batch_size, n);        // Imaginary parts of eigenvalues
        
        // Workspace (size = 2*n*n + 5*n)
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type>
          W("W", batch_size, 2*n*n + 5*n);
        
        // Fill matrices with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th matrix as a companion matrix
          // This has known eigenvalues for verification
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
              A(i, row, col) = 0.0;
              
              // Set the subdiagonal to 1
              if (row == col + 1) {
                A(i, row, col) = 1.0;
              }
              
              // Set the last row with specific coefficients
              if (row == n-1) {
                A(i, row, col) = -1.0 * (col + 1);
              }
            }
          }
        });
        
        Kokkos::fence();
        
        // Perform batched eigendecomposition using TeamVectorPolicy
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO, 32);
        
        Kokkos::parallel_for("batched_eigendecomposition", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto er_i = Kokkos::subview(er, i, Kokkos::ALL());
            auto ei_i = Kokkos::subview(ei, i, Kokkos::ALL());
            auto UL_i = Kokkos::subview(UL, i, Kokkos::ALL(), Kokkos::ALL());
            auto UR_i = Kokkos::subview(UR, i, Kokkos::ALL(), Kokkos::ALL());
            auto W_i = Kokkos::subview(W, i, Kokkos::ALL());
            
            // Perform eigendecomposition
            KokkosBatched::TeamVectorEigendecomposition<typename team_policy_type::member_type>
              ::invoke(member, A_i, er_i, ei_i, UL_i, UR_i, W_i);
          }
        );
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto er_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), er);
        auto ei_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), ei);
        
        // Check the eigenvalues for first matrix
        printf("Eigenvalues for first matrix:\n");
        for (int j = 0; j < n; ++j) {
          if (std::abs(ei_host(0, j)) < 1e-10) {
            printf("  λ%d = %.4f\n", j, er_host(0, j));
          } else {
            printf("  λ%d = %.4f + %.4fi\n", j, er_host(0, j), ei_host(0, j));
          }
        }
      }
      Kokkos::finalize();
      return 0;
    }
