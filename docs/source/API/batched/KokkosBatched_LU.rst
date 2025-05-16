KokkosBatched::LU
##################

Defined in header `KokkosBatched_LU_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_LU_Decl.hpp>`_

.. code:: c++

    template <typename MemberType, typename ArgMode, typename ArgAlgo>
    struct LU {
      template <typename AViewType>
      KOKKOS_FORCEINLINE_FUNCTION static int invoke(
        const MemberType &member, const AViewType &A,
        const typename MagnitudeScalarType<typename AViewType::non_const_value_type>::type tiny = 0);
    };

Performs batched LU decomposition without pivoting on a batch of small dense matrices. The operation decomposes each matrix A into:

.. math::

   A = LU

where:

- :math:`L` is a lower triangular matrix with unit diagonal
- :math:`U` is an upper triangular matrix

The LU decomposition is performed in-place, overwriting the input matrix A with both L and U factors. The unit diagonal of L is not stored.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:A: Input/output view for the matrix to decompose and store the LU factors
:tiny: Optional small value added to diagonal elements to prevent division by zero (default = 0)

Type Requirements
----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgMode`` must be one of:

  - ``Mode::Serial`` - for serial execution
  - ``Mode::Team`` - for team-based execution

- ``ArgAlgo`` must be one of:

  - ``Algo::LU::Unblocked`` - direct LU decomposition
  - ``Algo::LU::Blocked`` - blocked algorithm for larger matrices

- ``AViewType`` must be a rank-2 or rank-3 Kokkos View representing a batch of square matrices

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_LU_Decl.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define matrix dimensions
        int batch_size = 1000;  // Number of matrices in batch
        int n = 8;              // Size of each square matrix
        
        // Create views for batched matrices
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, n, n);
        
        // Fill matrices with data (diagonally dominant matrices for stability)
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th matrix in the batch as a diagonally dominant matrix
          for (int row = 0; row < n; ++row) {
            for (int col = 0; col < n; ++col) {
              if (row == col) {
                A(i, row, col) = n + 1.0; // Diagonal elements
              } else {
                A(i, row, col) = 1.0;     // Off-diagonal elements
              }
            }
          }
        });
        
        Kokkos::fence();
        
        // Small value to prevent division by zero
        scalar_type tiny_val = 1.0e-10;
        
        // Perform batched LU decomposition using TeamPolicy
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_lu", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slice for matrix A
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            
            // Perform LU decomposition using Team variant
            KokkosBatched::LU<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Mode::Team,               // ArgMode
              KokkosBatched::Algo::LU::Unblocked       // ArgAlgo
            >::invoke(member, A_i, tiny_val);
          }
        );
        
        Kokkos::fence();
        
        // At this point, each A(i) contains the LU factors
        // We could extract L and U or use them for solving linear systems
        
        // Example: Extract L and U from first matrix (on host)
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(A, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> 
          L_host("L_host", n, n), 
          U_host("U_host", n, n);
        
        // Extract L (with unit diagonal)
        for (int i = 0; i < n; ++i) {
          L_host(i, i) = 1.0; // Unit diagonal
          for (int j = 0; j < i; ++j) {
            L_host(i, j) = A_host(i, j);
          }
        }
        
        // Extract U
        for (int i = 0; i < n; ++i) {
          for (int j = i; j < n; ++j) {
            U_host(i, j) = A_host(i, j);
          }
        }
        
        // L and U could be used for further computations
      }
      Kokkos::finalize();
      return 0;
    }
