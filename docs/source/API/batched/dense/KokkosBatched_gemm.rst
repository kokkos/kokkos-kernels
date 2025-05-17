KokkosBatched::Gemm
###################

Defined in header `KokkosBatched_Gemm_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Gemm_Decl.hpp>`_

.. code:: c++

    template <typename MemberType, typename ArgTransA, typename ArgTransB, typename ArgMode, typename ArgAlgo>
    struct Gemm {
      template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
      KOKKOS_FORCEINLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, 
                                                    const AViewType &A, const BViewType &B, 
                                                    const ScalarType beta, const CViewType &C);
    };

Performs batched matrix-matrix multiplication across multiple small dense matrices. The operation is defined as:

.. math::

   C_i = \alpha \cdot \text{op}(A_i) \cdot \text{op}(B_i) + \beta \cdot C_i

where:

- :math:`\text{op}(X)` can be :math:`X`, :math:`X^T` (transpose), or :math:`X^H` (Hermitian transpose)
- :math:`A_i`, :math:`B_i`, and :math:`C_i` are the i-th matrices in the batch
- :math:`\alpha` and :math:`\beta` are scalar values

The batched GEMM operation performs this computation on a batch of matrices simultaneously, allowing for optimal performance when dealing with multiple small matrix-matrix multiplications.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:alpha: Scalar multiplier for the AB product
:A: Input view containing batch of matrices
:B: Input view containing batch of matrices
:beta: Scalar multiplier for C
:C: Input/output view for the result matrices

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgTransA`` and ``ArgTransB`` must be one of:

  - ``Trans::NoTranspose`` - use matrix as is
  - ``Trans::Transpose`` - use transpose of matrix
  - ``Trans::ConjTranspose`` - use conjugate transpose of matrix

- ``ArgMode`` must be one of:

  - ``Mode::Serial`` - for serial execution
  - ``Mode::Team`` - for team-based execution
  - ``Mode::TeamVector`` - for team-vector execution

- ``ArgAlgo`` must be one of:

  - ``Algo::Gemm::Unblocked`` - for small matrices
  - ``Algo::Gemm::Blocked`` - for larger matrices with blocking

- ``AViewType``, ``BViewType``, and ``CViewType`` must be rank-2 or rank-3 Kokkos Views

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Gemm_Decl.hpp>

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
        int m = 16;             // Rows in A and C
        int n = 16;             // Columns in B and C
        int k = 16;             // Columns in A, rows in B
        
        // Create views for batched matrices
        // Layout: (batch, row, col)
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, m, k),
          B("B", batch_size, k, n),
          C("C", batch_size, m, n);
        
        // Fill matrices with data (using simple RangePolicy)
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        // Initialize matrices (for example purposes)
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th matrix in each batch
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < k; ++col) {
              A(i, row, col) = 1.0; // Simple initialization
            }
          }
          
          for (int row = 0; row < k; ++row) {
            for (int col = 0; col < n; ++col) {
              B(i, row, col) = 1.0; // Simple initialization
            }
          }
          
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
              C(i, row, col) = 0.0; // Initialize C to zero
            }
          }
        });
        
        Kokkos::fence();
        
        // Define scalar multipliers
        scalar_type alpha = 1.0;
        scalar_type beta = 0.0;
        
        // Perform batched GEMM using TeamPolicy
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_gemm", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices for each matrix
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto B_i = Kokkos::subview(B, i, Kokkos::ALL(), Kokkos::ALL());
            auto C_i = Kokkos::subview(C, i, Kokkos::ALL(), Kokkos::ALL());
            
            // Perform GEMM using the Team variant
            KokkosBatched::Gemm<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Trans::NoTranspose,       // ArgTransA
              KokkosBatched::Trans::NoTranspose,       // ArgTransB
              KokkosBatched::Mode::Team,               // ArgMode
              KokkosBatched::Algo::Gemm::Unblocked     // ArgAlgo
            >::invoke(member, alpha, A_i, B_i, beta, C_i);
          }
        );
        
        Kokkos::fence();
        
        // Verify results or continue processing...
      }
      Kokkos::finalize();
      return 0;
    }
