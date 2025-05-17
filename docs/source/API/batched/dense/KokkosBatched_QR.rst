KokkosBatched::QR
#################

Defined in header: :code:`KokkosBatched_QR_Decl.hpp`

.. code:: c++

    template <typename MemberType, typename ArgMode, typename ArgAlgo>
    struct QR {
      template <typename AViewType, typename tViewType, typename wViewType>
      KOKKOS_FORCEINLINE_FUNCTION static int invoke(const MemberType &member, 
                                                   const AViewType &A, 
                                                   const tViewType &t, 
                                                   const wViewType &w);
    };

Performs batched QR decomposition on a batch of dense matrices. For each matrix A in the batch, the operation computes the factorization:

.. math::

   A = QR

where:

- :math:`Q` is an orthogonal (or unitary) matrix
- :math:`R` is an upper triangular matrix

The implementation uses Householder reflectors to compute the QR factorization. The matrix A is overwritten with the factorized result, with the R factor in the upper triangular portion and the Householder reflectors in the lower triangular portion.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:A: Input/output view for the matrix to decompose and store the QR factors
:t: Output view to store the Householder scalars
:w: Workspace view for temporary calculations

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgMode`` must be one of:

  - ``Mode::Serial`` - for serial execution
  - ``Mode::TeamVector`` - for team-vector execution

- ``ArgAlgo`` must be one of:

  - ``Algo::QR::Unblocked`` - for direct QR decomposition
  - ``Algo::QR::Blocked`` - for blocked algorithm (better for larger matrices)

- ``AViewType`` must be a rank-2 or rank-3 Kokkos View
- ``tViewType`` must be a rank-1 or rank-2 Kokkos View
- ``wViewType`` must be a rank-1 or rank-2 Kokkos View with sufficient workspace

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_QR_Decl.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define matrix dimensions
        int batch_size = 500;  // Number of matrices in batch
        int m = 10;            // Rows in A
        int n = 8;             // Columns in A (m >= n for QR)
        
        // Create views for batched matrices
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A("A", batch_size, m, n);
        
        // Create views for Householder scalars and workspace
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type>
          t("t", batch_size, n);  // Householder scalars
        
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type>
          w("w", batch_size, n);  // Workspace
        
        // Fill matrices with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th matrix in the batch
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
              // Create a matrix with known pattern
              // For simplicity, we'll use a matrix with predictable values
              A(i, row, col) = 1.0/(row + col + 1.0); // Hilbert-like matrix
            }
          }
        });
        
        Kokkos::fence();
        
        // Perform batched QR decomposition using TeamPolicy with TeamVector
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO, Kokkos::AUTO);
        
        Kokkos::parallel_for("batched_qr", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto A_i = Kokkos::subview(A, i, Kokkos::ALL(), Kokkos::ALL());
            auto t_i = Kokkos::subview(t, i, Kokkos::ALL());
            auto w_i = Kokkos::subview(w, i, Kokkos::ALL());
            
            // Perform QR decomposition
            KokkosBatched::QR<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Mode::TeamVector,         // ArgMode
              KokkosBatched::Algo::QR::Unblocked       // ArgAlgo
            >::invoke(member, A_i, t_i, w_i);
          }
        );
        
        Kokkos::fence();
        
        // At this point, each A(i) contains the QR factorization
        // with R in the upper triangular part and Householder reflectors
        // in the lower triangular part. The t(i) vectors contain the
        // Householder scalars.
        
        // Example: Extract R from first matrix (on host)
        auto A_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(A, 0, Kokkos::ALL(), Kokkos::ALL()));
        
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> 
          R_host("R_host", n, n);
        
        // Extract R (upper triangular part)
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            if (i <= j) {
              R_host(i, j) = A_host(i, j);
            } else {
              R_host(i, j) = 0.0;
            }
          }
        }
        
        // The R factor could be used for computing least squares solutions or
        // other applications
      }
      Kokkos::finalize();
      return 0;
    }
