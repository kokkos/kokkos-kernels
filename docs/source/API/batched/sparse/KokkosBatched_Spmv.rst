KokkosBatched::Spmv
###################

Defined in header `KokkosBatched_Spmv.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/src/batched/KokkosBatched_Spmv.hpp>`_

.. code-block:: c++

    // Serial version
    template <typename ArgTrans = Trans::NoTranspose>
    struct SerialSpmv {
      template <typename ValuesViewType, typename IntView, typename xViewType, 
                typename yViewType, typename alphaViewType, typename betaViewType, int dobeta>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const alphaViewType& alpha,
             const ValuesViewType& values,
             const IntView& row_ptr,
             const IntView& colIndices,
             const xViewType& x,
             const betaViewType& beta,
             const yViewType& Y);
    };
    
    // Team version
    template <typename MemberType, typename ArgTrans = Trans::NoTranspose>
    struct TeamSpmv {
      template <typename ValuesViewType, typename IntView, typename xViewType, 
                typename yViewType, typename alphaViewType, typename betaViewType, int dobeta>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType& member,
             const alphaViewType& alpha,
             const ValuesViewType& values,
             const IntView& row_ptr,
             const IntView& colIndices,
             const xViewType& x,
             const betaViewType& beta,
             const yViewType& y);
    };
    
    // TeamVector version
    template <typename MemberType, typename ArgTrans = Trans::NoTranspose, unsigned N_team = 1>
    struct TeamVectorSpmv {
      template <typename ValuesViewType, typename IntView, typename xViewType, 
                typename yViewType, typename alphaViewType, typename betaViewType, int dobeta>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType& member,
             const alphaViewType& alpha,
             const ValuesViewType& values,
             const IntView& row_ptr,
             const IntView& colIndices,
             const xViewType& x,
             const betaViewType& beta,
             const yViewType& y);
    };
    
    // Selective interface
    template <typename MemberType, typename ArgTrans, typename ArgMode>
    struct Spmv {
      template <typename ValuesViewType, typename IntView, typename xViewType, 
                typename yViewType, typename alphaViewType, typename betaViewType, int dobeta>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType& member,
             const alphaViewType& alpha,
             const ValuesViewType& values,
             const IntView& row_ptr,
             const IntView& colIndices,
             const xViewType& x,
             const betaViewType& beta,
             const yViewType& y);
    };

The ``Spmv`` (Sparse Matrix-Vector Multiplication) operation performs batched sparse matrix-vector multiplication for matrices sharing the same sparsity pattern. It computes:

.. math::

    y_l = \alpha_l \cdot A_l \cdot x_l + \beta_l \cdot y_l \quad \text{for all } l = 1, \ldots, N

where:

- :math:`N` is the number of matrices in the batch
- :math:`A_1, \ldots, A_N` are sparse matrices with the same sparsity pattern
- :math:`x_1, \ldots, x_N` are input vectors
- :math:`y_1, \ldots, y_N` are output vectors
- :math:`\alpha_1, \ldots, \alpha_N` are scaling factors for the matrix-vector products
- :math:`\beta_1, \ldots, \beta_N` are scaling factors for the input y vectors

The matrices are stored in Compressed Row Storage (CRS) format, where the sparsity pattern (row_ptr and colIndices) is shared across all matrices in the batch, but the values differ.

Parameters
==========

:member: Team execution policy instance (only for team/teamvector versions)
:alpha: Scaling factor(s) for the matrix-vector product(s)
:values: Values of the batched CRS matrix
:row_ptr: Row pointers of the CRS format (shared across all matrices)
:colIndices: Column indices of the CRS format (shared across all matrices)
:x: Input vector(s)
:beta: Scaling factor(s) for the input y vector(s)
:y: Input/output vector(s)

Template Parameters
-------------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgTrans`` must be the transpose option (typically ``KokkosBatched::Trans::NoTranspose``)
- ``ArgMode`` must be one of:
   - ``KokkosBatched::Mode::Serial`` for serial execution
   - ``KokkosBatched::Mode::Team`` for team-based execution
   - ``KokkosBatched::Mode::TeamVector`` for team-vector-based execution
- ``ValuesViewType`` must be a rank-2 view with dimensions (batch_size, nnz)
- ``IntView`` must be a rank-1 view for row pointers and column indices
- ``xViewType`` and ``yViewType`` must be rank-2 views with dimensions (batch_size, n)
- ``alphaViewType`` and ``betaViewType`` must be rank-1 views or scalar values
- ``dobeta`` must be 0 (don't use beta) or 1 (use beta)

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Spmv.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int batch_size = 10;  // Number of matrices
        int n = 100;          // Size of each matrix
        int nnz_per_row = 5;  // Non-zeros per row
        int nnz = n * nnz_per_row; // Total non-zeros
        
        // Create batched matrix in CRS format
        Kokkos::View<int*, memory_space> row_ptr("row_ptr", n+1);
        Kokkos::View<int*, memory_space> col_idx("col_idx", nnz);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          values("values", batch_size, nnz);
        
        // Create vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          x("x", batch_size, n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          y("y", batch_size, n);
        
        // Create alpha and beta
        Kokkos::View<scalar_type*, memory_space> alpha("alpha", batch_size);
        Kokkos::View<scalar_type*, memory_space> beta("beta", batch_size);
        
        // Initialize on host
        auto row_ptr_host = Kokkos::create_mirror_view(row_ptr);
        auto col_idx_host = Kokkos::create_mirror_view(col_idx);
        auto values_host = Kokkos::create_mirror_view(values);
        auto x_host = Kokkos::create_mirror_view(x);
        auto y_host = Kokkos::create_mirror_view(y);
        auto alpha_host = Kokkos::create_mirror_view(alpha);
        auto beta_host = Kokkos::create_mirror_view(beta);
        
        // Initialize matrix sparsity pattern (shared across all matrices)
        int nnz_count = 0;
        for (int i = 0; i < n; ++i) {
          row_ptr_host(i) = nnz_count;
          
          // Add diagonal element
          col_idx_host(nnz_count) = i;
          nnz_count++;
          
          // Add off-diagonal elements
          for (int k = 1; k < nnz_per_row; ++k) {
            int col = (i + k) % n;  // Simple pattern
            col_idx_host(nnz_count) = col;
            nnz_count++;
          }
        }
        row_ptr_host(n) = nnz_count;  // Finalize row_ptr
        
        // Initialize matrix values (different for each batch)
        for (int b = 0; b < batch_size; ++b) {
          for (int j = 0; j < nnz; ++j) {
            // Diagonal elements are larger for stability
            int row = 0;
            while (j >= row_ptr_host(row+1)) row++;
            
            if (col_idx_host(j) == row) {
              values_host(b, j) = 10.0 + 0.1 * b;  // Diagonal
            } else {
              values_host(b, j) = -1.0 + 0.05 * b;  // Off-diagonal
            }
          }
        }
        
        // Initialize vectors and coefficients
        for (int b = 0; b < batch_size; ++b) {
          alpha_host(b) = 1.0 + 0.1 * b;
          beta_host(b) = 0.5 + 0.05 * b;
          
          for (int i = 0; i < n; ++i) {
            x_host(b, i) = 1.0;  // Simple vector
            y_host(b, i) = 0.5;  // Initial y value
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(row_ptr, row_ptr_host);
        Kokkos::deep_copy(col_idx, col_idx_host);
        Kokkos::deep_copy(values, values_host);
        Kokkos::deep_copy(x, x_host);
        Kokkos::deep_copy(y, y_host);
        Kokkos::deep_copy(alpha, alpha_host);
        Kokkos::deep_copy(beta, beta_host);
        
        // Save original y for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          y_orig("y_orig", batch_size, n);
        Kokkos::deep_copy(y_orig, y);
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        int team_size = policy_type::team_size_recommended(
            [](const int &, const int &) {}, 
            Kokkos::ParallelForTag());
        policy_type policy(batch_size, team_size);
        
        // Perform batched SpMV with TeamVector mode (y = alpha*A*x + beta*y)
        Kokkos::parallel_for("BatchedSpMV", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            // Get current batch's vectors
            auto x_b = Kokkos::subview(x, b, Kokkos::ALL());
            auto y_b = Kokkos::subview(y, b, Kokkos::ALL());
            
            // Get current batch's values
            auto values_b = Kokkos::subview(values, b, Kokkos::ALL());
            
            // Perform SpMV: y = alpha*A*x + beta*y
            KokkosBatched::Spmv<typename policy_type::member_type, 
                              KokkosBatched::Trans::NoTranspose, 
                              KokkosBatched::Mode::TeamVector>
              ::template invoke<decltype(values_b), decltype(row_ptr), 
                              decltype(x_b), decltype(y_b), decltype(alpha), 
                              decltype(beta), 1>
              (member, alpha(b), values_b, row_ptr, col_idx, x_b, beta(b), y_b);
          }
        );
        
        // Copy results back to host
        Kokkos::deep_copy(y_host, y);
        auto y_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y_orig);
        
        // Verify results for first batch
        int b = 0;
        std::cout << "SpMV Results for batch " << b << ":" << std::endl;
        std::cout << "alpha = " << alpha_host(b) << ", beta = " << beta_host(b) << std::endl;
        
        // Print first few entries
        std::cout << "Original y: [";
        for (int i = 0; i < std::min(n, 5); ++i) {
          std::cout << y_orig_host(b, i) << " ";
        }
        std::cout << "...]" << std::endl;
        
        std::cout << "Result y: [";
        for (int i = 0; i < std::min(n, 5); ++i) {
          std::cout << y_host(b, i) << " ";
        }
        std::cout << "...]" << std::endl;
        
        // In a real application, you would implement a proper verification
        // by computing the expected result manually and comparing
      }
      Kokkos::finalize();
      return 0;
    }
