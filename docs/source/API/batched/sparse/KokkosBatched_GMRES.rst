KokkosBatched::GMRES
####################

Defined in header: :code:`KokkosBatched_GMRES.hpp`

.. code-block:: c++

    template <typename MemberType, typename ArgMode>
    struct GMRES {
      template <typename OperatorType, typename VectorViewType, typename KrylovHandleType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType& member,
             const OperatorType& A,
             const VectorViewType& B,
             const VectorViewType& X,
             const KrylovHandleType& handle);
    };

The ``GMRES`` (Generalized Minimal Residual Method) operation implements an iterative solver for batched sparse linear systems of the form :math:`Ax = b`. GMRES is particularly effective for non-symmetric systems, where CG may not be applicable.

The GMRES method works by constructing an orthogonal basis for the Krylov subspace and finding the solution that minimizes the residual norm over this subspace. It is more computationally intensive than CG but has better convergence properties for difficult systems.

Parameters
==========

:member: Team execution policy instance
:A: Operator (typically a batched sparse matrix or a preconditioned matrix)
:B: Input view containing the right-hand sides of the linear systems
:X: Input/output view containing the initial guess and the solution
:handle: Krylov handle providing solver parameters and workspace

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgMode`` must be one of:
   - ``KokkosBatched::Mode::Serial`` for serial execution
   - ``KokkosBatched::Mode::Team`` for team-based execution
   - ``KokkosBatched::Mode::TeamVector`` for team-vector-based execution
- ``OperatorType`` must support the application of a matrix or preconditioned matrix to a vector
- ``VectorViewType`` must be a rank-2 view containing the right-hand sides and solution vectors
- ``KrylovHandleType`` must provide solver parameters and workspace
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_GMRES.hpp>
    #include <KokkosBatched_Spmv.hpp>
    #include <KokkosBatched_Krylov_Handle.hpp>
    #include <KokkosBatched_JacobiPrec.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    using view_type = Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space>;
    using int_view_type = Kokkos::View<int*, memory_space>;
    
    // Matrix Operator for GMRES
    template <typename ScalarType, typename DeviceType>
    class BatchedCrsMatrixOperator {
    public:
      using execution_space = typename DeviceType::execution_space;
      using memory_space = typename DeviceType::memory_space;
      using device_type = DeviceType;
      using value_type = ScalarType;
      
      using values_view_type = Kokkos::View<ScalarType**, Kokkos::LayoutRight, memory_space>;
      using int_view_type = Kokkos::View<int*, memory_space>;
      using vector_view_type = Kokkos::View<ScalarType**, Kokkos::LayoutRight, memory_space>;
      
    private:
      values_view_type _values;
      int_view_type _row_ptr;
      int_view_type _col_idx;
      int _n_batch;
      int _n_rows;
      int _n_cols;
      
    public:
      BatchedCrsMatrixOperator(const values_view_type& values,
                              const int_view_type& row_ptr,
                              const int_view_type& col_idx)
        : _values(values), _row_ptr(row_ptr), _col_idx(col_idx) {
        _n_batch = values.extent(0);
        _n_rows = row_ptr.extent(0) - 1;
        _n_cols = _n_rows; // Assuming square matrix for GMRES
      }
      
      // Apply the operator to a vector
      template <typename MemberType, typename ArgMode>
      KOKKOS_INLINE_FUNCTION
      void apply(const MemberType& member,
                 const vector_view_type& X,
                 const vector_view_type& Y) const {
        // alpha = 1.0, beta = 0.0 (Y = A*X)
        const ScalarType alpha = 1.0;
        const ScalarType beta = 0.0;
        
        KokkosBatched::Spmv<MemberType, 
                           KokkosBatched::Trans::NoTranspose, 
                           ArgMode>
          ::template invoke<values_view_type, int_view_type, vector_view_type, vector_view_type, 0>
          (member, alpha, _values, _row_ptr, _col_idx, X, beta, Y);
      }
      
      KOKKOS_INLINE_FUNCTION
      int n_rows() const { return _n_rows; }
      
      KOKKOS_INLINE_FUNCTION
      int n_cols() const { return _n_cols; }
      
      KOKKOS_INLINE_FUNCTION
      int n_batch() const { return _n_batch; }
    };
    
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
        
        // Create vectors for the systems
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          B("B", batch_size, n);  // RHS
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          X("X", batch_size, n);  // Solution
        
        // Initialize on host
        auto row_ptr_host = Kokkos::create_mirror_view(row_ptr);
        auto col_idx_host = Kokkos::create_mirror_view(col_idx);
        auto values_host = Kokkos::create_mirror_view(values);
        auto B_host = Kokkos::create_mirror_view(B);
        auto X_host = Kokkos::create_mirror_view(X);
        
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
        // Creating a non-symmetric matrix to demonstrate GMRES
        for (int b = 0; b < batch_size; ++b) {
          for (int j = 0; j < nnz; ++j) {
            // Diagonal elements are larger for stability
            int row = 0;
            while (j >= row_ptr_host(row+1)) row++;
            
            int col = col_idx_host(j);
            if (col == row) {
              values_host(b, j) = 10.0 + 0.1 * b;  // Diagonal
            } else if (col > row) {
              values_host(b, j) = -1.0 + 0.05 * b;  // Upper triangular
            } else {
              values_host(b, j) = -0.5 + 0.025 * b;  // Lower triangular (asymmetric)
            }
          }
        }
        
        // Initialize right-hand side and initial guess
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            B_host(b, i) = 1.0;  // Simple RHS
            X_host(b, i) = 0.0;  // Initial guess = 0
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(row_ptr, row_ptr_host);
        Kokkos::deep_copy(col_idx, col_idx_host);
        Kokkos::deep_copy(values, values_host);
        Kokkos::deep_copy(B, B_host);
        Kokkos::deep_copy(X, X_host);
        
        // Create matrix operator
        using matrix_operator_type = BatchedCrsMatrixOperator<scalar_type, execution_space::device_type>;
        matrix_operator_type A_op(values, row_ptr, col_idx);
        
        // Configure Krylov handle for GMRES
        // For GMRES, we need workspace for the Arnoldi orthogonalization
        const int max_iterations = 100;
        const int n_team = 1;  // Number of systems per team
        const bool monitor_residual = true;
        
        using krylov_handle_type = KokkosBatched::KrylovHandle<view_type, int_view_type, view_type>;
        krylov_handle_type handle(batch_size, n_team, max_iterations, monitor_residual);
        
        // Set solver parameters
        handle.set_tolerance(1e-8);       // Convergence tolerance
        handle.set_max_iteration(100);    // Maximum iterations
        handle.set_ortho_strategy(1);     // Use Modified Gram-Schmidt (more stable)
        
        // Allocate workspace needed by GMRES
        // For GMRES, we need (n+max_iter+3) workspace per batch
        // This is for storing the Hessenberg matrix and Arnoldi vectors
        view_type Arnoldi_view("Arnoldi_view", batch_size, max_iterations, (n + max_iterations + 3));
        handle.Arnoldi_view = Arnoldi_view;
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        int team_size = policy_type::team_size_recommended(
          [](const int &, const int &) {}, 
          Kokkos::ParallelForTag());
        policy_type policy(batch_size, team_size);
        
        // Solve the linear systems using GMRES
        Kokkos::parallel_for("BatchedGMRES", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            // Get current batch's right-hand side and solution
            auto B_b = Kokkos::subview(B, b, Kokkos::ALL());
            auto X_b = Kokkos::subview(X, b, Kokkos::ALL());
            
            // Solve using GMRES
            KokkosBatched::GMRES<typename policy_type::member_type, 
                                KokkosBatched::Mode::TeamVector>
              ::invoke(member, A_op, B_b, X_b, handle);
          }
        );
        
        // Check convergence
        handle.synchronise_host();
        
        if (handle.is_converged_host()) {
          std::cout << "All linear systems converged!" << std::endl;
        } else {
          std::cout << "Some linear systems did not converge." << std::endl;
        }
        
        // Print convergence details for a few batches
        for (int b = 0; b < std::min(batch_size, 3); ++b) {
          if (handle.is_converged_host(b)) {
            std::cout << "Batch " << b << " converged in " 
                      << handle.get_iteration_host(b) << " iterations." << std::endl;
            
            if (monitor_residual) {
              std::cout << "  Final residual: " 
                        << handle.get_last_norm_host(b) << std::endl;
            }
          } else {
            std::cout << "Batch " << b << " did not converge." << std::endl;
          }
        }
        
        // Copy results back to host
        Kokkos::deep_copy(X_host, X);
        
        // Print first few entries of the solutions
        for (int b = 0; b < std::min(batch_size, 3); ++b) {
          std::cout << "Solution for batch " << b << ": [";
          for (int i = 0; i < std::min(n, 5); ++i) {
            std::cout << X_host(b, i) << " ";
          }
          std::cout << "...]" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
