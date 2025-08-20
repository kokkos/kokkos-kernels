KokkosBatched::JacobiPrec
#########################

Defined in header: :code:`KokkosBatched_JacobiPrec.hpp`

.. code-block:: c++

    template <class ValuesViewType>
    class JacobiPrec {
    public:
      // Constructor
      KOKKOS_INLINE_FUNCTION
      JacobiPrec(const ValuesViewType& _diag_values);
      
      // Compute inverse of diagonal elements
      KOKKOS_INLINE_FUNCTION 
      void setComputedInverse();
      
      // Compute inverse with team parallelism
      template <typename MemberType, typename ArgMode>
      KOKKOS_INLINE_FUNCTION 
      void computeInverse(const MemberType& member) const;
      
      // Compute inverse (serial version)
      KOKKOS_INLINE_FUNCTION 
      void computeInverse() const;
      
      // Apply preconditioner with team parallelism
      template <typename ArgTrans, typename ArgMode, int sameXY, 
                typename MemberType, typename XViewType, typename YViewType>
      KOKKOS_INLINE_FUNCTION
      void apply(const MemberType& member,
                 const XViewType& X,
                 const YViewType& Y) const;
      
      // Apply preconditioner (serial version)
      template <typename ArgTrans, int sameXY, 
                typename XViewType, typename YViewType>
      KOKKOS_INLINE_FUNCTION
      void apply(const XViewType& X,
                 const YViewType& Y) const;
    };

The ``JacobiPrec`` operation implements a Jacobi (diagonal) preconditioner for batched sparse linear systems. The Jacobi preconditioner is one of the simplest preconditioners and is often used to improve the convergence of iterative solvers like CG.

For a matrix :math:`A`, the Jacobi preconditioner :math:`M^{-1}` is defined as the inverse of the diagonal of :math:`A`:

.. math::

    M^{-1} = \text{diag}(A)^{-1}

When applied to a vector :math:`x`, the operation performs:

.. math::

    y = M^{-1} x

which is efficiently implemented as a Hadamard (element-wise) product:

.. math::

    y_i = (1/A_{ii}) \cdot x_i \quad \text{for all } i

Parameters
==========

Constructor Parameters
----------------------

:_diag_values: View containing the diagonal values of the matrices

Method Parameters
-----------------

computeInverse():
  Computes the inverse of the diagonal elements

computeInverse(member):
  :member: Team execution policy instance for parallel computation of inverse

apply():
  :X: Input vector view
  :Y: Output vector view

apply(member, X, Y):
  :member: Team execution policy instance
  :X: Input vector view
  :Y: Output vector view

Type Requirements
-----------------

- ``ValuesViewType`` must be a rank-2 view containing the diagonal values of the matrices
- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgMode`` must specify the execution mode (Serial, Team, or TeamVector)
- ``ArgTrans`` must specify transposition type (typically NoTranspose)
- ``XViewType`` and ``YViewType`` must be rank-2 views representing vectors
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_JacobiPrec.hpp>
    #include <KokkosBatched_Spmv.hpp>
    #include <KokkosBatched_CG.hpp>
    #include <KokkosBatched_Krylov_Handle.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    // Preconditioned Matrix Operator for CG
    template <typename ScalarType, typename DeviceType>
    class PreconditionedMatrixOperator {
    public:
      using execution_space = typename DeviceType::execution_space;
      using memory_space = typename DeviceType::memory_space;
      using device_type = DeviceType;
      using value_type = ScalarType;
      
      using values_view_type = Kokkos::View<ScalarType**, Kokkos::LayoutRight, memory_space>;
      using int_view_type = Kokkos::View<int*, memory_space>;
      using vector_view_type = Kokkos::View<ScalarType**, Kokkos::LayoutRight, memory_space>;
      using diag_view_type = Kokkos::View<ScalarType**, Kokkos::LayoutRight, memory_space>;
      
    private:
      values_view_type _values;
      int_view_type _row_ptr;
      int_view_type _col_idx;
      diag_view_type _diag_values;
      KokkosBatched::JacobiPrec<diag_view_type> _preconditioner;
      vector_view_type _temp;
      int _n_batch;
      int _n_rows;
      int _n_cols;
      
    public:
      PreconditionedMatrixOperator(const values_view_type& values,
                                  const int_view_type& row_ptr,
                                  const int_view_type& col_idx,
                                  const diag_view_type& diag_values,
                                  const vector_view_type& temp)
        : _values(values), _row_ptr(row_ptr), _col_idx(col_idx),
          _diag_values(diag_values), _preconditioner(diag_values), _temp(temp) {
        _n_batch = values.extent(0);
        _n_rows = row_ptr.extent(0) - 1;
        _n_cols = _n_rows; // Assuming square matrix for CG
      }
      
      // Initialize the preconditioner
      template <typename MemberType, typename ArgMode>
      KOKKOS_INLINE_FUNCTION
      void initialize(const MemberType& member) {
        _preconditioner.template computeInverse<MemberType, ArgMode>(member);
      }
      
      // Apply the preconditioned operator to a vector: P^-1 * A * x
      template <typename MemberType, typename ArgMode>
      KOKKOS_INLINE_FUNCTION
      void apply(const MemberType& member,
                 const vector_view_type& X,
                 const vector_view_type& Y) const {
        // Y = A*X
        const ScalarType alpha = 1.0;
        const ScalarType beta = 0.0;
        
        // First apply A*X -> temp
        KokkosBatched::Spmv<MemberType, 
                           KokkosBatched::Trans::NoTranspose, 
                           ArgMode>
          ::template invoke<values_view_type, int_view_type, vector_view_type, vector_view_type, 0>
          (member, alpha, _values, _row_ptr, _col_idx, X, beta, _temp);
        
        member.team_barrier();
        
        // Then apply P^-1 * temp -> Y
        _preconditioner.template apply<KokkosBatched::Trans::NoTranspose, ArgMode, 0>
          (member, _temp, Y);
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
        
        // Create batched matrix in CRS format
        // Note: In a real application, you would fill this with your actual matrix data
        
        // Allocate CRS arrays
        Kokkos::View<int*, memory_space> row_ptr("row_ptr", n+1);
        Kokkos::View<int*, memory_space> col_idx("col_idx", n*nnz_per_row);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          values("values", batch_size, n*nnz_per_row);
        
        // Diagonal values for preconditioner
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          diag_values("diag_values", batch_size, n);
        
        // Temporary workspace
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          temp("temp", batch_size, n);
        
        // Initialize row_ptr and col_idx for a simple 5-point stencil (on host)
        auto row_ptr_host = Kokkos::create_mirror_view(row_ptr);
        auto col_idx_host = Kokkos::create_mirror_view(col_idx);
        auto values_host = Kokkos::create_mirror_view(values);
        auto diag_values_host = Kokkos::create_mirror_view(diag_values);
        
        int nnz = 0;
        for (int i = 0; i < n; ++i) {
          row_ptr_host(i) = nnz;
          
          // For simplicity, create a symmetric diagonally dominant matrix
          // Add diagonal element
          col_idx_host(nnz) = i;
          for (int b = 0; b < batch_size; ++b) {
            values_host(b, nnz) = 2.0 * nnz_per_row;  // Diagonally dominant
            diag_values_host(b, i) = values_host(b, nnz);  // Store diagonal for preconditioner
          }
          nnz++;
          
          // Add off-diagonal elements
          for (int k = 1; k < nnz_per_row; ++k) {
            int col = (i + k) % n;  // Simple pattern
            col_idx_host(nnz) = col;
            for (int b = 0; b < batch_size; ++b) {
              values_host(b, nnz) = -1.0 + 0.1 * b;  // Slightly different for each batch
            }
            nnz++;
          }
        }
        row_ptr_host(n) = nnz;  // Finalize row_ptr
        
        // Copy to device
        Kokkos::deep_copy(row_ptr, row_ptr_host);
        Kokkos::deep_copy(col_idx, col_idx_host);
        Kokkos::deep_copy(values, values_host);
        Kokkos::deep_copy(diag_values, diag_values_host);
        
        // Create preconditioned matrix operator
        using matrix_operator_type = PreconditionedMatrixOperator<scalar_type, execution_space::device_type>;
        matrix_operator_type A_op(values, row_ptr, col_idx, diag_values, temp);
        
        // Create RHS and solution vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          B("B", batch_size, n);  // RHS
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          X("X", batch_size, n);  // Solution
        
        // Initialize RHS with a simple pattern and X with zeros
        auto B_host = Kokkos::create_mirror_view(B);
        auto X_host = Kokkos::create_mirror_view(X);
        
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            B_host(b, i) = 1.0;  // Simple RHS
            X_host(b, i) = 0.0;  // Initial guess = 0
          }
        }
        
        Kokkos::deep_copy(B, B_host);
        Kokkos::deep_copy(X, X_host);
        
        // Create Krylov handle with solver parameters
        using krylov_handle_type = KokkosBatched::KrylovHandle<scalar_type, memory_space>;
        krylov_handle_type handle;
        
        handle.set_max_iteration(100);     // Maximum iterations
        handle.set_rel_residual_tol(1e-8); // Convergence tolerance
        handle.set_verbose(true);          // Print convergence info
        
        // Set workspace for CG
        handle.allocate_workspace(batch_size, n);
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        int team_size = policy_type::team_size_recommended(
            [](const int &, const int &) {}, 
            Kokkos::ParallelForTag());
        policy_type policy(batch_size, team_size);
        
        // Initialize the preconditioner
        Kokkos::parallel_for("InitPreconditioner", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            A_op.template initialize<typename policy_type::member_type, KokkosBatched::Mode::TeamVector>(member);
          }
        );
        
        // Solve the linear systems using preconditioned CG
        Kokkos::parallel_for("PreconditionedCG", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            // Get current batch's right-hand side and solution
            auto B_b = Kokkos::subview(B, b, Kokkos::ALL());
            auto X_b = Kokkos::subview(X, b, Kokkos::ALL());
            
            // Solve using CG with preconditioned operator
            KokkosBatched::CG<typename policy_type::member_type, 
                            KokkosBatched::Mode::TeamVector>
              ::invoke(member, A_op, B_b, X_b, handle);
          }
        );
        
        // Copy results back to host
        Kokkos::deep_copy(X_host, X);
        
        // Check results - iteration count should be reduced with preconditioning
        std::cout << "Solutions for first few entries of each batch:" << std::endl;
        for (int b = 0; b < std::min(batch_size, 3); ++b) {
          std::cout << "Batch " << b << ": [";
          for (int i = 0; i < std::min(n, 5); ++i) {
            std::cout << X_host(b, i) << " ";
          }
          std::cout << "...]" << std::endl;
        }
        
        // Verify solution by computing residual ||Ax - b||
        // In a real application, you would implement the residual check
      }
      Kokkos::finalize();
      return 0;
    }
