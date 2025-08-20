KokkosBatched::KrylovHandle
###########################

Defined in header: :code:`KokkosBatched_Krylov_Handle.hpp`

.. code-block:: c++

    template <class NormViewType, class IntViewType, class ViewType3D>
    class KrylovHandle {
    public:
      using norm_type = typename NormViewType::non_const_value_type;
      typedef ViewType3D ArnoldiViewType;
      typedef Kokkos::View<typename ViewType3D::non_const_value_type **, 
                          typename ViewType3D::array_layout,
                          typename ViewType3D::execution_space> TemporaryViewType;
    
      // Constructor
      KrylovHandle(int _batched_size, int _N_team, 
                  int _max_iteration = 200, bool _monitor_residual = false);
    
      // Workspace data members
      NormViewType residual_norms;
      IntViewType iteration_numbers;
      typename NormViewType::HostMirror residual_norms_host;
      typename IntViewType::HostMirror iteration_numbers_host;
      IntViewType first_index;
      IntViewType last_index;
      ArnoldiViewType Arnoldi_view;
      TemporaryViewType tmp_view;
    
      // Methods
      int get_number_of_systems_per_team();
      int get_number_of_teams();
      void reset();
      void synchronise_host();
      
      // Convergence checking
      KOKKOS_INLINE_FUNCTION bool is_converged() const;
      bool is_converged_host();
      KOKKOS_INLINE_FUNCTION bool is_converged(int batched_id) const;
      bool is_converged_host(int batched_id);
      
      // Solver parameters
      KOKKOS_INLINE_FUNCTION void set_tolerance(norm_type _tolerance);
      KOKKOS_INLINE_FUNCTION norm_type get_tolerance() const;
      KOKKOS_INLINE_FUNCTION void set_max_tolerance(norm_type _max_tolerance);
      KOKKOS_INLINE_FUNCTION norm_type get_max_tolerance() const;
      KOKKOS_INLINE_FUNCTION void set_max_iteration(int _max_iteration);
      KOKKOS_INLINE_FUNCTION int get_max_iteration() const;
      
      // Residual norms
      KOKKOS_INLINE_FUNCTION norm_type get_norm(int batched_id, int iteration_id) const;
      norm_type get_norm_host(int batched_id, int iteration_id);
      KOKKOS_INLINE_FUNCTION norm_type get_last_norm(int batched_id) const;
      norm_type get_last_norm_host(int batched_id);
      
      // Iteration count
      KOKKOS_INLINE_FUNCTION int get_iteration(int batched_id) const;
      int get_iteration_host(int batched_id);
      
      // Solver configuration
      KOKKOS_INLINE_FUNCTION void set_ortho_strategy(int _ortho_strategy);
      KOKKOS_INLINE_FUNCTION int get_ortho_strategy() const;
      KOKKOS_INLINE_FUNCTION void set_scratch_pad_level(int _scratch_pad_level);
      KOKKOS_INLINE_FUNCTION int get_scratch_pad_level() const;
      KOKKOS_INLINE_FUNCTION void set_compute_last_residual(bool _compute_last_residual);
      KOKKOS_INLINE_FUNCTION bool get_compute_last_residual() const;
      KOKKOS_INLINE_FUNCTION void set_memory_strategy(int _memory_strategy);
      KOKKOS_INLINE_FUNCTION int get_memory_strategy() const;
    };

The ``KrylovHandle`` class serves as an interface between the Krylov solvers (CG, GMRES) and the calling code. It provides:

1. Workspace allocation for the solvers
2. Storage for convergence history 
3. Configuration parameters for the solver behavior
4. Methods to check convergence status
5. Access to the solution process results (iterations, residuals)

This handle is a critical component when working with KokkosBatched iterative solvers, as it manages both the solver configuration and its execution state.

Template Parameters
===================

- ``NormViewType``: View type for storing residual norms
- ``IntViewType``: View type for storing integer information like iteration counts
- ``ViewType3D``: View type for 3D data like the Arnoldi basis in GMRES

Constructor Parameters
======================

:_batched_size: Total number of linear systems to solve
:_N_team: Number of systems to be solved per team
:_max_iteration: Maximum number of iterations (default: 200)
:_monitor_residual: Whether to store convergence history (default: false)

Key Methods
===========

- ``set_tolerance()``: Set the convergence tolerance
- ``set_max_iteration()``: Set the maximum number of iterations
- ``is_converged()``: Check if all systems have converged
- ``is_converged(int batched_id)``: Check if a specific system has converged
- ``get_iteration(int batched_id)``: Get the iteration count for a specific system
- ``get_norm(int batched_id, int iteration_id)``: Get a specific residual norm
- ``reset()``: Reset the handle for solving new systems
- ``synchronise_host()``: Update host copies of device data

Examples
========

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Krylov_Handle.hpp>
    #include <KokkosBatched_CG.hpp>
    #include <KokkosBatched_CrsMatrix.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    using view_type = Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space>;
    using int_view_type = Kokkos::View<int*, memory_space>;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Setup parameters
        int batch_size = 10;     // Number of systems to solve
        int n = 100;             // System size
        int max_iterations = 50; // Maximum iterations
        int n_team = 2;          // Systems per team
        bool monitor_residual = true; // Track convergence history
        
        // Create the Krylov handle
        using krylov_handle_type = KokkosBatched::KrylovHandle<view_type, int_view_type, view_type>;
        krylov_handle_type handle(batch_size, n_team, max_iterations, monitor_residual);
        
        // Configure the solver
        handle.set_tolerance(1e-8);       // Convergence tolerance
        handle.set_max_tolerance(1e-30);  // Numerical zero tolerance
        
        // Allocate workspace for the solver (example for CG)
        // For CG, we need temporary vectors for the solver
        view_type tmp_view("tmp_view", batch_size, 3 * n); // For p, Ap, r vectors
        handle.tmp_view = tmp_view;
        
        // Create a view for residual norms (for manual monitoring if needed)
        view_type res_norms("res_norms", batch_size, max_iterations + 2);
        handle.residual_norms = res_norms;
        
        // [Create and set up your matrix and vectors here]
        // ...
        
        // After solving, check convergence
        handle.synchronise_host();
        
        if (handle.is_converged_host()) {
          std::cout << "All systems converged!" << std::endl;
        } else {
          std::cout << "Some systems did not converge." << std::endl;
        }
        
        // Print iteration counts and final residuals
        for (int b = 0; b < batch_size; ++b) {
          if (handle.is_converged_host(b)) {
            std::cout << "System " << b << " converged in " 
                      << handle.get_iteration_host(b) << " iterations." << std::endl;
            std::cout << "  Final residual: " << handle.get_last_norm_host(b) << std::endl;
            
            // Print convergence history for first system
            if (b == 0) {
              std::cout << "  Convergence history: ";
              for (int i = 0; i <= handle.get_iteration_host(b); ++i) {
                std::cout << handle.get_norm_host(b, i) << " ";
              }
              std::cout << std::endl;
            }
          } else {
            std::cout << "System " << b << " did not converge after " 
                      << max_iterations << " iterations." << std::endl;
          }
        }
        
        // Reset the handle to solve another set of systems
        handle.reset();
        
        // [Solve another set of systems here]
        // ...
      }
      Kokkos::finalize();
      return 0;
    }

Complete Example with CG Solver
-------------------------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Krylov_Handle.hpp>
    #include <KokkosBatched_CG.hpp>
    #include <KokkosBatched_Spmv.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    using view_type = Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space>;
    using int_view_type = Kokkos::View<int*, memory_space>;
    
    // Simple matrix operator
    template <typename ScalarType, typename DeviceType>
    class DiagonalOperator {
    public:
      using execution_space = typename DeviceType::execution_space;
      using memory_space = typename DeviceType::memory_space;
      using device_type = DeviceType;
      using value_type = ScalarType;
      
      using vector_view_type = Kokkos::View<ScalarType**, Kokkos::LayoutRight, memory_space>;
      
    private:
      vector_view_type _diag;
      int _n_batch;
      int _n_size;
      
    public:
      DiagonalOperator(const vector_view_type& diag)
        : _diag(diag) {
        _n_batch = diag.extent(0);
        _n_size = diag.extent(1);
      }
      
      // Apply the operator: y = D*x (diagonal matrix)
      template <typename MemberType, typename ArgMode>
      KOKKOS_INLINE_FUNCTION
      void apply(const MemberType& member,
                 const vector_view_type& X,
                 const vector_view_type& Y) const {
        const int b = member.league_rank();
        
        // Apply diagonal matrix via parallel loop
        Kokkos::parallel_for(Kokkos::TeamVectorRange(member, _n_size),
          [&](const int i) {
            Y(b, i) = _diag(b, i) * X(b, i);
          }
        );
      }
      
      KOKKOS_INLINE_FUNCTION
      int n_rows() const { return _n_size; }
      
      KOKKOS_INLINE_FUNCTION
      int n_cols() const { return _n_size; }
      
      KOKKOS_INLINE_FUNCTION
      int n_batch() const { return _n_batch; }
    };
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Setup parameters
        int batch_size = 5;      // Number of systems to solve
        int n = 100;             // System size
        int max_iterations = 50; // Maximum iterations
        int n_team = 1;          // Systems per team
        bool monitor_residual = true; // Track convergence history
        
        // Create the Krylov handle
        using krylov_handle_type = KokkosBatched::KrylovHandle<view_type, int_view_type, view_type>;
        krylov_handle_type handle(batch_size, n_team, max_iterations, monitor_residual);
        
        // Configure the solver
        handle.set_tolerance(1e-6);       // Convergence tolerance
        handle.set_max_iteration(max_iterations);
        
        // Allocate workspace for CG
        handle.allocate_workspace(batch_size, n);
        
        // Create a simple diagonal system to solve
        view_type diag("diag", batch_size, n);
        view_type B("B", batch_size, n);   // RHS
        view_type X("X", batch_size, n);   // Solution
        
        // Initialize on host
        auto diag_host = Kokkos::create_mirror_view(diag);
        auto B_host = Kokkos::create_mirror_view(B);
        auto X_host = Kokkos::create_mirror_view(X);
        
        // Create a simple problem with different condition number per batch
        for (int b = 0; b < batch_size; ++b) {
          // Condition number increases with batch index
          double condition = 1.0 + b * 10.0;
          
          for (int i = 0; i < n; ++i) {
            // Create a diagonal matrix with entries from 1 to condition
            diag_host(b, i) = 1.0 + (i * (condition - 1.0)) / (n - 1);
            
            // Set RHS to all ones
            B_host(b, i) = 1.0;
            
            // Initial guess = 0
            X_host(b, i) = 0.0;
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(diag, diag_host);
        Kokkos::deep_copy(B, B_host);
        Kokkos::deep_copy(X, X_host);
        
        // Create diagonal operator
        using matrix_operator_type = DiagonalOperator<scalar_type, execution_space::device_type>;
        matrix_operator_type A_op(diag);
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        int team_size = policy_type::team_size_recommended(
          [](const int &, const int &) {}, 
          Kokkos::ParallelForTag());
        policy_type policy(batch_size, team_size);
        
        // Solve the linear systems using CG
        Kokkos::parallel_for("DiagonalCG", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            // Get current batch's right-hand side and solution
            auto B_b = Kokkos::subview(B, b, Kokkos::ALL());
            auto X_b = Kokkos::subview(X, b, Kokkos::ALL());
            
            // Solve using CG
            KokkosBatched::CG<typename policy_type::member_type, 
                             KokkosBatched::Mode::TeamVector>
              ::invoke(member, A_op, B_b, X_b, handle);
          }
        );
        
        // Check convergence
        handle.synchronise_host();
        
        std::cout << "Diagonal system convergence results:" << std::endl;
        for (int b = 0; b < batch_size; ++b) {
          std::cout << "System " << b << ": ";
          if (handle.is_converged_host(b)) {
            std::cout << "Converged in " << handle.get_iteration_host(b) 
                      << " iterations, final residual = " 
                      << handle.get_last_norm_host(b) << std::endl;
          } else {
            std::cout << "Did not converge." << std::endl;
          }
        }
        
        // Copy solutions back to host for verification
        Kokkos::deep_copy(X_host, X);
        
        // For a diagonal system, the exact solution is x_i = b_i / diag_i
        bool all_correct = true;
        for (int b = 0; b < batch_size; ++b) {
          if (!handle.is_converged_host(b)) continue;
          
          double max_error = 0.0;
          for (int i = 0; i < n; ++i) {
            double exact = B_host(b, i) / diag_host(b, i);
            double error = std::abs(X_host(b, i) - exact);
            max_error = std::max(max_error, error);
          }
          
          std::cout << "System " << b << " max error: " << max_error << std::endl;
          if (max_error > 1e-4) all_correct = false;
        }
        
        if (all_correct) {
          std::cout << "All converged solutions are correct!" << std::endl;
        } else {
          std::cout << "Some solutions have significant errors." << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
