KokkosBatched::Identity
#######################

Defined in header `KokkosBatched_Identity.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/src/batched/KokkosBatched_Identity.hpp>`_

.. code-block:: c++

    class Identity {
    public:
      // Constructor
      KOKKOS_INLINE_FUNCTION
      Identity();
      
      // Destructor
      KOKKOS_INLINE_FUNCTION
      ~Identity();
      
      // Apply identity operator with team parallelism
      template <typename ArgTrans, typename ArgMode, int sameXY, 
                typename MemberType, typename XViewType, typename YViewType>
      KOKKOS_INLINE_FUNCTION
      void apply(const MemberType& member,
                 const XViewType& X,
                 const YViewType& Y) const;
      
      // Apply identity operator (serial version)
      template <typename ArgTrans, int sameXY, 
                typename XViewType, typename YViewType>
      KOKKOS_INLINE_FUNCTION
      void apply(const XViewType& X,
                 const YViewType& Y) const;
    };

The ``Identity`` class implements an identity operator that simply copies the input to the output. It's particularly useful as a "no-op" preconditioner in iterative solvers or as a placeholder when testing different operator configurations.

Mathematically, the identity operator :math:`I` applied to a vector :math:`x` simply returns the same vector:

.. math::

    I(x) = x

In the context of batched operations, the class efficiently copies batched vectors without additional computations.

Parameters
==========

Method Parameters (apply)
-------------------------

:member: Team execution policy instance (only for team version)
:X: Input vector view
:Y: Output vector view

Template Parameters
-------------------

- ``ArgTrans`` specifies the transposition mode (typically NoTranspose)
- ``ArgMode`` must be one of:
   - ``KokkosBatched::Mode::Serial`` for serial execution
   - ``KokkosBatched::Mode::Team`` for team-based execution
   - ``KokkosBatched::Mode::TeamVector`` for team-vector-based execution
- ``sameXY`` is a flag that optimizes when X and Y are the same view (0 = different, 1 = same)
- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``XViewType`` and ``YViewType`` must be views representing vectors
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Identity.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Vector dimensions
        int batch_size = 10;  // Number of vectors
        int n = 100;          // Vector length
        
        // Create input and output vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          x("x", batch_size, n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          y("y", batch_size, n);
        
        // Initialize x on host with some values
        auto x_host = Kokkos::create_mirror_view(x);
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            x_host(b, i) = b * 100 + i;  // Unique value per element
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(x, x_host);
        
        // Create identity operator
        KokkosBatched::Identity identity_op;
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        int team_size = policy_type::team_size_recommended(
          [](const int &, const int &) {}, 
          Kokkos::ParallelForTag());
        policy_type policy(batch_size, team_size);
        
        // Apply identity operator to copy x to y using team parallelism
        Kokkos::parallel_for("IdentityOperator", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            // Get current batch's vectors
            auto x_b = Kokkos::subview(x, b, Kokkos::ALL());
            auto y_b = Kokkos::subview(y, b, Kokkos::ALL());
            
            // Apply identity operator: y = x
            // sameXY = 0 indicates x and y are different views
            identity_op.template apply<KokkosBatched::Trans::NoTranspose, 
                                     KokkosBatched::Mode::TeamVector, 0>
              (member, x_b, y_b);
          }
        );
        
        // Copy results back to host
        auto y_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), y);
        
        // Verify that y contains the same values as x
        bool test_passed = true;
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            if (x_host(b, i) != y_host(b, i)) {
              test_passed = false;
              std::cout << "Mismatch at batch " << b << ", index " << i << ": " 
                        << x_host(b, i) << " vs " << y_host(b, i) << std::endl;
              break;
            }
          }
          if (!test_passed) break;
        }
        
        if (test_passed) {
          std::cout << "Identity operator test: PASSED" << std::endl;
          
          // Print first few values from the first batch
          std::cout << "First batch values: [";
          for (int i = 0; i < std::min(n, 5); ++i) {
            std::cout << y_host(0, i) << " ";
          }
          std::cout << "...]" << std::endl;
        } else {
          std::cout << "Identity operator test: FAILED" << std::endl;
        }
        
        // Demonstrate using Identity as a preconditioner
        std::cout << "\nDemonstrating Identity as a 'no-op' preconditioner:" << std::endl;
        
        // Reset y
        Kokkos::deep_copy(y, 0);
        
        // Create a simple preconditioned system using Identity
        // In a real application, you would use a more effective preconditioner
        Kokkos::parallel_for("PreconditionedSystem", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            // Get vectors for this batch
            auto x_b = Kokkos::subview(x, b, Kokkos::ALL());
            auto y_b = Kokkos::subview(y, b, Kokkos::ALL());
            
            // Apply "preconditioning" (which doesn't change anything)
            identity_op.template apply<KokkosBatched::Trans::NoTranspose, 
                                     KokkosBatched::Mode::TeamVector, 0>
              (member, x_b, y_b);
          }
        );
        
        // Copy results back to host
        Kokkos::deep_copy(y_host, y);
        
        // Verify again
        test_passed = true;
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            if (x_host(b, i) != y_host(b, i)) {
              test_passed = false;
              break;
            }
          }
          if (!test_passed) break;
        }
        
        if (test_passed) {
          std::cout << "Identity as preconditioner: PASSED" << std::endl;
        } else {
          std::cout << "Identity as preconditioner: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Using with Krylov Solvers
-------------------------

.. code-block:: cpp

    // Example of using Identity as a preconditioner with CG
    // Note: This is a simplified code snippet showing the pattern
    
    // Create operators
    matrix_operator_type A_op(values, row_ptr, col_idx);
    KokkosBatched::Identity I_op;
    
    // Create solver handle
    krylov_handle_type handle(batch_size, n_team);
    handle.set_max_iteration(100);
    handle.set_tolerance(1e-8);
    
    // Solve with "no preconditioning" (identity preconditioner)
    Kokkos::parallel_for("UnpreconditionedCG", policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const int b = member.league_rank();
        
        auto B_b = Kokkos::subview(B, b, Kokkos::ALL());
        auto X_b = Kokkos::subview(X, b, Kokkos::ALL());
        
        // Solve using CG without effective preconditioning
        KokkosBatched::CG<typename policy_type::member_type, 
                         KokkosBatched::Mode::TeamVector>
          ::invoke(member, A_op, B_b, X_b, handle);
      }
    );
