KokkosBatched::Krylov_Solvers
#############################

Defined in header `KokkosBatched_Krylov_Solvers.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/src/batched/KokkosBatched_Krylov_Solvers.hpp>`_

This header provides the core implementation structures for various Krylov subspace methods with different parallelism models. These are the implementations behind the higher-level interfaces like ``KokkosBatched::CG`` and ``KokkosBatched::GMRES``.

GMRES Implementations
=====================

.. code-block:: c++

    struct SerialGMRES {
      template <typename OperatorType, typename VectorViewType, 
                typename PrecOperatorType, typename KrylovHandleType>
      KOKKOS_INLINE_FUNCTION
      static int invoke(const OperatorType& A,
                        const VectorViewType& _B,
                        const VectorViewType& _X,
                        const PrecOperatorType& P,
                        const KrylovHandleType& handle,
                        const int GMRES_id);
                        
      template <typename OperatorType, typename VectorViewType, 
                typename KrylovHandleType>
      KOKKOS_INLINE_FUNCTION
      static int invoke(const OperatorType& A,
                        const VectorViewType& _B,
                        const VectorViewType& _X,
                        const KrylovHandleType& handle);
    };
    
    template <typename MemberType>
    struct TeamGMRES {
      // Various overloaded implementations
      // ...
    };
    
    template <typename MemberType>
    struct TeamVectorGMRES {
      // Various overloaded implementations
      // ...
    };

The GMRES (Generalized Minimal Residual Method) algorithm solves linear systems through an iterative process that minimizes the residual norm over a Krylov subspace. These implementations provide different parallelism models:

1. ``SerialGMRES``: Sequential implementation with no internal parallelism
2. ``TeamGMRES``: Implementation using team-based parallelism for matrix operations
3. ``TeamVectorGMRES``: Implementation using both team and vector parallelism for improved performance

Each implementation offers versions with and without preconditioning, as well as versions that accept custom workspace views for performance optimization.

CG Implementations
==================

.. code-block:: c++

    template <typename MemberType>
    struct TeamCG {
      template <typename OperatorType, typename VectorViewType, 
                typename KrylovHandleType, typename TMPViewType,
                typename TMPNormViewType>
      KOKKOS_INLINE_FUNCTION
      static int invoke(const MemberType& member,
                        const OperatorType& A,
                        const VectorViewType& _B,
                        const VectorViewType& _X,
                        const KrylovHandleType& handle,
                        const TMPViewType& _TMPView,
                        const TMPNormViewType& _TMPNormView);
                        
      template <typename OperatorType, typename VectorViewType, 
                typename KrylovHandleType>
      KOKKOS_INLINE_FUNCTION
      static int invoke(const MemberType& member,
                        const OperatorType& A,
                        const VectorViewType& _B,
                        const VectorViewType& _X,
                        const KrylovHandleType& handle);
    };
    
    template <typename MemberType>
    struct TeamVectorCG {
      // Similar overloads as TeamCG
      // ...
    };

The Conjugate Gradient (CG) method is an iterative algorithm for solving symmetric positive definite linear systems. These implementations provide different parallelism models:

1. ``TeamCG``: Implementation using team-based parallelism
2. ``TeamVectorCG``: Implementation using both team and vector parallelism for improved performance

Each implementation has overloads that accept custom workspace views for optimized memory usage.

Parameters Common to All Solvers
================================

:member: Team execution policy instance (not used in Serial versions)
:A: Operator representing the matrix of the linear system
:_B: View containing the right-hand sides
:_X: View containing the initial guess on input and the solution on output
:P: Optional preconditioner operator
:handle: Krylov handle containing solver parameters and workspace

Return
------

- ``0`` if the operation is successful
- If there are convergence issues, non-zero error codes may be returned

Implementation Notes
====================

These low-level solver implementations are used internally by the higher-level interfaces like ``KokkosBatched::CG`` and ``KokkosBatched::GMRES``. They provide the core algorithms while the higher-level interfaces provide a more user-friendly API that automatically dispatches to the appropriate implementation based on the requested execution mode.

Key features of these implementations:

1. **Preconditioning Support**: The GMRES implementations have overloads that accept a preconditioner operator to improve convergence.

2. **Custom Workspace**: Overloads accepting custom workspace views allow advanced users to manage memory allocation for better performance.

3. **Convergence Tracking**: Integration with the KrylovHandle allows tracking of convergence history, iteration counts, and residual norms.

4. **Arnoldi Process**: The GMRES implementations use the Arnoldi process to construct an orthogonal basis for the Krylov subspace.

5. **Parallelism Models**: Different implementations cater to different parallelism needs:
   - Serial: No internal parallelism
   - Team: Uses TeamThreadRange for parallelism
   - TeamVector: Uses both TeamThreadRange and ThreadVectorRange for hierarchical parallelism

Advanced Usage Example
======================

.. code-block:: cpp

    // Example of directly using the TeamVectorGMRES implementation
    // Note: Most users should use the higher-level GMRES interface instead
    
    // Create custom workspace
    view_type Arnoldi_view("Arnoldi", batch_size, max_iter, n + max_iter + 3);
    view_type tmp_view("tmp", batch_size, n + max_iter + 3);
    
    // Create team policy
    policy_type policy(batch_size, team_size);
    
    // Solve using direct implementation
    Kokkos::parallel_for(policy,
      KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
        const int b = member.league_rank();
        
        auto B_b = Kokkos::subview(B, b, Kokkos::ALL());
        auto X_b = Kokkos::subview(X, b, Kokkos::ALL());
        
        // Use preconditioned version with custom workspace
        KokkosBatched::TeamVectorGMRES<typename policy_type::member_type>
          ::invoke(member, A_op, B_b, X_b, precond, handle, 
                  Arnoldi_view, tmp_view);
      }
    );
