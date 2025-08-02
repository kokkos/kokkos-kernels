KokkosBatched::ApplyHouseholder
###############################

Defined in header: :code:`KokkosBatched_ApplyHouseholder_Decl.hpp`

.. code-block:: c++

    // Serial version
    template <typename ArgSide>
    struct SerialApplyHouseholder {
      template <typename uViewType, typename tauViewType, typename AViewType, typename wViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const uViewType& u2,
             const tauViewType& tau,
             const AViewType& A,
             const wViewType& w);
    };
    
    // Team Vector version
    template <typename MemberType, typename ArgSide>
    struct TeamVectorApplyHouseholder {
      template <typename uViewType, typename tauViewType, typename AViewType, typename wViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType& member,
             const uViewType& u2,
             const tauViewType& tau,
             const AViewType& A,
             const wViewType& w);
    };

The ``ApplyHouseholder`` operation applies a Householder transformation to a matrix. This is a fundamental building block for many linear algebra operations, including QR factorization.

A Householder transformation is defined by a vector ``u`` and a scalar ``tau``. The Householder vector ``u`` is typically defined as :math:`u = x - \|x\|e_0`, where :math:`e_0` is the first unit vector, and :math:`\tau` is defined as :math:`\tau = 2 / (u^H u)`. It applies the transformation matrix :math:`H = I - \tau u u^T` (for real matrices) or :math:`H = I - \tau u u^H` (for complex matrices) to a target matrix.

Mathematically, when applied from the left side:

.. math::

    A := H \cdot A = (I - \tau u u^T) \cdot A

When applied from the right side:

.. math::

    A := A \cdot H = A \cdot (I - \tau u u^T)

Parameters
==========

:member: Team policy member (only for team version)
:u2: View containing the entries of u starting at the second element (size n-1 if u is of size n)
:tau: View containing the scalar tau
:A: Input/output matrix view to which the transformation is applied
:w: Workspace view for temporary calculations

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgSide`` must be one of:
   - ``KokkosBatched::Side::Left`` to apply the transformation from the left
   - ``KokkosBatched::Side::Right`` to apply the transformation from the right
- ``uViewType`` must be a rank-1 view containing the Householder vector elements
- ``tauViewType`` must be a scalar or a rank-0 view containing the scalar tau
- ``AViewType`` must be a rank-2 view representing the matrix to transform
- ``wViewType`` must be a rank-1 view with sufficient workspace for the computation
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_ApplyHouseholder_Decl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int m = 5;  // Number of rows
        int n = 3;  // Number of columns
        
        // Create matrices and vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A("A", m, n);
        Kokkos::View<scalar_type*, memory_space> u("u", m);   // Householder vector
        Kokkos::View<scalar_type, memory_space> tau("tau");   // Scalar tau
        Kokkos::View<scalar_type*, memory_space> w("w", n);   // Workspace
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror(A);
        auto u_host = Kokkos::create_mirror(u);
        auto tau_host = Kokkos::create_mirror(tau);
        
        // Initialize A with recognizable pattern
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            A_host(i, j) = (i + 1) * 10 + (j + 1);
          }
        }
        
        // Initialize Householder vector (first element is 1.0, rest are zeros by convention)
        u_host(0) = 1.0;
        for (int i = 1; i < m; ++i) {
          u_host(i) = 0.5 * i;
        }
        
        // Set tau
        tau_host() = 0.5;
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(u, u_host);
        Kokkos::deep_copy(tau, tau_host);
        
        // Apply Householder transformation from the left
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialApplyHouseholder<KokkosBatched::Side::Left>
            ::invoke(u, tau, A, w);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Print results to demonstrate the function worked
        std::cout << "Matrix after ApplyHouseholder transformation:" << std::endl;
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            std::cout << A_host(i, j) << " ";
          }
          std::cout << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
