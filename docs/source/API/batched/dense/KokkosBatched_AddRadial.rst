KokkosBatched::AddRadial
########################

Defined in header: :code:`KokkosBatched_AddRadial_Decl.hpp`

.. code-block:: c++

    struct SerialAddRadial {
      template <typename ScalarType, typename AViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const ScalarType tiny,
             const AViewType& A);
    };
    
    template <typename MemberType>
    struct TeamAddRadial {
      template <typename ScalarType, typename AViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const MemberType& member,
             const ScalarType tiny,
             const AViewType& A);
    };

The ``AddRadial`` operation adds tiny values to the diagonal elements of a matrix to ensure that the absolute values of the diagonals become larger. This is typically used to improve numerical stability in matrix operations like LU or Cholesky factorizations by preventing division by zero or near-zero values.

Mathematically, the operation performs:

.. math::

    A_{ii} = A_{ii} + \text{tiny} \quad \text{for all diagonal elements } i

Parameters
==========

:member: Team policy member (only for team version)
:tiny: Scalar value to add to diagonal elements
:A: Input/output matrix view to which diagonal values are added

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ScalarType`` must be a scalar type compatible with the element type of the matrix view
- ``AViewType`` must be a rank-2 view representing a square matrix
- The view must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_AddRadial_Decl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int n = 5;  // Square matrix size
        
        // Create matrix
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> A("A", n, n);
        
        // Initialize matrix on host
        auto A_host = Kokkos::create_mirror(A);
        
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            if (i == j) {
              // Set diagonal elements to small values to demonstrate AddRadial effect
              A_host(i, j) = 1.0e-8;
            } else {
              // Off-diagonal elements
              A_host(i, j) = i*n + j + 1;
            }
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        
        // Value to add to diagonal
        scalar_type tiny = 1.0e-2;
        
        // Apply AddRadial operation
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialAddRadial::invoke(tiny, A);
        });
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Print results to demonstrate the function worked
        std::cout << "Matrix after AddRadial operation:" << std::endl;
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            std::cout << A_host(i, j) << " ";
          }
          std::cout << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Team Version Example
--------------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_AddRadial_Decl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Batch and matrix dimensions
        int batch_size = 10; // Number of matrices
        int n = 5;           // Square matrix size
        
        // Create batched matrices
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A("A", batch_size, n, n);
        
        // Initialize on host
        auto A_host = Kokkos::create_mirror(A);
        
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
              if (i == j) {
                // Set diagonal elements to small values for each batch
                A_host(b, i, j) = 1.0e-8 * (b + 1);
              } else {
                // Off-diagonal elements
                A_host(b, i, j) = b*n*n + i*n + j + 1;
              }
            }
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(A, A_host);
        
        // Values to add to diagonals (one per batch)
        Kokkos::View<scalar_type*, memory_space> tiny("tiny", batch_size);
        auto tiny_host = Kokkos::create_mirror(tiny);
        
        for (int b = 0; b < batch_size; ++b) {
          tiny_host(b) = 1.0e-2 * (b + 1);
        }
        
        Kokkos::deep_copy(tiny, tiny_host);
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        policy_type policy(batch_size, Kokkos::AUTO);
        
        // Apply AddRadial to each matrix using team parallelism
        Kokkos::parallel_for("BatchedAddRadial", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            auto A_b = Kokkos::subview(A, b, Kokkos::ALL(), Kokkos::ALL());
            
            KokkosBatched::TeamAddRadial<typename policy_type::member_type>
              ::invoke(member, tiny(b), A_b);
          }
        );
        
        // Copy results back to host
        Kokkos::deep_copy(A_host, A);
        
        // Print results for first batch to demonstrate the function worked
        std::cout << "First batch matrix after AddRadial operation:" << std::endl;
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < n; ++j) {
            std::cout << A_host(0, i, j) << " ";
          }
          std::cout << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
