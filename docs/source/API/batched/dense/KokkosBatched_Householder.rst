KokkosBatched::Householder
########################

Defined in header `KokkosBatched_Householder_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/batched/dense/src/KokkosBatched_Householder_Decl.hpp>`_

.. code:: c++

    template <typename ArgSide>
    struct SerialHouseholder {
      template <typename aViewType, typename tauViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const aViewType &a, const tauViewType &tau);
    };

    template <typename MemberType, typename ArgSide>
    struct TeamVectorHouseholder {
      template <typename aViewType, typename tauViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, 
                                              const aViewType &a, 
                                              const tauViewType &tau);
    };

Computes the Householder transformation for a vector. For each vector in the batch, computes:

.. math::

   v, \tau \text{ such that } H = I - \tau v v^T \text{ satisfies } Ha = \|a\|_2 e_1

where:

- :math:`a` is the input vector
- :math:`v` is the resulting Householder vector (stored in a, with v[0] = 1.0 implied)
- :math:`\tau` is the Householder scalar
- :math:`H` is the Householder transformation matrix
- :math:`e_1` is the first unit vector
- :math:`\|a\|_2` is the 2-norm of vector a

The Householder transformation is a fundamental operation used in many matrix factorizations, including QR and Hessenberg reduction. It creates a reflection that maps a vector to a multiple of the first unit vector.

Parameters
==========

:member: Team execution policy instance (not used in Serial mode)
:a: Input/output view for vectors (overwritten with Householder vectors)
:tau: Output view for Householder scalars

Type Requirements
----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``ArgSide`` must be one of:

  - ``Side::Left`` - for left side Householder transformation
  - ``Side::Right`` - for right side Householder transformation

- ``aViewType`` must be a rank-1 or rank-2 Kokkos View representing vectors
- ``tauViewType`` must be a rank-0 or rank-1 Kokkos View for scalars

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Householder_Decl.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 1000;  // Number of vectors
        int n = 5;              // Length of each vector
        
        // Create views for batched vectors and Householder scalars
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type> 
          a("a", batch_size, n),           // Input/output vectors
          a_copy("a_copy", batch_size, n); // Copy for verification
        
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, device_type>
          tau("tau", batch_size);          // Householder scalars
        
        // Fill vectors with data
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_vectors", policy, KOKKOS_LAMBDA(const int i) {
          // Initialize the i-th vector with a simple pattern
          for (int j = 0; j < n; ++j) {
            a(i, j) = j + 1.0;  // [1, 2, 3, 4, 5]
          }
          
          // Copy a for verification
          for (int j = 0; j < n; ++j) {
            a_copy(i, j) = a(i, j);
          }
          
          // Initialize tau to zero
          tau(i) = 0.0;
        });
        
        Kokkos::fence();
        
        // Compute Householder transformations for each vector
        Kokkos::parallel_for("batch_householder", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slices
          auto a_i = Kokkos::subview(a, i, Kokkos::ALL());
          auto tau_i = Kokkos::subview(tau, i);
          
          // Compute Householder vector and scalar
          KokkosBatched::SerialHouseholder<KokkosBatched::Side::Left>
            ::invoke(a_i, tau_i);
        });
        
        Kokkos::fence();
        
        // Copy results to host for verification
        auto a_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                         Kokkos::subview(a, 0, Kokkos::ALL()));
        auto a_copy_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                              Kokkos::subview(a_copy, 0, Kokkos::ALL()));
        auto tau_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), 
                                                           Kokkos::subview(tau, 0));
        
        // Verify the Householder transformation
        printf("Householder transformation results for first vector:\n");
        printf("Original vector a: [");
        for (int j = 0; j < n; ++j) {
          printf("%.1f%s", a_copy_host(j), (j < n-1) ? ", " : "");
        }
        printf("]\n");
        
        printf("Householder vector v: [1.0, ");  // v[0] = 1.0 is implied
        for (int j = 1; j < n; ++j) {
          printf("%.6f%s", a_host(j), (j < n-1) ? ", " : "");
        }
        printf("]\n");
        
        printf("Householder scalar tau: %.6f\n", tau_host());
        
        // Verify that the Householder transformation works correctly
        // H*a should be a multiple of e1 (i.e., [norm, 0, 0, ...])
        printf("\nVerifying H*a = ||a||*e1:\n");
        
        // Compute the norm of the original vector
        scalar_type norm = 0.0;
        for (int j = 0; j < n; ++j) {
          norm += a_copy_host(j) * a_copy_host(j);
        }
        norm = std::sqrt(norm);
        
        // Construct the full Householder vector (v[0] = 1.0)
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, Kokkos::HostSpace>
          v("v", n);
        
        v(0) = 1.0;
        for (int j = 1; j < n; ++j) {
          v(j) = a_host(j);
        }
        
        // Compute H*a = (I - tau*v*v^T)*a
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, Kokkos::HostSpace>
          Ha("Ha", n);
        
        // First compute v^T*a
        scalar_type vTa = 0.0;
        for (int j = 0; j < n; ++j) {
          vTa += v(j) * a_copy_host(j);
        }
        
        // Then compute H*a = a - tau * v * (v^T*a)
        for (int j = 0; j < n; ++j) {
          Ha(j) = a_copy_host(j) - tau_host() * v(j) * vTa;
        }
        
        // Check that Ha is a multiple of e1
        printf("H*a: [");
        for (int j = 0; j < n; ++j) {
          printf("%.6f%s", Ha(j), (j < n-1) ? ", " : "");
        }
        printf("]\n");
        
        printf("||a||*e1: [%.6f, ", norm);
        for (int j = 1; j < n; ++j) {
          printf("%.6f%s", 0.0, (j < n-1) ? ", " : "");
        }
        printf("]\n");
        
        // Check if the first element of Ha is ±||a||_2 and other elements are near zero
        bool correct = true;
        scalar_type error1 = std::abs(std::abs(Ha(0)) - norm);
        
        if (error1 > 1e-10) {
          printf("ERROR: First element of H*a (%.6f) does not match ±||a||_2 (%.6f)\n",
                 Ha(0), norm);
          correct = false;
        }
        
        for (int j = 1; j < n; ++j) {
          if (std::abs(Ha(j)) > 1e-10) {
            printf("ERROR: Element %d of H*a (%.6f) is not near zero\n", j, Ha(j));
            correct = false;
          }
        }
        
        if (correct) {
          printf("SUCCESS: H*a matches ±||a||_2 * e1 to within tolerance\n");
        }
        
        // Demonstrate TeamVectorHouseholder
        using team_policy_type = Kokkos::TeamPolicy<execution_space>;
        team_policy_type policy_team(batch_size, Kokkos::AUTO, Kokkos::AUTO);
        
        // Reset a to original values
        Kokkos::deep_copy(a, a_copy);
        
        // Compute Householder transformations using TeamVector variant
        Kokkos::parallel_for("batch_team_householder", policy_team, 
          KOKKOS_LAMBDA(const typename team_policy_type::member_type& member) {
            // Get batch index from team rank
            const int i = member.league_rank();
            
            // Extract batch slices
            auto a_i = Kokkos::subview(a, i, Kokkos::ALL());
            auto tau_i = Kokkos::subview(tau, i);
            
            // Compute Householder vector and scalar
            KokkosBatched::TeamVectorHouseholder<
              typename team_policy_type::member_type,  // MemberType
              KokkosBatched::Side::Left                // ArgSide
            >::invoke(member, a_i, tau_i);
          }
        );
        
        Kokkos::fence();
        
        // The results should be the same as with SerialHouseholder
        printf("\nTeamVectorHouseholder results should match SerialHouseholder results.\n");
      }
      Kokkos::finalize();
      return 0;
    }
