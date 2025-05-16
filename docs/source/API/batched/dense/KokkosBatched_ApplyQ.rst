KokkosBatched::ApplyQ
##################

Defined in header `KokkosBatched_ApplyQ_Decl.hpp <https://github.com/kokkos/kokkos-kernels/blob/master/src/batched/KokkosBatched_ApplyQ_Decl.hpp>`_

.. code-block:: c++

    // Serial version (Left, NoTranspose)
    template <typename AViewType, typename tViewType, typename vViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION
    int
    ApplyQ(const AViewType& A,
           const tViewType& t,
           const vViewType& v,
           const wViewType& w);
           
    // Serial version (Left, Transpose)
    template <typename TrType, typename AViewType, typename tViewType, typename vViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION
    int
    ApplyQ(const TrType tr,
           const AViewType& A,
           const tViewType& t,
           const vViewType& v,
           const wViewType& w);
           
    // Team version (Left, NoTranspose)
    template <typename MemberType, typename AViewType, typename tViewType, typename vViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION
    int
    ApplyQ(const MemberType& member,
           const AViewType& A,
           const tViewType& t,
           const vViewType& v,
           const wViewType& w);
           
    // Team version (Left, Transpose)
    template <typename MemberType, typename TrType, typename AViewType, typename tViewType, typename vViewType, typename wViewType>
    KOKKOS_INLINE_FUNCTION
    int
    ApplyQ(const MemberType& member,
           const TrType tr,
           const AViewType& A,
           const tViewType& t,
           const vViewType& v,
           const wViewType& w);

The ``ApplyQ`` operation applies an orthogonal matrix Q (obtained from QR decomposition) to another matrix. Instead of forming Q explicitly, it applies Q using its factored representation as a series of Householder reflectors, which is more numerically stable and computationally efficient.

Mathematically, when applied from the left:

.. math::

    A := Q \cdot A \quad \text{or} \quad A := Q^T \cdot A

When applied from the right:

.. math::

    A := A \cdot Q \quad \text{or} \quad A := A \cdot Q^T

where Q is the orthogonal matrix implicitly represented by Householder vectors v and scaling factors t.

Parameters
==========

:member: Team execution policy instance (only for team version)
:tr: Optional parameter indicating transposition (typically Trans::Transpose)
:A: Input/output matrix to which Q will be applied
:t: View containing Householder scaling factors (tau)
:v: View containing Householder vectors
:w: Workspace view for computation

Type Requirements
----------------

- ``MemberType`` must be a Kokkos TeamPolicy member type
- ``TrType`` must be a transposition type (typically Trans::NoTranspose or Trans::Transpose)
- ``AViewType`` must be a rank-2 view representing the matrix to which Q is applied
- ``tViewType`` must be a rank-1 view containing Householder scaling factors
- ``vViewType`` must be a rank-2 view containing Householder vectors
- ``wViewType`` must be a rank-1 workspace view with sufficient size

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_ApplyQ_Decl.hpp>
    #include <KokkosBatched_QR_Decl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define matrix dimensions
        int n = 5;  // Matrix rows
        int m = 3;  // Matrix columns
        int k = 2;  // Number of columns in B
        
        // Create views for matrices and vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          A("A", n, m),        // Matrix for QR factorization
          v("v", n, m),        // Householder vectors from QR factorization
          C("C", n, k);        // Matrix to apply Q to
        
        Kokkos::View<scalar_type*, memory_space> 
          t("t", m),           // Householder scalars (tau)
          w("w", n);           // Workspace
        
        // Fill A with data
        auto A_host = Kokkos::create_mirror_view(A);
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < m; ++j) {
            A_host(i, j) = (i+1) * 0.1 + (j+1) * 0.01;
          }
        }
        Kokkos::deep_copy(A, A_host);
        
        // Fill C with data
        auto C_host = Kokkos::create_mirror_view(C);
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < k; ++j) {
            C_host(i, j) = (i+1) + (j+1) * 10;
          }
        }
        Kokkos::deep_copy(C, C_host);
        
        // Copy A to v for QR factorization
        Kokkos::deep_copy(v, A);
        
        // Perform QR factorization to get Householder vectors and scaling factors
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialQR<KokkosBatched::Algo::QR::Unblocked>::invoke(v, t);
        });
        
        // Save a copy of C for verification
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> C_orig("C_orig", n, k);
        Kokkos::deep_copy(C_orig, C);
        
        // Apply Q from the left to C
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          // C = Q * C
          KokkosBatched::SerialApplyQ<KokkosBatched::Side::Left, KokkosBatched::Trans::NoTranspose, 
                                       KokkosBatched::Algo::Level2::Unblocked>::invoke(C, t, v, w);
        });
        
        // Apply Q^T from the left to revert back to original C
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          // C = Q^T * C
          KokkosBatched::SerialApplyQ<KokkosBatched::Side::Left, KokkosBatched::Trans::Transpose, 
                                       KokkosBatched::Algo::Level2::Unblocked>::invoke(C, t, v, w);
        });
        
        // Verify that applying Q followed by Q^T returns to the original matrix
        Kokkos::deep_copy(C_host, C);
        auto C_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C_orig);
        
        // Check if C approximately matches C_orig
        bool test_passed = true;
        for (int i = 0; i < n; ++i) {
          for (int j = 0; j < k; ++j) {
            if (std::abs(C_host(i, j) - C_orig_host(i, j)) > 1e-10) {
              test_passed = false;
              std::cout << "Mismatch at (" << i << ", " << j << "): " 
                        << C_host(i, j) << " vs " << C_orig_host(i, j) << std::endl;
            }
          }
        }
        
        if (test_passed) {
          std::cout << "ApplyQ test: PASSED" << std::endl;
        } else {
          std::cout << "ApplyQ test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }

Batched Example with Team Version
--------------------------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_ApplyQ_Decl.hpp>
    #include <KokkosBatched_QR_Decl.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define dimensions
        int batch_size = 10;  // Number of matrices
        int n = 5;            // Matrix rows
        int m = 3;            // Matrix columns
        int k = 2;            // Number of columns in B
        
        // Create batched views
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          A("A", batch_size, n, m),  // Matrices for QR factorization
          v("v", batch_size, n, m),  // Householder vectors
          C("C", batch_size, n, k);  // Matrices to apply Q to
        
        Kokkos::View<scalar_type**, memory_space> 
          t("t", batch_size, m);     // Householder scalars (tau)
        
        Kokkos::View<scalar_type**, memory_space> 
          w("w", batch_size, n);     // Workspaces
        
        // Fill matrices with data
        auto A_host = Kokkos::create_mirror_view(A);
        auto C_host = Kokkos::create_mirror_view(C);
        
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
              A_host(b, i, j) = (b+1) * 0.01 + (i+1) * 0.1 + (j+1) * 0.01;
            }
            
            for (int j = 0; j < k; ++j) {
              C_host(b, i, j) = (b+1) * 0.1 + (i+1) + (j+1) * 10;
            }
          }
        }
        
        Kokkos::deep_copy(A, A_host);
        Kokkos::deep_copy(C, C_host);
        
        // Copy A to v for QR factorization
        Kokkos::deep_copy(v, A);
        
        // Save copy of C for verification
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          C_orig("C_orig", batch_size, n, k);
        Kokkos::deep_copy(C_orig, C);
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        policy_type policy(batch_size, Kokkos::AUTO);
        
        // Perform QR factorization
        Kokkos::parallel_for("QR_factorization", policy, 
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            auto v_b = Kokkos::subview(v, b, Kokkos::ALL(), Kokkos::ALL());
            auto t_b = Kokkos::subview(t, b, Kokkos::ALL());
            
            KokkosBatched::TeamQR<typename policy_type::member_type, 
                                  KokkosBatched::Algo::QR::Unblocked>
              ::invoke(member, v_b, t_b);
          }
        );
        
        // Apply Q to C
        Kokkos::parallel_for("Apply_Q", policy, 
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            auto v_b = Kokkos::subview(v, b, Kokkos::ALL(), Kokkos::ALL());
            auto t_b = Kokkos::subview(t, b, Kokkos::ALL());
            auto C_b = Kokkos::subview(C, b, Kokkos::ALL(), Kokkos::ALL());
            auto w_b = Kokkos::subview(w, b, Kokkos::ALL());
            
            KokkosBatched::TeamApplyQ<typename policy_type::member_type,
                                     KokkosBatched::Side::Left,
                                     KokkosBatched::Trans::NoTranspose,
                                     KokkosBatched::Algo::Level2::Unblocked>
              ::invoke(member, v_b, t_b, C_b, w_b);
          }
        );
        
        // Apply Q^T to C (to verify)
        Kokkos::parallel_for("Apply_QT", policy, 
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            auto v_b = Kokkos::subview(v, b, Kokkos::ALL(), Kokkos::ALL());
            auto t_b = Kokkos::subview(t, b, Kokkos::ALL());
            auto C_b = Kokkos::subview(C, b, Kokkos::ALL(), Kokkos::ALL());
            auto w_b = Kokkos::subview(w, b, Kokkos::ALL());
            
            KokkosBatched::TeamApplyQ<typename policy_type::member_type,
                                     KokkosBatched::Side::Left,
                                     KokkosBatched::Trans::Transpose,
                                     KokkosBatched::Algo::Level2::Unblocked>
              ::invoke(member, v_b, t_b, C_b, w_b);
          }
        );
        
        // Verify results
        Kokkos::deep_copy(C_host, C);
        auto C_orig_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), C_orig);
        
        bool test_passed = true;
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            for (int j = 0; j < k; ++j) {
              if (std::abs(C_host(b, i, j) - C_orig_host(b, i, j)) > 1e-10) {
                test_passed = false;
                std::cout << "Batch " << b << " mismatch at (" << i << ", " << j << "): " 
                          << C_host(b, i, j) << " vs " << C_orig_host(b, i, j) << std::endl;
                break;
              }
            }
            if (!test_passed) break;
          }
          if (!test_passed) break;
        }
        
        if (test_passed) {
          std::cout << "Batched ApplyQ test: PASSED" << std::endl;
        } else {
          std::cout << "Batched ApplyQ test: FAILED" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
