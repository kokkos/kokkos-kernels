KokkosBatched::SVD
##################

Defined in header: :code:`KokkosBatched_SVD_Decl.hpp`

.. code:: c++

    struct SerialSVD {
      // Version to compute full factorization: A == U * diag(s) * Vt
      template <typename AViewType, typename UViewType, typename VtViewType, typename SViewType, typename WViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(
          SVD_USV_Tag, const AViewType &A, const UViewType &U, const SViewType &s, 
          const VtViewType &Vt, const WViewType &W,
          typename AViewType::const_value_type tol = Kokkos::ArithTraits<typename AViewType::value_type>::zero());

      // Version which computes only singular values
      template <typename AViewType, typename SViewType, typename WViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(
          SVD_S_Tag, const AViewType &A, const SViewType &s, const WViewType &W,
          typename AViewType::const_value_type tol = Kokkos::ArithTraits<typename AViewType::value_type>::zero());
    };

Performs Singular Value Decomposition (SVD) on a general matrix. For each matrix A in the batch, computes the factorization:

.. math::

   A = U \Sigma V^T

where:

- :math:`U` is an orthogonal (or unitary) matrix of left singular vectors
- :math:`\Sigma` is a diagonal matrix of singular values
- :math:`V^T` is an orthogonal (or unitary) matrix of right singular vectors

The implementation provides two versions:
1. Full SVD - computes U, Σ, and V matrices
2. Partial SVD - computes only the singular values Σ

Parameters
==========

:A: Input matrix to decompose (contents are overwritten during computation)
:U: Output view for the left singular vectors (only for full SVD)
:s: Output view for the singular values
:Vt: Output view for the right singular vectors (only for full SVD)
:W: Workspace view for temporary calculations
:tol: Optional tolerance value for convergence (default = 0)

Type Requirements
-----------------

- ``AViewType`` must be a rank-2 Kokkos View representing a matrix
- ``UViewType`` must be a rank-2 Kokkos View with dimensions (m×m)
- ``VtViewType`` must be a rank-2 Kokkos View with dimensions (n×n)
- ``SViewType`` must be a rank-1 Kokkos View with dimension min(m,n)
- ``WViewType`` must be a rank-1 Kokkos View with sufficient workspace:
  
  - At least max(m,n) elements
  - Must be contiguous in memory (stride = 1)

- All view value types must be compatible with real-valued computations

Example
=======

.. code:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_SVD_Decl.hpp>

    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    using device_type = Kokkos::Device<execution_space, memory_space>;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Define matrix dimensions
        int m = 5;        // Number of rows
        int n = 3;        // Number of columns
        int min_dim = std::min(m, n);
        int max_dim = std::max(m, n);
        
        // Create views for input matrix and results on host first for simplicity
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace> 
          A_host("A_host", m, n),        // Input matrix 
          U_host("U_host", m, m),        // Left singular vectors
          Vt_host("Vt_host", n, n);      // Right singular vectors (transposed)
        
        Kokkos::View<scalar_type*, Kokkos::LayoutRight, Kokkos::HostSpace>
          s_host("s_host", min_dim),     // Singular values
          W_host("W_host", max_dim);     // Workspace
        
        // Initialize the matrix with a known pattern
        // We'll use a simple diagonal-dominant matrix
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            if (i == j) {
              A_host(i, j) = 10.0 + i;  // Diagonal elements with different values
            } else {
              A_host(i, j) = 0.1;        // Small off-diagonal elements
            }
          }
        }
        
        // Create a copy of A for verification later
        auto A_orig = Kokkos::create_mirror_view(A_host);
        Kokkos::deep_copy(A_orig, A_host);
        
        // Perform SVD on host
        KokkosBatched::SerialSVD::invoke(
          KokkosBatched::SVD_USV_Tag(), 
          A_host, U_host, s_host, Vt_host, W_host);
        
        // Print the singular values
        std::cout << "Singular values:" << std::endl;
        for (int i = 0; i < min_dim; ++i) {
          std::cout << "  σ" << i << " = " << s_host(i) << std::endl;
        }
        
        // Verify the decomposition: A = U * Σ * V^T
        // Create Σ as a matrix with singular values on diagonal
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace>
          Sigma("Sigma", m, n);
        
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            Sigma(i, j) = 0.0;
            if (i == j && i < min_dim) {
              Sigma(i, j) = s_host(i);
            }
          }
        }
        
        // Compute U * Σ
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace>
          USigma("USigma", m, n);
        
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            USigma(i, j) = 0.0;
            for (int k = 0; k < m; ++k) {
              USigma(i, j) += U_host(i, k) * Sigma(k, j);
            }
          }
        }
        
        // Compute (U * Σ) * V^T
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, Kokkos::HostSpace>
          USigmaVt("USigmaVt", m, n);
        
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            USigmaVt(i, j) = 0.0;
            for (int k = 0; k < n; ++k) {
              USigmaVt(i, j) += USigma(i, k) * Vt_host(k, j);
            }
          }
        }
        
        // Check the error between original A and reconstructed A
        double max_error = 0.0;
        for (int i = 0; i < m; ++i) {
          for (int j = 0; j < n; ++j) {
            double error = std::abs(A_orig(i, j) - USigmaVt(i, j));
            max_error = std::max(max_error, error);
          }
        }
        
        std::cout << "Maximum reconstruction error: " << max_error << std::endl;
        
        // Now demonstrate the GPU version with batched operations
        int batch_size = 10;  // Number of matrices in batch
        
        // Create device views
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, device_type> 
          A_dev("A_dev", batch_size, m, n),
          U_dev("U_dev", batch_size, m, m),
          Vt_dev("Vt_dev", batch_size, n, n);
        
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, device_type>
          s_dev("s_dev", batch_size, min_dim),
          W_dev("W_dev", batch_size, max_dim);
        
        // Initialize matrices (with the same pattern as before)
        Kokkos::RangePolicy<execution_space> policy(0, batch_size);
        
        Kokkos::parallel_for("init_matrices", policy, KOKKOS_LAMBDA(const int i) {
          for (int row = 0; row < m; ++row) {
            for (int col = 0; col < n; ++col) {
              if (row == col) {
                A_dev(i, row, col) = 10.0 + row;
              } else {
                A_dev(i, row, col) = 0.1;
              }
            }
          }
        });
        
        Kokkos::fence();
        
        // Perform batched SVD on device
        Kokkos::parallel_for("batched_svd", policy, KOKKOS_LAMBDA(const int i) {
          // Extract batch slices
          auto A_i = Kokkos::subview(A_dev, i, Kokkos::ALL(), Kokkos::ALL());
          auto U_i = Kokkos::subview(U_dev, i, Kokkos::ALL(), Kokkos::ALL());
          auto s_i = Kokkos::subview(s_dev, i, Kokkos::ALL());
          auto Vt_i = Kokkos::subview(Vt_dev, i, Kokkos::ALL(), Kokkos::ALL());
          auto W_i = Kokkos::subview(W_dev, i, Kokkos::ALL());
          
          // Perform SVD
          KokkosBatched::SerialSVD::invoke(
            KokkosBatched::SVD_USV_Tag(), 
            A_i, U_i, s_i, Vt_i, W_i);
        });
        
        Kokkos::fence();
        
        // Copy singular values from first batch to host for verification
        auto s_first_batch = Kokkos::create_mirror_view_and_copy(
          Kokkos::HostSpace(), Kokkos::subview(s_dev, 0, Kokkos::ALL()));
        
        std::cout << "\nSingular values from device (first batch):" << std::endl;
        for (int i = 0; i < min_dim; ++i) {
          std::cout << "  σ" << i << " = " << s_first_batch(i) << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
