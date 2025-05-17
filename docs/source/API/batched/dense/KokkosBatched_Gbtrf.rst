KokkosBatched::Gbtrf
####################

Defined in header: :code:`KokkosBatched_Gbtrf.hpp`

.. code-block:: c++

    template <typename ArgAlgo>
    struct SerialGbtrf {
      template <typename ABViewType, typename PivViewType>
      KOKKOS_INLINE_FUNCTION
      static int
      invoke(const ABViewType& Ab,
             const PivViewType& piv,
             const int kl,
             const int ku,
             const int m = -1);
    };

The ``Gbtrf`` function computes an LU factorization of a general m-by-n band matrix A using partial pivoting with row interchanges. The factorization has the form:

.. math::

    A = P \cdot L \cdot U

where:

- :math:`P` is a permutation matrix
- :math:`L` is a lower triangular matrix with unit diagonal (unit diagonal elements are not stored)
- :math:`U` is an upper triangular matrix

The LU factorization is stored in the band format. In this format, a band matrix with kl subdiagonals and ku superdiagonals is stored in a two-dimensional array with (kl+ku+1) rows and n columns. The factored matrix components are stored as:

- The upper triangular band matrix U is stored in rows 1 to kl+ku+1
- The multipliers used during factorization are stored in rows kl+ku+2 to 2*kl+ku+1

Parameters
==========

:Ab: Input/output banded matrix view (stored in band format)
:piv: Output view for pivot indices
:kl: Number of subdiagonals within the band of A (kl ≥ 0)
:ku: Number of superdiagonals within the band of A (ku ≥ 0)
:m: Optional number of rows of matrix A (default -1, which means m = n)

Type Requirements
-----------------

- ``ArgAlgo`` must be ``KokkosBatched::Algo::Gbtrf::Unblocked`` for the unblocked algorithm
- ``ABViewType`` must be a rank-2 view containing the banded matrix in the appropriate format
- ``PivViewType`` must be a rank-1 view for storing the pivot indices
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Gbtrf.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions and band parameters
        int n = 10;        // Matrix dimension 
        int kl = 2;        // Number of subdiagonals
        int ku = 1;        // Number of superdiagonals
        int ldab = 2*kl+ku+1; // Leading dimension of band matrix
        
        // Create banded matrix and pivot vector
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> Ab("Ab", ldab, n);
        Kokkos::View<int*, memory_space> piv("piv", n);
        
        // Fill banded matrix with data
        auto Ab_host = Kokkos::create_mirror_view(Ab);
        
        // Initialize with a diagonally dominant matrix for stability
        for (int j = 0; j < n; ++j) {
          // Fill in diagonals and off-diagonals
          for (int i = std::max(0, j-ku); i <= std::min(n-1, j+kl); ++i) {
            // In band storage, element A(i,j) is stored at Ab(ku+i-j,j)
            int band_row = ku + i - j;
            
            if (i == j) {
              // Diagonal - make it dominant
              Ab_host(band_row, j) = 10.0;
            } else {
              // Off-diagonal
              Ab_host(band_row, j) = -1.0;
            }
          }
        }
        
        Kokkos::deep_copy(Ab, Ab_host);
        
        // Perform band LU factorization
        Kokkos::parallel_for(1, KOKKOS_LAMBDA(const int i) {
          KokkosBatched::SerialGbtrf<KokkosBatched::Algo::Gbtrf::Unblocked>::invoke(Ab, piv, kl, ku);
        });
        
        // Retrieve results to host
        auto piv_host = Kokkos::create_mirror_view_and_copy(Kokkos::HostSpace(), piv);
        Kokkos::deep_copy(Ab_host, Ab);
        
        // At this point, Ab_host contains the LU factorization in band format
        // and piv_host contains the pivot indices
        
        // Print the pivot indices
        std::cout << "Pivot indices:" << std::endl;
        for (int i = 0; i < n; ++i) {
          std::cout << piv_host(i) << " ";
        }
        std::cout << std::endl;
        
        // The factorization can be used with Gbtrs to solve linear systems
      }
      Kokkos::finalize();
      return 0;
    }

Batched Example
--------------

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_Gbtrf.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Batch and matrix dimensions
        int batch_size = 100;  // Number of matrices
        int n = 10;            // Matrix dimension 
        int kl = 2;            // Number of subdiagonals
        int ku = 1;            // Number of superdiagonals
        int ldab = 2*kl+ku+1;  // Leading dimension of band matrix
        
        // Create batched banded matrices and pivot vectors
        Kokkos::View<scalar_type***, Kokkos::LayoutRight, memory_space> 
          Ab("Ab", batch_size, ldab, n);
        Kokkos::View<int**, memory_space> piv("piv", batch_size, n);
        
        // Initialize matrices on host
        auto Ab_host = Kokkos::create_mirror_view(Ab);
        
        for (int b = 0; b < batch_size; ++b) {
          // Initialize each batch with a diagonally dominant matrix
          for (int j = 0; j < n; ++j) {
            for (int i = std::max(0, j-ku); i <= std::min(n-1, j+kl); ++i) {
              int band_row = ku + i - j;
              
              if (i == j) {
                // Diagonal - make it dominant
                Ab_host(b, band_row, j) = 10.0 + 0.1 * b;  // Slightly different per batch
              } else {
                // Off-diagonal
                Ab_host(b, band_row, j) = -1.0 - 0.01 * b;
              }
            }
          }
        }
        
        Kokkos::deep_copy(Ab, Ab_host);
        
        // Perform batch of LU factorizations
        Kokkos::parallel_for(batch_size, KOKKOS_LAMBDA(const int b) {
          auto Ab_b = Kokkos::subview(Ab, b, Kokkos::ALL(), Kokkos::ALL());
          auto piv_b = Kokkos::subview(piv, b, Kokkos::ALL());
          
          KokkosBatched::SerialGbtrf<KokkosBatched::Algo::Gbtrf::Unblocked>::invoke(Ab_b, piv_b, kl, ku);
        });
        
        // Results are now available in Ab and piv
        // Each Ab(b, :, :) contains an LU factorization
        // Each piv(b, :) contains the pivot indices for that factorization
      }
      Kokkos::finalize();
      return 0;
    }
