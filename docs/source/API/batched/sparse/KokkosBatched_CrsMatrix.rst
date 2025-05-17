KokkosBatched::CrsMatrix
########################

Defined in header: :code:`KokkosBatched_CrsMatrix.hpp`

.. code-block:: c++

    template <class ValuesViewType, class IntViewType>
    class CrsMatrix {
    public:
      using ScalarType = typename ValuesViewType::non_const_value_type;
      using MagnitudeType = typename Kokkos::ArithTraits<ScalarType>::mag_type;
      
      // Constructor
      KOKKOS_INLINE_FUNCTION
      CrsMatrix(const ValuesViewType& _values,
                const IntViewType& _row_ptr,
                const IntViewType& _colIndices);
      
      // Destructor
      KOKKOS_INLINE_FUNCTION
      ~CrsMatrix();
      
      // Apply matrix (team version)
      template <typename ArgTrans, typename ArgMode, typename MemberType, 
                typename XViewType, typename YViewType>
      KOKKOS_INLINE_FUNCTION
      void apply(const MemberType& member,
                 const XViewType& X,
                 const YViewType& Y,
                 MagnitudeType alpha = Kokkos::ArithTraits<MagnitudeType>::one(),
                 MagnitudeType beta = Kokkos::ArithTraits<MagnitudeType>::zero()) const;
      
      // Apply matrix (serial version)
      template <typename ArgTrans, typename XViewType, typename YViewType>
      KOKKOS_INLINE_FUNCTION
      void apply(const XViewType& X,
                 const YViewType& Y,
                 MagnitudeType alpha = Kokkos::ArithTraits<MagnitudeType>::one(),
                 MagnitudeType beta = Kokkos::ArithTraits<MagnitudeType>::zero()) const;
    };

The ``CrsMatrix`` class represents a batched compressed row storage matrix, where multiple matrices share the same sparsity pattern but have different values. This is particularly useful for sparse operations on multiple matrices that have identical structure, which is common in many applications including finite element analysis and graph algorithms.

The class provides a convenient interface to apply the sparse matrix to vectors (Spmv operation), with options for scaling and accumulation.

Parameters
==========

Constructor Parameters
----------------------

:_values: View containing the non-zero values of the matrices (batched)
:_row_ptr: View containing the row pointers (shared across all matrices)
:_colIndices: View containing the column indices (shared across all matrices)

Method Parameters (apply)
-------------------------

:member: Team execution policy instance (only for team version)
:X: Input vector view
:Y: Output vector view
:alpha: Scaling factor for the matrix-vector product (default: 1.0)
:beta: Scaling factor for the original Y values (default: 0.0)

Type Requirements
-----------------

- ``ValuesViewType`` must be a rank-2 view with dimensions (batch_size, nnz)
- ``IntViewType`` must be a rank-1 view containing integer indices
- ``XViewType`` and ``YViewType`` must be rank-2 views representing vectors with dimensions (batch_size, n)
- All views must be accessible in the execution space

Example
=======

.. code-block:: cpp

    #include <Kokkos_Core.hpp>
    #include <KokkosBatched_CrsMatrix.hpp>
    
    using execution_space = Kokkos::DefaultExecutionSpace;
    using memory_space = execution_space::memory_space;
    
    // Scalar type to use
    using scalar_type = double;
    
    int main(int argc, char* argv[]) {
      Kokkos::initialize(argc, argv);
      {
        // Matrix dimensions
        int batch_size = 10;  // Number of matrices
        int n = 100;          // Size of each matrix
        int nnz_per_row = 5;  // Non-zeros per row
        int nnz = n * nnz_per_row; // Total non-zeros
        
        // Create views for CRS format
        Kokkos::View<int*, memory_space> row_ptr("row_ptr", n+1);
        Kokkos::View<int*, memory_space> col_idx("col_idx", nnz);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          values("values", batch_size, nnz);
        
        // Create vectors
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          x("x", batch_size, n);
        Kokkos::View<scalar_type**, Kokkos::LayoutRight, memory_space> 
          y("y", batch_size, n);
        
        // Initialize on host
        auto row_ptr_host = Kokkos::create_mirror_view(row_ptr);
        auto col_idx_host = Kokkos::create_mirror_view(col_idx);
        auto values_host = Kokkos::create_mirror_view(values);
        auto x_host = Kokkos::create_mirror_view(x);
        auto y_host = Kokkos::create_mirror_view(y);
        
        // Initialize matrix sparsity pattern (shared across all matrices)
        int nnz_count = 0;
        for (int i = 0; i < n; ++i) {
          row_ptr_host(i) = nnz_count;
          
          // Add diagonal element
          col_idx_host(nnz_count) = i;
          nnz_count++;
          
          // Add off-diagonal elements
          for (int k = 1; k < nnz_per_row; ++k) {
            int col = (i + k) % n;  // Simple pattern
            col_idx_host(nnz_count) = col;
            nnz_count++;
          }
        }
        row_ptr_host(n) = nnz_count;  // Finalize row_ptr
        
        // Initialize matrix values (different for each batch)
        for (int b = 0; b < batch_size; ++b) {
          for (int j = 0; j < nnz; ++j) {
            // Diagonal elements are larger for stability
            int row = 0;
            while (j >= row_ptr_host(row+1)) row++;
            
            if (col_idx_host(j) == row) {
              values_host(b, j) = 10.0 + 0.1 * b;  // Diagonal
            } else {
              values_host(b, j) = -1.0 + 0.05 * b;  // Off-diagonal
            }
          }
        }
        
        // Initialize vectors
        for (int b = 0; b < batch_size; ++b) {
          for (int i = 0; i < n; ++i) {
            x_host(b, i) = 1.0;  // Simple vector
            y_host(b, i) = 0.0;  // Initial y value
          }
        }
        
        // Copy to device
        Kokkos::deep_copy(row_ptr, row_ptr_host);
        Kokkos::deep_copy(col_idx, col_idx_host);
        Kokkos::deep_copy(values, values_host);
        Kokkos::deep_copy(x, x_host);
        Kokkos::deep_copy(y, y_host);
        
        // Create CrsMatrix object
        using matrix_type = KokkosBatched::CrsMatrix<
            decltype(values), decltype(row_ptr)>;
        
        matrix_type A(values, row_ptr, col_idx);
        
        // Create team policy
        using policy_type = Kokkos::TeamPolicy<execution_space>;
        int team_size = policy_type::team_size_recommended(
            [](const int &, const int &) {}, 
            Kokkos::ParallelForTag());
        policy_type policy(batch_size, team_size);
        
        // Perform SpMV (y = A*x) using TeamVector mode for each batch
        Kokkos::parallel_for("BatchedCrsMatrixApply", policy,
          KOKKOS_LAMBDA(const typename policy_type::member_type& member) {
            const int b = member.league_rank();
            
            // Get current batch's vectors
            auto x_b = Kokkos::subview(x, b, Kokkos::ALL());
            auto y_b = Kokkos::subview(y, b, Kokkos::ALL());
            
            // Apply matrix: y = A*x
            A.template apply<KokkosBatched::Trans::NoTranspose, 
                           KokkosBatched::Mode::TeamVector>
              (member, x_b, y_b);
          }
        );
        
        // Copy results back to host
        Kokkos::deep_copy(y_host, y);
        
        // Print results for first few elements of first batch
        std::cout << "CrsMatrix SpMV Results for batch 0:" << std::endl;
        std::cout << "y = [";
        for (int i = 0; i < std::min(n, 5); ++i) {
          std::cout << y_host(0, i) << " ";
        }
        std::cout << "...]" << std::endl;
        
        // Example of serial usage
        if (batch_size <= 3) { // Only do this for small batch sizes
          // Reset y
          Kokkos::deep_copy(y, 0.0);
          
          // Perform serial SpMV on the host
          auto A_host = matrix_type(values, row_ptr, col_idx);
          
          for (int b = 0; b < batch_size; ++b) {
            auto x_b = Kokkos::subview(x, b, Kokkos::ALL());
            auto y_b = Kokkos::subview(y, b, Kokkos::ALL());
            
            // Apply matrix: y = A*x (serial version)
            A_host.template apply<KokkosBatched::Trans::NoTranspose>
              (x_b, y_b);
          }
          
          Kokkos::deep_copy(y_host, y);
          
          std::cout << "Serial CrsMatrix SpMV Results for batch 0:" << std::endl;
          std::cout << "y = [";
          for (int i = 0; i < std::min(n, 5); ++i) {
            std::cout << y_host(0, i) << " ";
          }
          std::cout << "...]" << std::endl;
        }
      }
      Kokkos::finalize();
      return 0;
    }
