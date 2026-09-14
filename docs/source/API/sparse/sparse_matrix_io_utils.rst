KokkosSparse::Impl Sparse Matrix I/O Utilities
###############################################

Defined in header: :code:`KokkosSparse_IOUtils.hpp`

.. code:: c++

  template <typename ScalarType, typename OrdinalType, typename SizeType>
  void kk_sparseMatrix_generate(OrdinalType nrows, OrdinalType ncols, SizeType &nnz,
                                OrdinalType row_size_variance, OrdinalType bandwidth,
                                ScalarType *&values, SizeType *&rowPtr, OrdinalType *&colInd,
                                OrdinalType block_elem_count = 1);

  template <typename ScalarType, typename OrdinalType, typename SizeType>
  void kk_sparseMatrix_generate_lower_upper_triangle(char uplo, OrdinalType nrows,
                                                     OrdinalType ncols, SizeType &nnz,
                                                     OrdinalType row_size_variance,
                                                     OrdinalType bandwidth,
                                                     ScalarType *&values, SizeType *&rowPtr,
                                                     OrdinalType *&colInd);

  template <typename ScalarType, typename OrdinalType, typename SizeType>
  void kk_diagonally_dominant_sparseMatrix_generate(OrdinalType nrows, OrdinalType ncols,
                                                    SizeType &nnz, OrdinalType row_size_variance,
                                                    OrdinalType bandwidth,
                                                    ScalarType *&values, SizeType *&rowPtr,
                                                    OrdinalType *&colInd);

Utility functions for generating sparse matrices in CRS (Compressed Row Storage) format. These functions
are primarily used for testing and benchmarking purposes.

- ``kk_sparseMatrix_generate``: Generates a random sparse matrix with specified characteristics.

- ``kk_sparseMatrix_generate_lower_upper_triangle``: Generates a lower or upper triangular sparse matrix.

- ``kk_diagonally_dominant_sparseMatrix_generate``: Generates a diagonally dominant sparse matrix.

Parameters
==========

:nrows: Number of rows in the sparse matrix.

:ncols: Number of columns in the sparse matrix.

:nnz: (in/out) Approximate number of nonzeros. On input, specifies the desired number of nonzeros.
      On output, contains the actual number of nonzeros generated.

:uplo: (``generate_lower_upper_triangle`` only) Character specifying 'L' for lower triangular
       or 'U' for upper triangular matrix.

:row_size_variance: Controls variance in the number of nonzeros per row.

:bandwidth: Controls the bandwidth of the matrix. Nonzeros are placed within a diagonal band
            around the main diagonal.

:values: (output) Pointer to dynamically allocated array of scalar values. Memory is allocated by the function.

:rowPtr: (output) Pointer to dynamically allocated array of row pointers for CRS format.
         Memory is allocated by the function.

:colInd: (output) Pointer to dynamically allocated array of column indices for CRS format.
         Memory is allocated by the function.

:block_elem_count: (``kk_sparseMatrix_generate`` only) Number of elements in block matrices
                   (default: 1 for standard sparse matrices).

Type Requirements
-----------------

- ``ScalarType`` must be a numeric type compatible with random number generation.
  Supports floating-point types (float, double, long double) and complex types.

- ``OrdinalType`` must be an integer type for matrix dimensions and indices.

- ``SizeType`` must be an integer type for array sizes.

Notes
=====

- All three functions allocate memory dynamically and return pointers. The caller is responsible
  for deallocating this memory.

- The generated matrices are deterministic and reproducible across calls with the same parameters.

- Matrix values are sampled uniformly from the range (-50, 50) for real types, or
  (-50 - 50i, 50 + 50i) for complex types.

- For ``generate_lower_upper_triangle``, the actual structure (lower or upper triangular)
  is determined by the ``uplo`` parameter.

Example
=======

.. code:: c++

  #include <KokkosSparse_IOUtils.hpp>

  void example() {
    using scalar_type = double;
    using ordinal_type = int;
    using size_type = size_t;

    ordinal_type nrows = 100;
    ordinal_type ncols = 100;
    size_type nnz = 1000;

    scalar_type *values = nullptr;
    size_type *rowPtr = nullptr;
    ordinal_type *colInd = nullptr;

    // Generate a random sparse matrix
    KokkosSparse::Impl::kk_sparseMatrix_generate(
        nrows, ncols, nnz, 10, 20, values, rowPtr, colInd);

    // Use the generated matrix...

    // Clean up allocated memory
    delete[] values;
    delete[] rowPtr;
    delete[] colInd;
  }
