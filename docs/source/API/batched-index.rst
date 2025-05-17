API: Batched
============

.. toctree::
   :maxdepth: 2
   :hidden:

   batched/dense/KokkosBatched_AddRadial

Overview
--------

KokkosBatched is a high-performance, portable library for batched linear algebra operations on manycore architectures. It addresses a critical performance gap in scientific computing: the ability to solve many small-to-medium sized linear algebra problems simultaneously. This computational pattern arises in numerous applications, including finite element analysis, computational fluid dynamics, quantum chemistry, machine learning, and uncertainty quantification.

The central premise of KokkosBatched is that for many small problems, traditional linear algebra approaches that process one problem at a time are suboptimal on modern hardware. By processing multiple independent problems simultaneously, KokkosBatched achieves:

1. Higher arithmetic intensity and improved data locality
2. Better utilization of vector / SIMD units
3. Reduced kernel launch overhead
4. Increased parallelism at multiple levels

The library is built on Kokkos Core, providing portability across diverse architectures including multicore CPUs, NVIDIA and AMD GPUs, and Intel XPUs, without requiring architecture-specific code rewrites.

Mathematical Foundation
-----------------------

Batched Representation
^^^^^^^^^^^^^^^^^^^^^^

At a fundamental level, the batched paradigm involves operating on tensor expressions where one dimension represents the batch index. For instance, the mathematical formulation of batched matrix-matrix multiplication is:

.. math::

   C_{b,i,j} = \sum_{k=0}^{n-1} A_{b,i,k} \cdot B_{b,k,j} \quad \forall b \in [0, \text{batch\_size} - 1]

Where :math:`b` represents the batch index, and the computation for each batch element is independent of others.

For sparse matrices, the representation becomes more nuanced. KokkosBatched offers an efficient approach where matrices with identical sparsity patterns (but different values) are stored together:

.. math::

   A_b = \begin{pmatrix} 
   a_{b,1,1} & a_{b,1,2} & \cdots & a_{b,1,n} \\
   a_{b,2,1} & a_{b,2,2} & \cdots & a_{b,2,n} \\
   \vdots & \vdots & \ddots & \vdots \\
   a_{b,m,1} & a_{b,m,2} & \cdots & a_{b,m,n}
   \end{pmatrix}

The library utilizes specialized data structures that minimize memory footprint while maximizing computational efficiency.

Hierarchical Parallelism
^^^^^^^^^^^^^^^^^^^^^^^^

KokkosBatched employs a sophisticated hierarchical parallelism model that maps efficiently to modern hardware architectures. This model incorporates:

1. **Batch-level parallelism**: Different batch elements are distributed across computational resources (e.g., thread blocks on GPUs, cores on CPUs)
2. **Team-level parallelism**: Multiple threads cooperate on individual problems
3. **Vector-level parallelism**: SIMD/vector instructions are leveraged for computational kernels

This can be mathematically conceptualized as a decomposition of the computation space. For an operation like batched matrix-vector multiplication:

.. math::

   y_{b,i} = \sum_{j=0}^{n-1} A_{b,i,j} \cdot x_{b,j}

Where:
- The batch dimension :math:`b \in [0, \text{batch\_size} - 1]` is mapped to the coarsest parallelism level
- The row dimension :math:`i \in [0, m - 1]` is potentially mapped to team-level parallelism
- The inner products are potentially mapped to vector-level parallelism

The library provides distinct implementation variants for each parallelism model:

- ``Serial``: No internal parallelism
- ``Team``: Team-based parallelism
- ``TeamVector``: Both team and vector parallelism

For example, the invocation pattern for a batched matrix-vector multiplication might follow:

.. math::

   \text{KokkosBatched::Spmv}<\text{MemberType}, \text{Trans::NoTranspose}, \text{Mode::TeamVector}>::\text{invoke}(\text{member}, \alpha, \text{values}, \text{row\_ptr}, \text{col\_idx}, x, \beta, y)

This mathematical formulation naturally extends to complex operations like iterative solvers, where multiple complex linear algebra operations are orchestrated to solve batched systems of equations.

Dense Linear Algebra Operations
-------------------------------

KokkosBatched offers a comprehensive suite of dense linear algebra operations organized in a BLAS-like hierarchy:

BLAS Level 1 (Vector-Vector Operations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These operations involve vectors and have computational complexity of :math:`O(n)`.

1. **AddRadial**: Adds scalar values to diagonal elements

   .. math::
   
      A_{ii} = A_{ii} + \alpha \quad \forall i

2. **Norm2**: Computes the Euclidean norm of a vector
   
   .. math::
   
      \|x\|_2 = \sqrt{\sum_{i=0}^{n-1} |x_i|^2}

3. **Axpy**: Scaled vector addition
   
   .. math::
   
      y_i = \alpha \cdot x_i + y_i \quad \forall i

4. **Dot**: Inner product of two vectors
   
   .. math::
   
      \text{dot}(x, y) = \sum_{i=0}^{n-1} x_i \cdot y_i

BLAS Level 2 (Matrix-Vector Operations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These operations involve a matrix and vector, with computational complexity of :math:`O(n^2)`.

1. **Gemv**: General matrix-vector multiplication
   
   .. math::
   
      y_i = \alpha \sum_{j=0}^{n-1} A_{ij} \cdot x_j + \beta \cdot y_i \quad \forall i

2. **Trsv**: Triangular solve
   
   .. math::
   
      A \cdot x = b

   Where :math:`A` is triangular, solved via forward or backward substitution.

3. **Syr**: Symmetric rank-1 update
   
   .. math::
   
      A = \alpha \cdot x \cdot x^T + A

BLAS Level 3 (Matrix-Matrix Operations)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

These involve matrices with computational complexity of :math:`O(n^3)`.

1. **Gemm**: General matrix-matrix multiplication
   
   .. math::
   
      C_{ij} = \alpha \sum_{k=0}^{n-1} A_{ik} \cdot B_{kj} + \beta \cdot C_{ij} \quad \forall i,j

2. **Trsm**: Triangular solve with multiple right-hand sides
   
   .. math::
   
      A \cdot X = B

   Where :math:`A` is triangular and :math:`X, B` are matrices.

Matrix Factorizations
^^^^^^^^^^^^^^^^^^^^^

KokkosBatched provides essential matrix factorization operations that decompose matrices into specialized forms:

1. **LU Factorization** (Getrf/Gbtrf/Pbtrf): Decomposes a matrix into lower and upper triangular factors

   .. math::
   
      A = P \cdot L \cdot U

   Where :math:`P` is a permutation matrix, :math:`L` is lower triangular with unit diagonal, and :math:`U` is upper triangular.

2. **QR Factorization**: Decomposes a matrix into an orthogonal matrix and an upper triangular matrix

   .. math::
   
      A = Q \cdot R

   Where :math:`Q` is orthogonal and :math:`R` is upper triangular.

3. **UTV Factorization**: Rank-revealing decomposition useful for ill-conditioned matrices

   .. math::
   
      A = U \cdot T \cdot V^T

   Where :math:`U` and :math:`V` are orthogonal, and :math:`T` is triangular.

4. **Cholesky Factorization**: For symmetric positive definite matrices

   .. math::
   
      A = L \cdot L^T

   Where :math:`L` is lower triangular.

Sparse Linear Algebra Operations
--------------------------------

KokkosBatched extends the batched paradigm to sparse matrices, offering significant performance advantages for applications with many sparse operations. The fundamental abstraction is the batched CRS (Compressed Row Storage) matrix, where matrices with identical sparsity patterns but different values are stored efficiently.

Sparse Matrix Representation
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

For a batch of :math:`B` sparse matrices sharing the same sparsity pattern, KokkosBatched uses:

1. A single row pointer array ``row_ptr`` (size :math:`n+1`)
2. A single column indices array ``col_idx`` (size :math:`nnz`)
3. A values array ``values`` (size :math:`B \times nnz`)

This representation minimizes memory consumption while enabling efficient computational kernels.

Sparse Matrix Operations
^^^^^^^^^^^^^^^^^^^^^^^^

1. **Spmv**: Sparse matrix-vector multiplication
   
   .. math::
   
      y_{b,i} = \alpha_b \sum_{j=0}^{n-1} A_{b,i,j} \cdot x_{b,j} + \beta_b \cdot y_{b,i}

   Where the summation only includes non-zero elements of the sparse matrix :math:`A_b`.

2. **CrsMatrix**: A class encapsulating the batched CRS matrix with optimized apply methods
   
   .. math::
   
      \text{apply}: Y = \alpha \cdot A \cdot X + \beta \cdot Y

   This operation leverages specialized kernels for different parallelism models.

Iterative Solvers
-----------------

KokkosBatched provides state-of-the-art iterative solvers for batched linear systems, essential for applications requiring the solution of many independent equations.

Krylov Subspace Methods
^^^^^^^^^^^^^^^^^^^^^^^

1. **Conjugate Gradient (CG)**: For symmetric positive definite systems

   The CG method minimizes the energy norm :math:`\|x - x^*\|_A` where :math:`x^*` is the exact solution, and :math:`\|x\|_A = \sqrt{x^T A x}`. Each iteration involves:

   .. math::
   
      \begin{align}
      r^{(k)} &= b - A x^{(k)} \\
      \alpha_k &= \frac{r^{(k)T} r^{(k)}}{p^{(k)T} A p^{(k)}} \\
      x^{(k+1)} &= x^{(k)} + \alpha_k p^{(k)} \\
      r^{(k+1)} &= r^{(k)} - \alpha_k A p^{(k)} \\
      \beta_k &= \frac{r^{(k+1)T} r^{(k+1)}}{r^{(k)T} r^{(k)}} \\
      p^{(k+1)} &= r^{(k+1)} + \beta_k p^{(k)}
      \end{align}

2. **Generalized Minimal Residual (GMRES)**: For general non-symmetric systems

   GMRES minimizes the Euclidean norm of the residual :math:`\|b - A x^{(k)}\|_2` over the Krylov subspace :math:`\mathcal{K}_k(A, r^{(0)}) = \text{span}\{r^{(0)}, A r^{(0)}, \ldots, A^{k-1} r^{(0)}\}`. 
   
   The method employs the Arnoldi process to construct an orthonormal basis :math:`\{v_1, v_2, \ldots, v_k\}` for the Krylov subspace:

   .. math::
   
      \begin{align}
      v_1 &= \frac{r^{(0)}}{\|r^{(0)}\|_2} \\
      h_{ij} &= (A v_j, v_i) \quad \text{for } i = 1, 2, \ldots, j \\
      \tilde{v}_{j+1} &= A v_j - \sum_{i=1}^{j} h_{ij} v_i \\
      h_{j+1,j} &= \|\tilde{v}_{j+1}\|_2 \\
      v_{j+1} &= \frac{\tilde{v}_{j+1}}{h_{j+1,j}}
      \end{align}

   The approximate solution is then computed as:

   .. math::
   
      x^{(k)} = x^{(0)} + V_k y_k

   Where :math:`y_k` minimizes :math:`\|\beta e_1 - \bar{H}_k y\|_2`, with :math:`\beta = \|r^{(0)}\|_2`, :math:`e_1` being the first unit vector, and :math:`\bar{H}_k` the upper Hessenberg matrix formed by the coefficients :math:`h_{ij}`.

Preconditioning
^^^^^^^^^^^^^^^

KokkosBatched offers several preconditioners to accelerate convergence:

1. **JacobiPrec**: Diagonal (Jacobi) preconditioner

   .. math::
   
      M^{-1} = \text{diag}(A)^{-1}

2. **Identity**: Identity preconditioner (no preconditioning)

   .. math::
   
      M^{-1} = I

The preconditioned systems transform the original system :math:`A x = b` into either:

- Left-preconditioned: :math:`M^{-1} A x = M^{-1} b`
- Right-preconditioned: :math:`A M^{-1} y = b`, where :math:`x = M^{-1} y`

The KrylovHandle Infrastructure
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The iterative solvers are orchestrated through the ``KrylovHandle`` class, which provides:

1. Workspace allocation for solver-specific data structures
2. Storage for convergence history
3. Tolerance and iteration control
4. Methods to query solution status

This infrastructure enables efficient and flexible configuration of the iterative solvers.

Implementation Details and Performance Optimization
---------------------------------------------------

Memory Layout Considerations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

KokkosBatched leverages Kokkos' powerful memory layout abstractions to optimize data access patterns. For optimal performance, batch dimensions are typically placed in the leftmost position, allowing for coalesced memory access in GPU implementations:

.. math::

   \text{data}[\text{batch\_index}][\text{row}][\text{column}]

This corresponds to a ``LayoutLeft`` in Kokkos terminology for row-major storage, ensuring efficient memory access patterns on both CPUs and GPUs.

Algorithmic Variants
^^^^^^^^^^^^^^^^^^^^

Many operations in KokkosBatched provide multiple algorithmic variants to accommodate different problem characteristics:

1. **Blocked vs. Unblocked**: For operations like LU factorization, blocked variants exploit cache hierarchy for large matrices, while unblocked variants minimize overhead for very small matrices.

2. **Orthogonalization Strategies**: For GMRES, both Classical and Modified Gram-Schmidt orthogonalization strategies are available, with different stability-performance tradeoffs.

Sparse-Specific Optimizations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Sparse operations employ specialized techniques:

1. **Thread cooperation schemes**: Threads within a team cooperate on rows or non-zero elements depending on matrix characteristics.

2. **Vectorization strategies**: For structured sparsity patterns, vector-level parallelism is exploited.

3. **Load balancing**: Sophisticated work distribution ensures balanced workloads despite irregular sparsity patterns.

Comparative Performance
^^^^^^^^^^^^^^^^^^^^^^^

KokkosBatched demonstrates significant performance advantages over traditional approaches. For example, batched LU factorization of 1000 matrices of size 32×32 can achieve speedups of 5-10× compared to sequential processing, with the advantage growing as batch sizes increase.

The figure below illustrates the typical performance scaling observed with KokkosBatched operations:

.. code-block:: text

   Performance scaling with batch size (schematic):
   
   ^
   |                                                 *
   |                                           *
   |                                     *
   |                               *
   |                         *
   |                   *
   |             *
   |       *
   | *
   +-------------------------------------------------->
     1    10    100   1000  10000 100000  Batch Size

This nonlinear scaling demonstrates the benefits of amortizing kernel launch overhead and improving computational intensity.

Usage Patterns and API Design
-----------------------------

KokkosBatched employs a consistent API design pattern across operations, making it intuitive to use once the basic paradigm is understood.

Execution Model Selection
^^^^^^^^^^^^^^^^^^^^^^^^^

Operations can be invoked with different execution models:

.. code-block:: cpp

   // Serial execution
   KokkosBatched::SerialLU::invoke(A);
   
   // Team execution
   KokkosBatched::TeamLU<MemberType>::invoke(member, A);
   
   // Team-Vector execution
   KokkosBatched::TeamVectorLU<MemberType>::invoke(member, A);

Alternatively, a unified interface with mode selection is provided:

.. code-block:: cpp

   KokkosBatched::LU<MemberType, KokkosBatched::Mode::TeamVector>::invoke(member, A);

Operator Models
^^^^^^^^^^^^^^^

For iterative solvers, KokkosBatched uses an operator model where matrix operations are encapsulated in classes providing an ``apply`` method:

.. code-block:: cpp

   template <typename MemberType, typename ArgMode>
   KOKKOS_INLINE_FUNCTION
   void apply(const MemberType& member,
              const XViewType& X,
              const YViewType& Y) const;

This allows composing complex operations from simpler building blocks, such as creating preconditioned operators.

Advanced Example: Preconditioned GMRES
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A sophisticated example demonstrating the library's capabilities is a preconditioned GMRES solve for multiple systems:

.. code-block:: cpp

   // Create batched CRS matrix
   auto A = KokkosBatched::CrsMatrix<values_type, index_type>(values, row_ptr, col_idx);
   
   // Create Jacobi preconditioner
   auto prec = KokkosBatched::JacobiPrec<diag_type>(diag_values);
   
   // Configure Krylov handle
   auto handle = KokkosBatched::KrylovHandle<norm_type, int_type, view3d_type>(batch_size, n_team);
   handle.set_tolerance(1e-8);
   handle.set_max_iteration(100);
   handle.allocate_workspace(batch_size, n, max_krylov_dim);
   
   // Solve with GMRES
   Kokkos::parallel_for(policy, KOKKOS_LAMBDA(const member_type& member) {
     const int b = member.league_rank();
     auto B_b = Kokkos::subview(B, b, Kokkos::ALL());
     auto X_b = Kokkos::subview(X, b, Kokkos::ALL());
     
     KokkosBatched::GMRES<member_type, KokkosBatched::Mode::TeamVector>
       ::invoke(member, A, B_b, X_b, prec, handle);
   });

This example demonstrates the elegant composition of components to solve complex problems with minimal code.

Applications and Use Cases
--------------------------

KokkosBatched finds applications across numerous domains:

1. **Computational Fluid Dynamics**: Implicit time-stepping schemes require solving many similar systems

2. **Finite Element Analysis**: Element-wise operations on unstructured meshes

3. **Quantum Chemistry**: Electronic structure calculations with many small dense matrices

4. **Uncertainty Quantification**: Ensemble methods requiring solutions for multiple parameter sets

5. **Machine Learning**: Batched operations for mini-batch processing

Each of these domains benefits from both the performance advantages and the simplified programming model that KokkosBatched provides.

Future Directions
-----------------

The development of KokkosBatched can evolve in several directions:

1. **Tensor Contractions**: Extending to higher-dimensional tensors

2. **Mixed Precision**: Leveraging specialized hardware for lower precision arithmetic

3. **Adaptive Algorithms**: Selecting optimal algorithms based on problem characteristics

4. **Sparse Tensor Operations**: Extending the batched paradigm to sparse tensors

5. **Domain-Specific Preconditioners**: Tailored preconditioning strategies for specific applications

Conclusion
----------

KokkosBatched represents a significant advancement in high-performance computing, addressing the critical need for efficient batched operations in scientific computing. By combining mathematical rigor with performance-oriented design, the library enables applications to achieve unprecedented performance on modern manycore architectures without sacrificing portability.

The library's comprehensive coverage of dense and sparse operations, sophisticated linear solvers, and flexible preconditioning options make it an invaluable tool for computational scientists and engineers dealing with multiple small-to-medium sized problems.

Through its integration with the Kokkos ecosystem, KokkosBatched ensures that applications can seamlessly leverage current and future high-performance computing architectures, providing a sustainable path forward for performance-critical scientific applications.

