API: Batched
============

.. toctree::
    :maxdepth: 2
    :hidden:

    batched/dense-index

.. toctree::
    :maxdepth: 2
    :hidden:

    batched/sparse-index

Overview
--------

The Kokkos batched interface provides multiple functor-level interfaces for dense linear algebra (DLA) or sparse linear algebra,
which correspond to Kokkos hierarchical parallelism. Unlike other batched BLAS and LAPACK interfaces, we do not provide a front-level (or subroutine) interface that launches a streaming parallel kernel.
Instead, we provide a functor-level interface that can be used in Kokkos parallel expressions (e.g., parallel for, reduce and scan).
The advantage of this approach is that a user can compose various batched dense linear algebra kernels, exploiting temporal locality via the functor-level interfaces.
For example, consider a case where small element matrices are created via dgemm and those matrices are triangular solved by lu and trsv.
An approach using conventional batched BLAS interfaces would be like the following pseudo code:

.. code-block:: c++

  int N = 1000; /// # of batch items
  int m = 8;    /// a square matrix size

  Kokkos::View<double***> AA("AA", N, m, m); /// element matrices
  Kokkos::View<double**>  LL("LL", N, m);    /// left basis vectors
  Kokkos::View<double**>  RR("RR", N, m);    /// right basis vectors
  Kokkos::View<double**>  BB("BB", N, m);    /// load vector and would be overwritten by a solution

  batched_dgemm(LL, RR, AA);     /// construct element matrices via batched dgemm
  batched_dgetrf(AA);            /// perform lu decomposition of each instance of A matrix array
  batched_dtrsv("Lower", AA, BB) /// perform forward substitution
  batched_dtrsv("Upper", AA, BB) /// perform backward substitution

Clearly, a performance problem of the above code comes from the fact that the sequence of batched functions does not exploit temporal locality between DLA functions;
each batched function sweeps the entire set of batched matrices or vectors in parallel for a single DLA operation.
On the other hand, Kokkos batched APIs provide functor-level interfaces so that a user can compose a new batched function.
The following example has the same functionality as the above:

.. code-block:: c++

  int N = 1000; /// # of batch items
  int m = 8;    /// a square matrix size

  Kokkos::View<double***> AA("AA", N, m, m); /// element matrices
  Kokkos::View<double**>  LL("LL", N, m);    /// left basis vectors
  Kokkos::View<double**>  RR("RR", N, m);    /// right basis vectors
  Kokkos::View<double**>  BB("BB", N, m);    /// load vector and would be overwritten by a solution

  using namespace KokkosBatched;
  Kokkos::parallel_for(N, KOKKOS_LAMBDA(const int i) {
    auto A = Kokkos::subview(AA, i, Kokkos::ALL(), Kokkos::ALL()); /// ith matrix
    auto L = Kokkos::subview(LL, i, Kokkos::ALL());                /// ith left vector
    auto R = Kokkos::subview(RR, i, Kokkos::ALL());                /// ith right vector
    auto B = Kokkos::subview(BB, i, Kokkos::ALL());                /// ith load/solution vector

    SerialGemm<Trans::NoTranspose,Trans::NoTranspose,Algo::Gemm::Unblocked>
      ::invoke(one, L, R, zero, A);
    SerialLU<Algo::LU::Unblocked>
      ::invoke(A);
    SerialTrsv<Uplo::Lower,Trans::NoTranspose,Diag::UnitDiag,Algo::Trsv::Unblocked>
      ::invoke(one, A, B);
  });

In this example, a single `parallel_for` is launched to compute a sequence of DLA operations i.e., `gemm`, `lu` and `trsv`.
Then, one may ask, "How is this different from putting BLAS and LAPACK functions inside a parallel loop ?".
The main difference is that Kokkos batched APIs are very lightweight generic implementations focusing on small matrix sizes (kernels are developed and tuned from the application context).
Most vendor provided DLA libraries such as Intel MKL and NVIDIA CUBLAS perform well for large problem sizes. For tiny problem sizes like 3x3 or 5x5 matrix problems,
it is not feasible to use vendor optimized DLA libraries as even error checking already puts a quite amount of overhead for such tiny problem sizes.
Furthermore, CUBLAS (or GPU numeric libraries) cannot be nested inside of a parallel region, which is why CUBLAS provides separate batched APIs.
Kokkos batched APIs provide generic header-only functor-level implementations that can be embedded in a parallel region.
The APIs can be mapped to Kokkos hierarchical parallelism and also provide various implementations of algorithms with a template argument,
which allows users to choose (or customize) the batched routines for their application context.
