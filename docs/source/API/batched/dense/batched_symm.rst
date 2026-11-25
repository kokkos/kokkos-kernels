KokkosBatched::Symm
###################

Defined in header: :code:`KokkosBatched_Symm.hpp`

.. code:: c++

    template <typename ArgSide, typename ArgUplo, bool Conj>
    struct SerialSymm {
      template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const ScalarType alpha, const AViewType &A, const BViewType &B, const ScalarType beta,
                                               const CViewType &C);
    };

    template <typename MemberType, typename ArgSide, typename ArgUplo, bool Conj>
    struct TeamSymm {
      template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                               const BViewType &B, const ScalarType beta, const CViewType &C);
    };

    template <typename MemberType, typename ArgSide, typename ArgUplo, bool Conj>
    struct TeamVectorSymm {
      template <typename ScalarType, typename AViewType, typename BViewType, typename CViewType>
      KOKKOS_INLINE_FUNCTION static int invoke(const MemberType &member, const ScalarType alpha, const AViewType &A,
                                               const BViewType &B, const ScalarType beta, const CViewType &C);
    };

Performs one of the matrix-matrix multiplication operations where :math:`A` is a symmetric (or hermitian) matrix and :math:`B` and :math:`C` are m by n matrices.

.. math::

   \begin{align}
   C &= \alpha A * B + \beta C \: \text{(if ArgSide == KokkosBatched::Side::Left)} \\
   C &= \alpha B * A + \beta C \: \text{(if ArgSide == KokkosBatched::Side::Right)}
   \end{align}

1. If ``Conj == false``, this operation is equivalent to the BLAS routine `SSYMM <https://www.netlib.org/blas/ssymm.f>`_ (`CSYMM <https://www.netlib.org/blas/csymm.f>`_) or `DSYMM <https://www.netlib.org/blas/dsymm.f>`_ (`ZSYMM <https://www.netlib.org/blas/zsymm.f>`_) for single or double precision for real (complex) symmetric matrix.

2. If ``Conj == true``,  this operation is equivalent to the BLAS routine `CHEMM <https://www.netlib.org/blas/chemm.f>`_ or `ZHEMM <https://www.netlib.org/blas/zhemm.f>`_ for single or double precision for complex hermitian matrix.

.. note::

   Only the upper or lower triangular part of A is referenced.
   Then, we compute C := alpha * A * B + beta * C or C := alpha * B * A + beta * C as if A were symmetric.

Parameters
==========

:alpha: Scaling factor
:A: :math:`A` is a symmetric (or hermitian) matrix. m by m if ``KokkosBatched::Side::Left`` or n by n if ``KokkosBatched::Side::Right``.
:B: :math:`B` is an m by n matrix.
:beta: Scaling factor
:C: :math:`C` is a m by n matrix.

Type Requirements
-----------------

- ``MemberType`` must be a Kokkos team member handle (only for ``TeamSymm`` and ``TeamVectorSymm``)
- ``ArgSide`` must be one of the following:
   - ``KokkosBatched::Side::Left`` :math:`A` appears on the left
   - ``KokkosBatched::Side::Right`` :math:`A` appears on the right
- ``ArgUplo`` must be one of the following:
   - ``KokkosBatched::Uplo::Upper`` to update the upper triangular part of :math:`A`
   - ``KokkosBatched::Uplo::Lower`` to update the lower triangular part of :math:`A`
- ``Conj`` must be a boolean indicating whether to perform the operation with the conjugate transpose of :math:`A` (for hermitian matrices) or not (for symmetric matrices)
- ``ScalarType`` must be a built-in floating point type (``float``, ``double``, ``Kokkos::complex<float>``, ``Kokkos::complex<double>``)
- ``AViewType`` must be a Kokkos `View <https://kokkos.org/kokkos-core-wiki/API/core/view/view.html>`_ of rank 2 containing a matrix :math:`A`
- ``BViewType`` must be a Kokkos `View <https://kokkos.org/kokkos-core-wiki/API/core/view/view.html>`_ of rank 2 containing a matrix :math:`B`
- ``CViewType`` must be a Kokkos `View <https://kokkos.org/kokkos-core-wiki/API/core/view/view.html>`_ of rank 2 containing a matrix :math:`C` that satisfies ``!std::is_const_v<typename CViewType::value_type>``

Example
=======

.. literalinclude:: ../../../../../example/batched_solve/serial_symm.cpp
  :language: c++
  :linenos:
  :lines: 4-

output:

.. code::

   symm works correctly!
