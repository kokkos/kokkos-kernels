KokkosBlas::nrm2w_squared
#########################

Defined in header: :code:`KokkosBlas1_nrm2w_squared.hpp`

.. code:: c++

  template <class execution_space, class XVector>
    requires Kokkos::is_execution_space_v<execution_space>
  typename KokkosKernels::Details::InnerProductSpaceTraits<typename XVector::non_const_value_type>::mag_type
  nrm2w(const execution_space& space, const XVector& x, const XVector& w);

  template <class XVector>
  typename KokkosKernels::Details::InnerProductSpaceTraits<typename XVector::non_const_value_type>::mag_type
  nrm2w(const XVector& x, const XVector& w);

  template <class execution_space, class RV, class XMV>
  void nrm2w(const execution_space& space, const RV& R, const XMV& X, const XMV& W);

  template <class RV, class XMV>
    requires Kokkos::is_view_v<RV>
  void nrm2w(const RV& R, const XMV& X, const XMV& W);

Computes the square of the nrm2 of vector(s) `X` weighted by vector(s) `W` and store the result(s) in `R`, if provided, otherwise return it.

.. math::

   R(j) = \sum_i \left(\frac{||X(i,j)||}{||W(i, j)||}\right)^2

1. accumulates the square of the values in ``X`` scaled by the values in ``W`` and fences the ``space`` instance
2. accumulates the square of the values in ``X`` scaled by the values in ``W`` and fences the default instance of ``typename XVector::execution_space``
3. accumulates the square of the values in each column of ``X`` scaled by the values in ``W`` and store it into ``R``, fences the ``space`` instance
4. accumulates the square of the values in each column of ``X`` scaled by the values in ``W`` and store it into ``R``and fences the default instance of ``typename XVector::execution_space``

The function will throw a runtime exception if ``X.extent(1) != R.extent(0)``

The result (returned or stored value) is undefined if ``X`` has no entries.

Parameters
==========

:space: execution space instance

:R: computed norm(s)

:X: input vector(s) for which we compute the norm

:W: input weight vector(s) to scale the values of `X`

Type Requirements
=================

- `execution_space` must be a Kokkos `execution space <https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html>`_

- `XVector` must be a Kokkos `View <https://kokkos.org/kokkos-core-wiki/API/core/view/view.html>`_ of rank 1 that satisfies

  - ``Kokkos::SpaceAccessibility<execution_space, typename XVector::memory_space>::accessible``

- `XMV` must be a Kokkos `View <https://kokkos.org/kokkos-core-wiki/API/core/view/view.html>`_ of rank 1 or 2 that satisfies

  - ``Kokkos::SpaceAccessibility<execution_space, typename XMV::memory_space>::accessible``

- `RV` must be a Kokkos `View <https://kokkos.org/kokkos-core-wiki/API/core/view/view.html>`_ that satisfies

  - ``RV::rank == XMV::rank - 1``
  - ``!std::is_const_v<typename RV::value_type>``
  - ``std::is_same_v<typename RV::value_type, typename KokkosKernels::Details::InnerProductSpaceTraits<typename XMV::non_const_value_type>::mag_type>``
  - ``RV::memory_space == Kokkos::HostSpace``

Example
=======

.. literalinclude:: ../../../../example/docs/blas/KokkosBlas1_docs_nrm2w_squared.cpp
  :language: c++

output:

.. code::

   X_nrm: 225 Expected: 225
