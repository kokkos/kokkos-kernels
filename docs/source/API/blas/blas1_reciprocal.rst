KokkosBlas::reciprocal
######################

Defined in header: :code:`KokkosBlas1_reciprocal.hpp`

.. code:: c++

  template <class execution_space, class RMV, class XMV>
  void reciprocal(const execution_space& space, const RMV& R, const XMV& X);

  template <class RMV, class XMV>
  void reciprocal(const RMV& R, const XMV& X);

Computes the square of the nrm2 of vector(s) `X` weighted by vector(s) `W` and store the result(s) in `R`, if provided, otherwise return it.

.. math::

   R(i, j) = \frac{1.0}{X(i, j)}

1. compute the inverse of each entry in ``X`` and store in the corresponding entry of ``R`` using the ``space`` instance
2. compute the inverse of each entry in ``X`` and store in the corresponding entry of ``R`` using the default instance of ``typename XVector::execution_space``

Parameters
==========

:space: execution space instance

:R: computed norm(s)

:X: input vector(s) for which we compute the norm

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

.. literalinclude:: ../../../../example/docs/blas/KokkosBlas1_docs_reciprocal.cpp
  :language: c++

output:

.. code::

   orignal values:
      x: {0.5, -1, -2, 3, -4}

   final values:
      y: {2, -1, -0.5, 0.333333, -0.25}
