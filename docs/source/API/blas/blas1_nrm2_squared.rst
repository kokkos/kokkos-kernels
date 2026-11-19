KokkosBlas::nrm2_squared
########################

Defined in header: :code:`KokkosBlas1_nrm2_squared.hpp`

.. code:: c++

  template <class execution_space, class XVector>
    requires Kokkos::is_execution_space_v<execution_space>
  typename KokkosKernels::Details::InnerProductSpaceTraits<typename XVector::non_const_value_type>::mag_type
  nrm2_squared(const execution_space& space, const XVector& x);

  template <class XVector>
  typename KokkosKernels::Details::InnerProductSpaceTraits<typename XVector::non_const_value_type>::mag_type
  nrm2_squared(const XVector& x);

  template <class execution_space, class RV, class XMV>
  void nrm2_squared(const execution_space& space, const RV& R, const XMV& X);

  template <class RV, class XMV>
    requires Kokkos::is_view_v<RV>
  void nrm2_squared(const RV& R, const XMV& X);

Computes the square of the norm 2 of the vector(s) stored column wise in X.

1. accumulates the square of the values in ``X`` and fences the ``space`` instance
2. accumulates the square of the values in ``X`` and fences the default instance of ``typename XVector::execution_space``
3. accumulates the square of the values in each column of ``X`` into ``R`` and fences the ``space`` instance
4. accumulates the square of the values in each column of ``X`` into ``R`` and fences the default instance of ``typename XVector::execution_space``

The function will throw a runtime exception if ``X.extent(1) != R.extent(0)``

The result (returned or stored value) is undefined if ``X`` has no entries.

Parameters
==========

:space: execution space instance

:X: vector(s) to compute the norm2

:R: computed norm(s)

Type Requirements
=================

- `execution_space` must be a Kokkos `execution space <https://kokkos.org/kokkos-core-wiki/API/core/execution_spaces.html>`_

- `XVector` must be a Kokkos `View <https://kokkos.org/kokkos-core-wiki/API/core/view/view.html>`_ of rank 1 that satisfies

  - ``Kokkos::SpaceAccessibility<execution_space, typename XVector::memory_space>::accessible == true``

- `XMV` must be a Kokkos `View <https://kokkos.org/kokkos-core-wiki/API/core/view/view.html>`_ of rank 1 or 2 that satisfies

  - ``Kokkos::SpaceAccessibility<execution_space, typename XMV::memory_space>::accessible == true``

- `RV` must be a Kokkos `View <https://kokkos.org/kokkos-core-wiki/API/core/view/view.html>`_ that satisfies

  - ``RV::rank == XMV::rank - 1``
  - ``std::is_same_v<typename RV::value_type, typename RV::non_const_value_type> == true``
  - ``std::is_same_v<typename RV::value_type, typename KokkosKernels::Details::InnerProductSpaceTraits<typename XMV::non_const_value_type>::mag_type> == true``


Example
=======

.. literalinclude:: ../../../../example/docs/blas/KokkosBlas1_docs_nrm2_squared.cpp
  :language: c++

output:

.. code::

   X_nrm: 900 Expected: 900
