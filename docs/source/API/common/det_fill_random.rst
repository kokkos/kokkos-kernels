KokkosKernels::Impl::det_fill_random
####################################

Defined in header: :code:`KokkosKernels_Utils.hpp`

.. code:: c++

  template <class ViewT>
  void det_fill_random(ViewT view, uint64_t seed,
                       typename ViewT::non_const_value_type min,
                       typename ViewT::non_const_value_type max);

  template <class ViewT>
  void det_fill_random(ViewT view, uint64_t seed,
                       typename ViewT::non_const_value_type max);

Deterministically fills a view with random values within a specified range. The randomness is deterministic and reproducible by using the same seed value.

The first overload fills the view with random values in the range ``[min, max)``.
The second overload is a convenience function that fills the view with random values in the range ``[0, max)``.

The function supports views of arbitrary rank and handles both contiguous and padded subviews
(e.g., row or column slices of a multi-dimensional array).

Parameters
==========

:view: A Kokkos view of arbitrary rank to be filled with random values. The view must have
       a contiguous layout (LayoutLeft or LayoutRight), but may represent a padded subview.

:seed: The seed value for the random number generator. The same seed will produce identical
       random values across different calls.

:min: (First overload) The minimum value (inclusive) for generated random numbers.

:max: The maximum value (exclusive) for generated random numbers.

Type Requirements
-----------------

- ``ViewT`` must be a Kokkos ``View`` with a contiguous array layout (LayoutLeft or LayoutRight).
  LayoutStride views are not supported.

- ``ViewT::non_const_value_type`` must be a scalar type that can be used with Kokkos' ``fill_random`` function.

Notes
=====

- The function uses ``Kokkos::Random_XorShift64_Pool<Kokkos::Serial>`` for deterministic random number generation.

- For views with non-contiguous memory layout (padded subviews), the function automatically
  creates a temporary contiguous view, fills it with random values, and copies the values
  back to the original view, preserving the layout.

- Contiguous views are handled more efficiently with direct memory access.

Example
=======

.. code:: c++

  #include <Kokkos_Core.hpp>
  #include <KokkosKernels_Utils.hpp>

  void example() {
    // Fill a 1D view with random values in [0, 10)
    Kokkos::View<double*> v1("v1", 100);
    KokkosKernels::Impl::det_fill_random(v1, 12345, 10.0);

    // Fill a 2D view with random values in [-1, 1)
    Kokkos::View<double**> v2("v2", 50, 40);
    KokkosKernels::Impl::det_fill_random(v2, 12345, -1.0, 1.0);

    // Fill a 3D view with random values in [0, 1)
    Kokkos::View<double***> v3("v3", 10, 20, 30);
    KokkosKernels::Impl::det_fill_random(v3, 54321, 0.0, 1.0);
  }
